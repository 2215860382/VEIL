"""Stage-1 experiment: N samples per MLVU task × 3 pipelines → accuracy table.

Resumable. Writes per-(sample, pipeline) records to outputs/results/mlvu/stage1.jsonl
as soon as they finish, so a crash/reboot at most loses one in-flight inference.
On rerun, already-recorded (sample × pipeline) combos are skipped.

All pipelines share the SAME pre-sampled frame set (fps / max_frames / resolution from
frame_sampling config), ensuring fair comparison.

Usage:
    PYTHONPATH=. python scripts/stage1.py --per-task 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_mlvu import load_mlvu, unique_videos
from eval.compute_accuracy import compute_accuracy
from eval.parse_answer import candidate_text_for_letter, parse_letter
from memory.build_memory import build_memory_bank, memory_bank_path
from memory.sample_frames import SampledVideo, sample_frames
from memory.schema import MemoryBank
from models.embedder import BGEM3Embedder
from models.llm_client import LLMClient
from models.reranker import BGEReranker
from models.vlm_client import VLMClient
from pipelines.direct_video_qa import run_direct_video_qa
from pipelines.naive_rag import run_naive_rag
from pipelines.veil_iterative import run_veil
from reasoning.answerer import VLAnswerer
from reasoning.planner import Planner
from reasoning.verifier import Verifier
from utils.config import load_config
from utils.logging import get_logger

log = get_logger("stage1")

PIPELINES = ("direct", "naive_rag", "veil")


def sample_key(s, pipeline) -> str:
    return f"{s.question_type}|{s.video_id}|{s.sample_idx}|{pipeline}"


def load_done_keys(jsonl_path: Path) -> set[str]:
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.open():
            try:
                r = json.loads(line)
                done.add(r["key"])
            except Exception:
                pass
    return done


def append_record(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mlvu.yaml")
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--vlm-gpu", default="cuda:0")
    ap.add_argument("--llm-gpu", default="cuda:4")
    ap.add_argument("--bge-gpu", default="cuda:2")
    ap.add_argument("--out", default="outputs/results/mlvu/stage1.jsonl")
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    ap.add_argument("--task-types", nargs="*", default=None)
    ap.add_argument("--min-duration", type=float, default=None,
                    help="Only include videos longer than this many seconds.")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="Exclude videos longer than this many seconds.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = cfg["benchmark"]
    task_types = args.task_types or bench["task_types"]
    fs_cfg = cfg["frame_sampling"]          # fps / max_frames / resolution
    mem_cfg = cfg["memory"]                  # chunk_size / stride
    dv_cfg = cfg.get("direct_video_qa", {}) # max_new_tokens

    # Sample selection.
    all_samples = []
    for task in task_types:
        ss = load_mlvu(
            bench["json_dir"], bench["video_dir"], bench["json_files"],
            task_types=[task],
        )
        if args.min_duration is not None:
            ss = [s for s in ss if s.duration >= args.min_duration]
        if args.max_duration is not None:
            ss = [s for s in ss if s.duration <= args.max_duration]
        if args.min_duration is not None or args.max_duration is not None:
            ss.sort(key=lambda s: s.duration, reverse=True)
        all_samples.extend(ss[: args.per_task])
    log.info("selected %d samples across %d tasks", len(all_samples), len(task_types))

    # Resume.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(out_path)
    log.info("resume: %d records already on disk", len(done_keys))

    # Plan work.
    todo = []
    for s in all_samples:
        for p in args.pipelines:
            if sample_key(s, p) not in done_keys:
                todo.append((s, p))
    if not todo:
        log.info("nothing to do — all combos already recorded")
        report(out_path, all_samples, args.pipelines, task_types)
        return
    log.info("todo: %d (sample, pipeline) combos", len(todo))

    needs_vlm = any(p in args.pipelines for p in ("direct", "naive_rag", "veil"))
    needs_llm = "veil" in args.pipelines
    needs_retr = any(p in args.pipelines for p in ("naive_rag", "veil"))
    cache_dir = Path(mem_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- Stage A: VLM + memory bank build -----
    vlm = None
    videos = unique_videos(all_samples)
    missing_banks = [s for s in videos if not memory_bank_path(cache_dir, s.video_id).exists()]
    log.info("memory banks: %d videos, %d missing", len(videos), len(missing_banks))

    if needs_vlm or missing_banks:
        cfg["models"]["vlm"]["device"] = args.vlm_gpu
        log.info("loading VLM on %s ...", args.vlm_gpu)
        t0 = time.time()
        vlm = VLMClient(**cfg["models"]["vlm"])
        log.info("  VLM ready (%.1fs)", time.time() - t0)

    # Lazy frame cache: keyed by video_id.
    _frame_cache: dict[str, SampledVideo] = {}

    def get_frames(video_path: str, video_id: str) -> SampledVideo:
        if video_id not in _frame_cache:
            log.info("sampling frames: %s", video_id)
            _frame_cache[video_id] = sample_frames(
                video_path,
                fps=fs_cfg["fps"],
                max_frames=fs_cfg["max_frames"],
                resolution=fs_cfg["resolution"],
            )
            sv = _frame_cache[video_id]
            log.info("  %d frames, %.1fs duration", len(sv.frames), sv.duration)
        return _frame_cache[video_id]

    # Build banks for missing videos.
    for sv in missing_banks:
        log.info("building memory: %s", sv.video_id)
        t0 = time.time()
        try:
            sampled = get_frames(sv.video_path, sv.video_id)
            bank = build_memory_bank(
                sampled=sampled,
                video_id=sv.video_id,
                vlm=vlm,
                chunk_size=mem_cfg.get("chunk_size", 8),
                stride=mem_cfg.get("stride", 4),
                progress=False,
            )
            bank.save(memory_bank_path(cache_dir, sv.video_id))
            log.info("  %d chunks in %.1fs", len(bank.chunks), time.time() - t0)
        except Exception as e:
            log.exception("memory build failed for %s: %s", sv.video_id, e)

    # ----- Stage B: load retrieval + LLM if needed -----
    embedder = reranker = llm = None
    planner = verifier = vl_answerer = None
    if needs_retr:
        cfg["models"]["embedder"]["device"] = args.bge_gpu
        cfg["models"]["reranker"]["device"] = args.bge_gpu
        log.info("loading BGE embedder + reranker on %s ...", args.bge_gpu)
        t0 = time.time()
        embedder = BGEM3Embedder(**cfg["models"]["embedder"])
        reranker = BGEReranker(**cfg["models"]["reranker"])
        log.info("  BGE ready (%.1fs)", time.time() - t0)
    if needs_llm:
        cfg["models"]["llm"]["device"] = args.llm_gpu
        log.info("loading LLM on %s ...", args.llm_gpu)
        t0 = time.time()
        llm = LLMClient(**cfg["models"]["llm"])
        log.info("  LLM ready (%.1fs)", time.time() - t0)
        planner = Planner(llm); verifier = Verifier(llm)
    if vlm is not None:
        vl_answerer = VLAnswerer(vlm)

    # ----- Stage C: run pipelines -----
    bank_cache: dict[str, MemoryBank] = {}

    def get_bank(vid: str) -> MemoryBank | None:
        if vid in bank_cache:
            return bank_cache[vid]
        p = memory_bank_path(cache_dir, vid)
        if not p.exists():
            return None
        bank_cache[vid] = MemoryBank.load(p)
        return bank_cache[vid]

    for idx, (s, pipeline) in enumerate(todo):
        key = sample_key(s, pipeline)
        log.info("[%d/%d] %s", idx + 1, len(todo), key)
        t0 = time.time()
        out, err = {}, None
        try:
            if pipeline == "direct":
                sampled = get_frames(s.video_path, s.video_id)
                out = run_direct_video_qa(
                    frames=sampled.frames,
                    question=s.question,
                    candidates=s.candidates,
                    vlm=vlm,
                    max_new_tokens=dv_cfg.get("max_new_tokens", 192),
                )
            else:
                bank = get_bank(s.video_id)
                if bank is None:
                    raise RuntimeError(f"missing memory bank for {s.video_id}")
                if pipeline == "naive_rag":
                    out = run_naive_rag(
                        s.question, s.candidates, bank, embedder, reranker, vl_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                    )
                else:  # veil
                    out = run_veil(
                        s.question, s.candidates, bank, planner, verifier,
                        embedder, reranker, vl_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                        max_iter=cfg["veil_loop"]["max_iter"],
                    )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.exception("pipeline failed: %s", err)

        raw = out.get("raw") or json.dumps({k: v for k, v in out.items() if k not in ("evidence_texts", "evidence_chunk_ids", "trace")})
        letter = parse_letter(raw, len(s.candidates))
        pred_text = candidate_text_for_letter(letter, s.candidates)
        record = {
            "key": key,
            "question_type": s.question_type,
            "video_id": s.video_id,
            "sample_idx": s.sample_idx,
            "pipeline": pipeline,
            "question": s.question,
            "candidates": s.candidates,
            "gold": s.answer,
            "pred_letter": letter,
            "pred_text": pred_text,
            "correct": pred_text == s.answer,
            "evidence_chunk_ids": out.get("evidence_chunk_ids", []),
            "trace_iters": len(out.get("trace", {}).get("iterations", [])) if pipeline == "veil" else None,
            "raw": (raw or "")[:1500],
            "elapsed": time.time() - t0,
            "error": err,
        }
        append_record(out_path, record)

    log.info("=== stage1 complete ===")
    report(out_path, all_samples, args.pipelines, task_types)


def report(jsonl_path: Path, samples: list, pipelines: list, task_types: list) -> None:
    """Read jsonl and print accuracy table per (task, pipeline)."""
    by = defaultdict(list)
    if not jsonl_path.exists():
        log.warning("no records to report")
        return
    for line in jsonl_path.open():
        try:
            r = json.loads(line)
            by[(r["question_type"], r["pipeline"])].append(r)
        except Exception:
            continue

    log.info("\n=== ACCURACY (per task × pipeline) ===")
    header = f"{'task':<20s} | " + " | ".join(f"{p:>12s}" for p in pipelines)
    print(header)
    print("-" * len(header))
    overall = {p: [0, 0] for p in pipelines}
    for task in task_types:
        cells = []
        for p in pipelines:
            rs = by.get((task, p), [])
            n = len(rs)
            c = sum(1 for r in rs if r.get("correct"))
            overall[p][0] += c; overall[p][1] += n
            cells.append(f"{c}/{n} ({100*c/n if n else 0:5.1f}%)")
        print(f"{task:<20s} | " + " | ".join(f"{c:>12s}" for c in cells))
    print("-" * len(header))
    cells = [f"{c}/{n} ({100*c/n if n else 0:5.1f}%)" for c, n in overall.values()]
    print(f"{'OVERALL':<20s} | " + " | ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
