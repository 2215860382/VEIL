"""Unified experiment runner: 5 pipelines × {MLVU, Video-MME, LongVideoBench}.

All pipelines share the same pre-sampled frame pool (fps/max_frames/resolution
from frame_sampling config). Memory banks are built once per video and cached.

Resumable: each (benchmark|video_id|sample_idx|pipeline) record is written to
disk immediately — a crash loses at most one in-flight inference.

Usage:
    cd /home2/ycj/Project/VEIL

    # MLVU, 80 samples, all 5 pipelines
    PYTHONPATH=. python scripts/run_experiments.py \\
        --config configs/mlvu.yaml --n-samples 80 \\
        --vlm-gpu cuda:0 --llm-gpu cuda:4 --bge-gpu cuda:2

    # Video-MME, 100 samples
    PYTHONPATH=. python scripts/run_experiments.py \\
        --config configs/videomme.yaml --n-samples 100

    # LongVideoBench, 120 samples
    PYTHONPATH=. python scripts/run_experiments.py \\
        --config configs/longvideobench.yaml --n-samples 120
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.build_memory import build_memory_bank, memory_bank_path
from memory.sample_frames import SampledVideo, sample_frames
from memory.schema import MemoryBank
from models.embedder import BGEM3Embedder
from models.llm_client import LLMClient
from models.reranker import BGEReranker
from models.vlm_client import VLMClient
from pipelines.bge_coarse_rag import run_bge_coarse_rag
from pipelines.direct_video_qa import run_direct_video_qa
from pipelines.naive_rag import run_naive_rag
from pipelines.uniform_memory import run_uniform_memory
from pipelines.veil_iterative import run_veil
from reasoning.answerer import TextAnswerer
from reasoning.planner import Planner
from reasoning.verifier import Verifier
from eval.parse_answer import candidate_text_for_letter, parse_letter
from utils.config import load_config
from utils.logging import get_logger

log = get_logger("run_experiments")

ALL_PIPELINES = ("direct", "uniform_memory", "bge_coarse", "bge_rerank", "veil")


# ── Sample loading ─────────────────────────────────────────────────────────────

def _load_mlvu(cfg, n: int, seed: int):
    from data.load_mlvu import load_mlvu
    bench = cfg["benchmark"]
    task_types = bench.get("task_types") or []
    if not task_types:
        raise ValueError("mlvu config missing task_types")
    n_tasks = len(task_types)
    base = n // n_tasks
    extra = n % n_tasks
    samples = []
    for i, task in enumerate(task_types):
        per = base + (1 if i < extra else 0)
        ss = load_mlvu(
            bench["json_dir"], bench["video_dir"], bench["json_files"],
            task_types=[task],
        )
        random.seed(seed + i)
        random.shuffle(ss)
        samples.extend(ss[:per])
    return samples


def _load_videomme(cfg, n: int, seed: int):
    from data.load_videomme import load_videomme
    bench = cfg["benchmark"]
    all_s = load_videomme(
        bench["parquet_path"], bench["video_dir"],
        duration_groups=bench.get("duration_groups"),
    )
    random.seed(seed)
    random.shuffle(all_s)
    return all_s[:n]


def _load_longvideobench(cfg, n: int, seed: int):
    from data.load_longvideobench import load_longvideobench
    bench = cfg["benchmark"]
    all_s = load_longvideobench(
        bench["json_path"], bench["video_dir"],
        duration_groups=bench.get("duration_groups"),
    )
    random.seed(seed)
    random.shuffle(all_s)
    return all_s[:n]


def load_benchmark_samples(cfg, n: int, seed: int) -> list:
    name = cfg["benchmark"]["name"]
    if name == "mlvu":
        return _load_mlvu(cfg, n, seed)
    if name == "videomme":
        return _load_videomme(cfg, n, seed)
    if name == "longvideobench":
        return _load_longvideobench(cfg, n, seed)
    raise ValueError(f"Unknown benchmark: {name}")


def sample_video_id(s) -> str:
    return getattr(s, "video_id", str(s.sample_idx))


def sample_video_path(s) -> str:
    return s.video_path


def sample_question_type(s) -> str:
    return getattr(s, "question_type", getattr(s, "question_category", "default"))


def unique_videos(samples) -> list:
    seen = {}
    for s in samples:
        vid = sample_video_id(s)
        if vid not in seen:
            seen[vid] = s
    return list(seen.values())


# ── Resumability ───────────────────────────────────────────────────────────────

def sample_key(benchmark: str, s, pipeline: str) -> str:
    return f"{benchmark}|{sample_video_id(s)}|{s.sample_idx}|{pipeline}"


def load_done_keys(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.open():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    return done


def append_record(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-samples", type=int, default=80)
    ap.add_argument("--pipelines", nargs="+", default=list(ALL_PIPELINES))
    ap.add_argument("--out", default=None,
                    help="Output jsonl path. Defaults to outputs/results/{benchmark}/experiments.jsonl")
    ap.add_argument("--vlm-gpu", default="cuda:0")
    ap.add_argument("--llm-gpu", default="cuda:4")
    ap.add_argument("--bge-gpu", default="cuda:2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    benchmark = cfg["benchmark"]["name"]
    fs_cfg = cfg["frame_sampling"]
    mem_cfg = cfg["memory"]
    dv_cfg = cfg.get("direct_video_qa", {})

    out_path = Path(args.out) if args.out else \
        Path(cfg["eval"]["output_dir"]) / "experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample selection.
    log.info("loading %d samples from %s ...", args.n_samples, benchmark)
    all_samples = load_benchmark_samples(cfg, args.n_samples, args.seed)
    log.info("  %d samples loaded", len(all_samples))

    # Resume.
    done_keys = load_done_keys(out_path)
    log.info("resume: %d records already on disk", len(done_keys))
    todo = [(s, p) for s in all_samples for p in args.pipelines
            if sample_key(benchmark, s, p) not in done_keys]
    if not todo:
        log.info("nothing to do")
        report(out_path, all_samples, args.pipelines, benchmark)
        return
    log.info("todo: %d (sample, pipeline) combos", len(todo))

    needs_vlm  = any(p in args.pipelines for p in ("direct",))
    needs_retr = any(p in args.pipelines for p in ("bge_coarse", "bge_rerank", "veil"))
    needs_llm  = any(p in args.pipelines for p in ("uniform_memory", "bge_coarse", "bge_rerank", "veil"))
    needs_veil = "veil" in args.pipelines

    cache_dir = Path(mem_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: VLM + memory bank build.
    vlm = None
    videos = unique_videos(all_samples)
    missing_banks = [s for s in videos if not memory_bank_path(cache_dir, sample_video_id(s)).exists()]
    log.info("memory banks: %d unique videos, %d missing", len(videos), len(missing_banks))

    if missing_banks or needs_vlm:
        cfg["models"]["vlm"]["device"] = args.vlm_gpu
        log.info("loading VLM on %s ...", args.vlm_gpu)
        t0 = time.time()
        vlm = VLMClient(**cfg["models"]["vlm"])
        log.info("  VLM ready (%.1fs)", time.time() - t0)

    _frame_cache: dict[str, SampledVideo] = {}

    def get_frames(video_path: str, video_id: str) -> SampledVideo:
        if video_id not in _frame_cache:
            log.info("sampling frames: %s", video_id)
            sv = sample_frames(video_path, fps=fs_cfg["fps"],
                               max_frames=fs_cfg["max_frames"],
                               resolution=fs_cfg["resolution"])
            log.info("  %d frames, %.1fs", len(sv.frames), sv.duration)
            _frame_cache[video_id] = sv
        return _frame_cache[video_id]

    for sv in missing_banks:
        vid = sample_video_id(sv)
        vpath = sample_video_path(sv)
        log.info("building memory: %s", vid)
        t0 = time.time()
        try:
            sampled = get_frames(vpath, vid)
            bank = build_memory_bank(
                sampled=sampled, video_id=vid, vlm=vlm,
                chunk_size=mem_cfg.get("chunk_size", 8),
                stride=mem_cfg.get("stride", 4),
                progress=False,
            )
            bank.save(memory_bank_path(cache_dir, vid))
            log.info("  %d chunks in %.1fs", len(bank.chunks), time.time() - t0)
        except Exception as e:
            log.exception("memory build failed for %s: %s", vid, e)

    # Stage B: load retrieval + LLM.
    embedder = reranker = None
    llm = text_answerer = None
    planner = verifier = None

    if needs_retr:
        cfg["models"]["embedder"]["device"] = args.bge_gpu
        cfg["models"]["reranker"]["device"] = args.bge_gpu
        log.info("loading BGE on %s ...", args.bge_gpu)
        t0 = time.time()
        embedder = BGEM3Embedder(**cfg["models"]["embedder"])
        reranker = BGEReranker(**cfg["models"]["reranker"])
        log.info("  BGE ready (%.1fs)", time.time() - t0)

    if needs_llm:
        cfg["models"]["llm"]["device"] = args.llm_gpu
        log.info("loading LLM on %s ...", args.llm_gpu)
        t0 = time.time()
        llm = LLMClient(**cfg["models"]["llm"])
        text_answerer = TextAnswerer(llm)
        log.info("  LLM ready (%.1fs)", time.time() - t0)

    if needs_veil:
        planner = Planner(llm)
        verifier = Verifier(llm)

    # Stage C: run pipelines.
    bank_cache: dict[str, MemoryBank] = {}

    def get_bank(video_id: str) -> MemoryBank | None:
        if video_id not in bank_cache:
            p = memory_bank_path(cache_dir, video_id)
            if not p.exists():
                return None
            bank_cache[video_id] = MemoryBank.load(p)
        return bank_cache[video_id]

    for idx, (s, pipeline) in enumerate(todo):
        key = sample_key(benchmark, s, pipeline)
        log.info("[%d/%d] %s", idx + 1, len(todo), key)
        t0 = time.time()
        out, err = {}, None
        vid = sample_video_id(s)
        try:
            if pipeline == "direct":
                sampled = get_frames(sample_video_path(s), vid)
                out = run_direct_video_qa(
                    frames=sampled.frames,
                    question=s.question,
                    candidates=s.candidates,
                    vlm=vlm,
                    max_new_tokens=dv_cfg.get("max_new_tokens", 192),
                )
            else:
                bank = get_bank(vid)
                if bank is None:
                    raise RuntimeError(f"missing memory bank for {vid}")
                if pipeline == "uniform_memory":
                    out = run_uniform_memory(s.question, s.candidates, bank, text_answerer)
                elif pipeline == "bge_coarse":
                    out = run_bge_coarse_rag(
                        s.question, s.candidates, bank, embedder, text_answerer,
                        top_k=cfg["retrieval"].get("rerank_top_k", 10),
                    )
                elif pipeline == "bge_rerank":
                    out = run_naive_rag(
                        s.question, s.candidates, bank, embedder, reranker, text_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                    )
                else:  # veil
                    out = run_veil(
                        s.question, s.candidates, bank, planner, verifier,
                        embedder, reranker, text_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                        max_iter=cfg["veil_loop"]["max_iter"],
                    )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.exception("pipeline failed: %s", err)

        raw = out.get("raw") or json.dumps(
            {k: v for k, v in out.items()
             if k not in ("evidence_texts", "evidence_chunk_ids", "trace")}
        )
        letter = parse_letter(raw, len(s.candidates))
        pred_text = candidate_text_for_letter(letter, s.candidates)
        record = {
            "key": key,
            "benchmark": benchmark,
            "question_type": sample_question_type(s),
            "video_id": vid,
            "sample_idx": s.sample_idx,
            "pipeline": pipeline,
            "question": s.question,
            "candidates": s.candidates,
            "gold": s.answer,
            "pred_letter": letter,
            "pred_text": pred_text,
            "correct": pred_text == s.answer,
            "evidence_chunk_ids": out.get("evidence_chunk_ids", []),
            "trace_iters": (len(out.get("trace", {}).get("iterations", []))
                            if pipeline == "veil" else None),
            "raw": (raw or "")[:1500],
            "elapsed": time.time() - t0,
            "error": err,
        }
        append_record(out_path, record)

    log.info("=== done ===")
    report(out_path, all_samples, args.pipelines, benchmark)


# ── Report ─────────────────────────────────────────────────────────────────────

def report(jsonl_path: Path, samples: list, pipelines: list, benchmark: str) -> None:
    by: dict = defaultdict(list)
    if not jsonl_path.exists():
        log.warning("no records"); return
    for line in jsonl_path.open():
        try:
            r = json.loads(line)
            by[(r["question_type"], r["pipeline"])].append(r)
        except Exception:
            continue

    task_types = list(dict.fromkeys(sample_question_type(s) for s in samples))
    print(f"\n=== {benchmark.upper()} — ACCURACY ===")
    header = f"{'task':<22s} | " + " | ".join(f"{p:>14s}" for p in pipelines)
    print(header)
    print("-" * len(header))
    overall = {p: [0, 0] for p in pipelines}
    for task in task_types:
        cells = []
        for p in pipelines:
            rs = by.get((task, p), [])
            n, c = len(rs), sum(1 for r in rs if r.get("correct"))
            overall[p][0] += c; overall[p][1] += n
            cells.append(f"{c}/{n}({100*c/n if n else 0:.1f}%)")
        print(f"{task:<22s} | " + " | ".join(f"{c:>14s}" for c in cells))
    print("-" * len(header))
    cells = [f"{c}/{n}({100*c/n if n else 0:.1f}%)" for c, n in overall.values()]
    print(f"{'OVERALL':<22s} | " + " | ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
