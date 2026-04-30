"""End-to-end smoke test on 1 video + 1 question.

Verifies that all moving parts run without crashing:
    model load → video decode → memory build → BGE retrieve+rerank → planner/verifier/answerer → final.

Does NOT verify accuracy — a wrong answer still counts as a successful smoke run.

Usage: PYTHONPATH=. python scripts/smoke.py
       PYTHONPATH=. python scripts/smoke.py --vlm-gpu cuda:0 --llm-gpu cuda:4 --bge-gpu cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reduce fragmentation OOM risk on shared GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_mlvu import load_mlvu
from eval.parse_answer import candidate_text_for_letter, parse_letter
from memory.build_memory import build_memory_bank, memory_bank_path
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

log = get_logger("smoke")


def _summary_line(label, letter, gold, candidates, elapsed, extra=""):
    pred = candidate_text_for_letter(letter, candidates)
    mark = "✓" if pred == gold else "✗"
    return f"[{label}] pred={letter}({pred!r}) gold={gold!r} {mark} {extra} took={elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mlvu.yaml")
    ap.add_argument("--vlm-gpu", default="cuda:0")
    ap.add_argument("--llm-gpu", default="cuda:4")
    ap.add_argument("--bge-gpu", default="cuda:2")
    ap.add_argument("--task", default="plotQA")
    ap.add_argument("--rebuild-bank", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = cfg["benchmark"]

    samples = load_mlvu(
        b["json_dir"], b["video_dir"], b["json_files"],
        task_types=[args.task], max_videos=1, max_questions_per_video=1,
    )
    if not samples:
        log.error("no samples found"); sys.exit(2)
    s = samples[0]
    log.info("sample: video_id=%s duration=%.0fs", s.video_id, s.duration)
    log.info("Q: %s", s.question)
    log.info("opts: %s | gold: %s", s.candidates, s.answer)

    # Smoke-test overrides — spread across GPUs, lighter sampling.
    cfg["models"]["vlm"]["device"] = args.vlm_gpu
    cfg["models"]["llm"]["device"] = args.llm_gpu
    cfg["models"]["embedder"]["device"] = args.bge_gpu
    cfg["models"]["reranker"]["device"] = args.bge_gpu
    cfg["memory"]["segment_seconds"] = 32
    cfg["memory"]["fps_per_segment"] = 0.5
    cfg["memory"]["max_frames_per_segment"] = 4
    log.info("device assignment: vlm=%s llm=%s bge=%s",
             args.vlm_gpu, args.llm_gpu, args.bge_gpu)

    cache_dir = Path(cfg["memory"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    bank_path = memory_bank_path(cache_dir, s.video_id)

    # ---- Stage 1: load VLM, build (or load) memory bank ----
    t0 = time.time()
    log.info("loading VLM ...")
    vlm = VLMClient(**cfg["models"]["vlm"])
    log.info("  VLM ready (%.1fs)", time.time() - t0)

    if bank_path.exists() and not args.rebuild_bank:
        log.info("loading cached memory bank: %s", bank_path)
        bank = MemoryBank.load(bank_path)
    else:
        log.info("building memory bank for %s ...", s.video_id)
        t0 = time.time()
        bank = build_memory_bank(
            video_path=s.video_path, video_id=s.video_id, vlm=vlm,
            segment_seconds=cfg["memory"]["segment_seconds"],
            fps_per_segment=cfg["memory"]["fps_per_segment"],
            max_frames_per_segment=cfg["memory"]["max_frames_per_segment"],
            progress=True,
        )
        bank.save(bank_path)
        log.info("  %d chunks built in %.1fs → %s", len(bank.chunks), time.time() - t0, bank_path)
    log.info("first chunk preview: %s", bank.chunks[0].memory_text[:300])

    # ---- Stage 2: Direct VideoQA baseline ----
    log.info("=== Pipeline: Direct Video QA ===")
    t0 = time.time()
    out_d = run_direct_video_qa(
        s.video_path, s.question, s.candidates, vlm,
        fps=0.5, max_pixels=128 * 28 * 28, max_new_tokens=128,
    )
    print(out_d["raw"][:400])
    letter_d = parse_letter(out_d["raw"], len(s.candidates))
    print(_summary_line("direct", letter_d, s.answer, s.candidates, time.time() - t0))

    # ---- Stage 3: BGE retrieval ----
    log.info("loading BGE embedder + reranker ...")
    embedder = BGEM3Embedder(**cfg["models"]["embedder"])
    reranker = BGEReranker(**cfg["models"]["reranker"])
    vl_answerer = VLAnswerer(vlm)

    log.info("=== Pipeline: Naive RAG ===")
    t0 = time.time()
    out_n = run_naive_rag(
        s.question, s.candidates, bank, embedder, reranker, vl_answerer,
        coarse_top_k=20, rerank_top_k=5,
    )
    raw_n = out_n.get("raw") or json.dumps({k: v for k, v in out_n.items() if k not in ("evidence_texts", "evidence_chunk_ids")})
    print(raw_n[:400])
    letter_n = parse_letter(raw_n, len(s.candidates))
    print(_summary_line("naive_rag", letter_n, s.answer, s.candidates, time.time() - t0,
                        extra=f"chunks={out_n.get('evidence_chunk_ids', [])}"))

    # ---- Stage 4: VEIL iterative ----
    log.info("loading LLM (Qwen3-8B) ...")
    llm = LLMClient(**cfg["models"]["llm"])
    planner = Planner(llm); verifier = Verifier(llm)

    log.info("=== Pipeline: VEIL ===")
    t0 = time.time()
    out_v = run_veil(
        s.question, s.candidates, bank, planner, verifier, embedder, reranker, vl_answerer,
        coarse_top_k=20, rerank_top_k=5, max_iter=2,
    )
    raw_v = out_v.get("raw") or json.dumps({k: v for k, v in out_v.items() if k not in ("evidence_texts", "evidence_chunk_ids", "trace")})
    print(raw_v[:400])
    letter_v = parse_letter(raw_v, len(s.candidates))
    iters = len(out_v["trace"]["iterations"])
    print(_summary_line("veil", letter_v, s.answer, s.candidates, time.time() - t0,
                        extra=f"iters={iters} queries={out_v['trace']['queries']} chunks={out_v.get('evidence_chunk_ids', [])}"))

    log.info("=== smoke test complete ===")


if __name__ == "__main__":
    main()
