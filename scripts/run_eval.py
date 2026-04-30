"""Entry point: run a pipeline on a benchmark and compute accuracy.

Usage:
    python scripts/run_eval.py --config configs/mlvu.yaml --pipeline veil
    python scripts/run_eval.py --config configs/mlvu.yaml --pipeline direct
    python scripts/run_eval.py --config configs/mlvu.yaml --pipeline naive_rag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from data.load_mlvu import load_mlvu
from eval.compute_accuracy import compute_accuracy
from eval.parse_answer import candidate_text_for_letter, parse_letter
from memory.build_memory import memory_bank_path
from memory.schema import MemoryBank
from pipelines.direct_video_qa import run_direct_video_qa
from pipelines.naive_rag import run_naive_rag
from pipelines.veil_iterative import run_veil
from models.embedder import BGEM3Embedder
from models.llm_client import LLMClient
from models.reranker import BGEReranker
from models.vlm_client import VLMClient
from reasoning.answerer import VLAnswerer
from reasoning.planner import Planner
from reasoning.verifier import Verifier
from utils.config import load_config
from utils.logging import get_logger

log = get_logger("veil.run_eval")

PIPELINES = {"direct", "naive_rag", "veil"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", choices=sorted(PIPELINES), required=True)
    p.add_argument("--task-types", nargs="*", default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--max-questions-per-video", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Cap total questions evaluated.")
    p.add_argument("--out-suffix", default="", help="Append to output filename.")
    return p.parse_args()


def _need_models(pipeline: str):
    """Return tuple (need_vlm, need_llm, need_retriever)."""
    if pipeline == "direct":
        return True, False, False
    if pipeline == "naive_rag":
        return True, False, True
    return True, True, True  # veil


def main():
    args = parse_args()
    cfg = load_config(args.config)
    bench = cfg["benchmark"]
    cache_dir = Path(cfg["memory"]["cache_dir"])
    out_dir = Path(cfg["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    task_types = args.task_types if args.task_types is not None else bench.get("task_types")
    samples = load_mlvu(
        json_dir=bench["json_dir"],
        video_dir=bench["video_dir"],
        json_files=bench["json_files"],
        task_types=task_types,
        max_videos=args.max_videos,
        max_questions_per_video=args.max_questions_per_video,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    log.info("Eval %d samples on pipeline=%s", len(samples), args.pipeline)

    # Lazy load only what's needed.
    need_vlm, need_llm, need_retriever = _need_models(args.pipeline)
    vlm = VLMClient(**cfg["models"]["vlm"]) if need_vlm else None
    llm = LLMClient(**cfg["models"]["llm"]) if need_llm else None
    embedder = BGEM3Embedder(**cfg["models"]["embedder"]) if need_retriever else None
    reranker = BGEReranker(**cfg["models"]["reranker"]) if need_retriever else None

    planner = Planner(llm) if llm is not None else None
    verifier = Verifier(llm) if llm is not None else None
    vl_answerer = VLAnswerer(vlm) if vlm is not None else None

    # Cache memory banks per video to avoid reloading.
    bank_cache: dict[str, MemoryBank] = {}

    def get_bank(video_id: str) -> MemoryBank | None:
        if video_id in bank_cache:
            return bank_cache[video_id]
        p = memory_bank_path(cache_dir, video_id)
        if not p.exists():
            return None
        bank_cache[video_id] = MemoryBank.load(p)
        return bank_cache[video_id]

    records = []
    for s in tqdm(samples, desc=args.pipeline):
        n_opt = len(s.candidates)
        t0 = time.time()
        try:
            if args.pipeline == "direct":
                out = run_direct_video_qa(s.video_path, s.question, s.candidates, vlm)
            else:
                bank = get_bank(s.video_id)
                if bank is None:
                    log.warning("Missing memory bank for %s — skipping", s.video_id)
                    continue
                if args.pipeline == "naive_rag":
                    out = run_naive_rag(
                        s.question, s.candidates, bank, embedder, reranker, vl_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                    )
                else:
                    out = run_veil(
                        s.question, s.candidates, bank, planner, verifier, embedder, reranker, vl_answerer,
                        coarse_top_k=cfg["retrieval"]["coarse_top_k"],
                        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
                        max_iter=cfg["veil_loop"]["max_iter"],
                    )
        except Exception as e:
            log.exception("pipeline failed on %s: %s", s.video_id, e)
            out = {"raw": "", "answer": "", "evidence_texts": [], "evidence_chunk_ids": []}

        raw = out.get("raw") or json.dumps({k: v for k, v in out.items() if k != "trace"})
        letter = parse_letter(raw, n_opt)
        pred_text = candidate_text_for_letter(letter, s.candidates)
        records.append({
            "video_id": s.video_id,
            "question_type": s.question_type,
            "question": s.question,
            "candidates": s.candidates,
            "gold": s.answer,
            "pred_letter": letter,
            "pred_text": pred_text,
            "evidence_chunk_ids": out.get("evidence_chunk_ids", []),
            "raw": raw[:1500],
            "elapsed": time.time() - t0,
        })

    metrics = compute_accuracy(records)
    log.info("Metrics: %s", json.dumps(metrics, indent=2))

    suffix = f".{args.out_suffix}" if args.out_suffix else ""
    out_path = out_dir / f"{args.pipeline}{suffix}.json"
    out_path.write_text(json.dumps({"config": str(args.config), "metrics": metrics, "records": records}, indent=2))
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
