#!/usr/bin/env python
"""Coarse24 — 极简基线：单query → BGE top-24 → LLM判断
"""
import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataloader.videomme import load_videomme
from src.models.embedder import BGEM3Embedder
from src.models.llm_client import LLMClient
from src.utils.logging import get_logger

PIPELINE_NAME = "coarse24"
log = get_logger(PIPELINE_NAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bge-gpu", default="cuda:3")
    ap.add_argument("--llm-api-url", default=None)
    ap.add_argument("--llm-api-model", default="Qwen3.5-27B")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = cfg["benchmark"]["name"]
    out_root = Path(cfg.get("paths", {}).get("outputs_root", "outputs"))

    # 加载样本
    b = cfg["benchmark"]
    samples = load_videomme(
        parquet_path=b["parquet_path"],
        video_dir=b["video_dir"],
        duration_groups=b.get("duration_groups"),
    )
    log.info("loaded %d samples", len(samples))

    # 输出路径
    out_path = Path(args.out) if args.out else (out_root / "results" / "videommeL" / "coarse24.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_keys.add(json.loads(line)["key"])
            except:
                pass
    log.info("already done: %d", len(done_keys))

    # 记忆库
    memory_dir = Path(args.memory_dir) if args.memory_dir else (out_root / "memory" / "videomme_L_27b_27b")

    def load_bank(vid):
        bp = memory_dir / f"{vid}.json"
        return json.load(open(bp)) if bp.exists() else None

    # 模型
    log.info("loading BGE on %s", args.bge_gpu)
    embedder = BGEM3Embedder(
        model_path=cfg["models"]["embedder"]["model_path"],
        use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
        device=args.bge_gpu,
    )

    log.info("loading LLM API %s", args.llm_api_url)
    llm = LLMClient(
        model_path=args.llm_api_model,
        api_url=args.llm_api_url,
        api_model=args.llm_api_model,
    )

    out_fh = out_path.open("a")

    def run_sample(s):
        bank = load_bank(s.video_id)
        if not bank:
            return None, "bank_missing"

        # 单 query + top-24
        chunks = bank.get("chunks", [])
        if not chunks:
            return None, "no_chunks"

        texts = [c.get("memory_text", "") if isinstance(c, dict) else str(c) for c in chunks]

        # BGE 检索 top-24
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            query_emb = embedder.encode([s.question])[0:1]
            texts_embs = embedder.encode(texts)

            # Check for NaN values
            if np.isnan(query_emb).any() or np.isnan(texts_embs).any():
                log.warning("NaN detected in embeddings, using fallback ranking")
                top_indices = list(range(min(24, len(texts))))
            else:
                similarities = cosine_similarity(query_emb, texts_embs)[0]
                top_indices = np.argsort(similarities)[::-1][:24].tolist()
        except Exception as e:
            log.error("search error: %s", e)
            top_indices = list(range(min(24, len(texts))))

        evidence_text = "\n".join([
            f"{j+1}. {texts[i][:100]}"
            for j, i in enumerate(top_indices)
        ])

        # LLM 判断
        prompt = f"""Question: {s.question}

Options:
{chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(s.candidates)])}

Evidence (top-24 relevant chunks):
{evidence_text}

Based on the evidence above, which option is correct? Answer with just the letter (A/B/C/D)."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm.chat(messages, max_new_tokens=10, enable_thinking=False)
            pred = response.strip()[:1].upper() if response else ""
        except Exception as e:
            log.error("LLM error: %s", e)
            pred = ""

        return {
            "pred": pred,
            "chunk_ids": top_indices,
        }, None

    total = len(samples)
    done_count = [len(done_keys)]
    results = {}

    def process_item(s):
        key = f"{bench}|{s.video_id}|{s.sample_idx}|{PIPELINE_NAME}"

        if key in done_keys:
            return

        t0 = time.time()
        try:
            result, err = run_sample(s)
        except Exception as e:
            log.error("[%s] %s", s.video_id, e)
            result, err = None, str(e)
        elapsed = time.time() - t0

        if result is None:
            pred, pred_text, correct = "", "", False
        else:
            pred = result["pred"]
            idx = ord(pred) - ord("A") if pred else -1
            pred_text = s.candidates[idx] if 0 <= idx < len(s.candidates) else ""
            correct = (pred_text == s.answer)

        rec = {
            "key": key,
            "benchmark": bench,
            "question_type": s.question_type,
            "video_id": s.video_id,
            "sample_idx": s.sample_idx,
            "pipeline": PIPELINE_NAME,
            "question": s.question,
            "candidates": s.candidates,
            "gold_answer": s.answer,
            "pred_letter": pred,
            "pred_text": pred_text,
            "correct": correct,
            "evidence_chunk_ids": result.get("chunk_ids", []) if result else [],
            "elapsed": round(elapsed, 2),
            "error": err,
        }

        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()

        done_count[0] += 1
        results.setdefault(s.question_type, []).append(int(correct))

        if done_count[0] % 10 == 0:
            log.info("[%d/%d] %s %s %.1fs", done_count[0], total, pred, "✓" if correct else "✗", elapsed)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(process_item, s) for s in samples]))

    out_fh.close()

    # 准确率
    print(f"\n=== {PIPELINE_NAME} ===")
    for qt in sorted(results):
        c = results[qt]
        acc = sum(c) / len(c) * 100 if c else 0
        print(f"{qt:20} {acc:6.1f}%")
    all_c = sum(sum(v) for v in results.values())
    all_n = sum(sum(1 for _ in v) for v in results.values())
    print(f"{'Overall':20} {all_c/all_n*100:6.1f}%")


if __name__ == "__main__":
    main()
