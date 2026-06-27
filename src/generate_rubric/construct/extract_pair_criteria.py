"""Extract criterion candidates from evidence-chain pairs."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.generate_rubric.construct.io import read_jsonl, write_jsonl
from src.generate_rubric.construct.llm import LLMRouter
from src.generate_rubric.construct.prompts import (
    PAIRWISE_SYSTEM,
    PAIRWISE_USER_TEMPLATE,
    format_evidence,
    format_initial_rubric,
    format_options,
)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"raw": text, "new_criteria": []}
    try:
        return json.loads(m.group())
    except Exception:
        return {"raw": text, "new_criteria": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="outputs/rubric/pair_criteria.jsonl")
    ap.add_argument("--rubric-path", default="")
    ap.add_argument("--llm-api-url", default="http://127.0.0.1:8003,http://127.0.0.1:8004")
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    dev = {int(r["sample_idx"]): r for r in read_jsonl(args.dev)}
    chains = {r["chain_id"]: r for r in read_jsonl(args.chains)}
    pairs = list(read_jsonl(args.pairs))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    done = set()
    try:
        done = {r["pair_id"] for r in read_jsonl(args.out)}
    except FileNotFoundError:
        done = set()
    pairs = [p for p in pairs if p["pair_id"] not in done]

    llm = LLMRouter.from_urls(args.llm_api_url, model=args.model)
    if args.rubric_path:
        initial_rubric = Path(args.rubric_path).read_text(encoding="utf-8")
    else:
        initial_rubric = format_initial_rubric()

    def process_pair(p: dict) -> dict:
        q = dev[int(p["sample_idx"])]
        weaker = chains[p["weaker_chain_id"]]
        stronger = chains[p["stronger_chain_id"]]
        user = PAIRWISE_USER_TEMPLATE.format(
            initial_rubric=initial_rubric,
            question_type=q["question_type"],
            question=q["question"],
            options=format_options(q["candidates"]),
            weaker_quality=weaker["quality"],
            weaker_id=weaker["chain_id"],
            weaker_evidence=format_evidence(weaker["evidence"]),
            stronger_quality=stronger["quality"],
            stronger_id=stronger["chain_id"],
            stronger_evidence=format_evidence(stronger["evidence"]),
            pair_type=p["pair_type"],
        )
        raw = llm.chat(PAIRWISE_SYSTEM, user, max_tokens=2400)
        parsed = _parse_json(raw)
        return {
            **p,
            "question": q["question"],
            "candidates": q["candidates"],
            "llm_raw": raw,
            "extraction": parsed,
            "new_criteria": parsed.get("new_criteria", []),
        }

    print(f"pairs to process: {len(pairs)} (already done: {len(done)}, workers={args.workers})")
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process_pair, p) for p in pairs]
        for fut in as_completed(futs):
            row = fut.result()
            write_jsonl(args.out, [row], append=True)
            completed += 1
            print(f"[{completed}/{len(pairs)}] {row['pair_id']} criteria={len(row.get('new_criteria') or [])}", flush=True)
    print(f"wrote pair criteria to {args.out}")


if __name__ == "__main__":
    main()
