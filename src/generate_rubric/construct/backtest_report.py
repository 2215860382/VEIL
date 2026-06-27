"""Lightweight diagnostics for rubric-construction artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from src.generate_rubric.construct.io import dump_json, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--out", default="outputs/rubric/backtest_report.json")
    args = ap.parse_args()

    dev = list(read_jsonl(args.dev))
    chains = list(read_jsonl(args.chains))
    pairs = list(read_jsonl(args.pairs))
    crit_rows = list(read_jsonl(args.criteria))

    criteria = []
    for row in crit_rows:
        for c in row.get("new_criteria") or []:
            criteria.append({**c, "question_type": row.get("question_type"), "pair_id": row.get("pair_id")})

    report = {
        "n_dev_questions": len(dev),
        "question_type_counts": Counter(r["question_type"] for r in dev),
        "chain_quality_counts": Counter(r["quality"] for r in chains),
        "pair_type_counts": Counter(r["pair_type"] for r in pairs),
        "n_extracted_criteria": len(criteria),
        "criteria_by_question_type": Counter(c["question_type"] for c in criteria),
        "criteria_by_repair_action": Counter(c.get("repair_action", "") for c in criteria),
        "needs_more_sampling": [],
    }

    by_type_pairs = Counter(r["question_type"] for r in pairs)
    by_type_criteria = Counter(c["question_type"] for c in criteria)
    all_types = sorted(set(report["question_type_counts"]))
    for qtype in all_types:
        if by_type_pairs[qtype] < 5 or by_type_criteria[qtype] < 3:
            report["needs_more_sampling"].append({
                "question_type": qtype,
                "pairs": by_type_pairs[qtype],
                "criteria": by_type_criteria[qtype],
            })

    dump_json(args.out, report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

