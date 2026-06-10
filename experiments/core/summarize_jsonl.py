"""Summarize VEIL JSONL result files.

Outputs per-pipeline accuracy, elapsed time, and average unique evidence count.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def iter_records(paths: list[Path]):
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_source"] = str(path)
                yield rec


def unique_evidence_count(rec: dict) -> int:
    ids = rec.get("evidence_chunk_ids") or []
    return len({str(x) for x in ids})


def summarize(paths: list[Path]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in iter_records(paths):
        grouped[str(rec.get("pipeline", ""))].append(rec)

    out = {}
    for pipeline, records in sorted(grouped.items()):
        n = len(records)
        correct = sum(1 for r in records if bool(r.get("correct")))
        ev_counts = [unique_evidence_count(r) for r in records]
        elapsed = [float(r.get("elapsed") or 0.0) for r in records]
        avg_ev = sum(ev_counts) / n if n else 0.0
        out[pipeline] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "avg_unique_evidence_count": avg_ev,
            "ceil_avg_unique_evidence_count": int(math.ceil(avg_ev)),
            "total_elapsed_sec": sum(elapsed),
            "avg_elapsed_sec": sum(elapsed) / n if n else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--field", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = summarize(args.jsonl)
    if args.pipeline:
        data = {args.pipeline: data.get(args.pipeline, {})}

    if args.field:
        row = next(iter(data.values()), {})
        val = row.get(args.field)
        if isinstance(val, float):
            print(f"{val:.6f}")
        elif val is not None:
            print(val)
        return

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print(f"{'pipeline':<28} {'n':>5} {'acc':>8} {'avg_ev':>8} {'ceil_ev':>8} {'total_h':>8}")
    for pipeline, row in data.items():
        print(
            f"{pipeline:<28} {row['n']:>5} {row['accuracy'] * 100:>7.2f}% "
            f"{row['avg_unique_evidence_count']:>8.2f} "
            f"{row['ceil_avg_unique_evidence_count']:>8} "
            f"{row['total_elapsed_sec'] / 3600:>8.2f}"
        )


if __name__ == "__main__":
    main()
