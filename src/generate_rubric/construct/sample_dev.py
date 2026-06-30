"""Select stratified train/val/test splits from Video-MME-L results."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.config import load_config
from src.generate_rubric.construct.io import read_jsonl, write_jsonl
from src.generate_rubric.construct.schema import DevQuestion
from experiments.tuning.veil_27b import load_samples


def _reasons(row: dict) -> list[str]:
    reasons = []
    if row.get("correct") is False:
        reasons.append("system_wrong")
    trace = row.get("trace_iters") or []
    if trace:
        last = trace[-1]
        if str(last.get("verdict", "")).upper() == "INSUFFICIENT":
            reasons.append("final_insufficient")
        if any(last.get("option_judgment", {}).get(k) == "unknown" for k in last.get("option_judgment", {})):
            reasons.append("has_unknown_options")
        if len(trace) >= 3:
            reasons.append("multi_round")
    q = (row.get("question") or "").lower()
    if any(w in q for w in ["before", "after", "first", "last", "when", "changed", "become"]):
        reasons.append("temporal_or_state")
    if any(w in q for w in ["where", "left", "right", "behind", "front", "next to"]):
        reasons.append("spatial")
    if any(w in q for w in ["why", "because", "reason", "cause"]):
        reasons.append("causal")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--result", required=True, help="Existing VEIL result JSONL used only for prioritization.")
    ap.add_argument("--out-dir", default="outputs/rubric/splits")
    ap.add_argument("--n-dev", type=int, default=720)
    ap.add_argument("--n-val", type=int, default=90)
    ap.add_argument("--n-test", type=int, default=90)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cfg = load_config(args.config)
    samples = {s.sample_idx: s for s in load_samples(cfg)}
    result_by_idx = {int(r["sample_idx"]): r for r in read_jsonl(args.result)}

    grouped: dict[str, list[DevQuestion]] = defaultdict(list)
    for idx, sample in samples.items():
        row = result_by_idx.get(idx, {})
        reasons = _reasons(row)
        if not reasons:
            reasons = ["stratified_background"]
        grouped[sample.question_type].append(DevQuestion(
            sample_idx=sample.sample_idx,
            video_id=sample.video_id,
            question_id=sample.question_id,
            question_type=sample.question_type,
            question=sample.question,
            candidates=sample.candidates,
            priority_reasons=reasons,
            source_result={
                "correct": row.get("correct"),
                "final_verdict": (row.get("trace_iters") or [{}])[-1].get("verdict") if row else None,
            },
        ))

    rng = random.Random(args.seed)
    def sort_key(d: DevQuestion):
        return (
            "system_wrong" not in d.priority_reasons,
            "final_insufficient" not in d.priority_reasons,
            "has_unknown_options" not in d.priority_reasons,
            rng.random(),
        )

    pool: list[DevQuestion] = []
    per_type = max(1, args.n_dev // max(1, len(grouped)))
    for qtype, rows in grouped.items():
        rows.sort(key=sort_key)
        pool.extend(rows[:per_type])

    total_needed = args.n_dev + args.n_val + args.n_test
    if len(pool) < total_needed:
        seen = {d.sample_idx for d in pool}
        rest = [d for rows in grouped.values() for d in rows if d.sample_idx not in seen]
        rest.sort(key=sort_key)
        pool.extend(rest[: total_needed - len(pool)])

    pool = pool[:total_needed]
    rng.shuffle(pool)

    dev = pool[: args.n_dev]
    val = pool[args.n_dev: args.n_dev + args.n_val]
    test = pool[args.n_dev + args.n_val: args.n_dev + args.n_val + args.n_test]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "dev_questions.jsonl", (d.to_dict() for d in dev))
    write_jsonl(out_dir / "val_questions.jsonl", (d.to_dict() for d in val))
    write_jsonl(out_dir / "test_questions.jsonl", (d.to_dict() for d in test))
    heldout = sorted(set(samples) - {d.sample_idx for d in pool})
    (out_dir / "heldout_remaining.txt").write_text(
        "\n".join(str(i) for i in heldout) + "\n",
        encoding="utf-8",
    )
    print(f"wrote splits to {out_dir}: dev={len(dev)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
