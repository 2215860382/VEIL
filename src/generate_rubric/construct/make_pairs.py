"""Create prioritized weak/good/great/excellent evidence-chain pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.generate_rubric.construct.io import read_jsonl, write_jsonl
from src.generate_rubric.construct.schema import ChainPair


QUALITY_RANK = {"weak": 0, "good": 1, "great": 2, "excellent": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--out", default="outputs/rubric/pairs.jsonl")
    ap.add_argument("--max-pairs-per-question", type=int, default=3)
    args = ap.parse_args()

    by_idx = defaultdict(list)
    for c in read_jsonl(args.chains):
        by_idx[int(c["sample_idx"])].append(c)

    pairs = []
    for idx, chains in by_idx.items():
        chains.sort(key=lambda c: QUALITY_RANK.get(c.get("quality", ""), -1))
        by_q = defaultdict(list)
        for c in chains:
            by_q[c.get("quality", "")].append(c)

        candidates = []
        if by_q["weak"] and by_q["excellent"]:
            candidates.append((by_q["weak"][0], by_q["excellent"][0], "weak_vs_excellent"))
        if by_q["good"] and by_q["great"]:
            candidates.append((by_q["good"][0], by_q["great"][0], "good_vs_great"))
        if by_q["great"] and by_q["excellent"]:
            candidates.append((by_q["great"][0], by_q["excellent"][0], "great_vs_excellent"))
        if len(by_q["excellent"]) >= 2:
            candidates.append((by_q["excellent"][0], by_q["excellent"][1], "excellent_vs_excellent_diverse"))
        if by_q["weak"] and by_q["good"] and not any(pt == "good_vs_great" for _, _, pt in candidates):
            candidates.append((by_q["weak"][0], by_q["good"][0], "weak_vs_good"))
        if not candidates and len(chains) >= 2:
            candidates.append((chains[0], chains[-1], f"{chains[0]['quality']}_vs_{chains[-1]['quality']}"))

        for weaker, stronger, pair_type in candidates[: args.max_pairs_per_question]:
            pair_id = f"{idx}:{pair_type}:{weaker['chain_id']}->{stronger['chain_id']}"
            pairs.append(ChainPair(
                sample_idx=idx,
                video_id=weaker["video_id"],
                question_type=weaker.get("question_type", ""),
                pair_id=pair_id,
                weaker_chain_id=weaker["chain_id"],
                stronger_chain_id=stronger["chain_id"],
                pair_type=pair_type,
            ).to_dict())

    write_jsonl(args.out, pairs)
    print(f"wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
