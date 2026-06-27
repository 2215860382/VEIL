"""Build weak/good/great evidence-chain records from existing VEIL traces."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.build_memory.core.bank_loader import load_bank
from src.config import load_config
from src.generate_rubric.construct.io import read_jsonl, write_jsonl
from src.generate_rubric.construct.schema import EvidenceChain, EvidenceItem


def _chunk_map(memory_dir: Path, video_id: str) -> dict[int, object]:
    bank = load_bank(memory_dir / video_id)
    return {int(c.chunk_id): c for c in bank.chunks}


def _items(cmap: dict[int, object], ids: list[int]) -> list[EvidenceItem]:
    out = []
    seen = set()
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        c = cmap.get(int(cid))
        if c is None:
            continue
        text = c.memory_text or c.visual_caption or c.asr or ""
        if c.asr:
            text = f"{text}\nASR: {c.asr}"
        out.append(EvidenceItem(
            chunk_id=int(cid),
            start_time=getattr(c, "start_time", None),
            end_time=getattr(c, "end_time", None),
            text=text,
        ))
    return out


def _chain_quality(iter_idx: int, n_iters: int, correct: bool | None) -> str:
    if iter_idx <= 0:
        return "weak"
    if iter_idx < max(1, n_iters - 1):
        return "good"
    if correct:
        return "great"
    return "good"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/veil.yaml")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--memory-dir", default="outputs/memory/videomme_multiframe")
    ap.add_argument("--out", default="outputs/rubric/chains.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    del cfg
    dev_idxs = {int(r["sample_idx"]): r for r in read_jsonl(args.dev)}
    results = [r for r in read_jsonl(args.result) if int(r["sample_idx"]) in dev_idxs]
    memory_dir = Path(args.memory_dir)

    rows = []
    cache: dict[str, dict[int, object]] = {}
    for r in results:
        idx = int(r["sample_idx"])
        video_id = r["video_id"]
        if video_id not in cache:
            cache[video_id] = _chunk_map(memory_dir, video_id)
        cmap = cache[video_id]
        trace = r.get("trace_iters") or []
        accumulated: list[int] = []
        for it in trace:
            accumulated.extend(it.get("new_ids") or [])
            q = _chain_quality(int(it.get("iter", 0)), len(trace), r.get("correct"))
            chain_id = f"{idx}:{q}:iter{int(it.get('iter', 0))}"
            rows.append(EvidenceChain(
                sample_idx=idx,
                video_id=video_id,
                question_type=r.get("question_type", ""),
                chain_id=chain_id,
                quality=q,
                source=f"iter{int(it.get('iter', 0))}_accumulated",
                evidence=_items(cmap, accumulated),
                notes=(
                    "Accumulated evidence up to this iteration; later iterations should "
                    "provide progressively stronger sufficiency."
                ),
            ).to_dict())
        final_ids = r.get("evidence_chunk_ids") or accumulated
        if final_ids:
            rows.append(EvidenceChain(
                sample_idx=idx,
                video_id=video_id,
                question_type=r.get("question_type", ""),
                chain_id=f"{idx}:great:final",
                quality="great" if r.get("correct") else "good",
                source="final_multi_round",
                evidence=_items(cmap, final_ids),
                notes="Final accumulated chain from current system.",
            ).to_dict())

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} chains to {args.out}")


if __name__ == "__main__":
    main()
