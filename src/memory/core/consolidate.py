"""Post-process a MemoryBank: merge adjacent near-duplicate chunks.

Inspired by WorldMM's consolidate_semantic_memory.py, but adapted for
VEIL's chunk structure: instead of deduplicating S-V-O triples, we merge
adjacent video chunks whose BGE-M3 semantic vectors are nearly identical
(cosine similarity >= theta). This collapses static shots and repetitive
segments into a single representative chunk.

Usage:
    from src.memory.core.consolidate import consolidate_bank
    consolidated = consolidate_bank(bank, theta=0.92)
    consolidated.save("outputs/memory/video_id_consolidated.json")

CLI:
    python -m src.memory.core.consolidate \
        --input outputs/memory/mlvu_similarity \
        --output outputs/memory/mlvu_similarity_cons \
        --theta 0.92 --max-gap 2.0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from src.memory.core.schema import MemoryBank, MemoryChunk

log = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two pre-L2-normalised vectors."""
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def _merge(a: MemoryChunk, b: MemoryChunk) -> MemoryChunk:
    """Merge chunk b into chunk a (a precedes b in time)."""
    # Prefer the longer description; append b's if it adds content
    if b.memory_text and b.memory_text not in a.memory_text:
        memory_text = a.memory_text.rstrip(".") + ". " + b.memory_text
    else:
        memory_text = a.memory_text

    # Average the semantic vector then re-normalise
    if a.v_semantic and b.v_semantic:
        avg = np.array(a.v_semantic, dtype=np.float32) + np.array(b.v_semantic, dtype=np.float32)
        norm = np.linalg.norm(avg)
        v_semantic = (avg / norm).tolist() if norm > 0 else a.v_semantic
    else:
        v_semantic = a.v_semantic or b.v_semantic

    return a.model_copy(update={
        "end_time": b.end_time,
        "memory_text": memory_text,
        "objects": list(dict.fromkeys(a.objects + b.objects)),
        "persons": list(dict.fromkeys(a.persons + b.persons)),
        "actions": list(dict.fromkeys(a.actions + b.actions)),
        "asr": (a.asr.rstrip() + " " + b.asr.lstrip()).strip() if b.asr else a.asr,
        "ocr": a.ocr or b.ocr,
        "state_change": b.state_change or a.state_change,
        "sampled_frames": sorted(set(a.sampled_frames + b.sampled_frames)),
        "v_semantic": v_semantic,
        # Keep a's keyframe (first / most representative)
        "v_visual": a.v_visual,
        "keyframe_path": a.keyframe_path,
        "keyframe_ts": a.keyframe_ts,
    })


def consolidate_bank(
    bank: MemoryBank,
    theta: float = 0.92,
    max_gap_sec: float = 2.0,
) -> MemoryBank:
    """Merge adjacent chunks with cosine(v_semantic) >= theta.

    Args:
        bank:        Source MemoryBank (must have v_semantic populated).
        theta:       Similarity threshold (0-1). Higher = more conservative merging.
        max_gap_sec: Only merge if the time gap between chunks is <= this value (seconds).

    Returns:
        A new MemoryBank with merged chunks and memory_kind suffixed with "_cons".
        If v_semantic is absent the original bank is returned unchanged.
    """
    chunks = list(bank.chunks)
    if len(chunks) <= 1 or not chunks[0].v_semantic:
        log.warning("consolidate_bank: no v_semantic vectors found, skipping.")
        return bank

    merged: List[MemoryChunk] = [chunks[0]]
    n_merges = 0

    for chunk in chunks[1:]:
        prev = merged[-1]
        gap = chunk.start_time - prev.end_time
        if gap <= max_gap_sec and prev.v_semantic and chunk.v_semantic:
            sim = _cosine(prev.v_semantic, chunk.v_semantic)
            if sim >= theta:
                merged[-1] = _merge(prev, chunk)
                n_merges += 1
                continue
        merged.append(chunk)

    # Re-index chunk_ids sequentially
    merged = [c.model_copy(update={"chunk_id": i}) for i, c in enumerate(merged)]

    log.info(
        "consolidate_bank [%s]: %d → %d chunks (%d merges, theta=%.2f)",
        bank.video_id, len(chunks), len(merged), n_merges, theta,
    )

    return bank.model_copy(update={
        "chunks": merged,
        "memory_kind": bank.memory_kind.removesuffix("_cons") + "_cons",
    })


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Consolidate a directory of MemoryBank JSON files by merging adjacent similar chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",   required=True, help="Directory containing *.json MemoryBank files.")
    ap.add_argument("--output",  required=True, help="Output directory for consolidated MemoryBank files.")
    ap.add_argument("--theta",   type=float, default=0.92,
                    help="Cosine similarity threshold for merging adjacent chunks.")
    ap.add_argument("--max-gap", type=float, default=2.0, dest="max_gap",
                    help="Maximum allowed time gap (seconds) between chunks to merge.")
    ap.add_argument("--suffix",  default="_cons",
                    help="Suffix appended to output filenames (default: _cons → video_id_cons.json).")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_cli().parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        log.error("No *.json files found in %s", in_dir)
        return

    before_total = after_total = 0
    for path in json_files:
        try:
            bank = MemoryBank.load(path)
        except Exception as e:
            log.warning("Skipping %s: %s", path.name, e)
            continue

        before = len(bank.chunks)
        consolidated = consolidate_bank(bank, theta=args.theta, max_gap_sec=args.max_gap)
        after = len(consolidated.chunks)
        before_total += before
        after_total += after

        stem = path.stem + args.suffix
        consolidated.save(out_dir / f"{stem}.json")

    if before_total:
        reduction = (before_total - after_total) / before_total * 100
        log.info("Done: %d → %d chunks total (%.1f%% reduction)", before_total, after_total, reduction)


if __name__ == "__main__":
    main()
