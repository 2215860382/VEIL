"""Dedupe near-identical frames in already-built attribute chunks.

For each chunk whose `frame_paths` is populated:
  1. Compute dHash for every frame file listed
  2. Cluster frames by Hamming distance (≤ DEDUP_HAMMING_THRESHOLD = 4)
  3. Keep the first frame per cluster, delete the rest from disk
  4. Renumber the kept files to contiguous indices: {cid:04d}_0.jpg, _1.jpg, ...
  5. Update `frame_paths` in attributes.json

`static_attributes` / `static_index_text` are left unchanged — they were derived
from the full frame set (dups don't usually add new entities anyway), so dedup
is purely a disk/cleanup pass, no VLM redo.

Empty / unfilled chunks (no `static_index_text` or empty `frame_paths`) are
skipped to avoid racing with a concurrent build_attribute_layer respawn.

Run:
    PYTHONPATH=. python scripts/cleanup_dedup_frames.py [--dry-run] [--video-id VID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")
DEDUP_HAMMING_THRESHOLD = 4


def compute_dhash(path: Path, size: int = 8) -> str | None:
    try:
        img = Image.open(path).convert("L").resize((size + 1, size))
        arr = np.array(img)
        bits = []
        for i in range(size):
            for j in range(size):
                bits.append("1" if arr[i, j + 1] > arr[i, j] else "0")
        return "".join(bits)
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def cleanup_chunk(video_dir: Path, cid: int, frame_paths: list[str], dry_run: bool) -> tuple[list[str], int]:
    """Return (new_frame_paths, n_dropped). frame_paths are relative to video_dir."""
    kept_hashes: list[str] = []
    kept_paths: list[str] = []
    drop_paths: list[Path] = []
    for rel in frame_paths:
        abs_p = video_dir / rel
        if not abs_p.exists():
            continue
        h = compute_dhash(abs_p)
        if h is None:
            kept_paths.append(rel)
            continue
        is_dup = any(hamming(h, prev) <= DEDUP_HAMMING_THRESHOLD for prev in kept_hashes)
        if is_dup:
            drop_paths.append(abs_p)
        else:
            kept_hashes.append(h)
            kept_paths.append(rel)
    n_dropped = len(drop_paths)
    if n_dropped == 0:
        return frame_paths, 0  # no change, keep original order/paths intact

    if dry_run:
        return [f"frames/{cid:04d}_{i}.jpg" for i in range(len(kept_paths))], n_dropped

    # Delete duplicate files
    for p in drop_paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # Renumber kept files to contiguous 0..k-1.
    # Two-phase rename via temp suffix to avoid collisions between source/target.
    new_paths: list[str] = []
    tmp_paths: list[Path] = []
    for new_i, old_rel in enumerate(kept_paths):
        target_rel = f"frames/{cid:04d}_{new_i}.jpg"
        new_paths.append(target_rel)
        if old_rel == target_rel:
            tmp_paths.append(video_dir / target_rel)  # no move needed
            continue
        old_abs = video_dir / old_rel
        if not old_abs.exists():
            tmp_paths.append(None)
            continue
        tmp_abs = video_dir / f"{target_rel}.tmprename"
        old_abs.rename(tmp_abs)
        tmp_paths.append(tmp_abs)
    # Phase 2: move tmp → final
    for new_i, tmp_abs in enumerate(tmp_paths):
        if tmp_abs is None:
            continue
        target_abs = video_dir / f"frames/{cid:04d}_{new_i}.jpg"
        if tmp_abs == target_abs:
            continue
        if target_abs.exists():
            target_abs.unlink()
        tmp_abs.rename(target_abs)

    return new_paths, n_dropped


def process_video(video_dir: Path, dry_run: bool, only_video_id: str | None = None) -> tuple[int, int, int]:
    """Returns (chunks_modified, frames_dropped, chunks_skipped)."""
    if only_video_id and video_dir.name != only_video_id:
        return 0, 0, 0
    attr_path = video_dir / "attributes.json"
    if not attr_path.exists():
        return 0, 0, 0
    try:
        data = json.loads(attr_path.read_text())
    except Exception:
        return 0, 0, 0

    chunks = data.get("chunks", [])
    n_modified = 0
    n_dropped_total = 0
    n_skipped = 0
    for c in chunks:
        cid = c.get("chunk_id")
        if cid is None:
            continue
        # Skip chunks that aren't yet completed by the build (avoid racing it)
        if not (c.get("static_index_text") or "").strip():
            n_skipped += 1
            continue
        frame_paths = c.get("frame_paths") or []
        if len(frame_paths) <= 1:
            continue
        new_paths, n_dropped = cleanup_chunk(video_dir, cid, frame_paths, dry_run)
        if n_dropped > 0:
            c["frame_paths"] = new_paths
            n_modified += 1
            n_dropped_total += n_dropped

    if n_modified > 0 and not dry_run:
        attr_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return n_modified, n_dropped_total, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted, don't touch files or attr_data")
    ap.add_argument("--video-id", default=None,
                    help="Process only this video (useful for testing)")
    args = ap.parse_args()

    if not BANK_DIR.exists():
        print(f"BANK_DIR not found: {BANK_DIR}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(d for d in BANK_DIR.iterdir() if d.is_dir())
    print(f"scanning {len(videos)} videos, threshold ≤ {DEDUP_HAMMING_THRESHOLD}, dry_run={args.dry_run}\n")

    total_videos_touched = 0
    total_chunks_modified = 0
    total_frames_dropped = 0
    for i, vd in enumerate(videos):
        if args.video_id and vd.name != args.video_id:
            continue
        n_chunks, n_drop, n_skip = process_video(vd, args.dry_run, args.video_id)
        if n_chunks > 0:
            total_videos_touched += 1
            total_chunks_modified += n_chunks
            total_frames_dropped += n_drop
            print(f"[{i+1}/{len(videos)}] {vd.name}: {n_chunks} chunks deduped, "
                  f"{n_drop} frames dropped (skipped {n_skip} empty chunks)", flush=True)

    print()
    print(f"summary: {total_videos_touched} videos modified, "
          f"{total_chunks_modified} chunks deduped, {total_frames_dropped} frames dropped")
    if args.dry_run:
        print("(dry-run — no files or attr_data changed)")


if __name__ == "__main__":
    main()
