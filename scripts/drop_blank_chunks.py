"""Remove blank/empty chunks from a video bank and renumber chunk_ids.

A "blank chunk" is one whose attribute layer ended up with `static_index_text
== "[blank frames]"`. Those chunks had all-blank or all-deleted frames, so
their attribute layer is degenerate — drop them entirely.

Per video, this script:
  1. Identify blank chunk_ids in attributes.json
  2. Drop those chunks from BOTH attributes.json and narrative.json
  3. Renumber surviving chunks to contiguous chunk_id = 0..N-1
  4. Rename frame files on disk: frames/{old_cid:04d}_{i}.jpg → frames/{new_cid:04d}_{i}.jpg
  5. Delete orphan frame files belonging to dropped chunks
  6. Rebuild vectors.npz (if present) with new chunk_id order

The narrative ↔ attribute join is by chunk_id, so both must be updated atomically.

Run:
    PYTHONPATH=. python scripts/drop_blank_chunks.py [--dry-run] [--video-id VID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")
BLANK_SENTINEL = "[blank frames]"


def process_video(video_dir: Path, dry_run: bool) -> tuple[int, int, bool]:
    """Returns (n_blank_dropped, n_chunks_after, vectors_rebuilt)."""
    attr_path = video_dir / "attributes.json"
    narr_path = video_dir / "narrative.json"
    vec_path = video_dir / "vectors.npz"
    if not attr_path.exists() or not narr_path.exists():
        return 0, 0, False

    try:
        attr = json.loads(attr_path.read_text())
        narr = json.loads(narr_path.read_text())
    except Exception:
        return 0, 0, False

    blank_cids: set[int] = {
        c["chunk_id"] for c in attr.get("chunks", [])
        if (c.get("static_index_text") or "").strip() == BLANK_SENTINEL
    }
    if not blank_cids:
        return 0, len(attr.get("chunks", [])), False

    # Keep + sort surviving chunks by original chunk_id (attribute side is authoritative)
    surv_attr = sorted(
        [c for c in attr["chunks"] if c["chunk_id"] not in blank_cids],
        key=lambda c: c["chunk_id"],
    )
    surv_narr = sorted(
        [c for c in narr["chunks"] if c["chunk_id"] not in blank_cids],
        key=lambda c: c["chunk_id"],
    )
    # Build old → new chunk_id mapping
    cid_map: dict[int, int] = {c["chunk_id"]: new_idx for new_idx, c in enumerate(surv_attr)}

    if dry_run:
        return len(blank_cids), len(surv_attr), vec_path.exists()

    # Rename frame files in ASCENDING old_cid order to avoid overwriting kept files
    # (all renames are downward shifts since we only drop, never insert).
    for c in surv_attr:
        old_cid = c["chunk_id"]
        new_cid = cid_map[old_cid]
        new_frame_paths = []
        for old_rel in c.get("frame_paths") or []:
            base = Path(old_rel).name  # "0042_3.jpg"
            try:
                _, suffix = base.split("_", 1)  # "3.jpg"
            except ValueError:
                new_frame_paths.append(old_rel)
                continue
            new_rel = f"frames/{new_cid:04d}_{suffix}"
            if old_rel != new_rel:
                old_abs = video_dir / old_rel
                new_abs = video_dir / new_rel
                if old_abs.exists():
                    if new_abs.exists():
                        new_abs.unlink()  # safety: stale leftover
                    old_abs.rename(new_abs)
            new_frame_paths.append(new_rel)
        c["frame_paths"] = new_frame_paths
        c["chunk_id"] = new_cid

    for c in surv_narr:
        c["chunk_id"] = cid_map[c["chunk_id"]]

    # Delete orphan frame files belonging to dropped chunks
    frames_dir = video_dir / "frames"
    if frames_dir.exists():
        for cid in blank_cids:
            for jpg in frames_dir.glob(f"{cid:04d}_*.jpg"):
                try:
                    jpg.unlink()
                except Exception:
                    pass

    attr["chunks"] = surv_attr
    attr["num_chunks"] = len(surv_attr)
    narr["chunks"] = surv_narr
    narr["num_chunks"] = len(surv_narr)

    attr_path.write_text(json.dumps(attr, ensure_ascii=False, indent=2))
    narr_path.write_text(json.dumps(narr, ensure_ascii=False, indent=2))

    vectors_rebuilt = False
    if vec_path.exists():
        try:
            with np.load(vec_path) as data:
                old_cids = data["chunk_ids"]
                narrative_vecs = data["narrative_vecs"]
                attribute_vecs = data["attribute_vecs"]
            keep_mask = np.array([int(c) not in blank_cids for c in old_cids])
            kept_old_cids = old_cids[keep_mask]
            new_cids_arr = np.array(
                [cid_map[int(c)] for c in kept_old_cids], dtype=np.int32
            )
            # Sort by new_cid to keep vectors in same order as chunks list
            order = np.argsort(new_cids_arr)
            np.savez_compressed(
                vec_path,
                narrative_vecs=narrative_vecs[keep_mask][order].astype(np.float32),
                attribute_vecs=attribute_vecs[keep_mask][order].astype(np.float32),
                chunk_ids=new_cids_arr[order],
            )
            vectors_rebuilt = True
        except Exception as e:
            print(f"  [vec error] {video_dir.name}: {type(e).__name__}: {e}", flush=True)

    return len(blank_cids), len(surv_attr), vectors_rebuilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    if not BANK_DIR.exists():
        print(f"BANK_DIR not found: {BANK_DIR}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(d for d in BANK_DIR.iterdir() if d.is_dir())
    if args.video_id:
        videos = [d for d in videos if d.name == args.video_id]
    print(f"scanning {len(videos)} videos, dry_run={args.dry_run}\n")

    total_videos = 0
    total_dropped = 0
    total_vec_rebuilt = 0
    for i, vd in enumerate(videos):
        n_drop, n_after, vec_rebuilt = process_video(vd, args.dry_run)
        if n_drop > 0:
            total_videos += 1
            total_dropped += n_drop
            total_vec_rebuilt += int(vec_rebuilt)
            print(f"[{i+1}/{len(videos)}] {vd.name}: dropped {n_drop} blank chunks, "
                  f"{n_after} chunks remain"
                  + (" (vectors rebuilt)" if vec_rebuilt else ""), flush=True)

    print()
    print(f"summary: {total_videos} videos modified, {total_dropped} blank chunks dropped, "
          f"{total_vec_rebuilt} vectors.npz rebuilt")
    if args.dry_run:
        print("(dry-run — no files changed)")


if __name__ == "__main__":
    main()
