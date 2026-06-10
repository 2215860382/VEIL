"""Restore narrative chunks that were mistakenly dropped by drop_blank_chunks.py.

Source of truth: /home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b/{vid}.json
This file is the LEGACY bank where the narrative layer was originally built
with denser frame sampling — so it has narrative content even for chunks where
the attribute-layer build at fps=0.5 happened to land on all-black frames.

Per affected video:
  1. Load legacy bank — authoritative for full chunk list + narrative text
  2. Load current narrative.json + attributes.json (post-drop, renumbered)
  3. Match current chunks to legacy by start_time (round to 0.1s for safety)
  4. For each legacy chunk:
       - if matched in current → keep current data, restore original chunk_id
       - if missing in current → restore narrative from legacy; attribute layer
         gets an empty placeholder so the build pipeline will fill it next pass
  5. Rename frame files: from current new_cid back to legacy original chunk_id
     (two-phase rename to avoid collisions)
  6. Delete vectors.npz — next build pass will regenerate it

Field mapping (legacy → current narrative chunk):
  memory_text     → narrative
  visual_caption  → caption
  sampled_frames  → frame_timestamps, sampled_frames
  asr             → speech_text
  keyframe_ts     → keyframe_ts

Run:
    PYTHONPATH=. python scripts/restore_dropped_chunks.py [--dry-run] [--video-id VID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")
LEGACY_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b")


_VISUALLY_BLANK_PHRASES = [
    "entirely black", "only black screen", "completely black",
    "consists entirely of black", "all black", "black screens",
    "no visible content", "no visible imagery",
]
_VISIBLE_OBJECT_PHRASES = [
    "circle", "logo", "figure", "individual", "person", "object",
    "standing", "sitting", "hoodie", "illuminated", "dimly lit",
    "background music", "camera viewfinder",
]


def is_truly_empty(lc: dict) -> bool:
    """Truly droppable: narrative says all-black AND no speech AND no described objects."""
    text = (lc.get("memory_text") or "").lower()
    asr = (lc.get("asr") or "").strip()
    if asr:
        return False
    if not any(p in text for p in _VISUALLY_BLANK_PHRASES):
        return False
    if any(p in text for p in _VISIBLE_OBJECT_PHRASES):
        return False
    return True


def _legacy_to_narr(lc: dict) -> dict:
    """Convert a legacy chunk into current-format narrative chunk."""
    return {
        "chunk_id": lc["chunk_id"],
        "start_time": lc["start_time"],
        "end_time": lc["end_time"],
        "narrative": lc.get("memory_text", "") or "",
        "caption": lc.get("visual_caption", "") or [],
        "frame_timestamps": list(lc.get("sampled_frames", []) or []),
        "speech_text": lc.get("asr", "") or "",
        "keyframe_ts": lc.get("keyframe_ts", lc.get("start_time", 0.0)),
        "sampled_frames": list(lc.get("sampled_frames", []) or []),
    }


def _empty_attr_chunk(lc: dict) -> dict:
    """Skeleton attribute chunk for a restored narrative chunk — empty so build refills."""
    return {
        "chunk_id": lc["chunk_id"],
        "start_time": lc["start_time"],
        "end_time": lc["end_time"],
        "static_index_text": "",
        "static_attributes": {"entities": []},
        "frame_paths": [],
    }


def process_video(video_dir: Path, dry_run: bool) -> tuple[int, int, int]:
    """Returns (restored, surviving_renumbered, total_after)."""
    vid = video_dir.name
    legacy_path = LEGACY_DIR / f"{vid}.json"
    narr_path = video_dir / "narrative.json"
    attr_path = video_dir / "attributes.json"
    vec_path = video_dir / "vectors.npz"
    if not legacy_path.exists() or not narr_path.exists() or not attr_path.exists():
        return 0, 0, 0

    try:
        legacy = json.loads(legacy_path.read_text())
        narr = json.loads(narr_path.read_text())
        attr = json.loads(attr_path.read_text())
    except Exception as e:
        print(f"  [load error] {vid}: {e}", flush=True)
        return 0, 0, 0

    legacy_chunks = sorted(legacy.get("chunks", []), key=lambda c: c["chunk_id"])
    cur_narr_chunks = sorted(narr.get("chunks", []), key=lambda c: c["chunk_id"])
    cur_attr_chunks = sorted(attr.get("chunks", []), key=lambda c: c["chunk_id"])

    # Map current chunks by rounded start_time (matches legacy precisely if narrative came from same chunker)
    def _tkey(c):
        return round(c["start_time"] * 10)

    cur_narr_by_t = {_tkey(c): c for c in cur_narr_chunks}
    cur_attr_by_t = {_tkey(c): c for c in cur_attr_chunks}

    if len(cur_narr_chunks) == len(legacy_chunks):
        # Nothing to restore — current already has all chunks
        return 0, 0, len(cur_narr_chunks)

    new_narr_chunks: list[dict] = []
    new_attr_chunks: list[dict] = []
    # rename map: current chunk_id (post-drop) → new contiguous chunk_id
    rename_map: dict[int, int] = {}
    restored = 0
    renumbered = 0
    truly_dropped = 0

    new_cid = 0
    for lc in legacy_chunks:
        t = _tkey(lc)
        if t in cur_narr_by_t:
            # Surviving chunk — keep current data, rebind chunk_id to new contiguous index
            cur_nc = cur_narr_by_t[t]
            cur_ac = cur_attr_by_t.get(t)
            old_cid = cur_nc["chunk_id"]
            new_narr_chunks.append({**cur_nc, "chunk_id": new_cid})
            if cur_ac is not None:
                new_ac = {**cur_ac, "chunk_id": new_cid}
                if cur_ac.get("frame_paths"):
                    new_paths = []
                    for old_rel in cur_ac["frame_paths"]:
                        base = Path(old_rel).name  # "{old_cid:04d}_{i}.jpg"
                        try:
                            _, suffix = base.split("_", 1)
                        except ValueError:
                            new_paths.append(old_rel)
                            continue
                        new_paths.append(f"frames/{new_cid:04d}_{suffix}")
                    new_ac["frame_paths"] = new_paths
                new_attr_chunks.append(new_ac)
            else:
                new_attr_chunks.append(_empty_attr_chunk(lc) | {"chunk_id": new_cid})
            if old_cid != new_cid:
                rename_map[old_cid] = new_cid
                renumbered += 1
            new_cid += 1
        else:
            # Dropped chunk — check if it's truly empty (legitimately deletable)
            if is_truly_empty(lc):
                truly_dropped += 1
                continue  # leave deleted
            # Otherwise restore from legacy with new contiguous chunk_id
            nc = _legacy_to_narr(lc)
            nc["chunk_id"] = new_cid
            ac = _empty_attr_chunk(lc)
            ac["chunk_id"] = new_cid
            new_narr_chunks.append(nc)
            new_attr_chunks.append(ac)
            restored += 1
            new_cid += 1

    if restored == 0 and not rename_map:
        return 0, 0, len(legacy_chunks)

    if dry_run:
        return restored, renumbered, len(legacy_chunks)

    # Two-phase rename of frame files to avoid collision
    frames_dir = video_dir / "frames"
    if frames_dir.exists() and rename_map:
        tmp_renamed: list[tuple[Path, Path]] = []
        for jpg in frames_dir.iterdir():
            if not jpg.is_file() or jpg.suffix != ".jpg":
                continue
            name = jpg.name  # "{cid:04d}_{i}.jpg"
            try:
                cid_str, rest = name.split("_", 1)
                old_cid = int(cid_str)
            except (ValueError, IndexError):
                continue
            if old_cid not in rename_map:
                continue
            new_cid = rename_map[old_cid]
            new_name = f"{new_cid:04d}_{rest}"
            tmp_abs = frames_dir / f".tmprename.{new_name}"
            jpg.rename(tmp_abs)
            tmp_renamed.append((tmp_abs, frames_dir / new_name))
        for tmp_abs, final_abs in tmp_renamed:
            if final_abs.exists():
                final_abs.unlink()
            tmp_abs.rename(final_abs)

    narr["chunks"] = sorted(new_narr_chunks, key=lambda c: c["chunk_id"])
    narr["num_chunks"] = len(new_narr_chunks)
    attr["chunks"] = sorted(new_attr_chunks, key=lambda c: c["chunk_id"])
    attr["num_chunks"] = len(new_attr_chunks)

    narr_path.write_text(json.dumps(narr, ensure_ascii=False, indent=2))
    attr_path.write_text(json.dumps(attr, ensure_ascii=False, indent=2))

    # Vectors.npz is now stale (chunk order changed, restored chunks unembedded) — delete it
    if vec_path.exists():
        try:
            vec_path.unlink()
        except Exception:
            pass

    return restored, renumbered, len(legacy_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    if not BANK_DIR.exists():
        print(f"BANK_DIR not found: {BANK_DIR}", file=sys.stderr)
        sys.exit(1)
    if not LEGACY_DIR.exists():
        print(f"LEGACY_DIR not found: {LEGACY_DIR}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(d for d in BANK_DIR.iterdir() if d.is_dir())
    if args.video_id:
        videos = [d for d in videos if d.name == args.video_id]
    print(f"scanning {len(videos)} videos, dry_run={args.dry_run}\n")

    total_videos = 0
    total_restored = 0
    for i, vd in enumerate(videos):
        n_rest, n_renum, n_after = process_video(vd, args.dry_run)
        if n_rest > 0 or n_renum > 0:
            total_videos += 1
            total_restored += n_rest
            print(f"[{i+1}/{len(videos)}] {vd.name}: restored {n_rest} chunks, "
                  f"renumbered {n_renum} survivors, total {n_after} chunks", flush=True)

    print()
    print(f"summary: {total_videos} videos restored, {total_restored} narrative chunks recovered")
    if args.dry_run:
        print("(dry-run — no files changed)")


if __name__ == "__main__":
    main()
