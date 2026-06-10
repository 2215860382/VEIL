"""Delete frame JPGs that aren't referenced by any chunk's frame_paths.

After the build pipeline went through a few iterations of dedup/blank-detect
logic changes, some chunks' attributes.json correctly excludes blank or duplicate
frames, but the actual JPG files still sit on disk as orphans. Sweep them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")


def sweep_video(video_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (orphans_found, bytes_freed)."""
    attr_path = video_dir / "attributes.json"
    frames_dir = video_dir / "frames"
    if not attr_path.exists() or not frames_dir.exists():
        return 0, 0
    try:
        data = json.loads(attr_path.read_text())
    except Exception:
        return 0, 0

    referenced: set[str] = set()
    for c in data.get("chunks", []):
        for rel in c.get("frame_paths") or []:
            referenced.add(rel)

    orphans = 0
    bytes_freed = 0
    for jpg in frames_dir.iterdir():
        if not jpg.is_file() or jpg.suffix != ".jpg":
            continue
        rel = f"frames/{jpg.name}"
        if rel in referenced:
            continue
        orphans += 1
        bytes_freed += jpg.stat().st_size
        if not dry_run:
            try:
                jpg.unlink()
            except Exception:
                pass
    return orphans, bytes_freed


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

    total_orphans = 0
    total_bytes = 0
    touched = 0
    for i, vd in enumerate(videos):
        n, b = sweep_video(vd, args.dry_run)
        if n > 0:
            total_orphans += n
            total_bytes += b
            touched += 1
            print(f"[{i+1}/{len(videos)}] {vd.name}: {n} orphans ({b/1024:.1f} KB)", flush=True)

    print()
    print(f"summary: {touched} videos touched, {total_orphans} orphan frames, "
          f"{total_bytes/1024/1024:.1f} MB freed")
    if args.dry_run:
        print("(dry-run — no files changed)")


if __name__ == "__main__":
    main()
