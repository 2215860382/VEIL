"""One-shot bank repair: refill ``frames/{cid:04d}_*.jpg`` for chunks the legacy
``build_attribute_layer.py`` dropped entirely. Re-extracts via ffmpeg on the
chunk's [t_start, t_end] window and runs the new ``frame_pipeline.py`` filters
(blank/blur/dedup). Chunks with truly no content (all-black) stay empty.

Inference (``experiments/core/_keyframes.py``) globs ``frames/`` directly, so
narrative.json doesn't need updating — just dropping the JPGs is enough.

Usage:
    PYTHONPATH=. python scripts/refill_missing_chunk_frames.py \\
        --bank-root outputs/memory/videomme_L_27B \\
        --video-dir /home2/ycj/Datas/VideoMME/videos \\
        --embed-api-url http://localhost:9000 \\
        [--dry-run] [--video-id <vid>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from src.build_memory.core.frame_pipeline import (
    is_blank, is_blurry, _dedup_chunk,
    DEFAULT_BLANK_BYTES_MIN, DEFAULT_BLUR_THRESHOLD, BLUR_FAILSAFE_RATIO,
    DEFAULT_DEDUP_HAMMING, DEFAULT_DEDUP_SIGLIP_COS,
)


def extract_window(video_path: Path, t_start: float, t_end: float,
                   out_dir: Path, fps: float = 1.0) -> List[Path]:
    """ffmpeg extract [t_start, t_end] at given fps into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}", "-to", f"{t_end:.3f}",
        "-i", str(video_path),
        "-vf", f"fps={fps}", "-q:v", "3",
        str(out_dir / "frame_%03d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def refill_chunk(video_path: Path, t_start: float, t_end: float,
                 cid: int, frames_dir: Path, siglip) -> dict:
    """Refill one chunk's frames. Returns a status dict."""
    with tempfile.TemporaryDirectory(prefix=f"refill_{cid}_") as tmp:
        tmp_dir = Path(tmp)
        raw = extract_window(video_path, t_start, t_end, tmp_dir, fps=1.0)
        if not raw:
            return {"cid": cid, "status": "no_frames_extracted"}

        non_blank = [p for p in raw if not is_blank(p, DEFAULT_BLANK_BYTES_MIN)]
        if not non_blank:
            return {"cid": cid, "status": "all_blank",
                    "n_extracted": len(raw)}

        blurry_flags = [is_blurry(p, DEFAULT_BLUR_THRESHOLD) for p in non_blank]
        if sum(blurry_flags) / len(blurry_flags) > BLUR_FAILSAFE_RATIO:
            survived = non_blank
            blur_skipped = True
        else:
            survived = [p for p, bl in zip(non_blank, blurry_flags) if not bl]
            blur_skipped = False

        if not survived:
            return {"cid": cid, "status": "all_blurry",
                    "n_extracted": len(raw), "n_non_blank": len(non_blank)}

        v = siglip.encode_images([str(p) for p in survived])
        kept_idx = _dedup_chunk(
            survived, np.asarray(v),
            hamming_threshold=DEFAULT_DEDUP_HAMMING,
            siglip_cos_threshold=DEFAULT_DEDUP_SIGLIP_COS,
        )
        if not kept_idx:
            return {"cid": cid, "status": "all_dedup'd",
                    "n_extracted": len(raw), "n_survived": len(survived)}

        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, local_i in enumerate(kept_idx):
            dst = frames_dir / f"{cid:04d}_{i}.jpg"
            shutil.copyfile(survived[local_i], dst)

        return {
            "cid": cid, "status": "refilled",
            "n_extracted": len(raw),
            "n_non_blank": len(non_blank),
            "n_survived_blur": len(survived),
            "n_kept": len(kept_idx),
            "blur_failsafe": blur_skipped,
        }


def scan_bank(bank_root: Path, video_id: str | None = None) -> list:
    """Find chunks whose narrative.json entry exists but frames/{cid:04d}_*.jpg is empty."""
    todo = []
    vids = ([bank_root / video_id] if video_id else
            sorted(p for p in bank_root.iterdir() if p.is_dir()))
    for vd in vids:
        nj = vd / "narrative.json"
        if not nj.exists():
            continue
        narr = json.loads(nj.read_text())
        frames_dir = vd / "frames"
        for c in narr.get("chunks", []):
            cid = c["chunk_id"]
            if not list(frames_dir.glob(f"{cid:04d}_*.jpg")):
                todo.append((vd.name, cid,
                             c.get("start_time", 0.0),
                             c.get("end_time", 0.0)))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root",    type=Path, required=True)
    ap.add_argument("--video-dir",    type=Path, required=True)
    ap.add_argument("--embed-api-url", default="http://localhost:9000")
    ap.add_argument("--video-id",     default=None,
                    help="Restrict to a single video")
    ap.add_argument("--dry-run",      action="store_true",
                    help="List missing chunks, do nothing")
    args = ap.parse_args()

    todo = scan_bank(args.bank_root, args.video_id)
    n_videos = len({v for v, *_ in todo})
    print(f"Found {len(todo)} chunks with empty frames/ across {n_videos} videos")

    if args.dry_run or not todo:
        for v, cid, ts, te in todo[:20]:
            print(f"  {v}  cid={cid}  [{ts:.0f}-{te:.0f}s]")
        if len(todo) > 20:
            print(f"  ... ({len(todo) - 20} more)")
        return

    from src.clients.siglip_embedder import SigLIPEmbedder
    siglip = SigLIPEmbedder(api_url=args.embed_api_url)

    stats: dict = {}
    for i, (vid, cid, ts, te) in enumerate(todo):
        video_path = args.video_dir / f"{vid}.mp4"
        if not video_path.exists():
            print(f"  [{i+1}/{len(todo)}] {vid}#{cid}: VIDEO NOT FOUND")
            stats["video_missing"] = stats.get("video_missing", 0) + 1
            continue
        frames_dir = args.bank_root / vid / "frames"
        try:
            r = refill_chunk(video_path, ts, te, cid, frames_dir, siglip)
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] {vid}#{cid}: ERROR {e}")
            stats["error"] = stats.get("error", 0) + 1
            continue
        stats[r["status"]] = stats.get(r["status"], 0) + 1
        if r["status"] == "refilled":
            print(f"  [{i+1}/{len(todo)}] {vid}#{cid} [{ts:.0f}-{te:.0f}s]: "
                  f"+{r['n_kept']} frames (extracted={r['n_extracted']}, "
                  f"blur_safe={r['blur_failsafe']})")
        else:
            print(f"  [{i+1}/{len(todo)}] {vid}#{cid} [{ts:.0f}-{te:.0f}s]: "
                  f"{r['status']}")

    print(f"\nSummary: {stats}")


if __name__ == "__main__":
    main()
