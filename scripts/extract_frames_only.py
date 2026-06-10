"""Extract + dedup keyframes for a list of videos. No VLM, no attributes.json changes.

Reuses helpers from build_attribute_layer (ffmpeg call, dHash, SigLIP dedup).
Output: ``{BANK_DIR}/{video_id}/frames/{cid:04d}_{i}.jpg`` (post-dedup, renumbered).

Usage:
    PYTHONPATH=. python scripts/extract_frames_only.py \\
        --video-ids-file /tmp/rest_196.txt \\
        --siglip-device cuda:4 \\
        --concurrency 32 \\
        --ffmpeg-workers 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_memory.build_attribute_layer import (
    BANK_DIR,
    DEDUP_HAMMING_THRESHOLD,
    SIGLIP_COS_THRESHOLD,
    VIDEO_DIR,
    compute_dhash,
    dhash_hamming,
    extract_frame_from_video,
    select_frame_positions,
)


def _siglip_encode_sync(siglip, p: Path):
    if siglip is None:
        return None
    try:
        return siglip.encode_images([str(p)])[0]
    except Exception:
        return None


async def extract_for_video(
    video_id: str,
    siglip,
    ffmpeg_executor,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    """Returns (n_extracted, n_dup, n_blank, n_failed)."""
    video_dir = BANK_DIR / video_id
    narr_path = video_dir / "narrative.json"
    if not narr_path.exists():
        print(f"  [skip] {video_id}: no narrative.json", flush=True)
        return 0, 0, 0, 0
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        print(f"  [skip] {video_id}: video file missing", flush=True)
        return 0, 0, 0, 0

    narrative = json.loads(narr_path.read_text())
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build extraction list — skip chunks that already have at least one frame on disk
    frames_to_extract: list[tuple[int, int, Path, float]] = []
    for nchunk in narrative.get("chunks", []):
        cid = nchunk["chunk_id"]
        if not overwrite and list(frames_dir.glob(f"{cid:04d}_*.jpg")):
            continue
        positions = select_frame_positions(nchunk.get("frame_timestamps", []))
        for i, (_local_idx, ts) in enumerate(positions):
            fpath = frames_dir / f"{cid:04d}_{i}.jpg"
            frames_to_extract.append((cid, i, fpath, ts))

    if not frames_to_extract:
        return 0, 0, 0, 0

    # Per-chunk dedup state
    chunk_state: dict[int, dict] = {}
    for cid, _, _, _ in frames_to_extract:
        chunk_state.setdefault(cid, {"lock": asyncio.Lock(), "hashes": [], "embs": [], "kept": []})

    loop = asyncio.get_running_loop()

    async def extract_one_frame(meta) -> str:
        cid, i, fpath, ts = meta
        try:
            status = await loop.run_in_executor(
                ffmpeg_executor, extract_frame_from_video, video_path, ts, fpath
            )
            if status == "failed":
                return "FAILED"
            if status == "blank":
                try: fpath.unlink(missing_ok=True)
                except: pass
                return "BLANK"
            h = await loop.run_in_executor(ffmpeg_executor, compute_dhash, fpath)
            st = chunk_state[cid]
            async with st["lock"]:
                if h is not None and any(
                    dhash_hamming(h, prev) <= DEDUP_HAMMING_THRESHOLD for prev in st["hashes"]
                ):
                    try: fpath.unlink(missing_ok=True)
                    except: pass
                    return "DUP"
                emb = None
                if siglip is not None:
                    emb = await loop.run_in_executor(ffmpeg_executor, _siglip_encode_sync, siglip, fpath)
                    if emb is not None and st["embs"]:
                        prev = np.stack(st["embs"])
                        if float((prev @ emb).max()) >= SIGLIP_COS_THRESHOLD:
                            try: fpath.unlink(missing_ok=True)
                            except: pass
                            return "DUP"
                if h is not None: st["hashes"].append(h)
                if emb is not None: st["embs"].append(emb)
                st["kept"].append((i, fpath))
            return "OK"
        except Exception as e:
            print(f"  [err] {video_id} c{cid} f{i}: {e}", flush=True)
            return "FAILED"

    results = await asyncio.gather(
        *(extract_one_frame(m) for m in frames_to_extract), return_exceptions=False
    )
    n_ok = sum(1 for r in results if r == "OK")
    n_dup = sum(1 for r in results if r == "DUP")
    n_blank = sum(1 for r in results if r == "BLANK")
    n_failed = sum(1 for r in results if r == "FAILED")

    # Renumber kept frames to contiguous _0, _1, ... per chunk (DUP holes filled)
    for cid, st in chunk_state.items():
        kept = sorted(st["kept"], key=lambda x: x[0])
        for new_i, (old_i, old_path) in enumerate(kept):
            new_path = frames_dir / f"{cid:04d}_{new_i}.jpg"
            if old_path == new_path:
                continue
            try:
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)
            except Exception as e:
                print(f"  [rename-fail] {video_id} c{cid} {old_path.name}→{new_path.name}: {e}", flush=True)

    return n_ok, n_dup, n_blank, n_failed


async def main_async(args):
    siglip = None
    if not args.no_siglip:
        from src.clients.siglip_embedder import SigLIPEmbedder
        siglip = SigLIPEmbedder(model_path=args.siglip_model, device=args.siglip_device)
        print(f"SigLIP ready on {args.siglip_device}", flush=True)

    video_ids = [l.strip() for l in open(args.video_ids_file) if l.strip()]
    print(f"videos to process: {len(video_ids)}", flush=True)

    ffmpeg_executor = ThreadPoolExecutor(max_workers=args.ffmpeg_workers)

    t_total = time.time()
    for i, vid in enumerate(video_ids):
        t0 = time.time()
        try:
            n_ok, n_dup, n_blank, n_failed = await extract_for_video(
                vid, siglip, ffmpeg_executor, args.overwrite
            )
        except Exception as e:
            print(f"[{i+1}/{len(video_ids)}] {vid}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(video_ids)}] {vid}: ok={n_ok} dup={n_dup} blank={n_blank} fail={n_failed}  {elapsed:.1f}s", flush=True)

    print(f"\nTOTAL: {time.time()-t_total:.1f}s for {len(video_ids)} videos", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids-file", required=True)
    ap.add_argument("--siglip-model", default="/home2/ycj/Models/google/siglip-large-patch16-384")
    ap.add_argument("--siglip-device", default="cuda:4")
    ap.add_argument("--no-siglip", action="store_true",
                    help="Disable SigLIP semantic dedup (dHash only)")
    ap.add_argument("--ffmpeg-workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract chunks that already have frames")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
