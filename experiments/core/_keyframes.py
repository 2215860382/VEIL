"""Shared keyframe loading and visual deduplication helpers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_keyframe_pil(path: str):
    try:
        from PIL import Image
        p = Path(path)
        if p.is_file():
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


DEFAULT_KEYFRAME_SUBDIR = "keyframes_origin"
LEGACY_KEYFRAME_SUBDIR  = "frames"


def _resolve_subdir(video_dir: Path, subdir: str) -> str:
    """Pick the actual keyframe subdir for this bank.

    New layout is ``keyframes_origin/`` (+ ``keyframes_resized/``); legacy banks
    only have ``frames/``. If the requested subdir is missing, fall back to
    legacy ``frames/`` so old banks still work.
    """
    if (video_dir / subdir).is_dir():
        return subdir
    if (video_dir / LEGACY_KEYFRAME_SUBDIR).is_dir():
        return LEGACY_KEYFRAME_SUBDIR
    return subdir


def keyframe_path(keyframe_dir, video_id: str, chunk_id: int, chunk=None,
                  subdir: str = DEFAULT_KEYFRAME_SUBDIR) -> str:
    """Return the sharpest representative frame for a chunk.

    Looks under ``{keyframe_dir}/{video_id}/{subdir}/`` (default
    ``keyframes_origin``). Falls back to ``frames/`` for legacy banks that
    predate the origin/resized split. ``chunk.keyframe_path`` is consulted
    only when its on-disk file exists; otherwise the chunk_id glob is used so
    the subdir switch (origin ↔ resized) is honoured.
    """
    video_dir = Path(keyframe_dir) / video_id
    actual = _resolve_subdir(video_dir, subdir)

    if chunk is not None:
        kp = getattr(chunk, "keyframe_path", "") or ""
        if kp:
            kp_path = Path(kp)
            if kp_path.is_absolute() and kp_path.is_file():
                return str(kp_path)
            # If chunk.keyframe_path points to a different subdir than the
            # caller wants, swap to the requested actual subdir and try glob.
            name = kp_path.name
            alt = video_dir / actual / name
            if alt.is_file():
                return str(alt)

    matches = sorted((video_dir / actual).glob(f"{chunk_id:04d}_*.jpg"))
    return str(matches[0]) if matches else ""


def _sharpest(paths: List[str]) -> str:
    """Return the path with highest Laplacian variance (sharpest). Falls back to paths[0]."""
    import cv2
    best_path, best_score = paths[0], -1.0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if score > best_score:
            best_score, best_path = score, p
    return best_path


def keyframe_paths(keyframe_dir, video_id: str, chunk_id: int, cap: int = 3,
                   chunk=None, subdir: str = DEFAULT_KEYFRAME_SUBDIR) -> List[str]:
    """All surviving frames for a chunk, capped to ``cap``.

    The sharpest frame (by Laplacian variance over existing disk files) is placed
    first so that cap=1 always returns the best frame. ``subdir`` selects
    ``keyframes_origin`` vs ``keyframes_resized``; falls back to legacy
    ``frames/`` if the requested subdir is missing.
    """
    video_dir = Path(keyframe_dir) / video_id
    actual = _resolve_subdir(video_dir, subdir)
    frames_dir = video_dir / actual
    all_matches = [str(p) for p in sorted(frames_dir.glob(f"{chunk_id:04d}_*.jpg"))]
    # pyramid naming: c{chunk_id:05d}_f*.jpg
    if not all_matches:
        all_matches = [str(p) for p in sorted(frames_dir.glob(f"c{chunk_id:05d}_f*.jpg"))]
    if not all_matches:
        return []

    best = _sharpest(all_matches) if len(all_matches) > 1 else all_matches[0]
    rest = [p for p in all_matches if p != best]
    return ([best] + rest)[:cap]


def visual_dedup(chunks_with_imgs: List[Tuple], threshold: float = 0.92) -> List:
    """Return visually-unique PIL images (cosine sim of v_visual < threshold)."""
    kept = []
    kept_vecs = []
    for c, img in chunks_with_imgs:
        if img is None:
            continue
        if c.v_visual and kept_vecs:
            v = np.array(c.v_visual, dtype=np.float32)
            mat = np.array(kept_vecs, dtype=np.float32)
            if float(np.max(mat @ v)) >= threshold:
                continue
        kept.append(img)
        kept_vecs.append(c.v_visual if c.v_visual else [])
    return kept


def load_keyframes(
    chunks,
    keyframe_dir,
    video_id: str,
    dedup_threshold: float = 0.92,
    cap: int = 8,
    subdir: str = DEFAULT_KEYFRAME_SUBDIR,
) -> List:
    """Load, dedup, and cap keyframe images for a list of MemoryChunks."""
    pairs = []
    for c in chunks:
        kp = keyframe_path(keyframe_dir, video_id, c.chunk_id, chunk=c, subdir=subdir)
        pairs.append((c, load_keyframe_pil(kp)))
    return visual_dedup(pairs, dedup_threshold)[:cap]
