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


def keyframe_path(keyframe_dir, video_id: str, chunk_id: int, chunk=None) -> str:
    """Return the sharpest representative frame for a chunk.

    Prefers ``chunk.keyframe_path`` (set at build time to the sharpest frame by
    Laplacian variance) when it exists on disk. Falls back to the lowest-index
    glob match so old banks without the field still work.
    """
    if chunk is not None:
        kp = getattr(chunk, "keyframe_path", "") or ""
        if kp and Path(kp).is_file():
            return kp
    frames_dir = Path(keyframe_dir) / video_id / "frames"
    matches = sorted(frames_dir.glob(f"{chunk_id:04d}_*.jpg"))
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
                   chunk=None) -> List[str]:
    """All surviving frames for a chunk, capped to ``cap``.

    The sharpest frame (by Laplacian variance over existing disk files) is placed
    first so that cap=1 always returns the best frame.
    """
    frames_dir = Path(keyframe_dir) / video_id / "frames"
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
) -> List:
    """Load, dedup, and cap keyframe images for a list of MemoryChunks."""
    pairs = []
    for c in chunks:
        kp = keyframe_path(keyframe_dir, video_id, c.chunk_id, chunk=c)
        pairs.append((c, load_keyframe_pil(kp)))
    return visual_dedup(pairs, dedup_threshold)[:cap]
