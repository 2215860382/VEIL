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


def keyframe_path(keyframe_dir, video_id: str, chunk_id: int) -> str:
    return str(Path(keyframe_dir) / video_id / "keyframes" / f"{chunk_id:04d}.jpg")


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
        kp = keyframe_path(keyframe_dir, video_id, c.chunk_id)
        pairs.append((c, load_keyframe_pil(kp)))
    return visual_dedup(pairs, dedup_threshold)[:cap]
