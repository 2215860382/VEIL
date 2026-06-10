"""Dynamic frame grouping based on CLIP cosine similarity.

Groups consecutive similar frames into segments ("shots"), mirroring the
SigLIP-based approach from the 3-tier memory architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FrameGroup:
    frame_indices: List[int]   # indices into the original 1-fps frame list
    t_start: float
    t_end: float
    center_idx: int            # index of the center/sharpest frame
    size: int


def group_frames(
    v_frames: np.ndarray,      # (T, D) L2-normalized CLIP embeddings
    timestamps: List[float],   # length T, seconds
    theta: float = 0.80,
    n_max: int = 30,
    min_size: int = 3,
) -> List[FrameGroup]:
    """Greedy cosine-similarity grouping of frames.

    Adjacent frames with cosine similarity >= theta are merged into one group,
    up to n_max frames.  Short groups (< min_size) are merged into the left
    neighbour.
    """
    T = len(v_frames)
    if T == 0:
        return []

    groups: List[List[int]] = []
    current: List[int] = [0]

    for t in range(1, T):
        sim = float(v_frames[t] @ v_frames[t - 1])
        if sim >= theta and len(current) < n_max:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)

    # Merge short groups into left neighbour
    merged: List[List[int]] = []
    for g in groups:
        if len(g) < min_size and merged:
            merged[-1].extend(g)
        else:
            merged.append(g)

    result: List[FrameGroup] = []
    for g in merged:
        center = g[len(g) // 2]
        result.append(FrameGroup(
            frame_indices=g,
            t_start=timestamps[g[0]],
            t_end=timestamps[g[-1]] + 1.0,  # +1s so end > start
            center_idx=center,
            size=len(g),
        ))
    return result


def select_sharpest(frame_paths: List[str], indices: List[int]) -> int:
    """Return the index (into indices) with highest Laplacian variance."""
    import cv2
    best_idx, best_score = 0, -1.0
    for i, fi in enumerate(indices):
        img = cv2.imread(frame_paths[fi], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if score > best_score:
            best_score, best_idx = score, i
    return indices[best_idx]
