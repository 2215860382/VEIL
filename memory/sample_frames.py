"""Unified frame sampler — one call site for ALL pipelines.

All methods (direct VQA, memory bank building, retrieval answering) draw frames
from this function to guarantee identical fps / max_frames / resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image


@dataclass
class SampledVideo:
    frames: List[np.ndarray]   # RGB uint8, shape (resolution, resolution, 3)
    timestamps: List[float]    # seconds from video start, one per frame
    fps_native: float
    duration: float
    # Sampling parameters (what was requested, not native video properties)
    target_fps: float = 1.0
    target_max_frames: int = 256
    target_resolution: int = 448


def _resize_and_crop(frame_rgb: np.ndarray, resolution: int) -> np.ndarray:
    """Resize shorter side to `resolution`, center-crop to square."""
    img = Image.fromarray(frame_rgb)
    w, h = img.size
    if w <= h:
        new_w, new_h = resolution, max(resolution, int(h * resolution / w))
    else:
        new_w, new_h = max(resolution, int(w * resolution / h)), resolution
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    return np.array(img)


def sample_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 256,
    resolution: int = 448,
) -> SampledVideo:
    """Decode video, sample at `fps`, cap at `max_frames`, resize+crop to resolution×resolution.

    If the raw sampled count exceeds `max_frames`, frames are uniformly subsampled so that
    all `max_frames` span the full video duration (no content is silently dropped).
    """
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=2)
    fps_native = float(vr.get_avg_fps() or 25.0)
    n_total = len(vr)
    duration = n_total / fps_native

    # Build candidate indices at target fps
    step = fps_native / max(fps, 1e-6)
    raw_idxs = np.unique(np.clip(
        np.arange(0, n_total, step).astype(int), 0, n_total - 1
    ))

    # Uniform subsample if over cap
    if len(raw_idxs) > max_frames:
        sel = np.round(np.linspace(0, len(raw_idxs) - 1, max_frames)).astype(int)
        raw_idxs = raw_idxs[sel]

    timestamps = (raw_idxs / fps_native).tolist()
    raw_frames = vr.get_batch(raw_idxs.tolist()).asnumpy()  # (T, H, W, 3) uint8
    frames = [_resize_and_crop(f, resolution) for f in raw_frames]

    return SampledVideo(
        frames=frames,
        timestamps=timestamps,
        fps_native=fps_native,
        duration=duration,
        target_fps=fps,
        target_max_frames=max_frames,
        target_resolution=resolution,
    )


def sliding_windows(n: int, chunk_size: int, stride: int):
    """Yield (start, end) index pairs for a sliding window over n frames."""
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end
        if end == n:
            break
        start += stride
