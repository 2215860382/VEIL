"""Segment a video into fixed-duration windows and sample frames per window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class VideoSegment:
    chunk_id: int
    start_time: float
    end_time: float
    frame_indices: List[int]              # indices within the original decoded stream
    frames: List[np.ndarray] = field(default_factory=list)  # (H, W, 3) uint8 RGB
    fps_native: float = 0.0


def _open_video(video_path: str):
    """Open with decord; return (vr, fps, total_frames)."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = vr.get_avg_fps() or 25.0
    return vr, fps, len(vr)


def segment_video(
    video_path: str,
    segment_seconds: float = 16.0,
    fps_per_segment: float = 1.0,
    max_frames_per_segment: int = 8,
) -> List[VideoSegment]:
    """Cut [video] into windows of `segment_seconds`; sample `fps_per_segment` frames/s within each.

    Returns a list of VideoSegment with frames as RGB uint8 arrays.
    The last segment may be shorter than `segment_seconds`.
    """
    vr, fps, n = _open_video(video_path)
    duration = n / fps
    seg_n = int(np.ceil(duration / segment_seconds))

    segments: List[VideoSegment] = []
    for i in range(seg_n):
        s = i * segment_seconds
        e = min((i + 1) * segment_seconds, duration)
        n_frames = max(1, min(max_frames_per_segment, int(round((e - s) * fps_per_segment))))
        # Evenly spaced timestamps in [s, e)
        ts = np.linspace(s, e, n_frames + 2)[1:-1]
        idxs = np.clip((ts * fps).astype(int), 0, n - 1).tolist()
        frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3) uint8
        segments.append(
            VideoSegment(
                chunk_id=i,
                start_time=float(s),
                end_time=float(e),
                frame_indices=list(map(int, idxs)),
                frames=[f for f in frames],
                fps_native=float(fps),
            )
        )
    return segments


def video_duration(video_path: str) -> float:
    _, fps, n = _open_video(video_path)
    return n / fps
