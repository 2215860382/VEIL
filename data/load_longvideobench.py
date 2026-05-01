"""Loader for LongVideoBench benchmark.

JSON layout (lvb_val.json):
  video_id, video_path ({videoID}.mp4), question, candidates (list),
  correct_choice (int, 0-indexed), duration (seconds), duration_group,
  position, topic_category, question_category, level

Videos live at {video_dir}/{video_path}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class LongVideoBenchSample:
    video_id: str
    video_path: str
    duration: float
    duration_group: int     # bucket in seconds (e.g. 600, 900, ...)
    question: str
    candidates: List[str]
    answer: str             # gold answer text
    question_category: str
    sample_idx: int


def load_longvideobench(
    json_path: str | Path,
    video_dir: str | Path,
    duration_groups: Optional[Iterable[int]] = None,
    max_samples: Optional[int] = None,
) -> List[LongVideoBenchSample]:
    """Load LongVideoBench samples from JSON.

    Args:
        json_path:       path to lvb_val.json (or lvb_test_wo_gt.json)
        video_dir:       dir containing {video_id}.mp4 files
        duration_groups: filter by duration_group bucket values; None = all
        max_samples:     cap total samples
    """
    video_dir = Path(video_dir)
    entries = json.loads(Path(json_path).read_text())

    if duration_groups is not None:
        dg_set = set(duration_groups)
        entries = [e for e in entries if e.get("duration_group") in dg_set]

    samples: List[LongVideoBenchSample] = []
    for idx, entry in enumerate(entries):
        if max_samples is not None and len(samples) >= max_samples:
            break
        candidates = list(entry["candidates"])
        correct_idx = int(entry["correct_choice"])
        answer_text = candidates[correct_idx] if correct_idx < len(candidates) else ""

        samples.append(LongVideoBenchSample(
            video_id=Path(entry["video_path"]).stem,
            video_path=str(video_dir / entry["video_path"]),
            duration=float(entry.get("duration", 0)),
            duration_group=int(entry.get("duration_group", 0)),
            question=entry["question"],
            candidates=candidates,
            answer=answer_text,
            question_category=entry.get("question_category", ""),
            sample_idx=idx,
        ))
    return samples
