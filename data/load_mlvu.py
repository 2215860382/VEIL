"""Loader for MLVU-Dev (and MLVU_Test once available)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class MLVUSample:
    video_id: str          # filename without extension, used as memory bank key
    video_path: str
    duration: float
    question: str
    candidates: List[str]
    answer: str            # gold-answer string (one of candidates) for Dev; "" for Test
    question_type: str
    sample_idx: int        # within-task index, for traceability


def _video_path(video_dir: Path, video_subfolder: str, video_filename: str) -> Path:
    """MLVU layout: video_dir/{1_plotQA|2_needle|...}/{video_filename}.

    `video_subfolder` is the JSON filename stem (e.g. "1_plotQA"), which matches
    the actual on-disk video folder. The shorter `question_type` field inside
    the JSON ("plotQA") is NOT the folder name.
    """
    return video_dir / video_subfolder / video_filename


def load_mlvu(
    json_dir: str | Path,
    video_dir: str | Path,
    json_files: dict[str, str],
    task_types: Optional[Iterable[str]] = None,
    max_videos: Optional[int] = None,
    max_questions_per_video: Optional[int] = None,
) -> List[MLVUSample]:
    """Walk MLVU per-task JSON files and yield flattened samples.

    Args:
        json_dir: dir containing 1_plotQA.json ... 9_summary.json
        video_dir: dir whose subfolders match question_type
        json_files: mapping {task_type: json_filename}
        task_types: subset of tasks; None = all
        max_videos: cap unique videos sampled (rough budget control)
        max_questions_per_video: cap questions per video
    """
    json_dir, video_dir = Path(json_dir), Path(video_dir)
    if task_types is None:
        task_types = list(json_files.keys())

    samples: List[MLVUSample] = []
    seen_videos: set[str] = set()
    per_video_count: dict[str, int] = {}

    for task in task_types:
        fname = json_files.get(task)
        if fname is None:
            continue
        video_subfolder = Path(fname).stem  # "1_plotQA.json" -> "1_plotQA"
        with open(json_dir / fname) as f:
            entries = json.load(f)

        for i, entry in enumerate(entries):
            video_filename = entry["video"]
            video_path = _video_path(video_dir, video_subfolder, video_filename)
            video_id = Path(video_filename).stem

            if max_videos is not None and video_id not in seen_videos and len(seen_videos) >= max_videos:
                continue
            if max_questions_per_video is not None:
                if per_video_count.get(video_id, 0) >= max_questions_per_video:
                    continue

            samples.append(
                MLVUSample(
                    video_id=video_id,
                    video_path=str(video_path),
                    duration=float(entry.get("duration", 0)),
                    question=entry["question"],
                    candidates=list(entry["candidates"]),
                    answer=entry.get("answer", ""),
                    question_type=task,
                    sample_idx=i,
                )
            )
            seen_videos.add(video_id)
            per_video_count[video_id] = per_video_count.get(video_id, 0) + 1
    return samples


def unique_videos(samples: List[MLVUSample]) -> List[MLVUSample]:
    """Return one MLVUSample per unique video (first occurrence). Useful for memory build."""
    seen: set[str] = set()
    out = []
    for s in samples:
        if s.video_id in seen:
            continue
        seen.add(s.video_id)
        out.append(s)
    return out
