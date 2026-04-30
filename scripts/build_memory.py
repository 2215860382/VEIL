"""Entry point: build memory banks for all unique videos in a benchmark.

Usage:
    cd /home2/ycj/Project/VEIL
    python scripts/build_memory.py --config configs/mlvu.yaml \
        --task-types plotQA --max-videos 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local package import works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_mlvu import load_mlvu, unique_videos
from memory.build_memory import build_memory_bank, memory_bank_path
from memory.schema import MemoryBank
from models.vlm_client import VLMClient
from utils.config import load_config
from utils.logging import get_logger

log = get_logger("veil.build_memory")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task-types", nargs="*", default=None,
                   help="Subset of MLVU task types to include. Default: all from config.")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Rebuild even if cached.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    bench = cfg["benchmark"]
    task_types = args.task_types if args.task_types is not None else bench.get("task_types")
    cache_dir = Path(cfg["memory"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = load_mlvu(
        json_dir=bench["json_dir"],
        video_dir=bench["video_dir"],
        json_files=bench["json_files"],
        task_types=task_types,
        max_videos=args.max_videos,
    )
    videos = unique_videos(samples)
    log.info("Found %d unique videos across %s", len(videos), task_types or "all")

    # Lazy-load VLM only if there's work to do.
    pending = []
    for s in videos:
        out = memory_bank_path(cache_dir, s.video_id)
        if out.exists() and not args.force:
            continue
        pending.append((s, out))

    if not pending:
        log.info("All memory banks already cached at %s", cache_dir)
        return

    log.info("Loading VLM %s", cfg["models"]["vlm"]["model_path"])
    vlm = VLMClient(**cfg["models"]["vlm"])

    mem_cfg = cfg["memory"]
    for s, out in pending:
        log.info("Building memory: %s (%s)", s.video_id, s.video_path)
        bank = build_memory_bank(
            video_path=s.video_path,
            video_id=s.video_id,
            vlm=vlm,
            segment_seconds=mem_cfg["segment_seconds"],
            fps_per_segment=mem_cfg["fps_per_segment"],
            max_frames_per_segment=mem_cfg["max_frames_per_segment"],
        )
        bank.save(out)
        log.info("  → %s (%d chunks)", out, len(bank.chunks))


if __name__ == "__main__":
    main()
