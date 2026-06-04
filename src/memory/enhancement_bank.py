"""Phase 2: Multi-frame supplement + VLM enhancement for two-layer banks.

Takes Phase 1 converted banks (single keyframe + empty narrative fields)
and enhances them with:
1. Multiple frames extracted from video (mid-frame, end-frame)
2. VLM-generated attributes for each frame (colors, clothing, spatial_layout, etc.)
3. VLM-generated narrative fields (state_changes, temporal_relations, causal_clues)
4. Re-computed v_static vectors (merged from multi-frame attributes)

Usage:
    cd /home2/ycj/Project/VEIL_twoLayer
    python -m src.memory.enhancement_bank \
        --bank-dir /path/to/phase1_banks \
        --video-dir /path/to/videos \
        --vlm-model /path/to/Qwen-VL \
        --device cuda:0 \
        --output-dir /path/to/output_banks
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.core.schema import MemoryBank, StaticAttributeFrame
from src.memory.similarity import (
    DYNAMIC_SYS,
    STATIC_ATTR_SYS,
    extract_static_attributes,
    build_static_index_text,
)
from src.models.embedder import BGEM3Embedder
from src.models.vlm_client import VLMClient
from src.utils.logging import get_logger

log = get_logger("enhancement_bank")


def _extract_frames_from_video(
    video_path: str,
    sampled_frames: List[float],
    out_dir: Path,
    fps: float = 1.0,
    max_frames: int = 3,
    timeout: int = 30,
) -> List[Path]:
    """Extract specific frames from video at sampled_frames timestamps.

    Args:
        video_path: Path to video file
        sampled_frames: List of timestamps (in seconds) to extract
        out_dir: Output directory for frames
        fps: Video FPS (for frame number calculation)
        max_frames: Max frames to extract (evenly spaced from sampled_frames)
        timeout: Timeout per frame extraction in seconds (default 30)

    Returns:
        List of extracted frame paths
    """
    if not sampled_frames:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    # Select frames: first, middle, last
    indices = [0, len(sampled_frames) // 2, len(sampled_frames) - 1]
    indices = list(set(indices))[:max_frames]  # Remove duplicates, keep up to max_frames
    selected_times = [sampled_frames[i] for i in indices]

    frame_paths = []
    for frame_idx, t in enumerate(selected_times):
        frame_path = out_dir / f"frame_{frame_idx:02d}_{t:.1f}s.jpg"

        # Use ffmpeg to extract frame at timestamp
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(t),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
            "-hide_banner", "-loglevel", "error", "-y",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
            if frame_path.exists():
                frame_paths.append(frame_path)
        except subprocess.TimeoutExpired:
            log.warning("Frame extraction timeout at %.1fs (exceeds %ds): skipping", t, timeout)
        except Exception as e:
            log.warning("Failed to extract frame at %.1fs: %s", t, e)

    return frame_paths


def _enhance_chunk(
    chunk,
    video_path: Optional[str],
    vlm,
    embedder,
) -> None:
    """Phase 2: Enhance a single chunk with multi-frame attributes + VLM fields.

    Args:
        chunk: MemoryChunk to enhance
        video_path: Path to source video (for frame extraction)
        vlm: VLMClient for attribute + narrative generation
        embedder: BGE embedder for v_static computation
    """
    # ─────────────────────────────────────────────────────────────
    # 1. Extract multi-frames from video
    # ─────────────────────────────────────────────────────────────

    additional_frames: List[StaticAttributeFrame] = []

    if video_path and Path(video_path).exists() and chunk.sampled_frames:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = _extract_frames_from_video(
                video_path,
                chunk.sampled_frames,
                Path(tmpdir),
                max_frames=3,  # Extract up to 3 frames
                timeout=30,  # 30 seconds per frame (increased from 10)
            )

            # Extract attributes for each frame
            for frame_idx, frame_path in enumerate(frame_paths):
                try:
                    static_attrs = extract_static_attributes(
                        str(frame_path),
                        frame_id=f"{chunk.video_id}_chunk{chunk.chunk_id:03d}_f{frame_idx}",
                        timestamp=chunk.sampled_frames[min(
                            frame_idx * (len(chunk.sampled_frames) - 1) // max(len(frame_paths) - 1, 1),
                            len(chunk.sampled_frames) - 1
                        )],
                        vlm=vlm,
                    )
                    if static_attrs:
                        additional_frames.append(static_attrs)
                except Exception as e:
                    log.debug("Static attribute extraction failed for frame %d: %s", frame_idx, e)

    # ─────────────────────────────────────────────────────────────
    # 2. Update static_frames with multi-frame attributes
    # ─────────────────────────────────────────────────────────────

    if additional_frames:
        # Merge with existing keyframe
        all_frames = chunk.static_frames + additional_frames
        chunk.static_frames = all_frames[:3]  # Keep up to 3 frames

        # Rebuild static_index_text from merged frames
        combined_attrs = {
            "ocr_text": [],
            "numbers": [],
            "colors": [],
            "objects": [],
            "people_appearance": [],
            "clothing": [],
            "spatial_layout": [],
            "textures": [],
            "scene_attributes": [],
        }

        for frame in chunk.static_frames:
            # Convert StaticAttributeFrame to dict if needed
            frame_dict = frame.dict() if hasattr(frame, 'dict') else frame
            for key in combined_attrs:
                val = frame_dict.get(key, []) if isinstance(frame_dict, dict) else getattr(frame, key, [])
                if isinstance(val, list):
                    # Convert all values to strings to handle mixed types
                    str_val = [str(v) for v in val if v]
                    combined_attrs[key].extend(str_val)

        # Remove duplicates while preserving order
        for key in combined_attrs:
            # Filter and stringify to avoid unhashable types
            unique_vals = []
            seen = set()
            for v in combined_attrs[key]:
                v_str = str(v) if not isinstance(v, str) else v
                if v_str and v_str not in seen:
                    unique_vals.append(v_str)
                    seen.add(v_str)
            combined_attrs[key] = unique_vals

        chunk.static_index_text = build_static_index_text(combined_attrs)

        # Recompute v_static from merged attributes
        text_to_encode = chunk.static_index_text if chunk.static_index_text else " "
        v_static = embedder.encode([text_to_encode])[0].tolist()
        chunk.v_static = v_static

    # ─────────────────────────────────────────────────────────────
    # 3. VLM enhancement for narrative layer
    # ─────────────────────────────────────────────────────────────

    if chunk.event_summary or chunk.asr:  # Only if we have source material
        narrative_input = f"Event summary: {chunk.event_summary}\n\nOverall summary: {chunk.memory_text}"
        if chunk.asr:
            narrative_input += f"\n\nSpoken text: {chunk.asr}"

        messages = [
            {"role": "system", "content": DYNAMIC_SYS},
            {"role": "user", "content": narrative_input},
        ]

        try:
            raw = vlm._generate(messages, max_new_tokens=256).strip()
            import json as _json
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = _json.loads(m.group())
                # Update narrative fields (keep summary unchanged)
                chunk.key_events = d.get("key_events", []) if isinstance(d.get("key_events"), list) else []
                chunk.actors = d.get("actors", []) if isinstance(d.get("actors"), list) else []
                chunk.state_changes = d.get("state_changes", []) if isinstance(d.get("state_changes"), list) else []
                chunk.temporal_relations = d.get("temporal_relations", []) if isinstance(d.get("temporal_relations"), list) else []
                chunk.causal_clues = d.get("causal_clues", []) if isinstance(d.get("causal_clues"), list) else []
        except Exception as e:
            log.debug("Narrative enhancement failed: %s", e)


def enhance_bank_file(
    input_path: Path,
    video_path: Optional[str],
    vlm,
    embedder,
    output_path: Optional[Path] = None,
) -> Path:
    """Phase 2: Enhance a single bank file with multi-frame + VLM fields.

    Args:
        input_path: Path to Phase 1 converted bank JSON
        video_path: Path to source video (optional)
        vlm: VLMClient for enhancement
        embedder: BGE embedder for v_static vectors
        output_path: Where to save enhanced bank (default: same as input, in-place)

    Returns:
        Path to enhanced bank file
    """
    if output_path is None:
        output_path = input_path

    log.info("Loading bank: %s", input_path)
    bank = MemoryBank.load(input_path)

    log.info("Phase 2: Enhancing %d chunks with multi-frame + VLM...", len(bank.chunks))

    for i, chunk in enumerate(bank.chunks):
        if i % 100 == 0:
            log.info("  [%d/%d]", i, len(bank.chunks))
        try:
            _enhance_chunk(chunk, video_path, vlm, embedder)
        except Exception as e:
            log.warning("  chunk %d failed: %s", i, e)
            # Continue processing other chunks even if one fails

    log.info("Saving enhanced bank: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output_path)
    return output_path


def _get_video_path(bank_dir: Path, video_id: str, video_dir: Optional[Path]) -> Optional[str]:
    """Locate video file for a given video_id."""
    if not video_dir:
        return None

    video_dir = Path(video_dir)
    if not video_dir.exists():
        return None

    # Common extensions
    for ext in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
        video_path = video_dir / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)

    return None


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: Multi-frame enhancement + VLM fields for two-layer banks"
    )
    ap.add_argument(
        "--bank-dir",
        required=True,
        help="Directory containing Phase 1 converted bank JSON files",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as bank-dir, in-place upgrade)",
    )
    ap.add_argument(
        "--video-dir",
        default=None,
        help="Directory containing source videos (optional, for multi-frame extraction)",
    )
    ap.add_argument(
        "--vlm-model",
        default=None,
        help="Path to VLM checkpoint (or set --vlm-api-url instead)",
    )
    ap.add_argument(
        "--vlm-api-url",
        default=None,
        help="VLM API server URL (e.g., http://localhost:8000). If set, --vlm-model is ignored",
    )
    ap.add_argument(
        "--bge-model",
        default=None,
        help="Path to BGE model (default: from config)",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Device for VLM/BGE (default: cuda:0)",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard ID for multi-GPU processing (0 or 1, default 0)",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (default 1, set to 2 for multi-GPU)",
    )
    args = ap.parse_args()

    bank_dir = Path(args.bank_dir)
    output_dir = Path(args.output_dir) if args.output_dir else bank_dir
    video_dir = Path(args.video_dir) if args.video_dir else None

    if not bank_dir.exists():
        ap.error(f"Bank directory not found: {bank_dir}")

    if not args.vlm_model and not args.vlm_api_url:
        ap.error("Either --vlm-model or --vlm-api-url must be provided")

    log.info("Phase 2: Multi-frame enhancement + VLM")
    if args.vlm_api_url:
        log.info("Connecting to VLM API: %s", args.vlm_api_url)
        vlm = VLMClient(model_path=args.vlm_model or "qwen-vl", api_url=args.vlm_api_url, device=args.device, max_new_tokens=256)
    else:
        log.info("Loading VLM: %s", args.vlm_model)
        vlm = VLMClient(model_path=args.vlm_model, device=args.device, max_new_tokens=256)

    log.info("Loading BGE embedder...")
    if args.bge_model:
        embedder = BGEM3Embedder(model_path=args.bge_model, device=args.device)
    else:
        from src.config import load_config
        from src.memory.core import specs
        cfg = specs.cfg_for_similarity_build("videomme")
        embedder = BGEM3Embedder(
            model_path=cfg["models"]["embedder"]["model_path"],
            device=args.device,
        )

    # Process all banks (with shard support for multi-GPU)
    bank_files = sorted(bank_dir.glob("*.json"))

    # Shard the files
    if args.num_shards > 1:
        shard_size = (len(bank_files) + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min((args.shard_id + 1) * shard_size, len(bank_files))
        bank_files = bank_files[start_idx:end_idx]
        log.info("Processing shard %d/%d: banks [%d:%d] (%d files)",
                 args.shard_id + 1, args.num_shards, start_idx, end_idx, len(bank_files))
    else:
        log.info("Found %d bank files to enhance (no sharding)", len(bank_files))

    for idx, bank_file in enumerate(bank_files):
        if idx % 50 == 0:
            log.info("Progress: [%d/%d]", idx, len(bank_files))

        # Try to find corresponding video
        video_id = bank_file.stem
        video_path = _get_video_path(bank_dir, video_id, video_dir)

        output_file = output_dir / bank_file.name if output_dir != bank_dir else bank_file
        try:
            enhance_bank_file(bank_file, video_path, vlm, embedder, output_file)
        except Exception as e:
            log.error("Failed to enhance %s: %s", bank_file.name, e)

    log.info("Shard %d complete! %d banks enhanced", args.shard_id, len(bank_files))


if __name__ == "__main__":
    main()
