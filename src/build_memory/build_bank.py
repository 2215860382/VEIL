"""从零构建两层记忆库：直接从视频生成动态层 + 静态层。

单步建库流程（无分阶段改造）：
1. 帧采样 (decord, 1fps)
2. SigLIP 编码 → 视觉向量
3. 动态分组 (余弦相似度)
4. 帧描述生成 (caption，VLM)
5. SRT 字幕对齐（可选）
6. 生成动态层 (summary + key_events + actors + state_changes + temporal_relations + causal_clues)
7. 生成静态层 (ocr_text + numbers + colors + objects + clothing + spatial_layout + ...)
8. BGE 向量计算 (v_dynamic + v_static) + MemoryBank 保存

用法：
    cd /home2/ycj/Project/VEIL

    # 单视频，本地VLM
    PYTHONPATH=. python -m src.build_memory.build_bank \\
        --video-file /path/to/video.mp4 \\
        --out-dir outputs/memory/mybank \\
        --vlm-model /path/to/Qwen-VL \\
        --siglip-model /path/to/SigLIP \\
        --bge-model /path/to/bge-m3 \\
        --device cuda:0

    # 单视频，API 模式
    PYTHONPATH=. python -m src.build_memory.build_bank \\
        --video-file /path/to/video.mp4 \\
        --out-dir outputs/memory/mybank \\
        --vlm-api-url http://localhost:8000 \\
        --siglip-model /path/to/SigLIP \\
        --bge-model /path/to/bge-m3

    # 批量处理（多GPU 分片）
    PYTHONPATH=. python -m src.build_memory.build_bank \\
        --video-dir /home/videos \\
        --out-dir outputs/memory/mybank \\
        --vlm-model /path/to/Qwen-VL \\
        --siglip-model /path/to/SigLIP \\
        --bge-model /path/to/bge-m3 \\
        --device cuda:0 \\
        --shard-id 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_memory.core.dynamic_grouper import group_frames
from src.build_memory.core.readme import write_memory_build_readme
from src.build_memory.core.sample_frames import sample_frames
from src.build_memory.core.schema import MemoryBank, MemoryChunk, StaticAttribute
from src.build_memory.similarity import (
    DYNAMIC_SYS,
    FRAME_CAPTION_PROMPT,
    STATIC_ATTR_SYS,
    align_subtitles,
    build_static_index_text,
    extract_frames,
    extract_static_attributes,
    parse_srt,
    summarize_group,
)
from src.dataloader.videomme import load_videomme
from src.clients.embedder import BGEM3Embedder
from src.clients.siglip_embedder import SigLIPEmbedder
from src.clients.vlm_client import VLMClient
from src.utils.logging import get_logger

log = get_logger("build_bank")


# ── Chunk Quality Rubric ────────────────────────────────────────────────────────


class ChunkQualityRubric:
    """Evaluate MemoryChunk quality based on information richness and structure.

    Extensible framework for chunk validation. Specific criteria can be added later.
    """

    def evaluate(self, chunk: MemoryChunk) -> dict:
        """Evaluate chunk quality across multiple dimensions.

        Returns:
            {
                "score": float (0.0-1.0),
                "passed": bool,
                "feedback": str,
                "details": {
                    "has_summary": bool,
                    "summary_length": int,
                    "has_dynamic_fields": bool,
                    "has_static_attributes": bool,
                    "has_vectors": bool,
                    "issues": List[str],
                }
            }
        """
        details = {
            "has_summary": bool(chunk.memory_text and chunk.memory_text.strip()),
            "summary_length": len(chunk.memory_text or ""),
            "has_dynamic_fields": self._check_dynamic_fields(chunk),
            "has_static_attributes": len(chunk.static_attributes) > 0,
            "has_vectors": bool(chunk.v_dynamic and chunk.v_static),
            "issues": [],
        }

        # Collect issues
        if not details["has_summary"]:
            details["issues"].append("Missing summary")
        if details["summary_length"] < 10:
            details["issues"].append("Summary too short")
        if not details["has_dynamic_fields"]:
            details["issues"].append("Missing dynamic layer fields")
        if not details["has_static_attributes"]:
            details["issues"].append("Missing static attributes")
        if not details["has_vectors"]:
            details["issues"].append("Missing embeddings")

        # Calculate score (0-1)
        total_checks = 5
        passed_checks = sum([
            details["has_summary"],
            details["has_dynamic_fields"],
            details["has_static_attributes"],
            details["has_vectors"],
            details["summary_length"] >= 10,
        ])
        score = passed_checks / total_checks

        feedback = (
            f"Chunk quality: {score:.2%} "
            f"({', '.join(details['issues']) if details['issues'] else 'OK'})"
        )

        return {
            "score": score,
            "passed": score >= 0.8,  # Threshold: 80%
            "feedback": feedback,
            "details": details,
        }

    def _check_dynamic_fields(self, chunk: MemoryChunk) -> bool:
        """Check if dynamic layer has key fields populated."""
        return bool(
            chunk.key_events
            or chunk.actors
            or chunk.state_changes
            or chunk.temporal_relations
            or chunk.causal_clues
        )


def _extract_keyframe(
    frames: List[np.ndarray],
    frame_indices: List[int],
    resolution: int = 448,
) -> Tuple[Image.Image, float]:
    """Extract sharpest frame from a group and return as PIL image + timestamp offset."""
    if not frame_indices:
        return None, 0.0

    if len(frame_indices) == 1:
        idx = frame_indices[0]
        img = Image.fromarray(frames[idx])
        return img, float(idx)

    # Simple focus/sharpness detection: Laplacian variance
    try:
        import cv2

        sharpness = []
        for idx in frame_indices:
            gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness.append(laplacian.var())
        best_idx = frame_indices[np.argmax(sharpness)]
    except Exception:
        # Fallback: use center frame
        best_idx = frame_indices[len(frame_indices) // 2]

    img = Image.fromarray(frames[best_idx])
    return img, float(best_idx)


def _save_keyframe(
    keyframe: Image.Image,
    video_id: str,
    chunk_id: int,
    frame_idx: int,
    out_dir: Path,
    quality: int = 95,
) -> str:
    """Save keyframe to chunk-specific directory. Return relative path.

    Args:
        keyframe: PIL Image
        video_id: Video ID
        chunk_id: Chunk ID
        frame_idx: Frame index within chunk (0=first, 1=middle, 2=last)
        out_dir: Output directory (base directory)
        quality: JPEG quality

    Returns:
        Relative path from chunk directory: keyframes/frame_00X.jpg
    """
    chunk_keyframe_dir = out_dir / video_id / f"chunk{chunk_id:03d}" / "keyframes"
    chunk_keyframe_dir.mkdir(parents=True, exist_ok=True)

    # Resize to 448x448 if needed
    if keyframe.size != (448, 448):
        keyframe = keyframe.resize((448, 448), Image.BILINEAR)

    filename = f"frame_{frame_idx + 1:03d}.jpg"
    path = chunk_keyframe_dir / filename
    keyframe.save(path, "JPEG", quality=quality)

    # Return relative path from chunk directory
    return str(path.relative_to(out_dir / video_id / f"chunk{chunk_id:03d}"))


def _caption_frames(vlm, frames: List[np.ndarray], stride: int = 1) -> List[str]:
    """Generate captions for frames with stride (default stride=1: all frames, like old library).

    stride=1: caption all frames (100 frames → 100 captions, matches old library default)
    stride=N: caption every N-th frame (optimization if needed)
    """
    if not frames:
        return []

    captions = [""] * len(frames)  # Pre-allocate with empty strings
    for i in range(0, len(frames), stride):
        try:
            frame = frames[i]
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)

            caption = vlm.chat_with_frames([frame], FRAME_CAPTION_PROMPT, max_new_tokens=50)
            caption = caption.strip()
            if not caption or caption.startswith("[frame at"):
                caption = "[frame]"
            captions[i] = caption
        except Exception as e:
            log.debug("Frame %d caption failed: %s", i, e)
            captions[i] = "[frame]"

    # Fill in skipped frames with empty (they'll be filtered by summarize_group)
    return captions


def _extract_static_attributes_concurrent(
    keyframe_paths_and_info: List[Tuple[str, str, float]],
    vlm,
    max_workers: int = 4,
) -> List[Optional[dict]]:
    """Extract static attributes from multiple frames concurrently.

    Args:
        keyframe_paths_and_info: List of (path, frame_id, timestamp) tuples
        vlm: VLM client
        max_workers: Number of concurrent threads

    Returns:
        List of attribute dicts (None if extraction failed)
    """
    results = [None] * len(keyframe_paths_and_info)

    def extract_one(idx: int, path: str, frame_id: str, timestamp: float) -> Tuple[int, Optional[dict]]:
        try:
            attr_dict = extract_static_attributes(path, frame_id, timestamp, vlm)
            return idx, attr_dict
        except Exception as e:
            log.debug(f"Static attr extraction failed for {frame_id}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_one, i, path, fid, ts): i
            for i, (path, fid, ts) in enumerate(keyframe_paths_and_info)
        }
        for future in as_completed(futures):
            idx, attr_dict = future.result()
            results[idx] = attr_dict

    return results


def _save_captions(
    video_id: str,
    chunk_id: int,
    frame_captions: List[str],
    frame_timestamps: List[float],
    speech_text: str,
    out_dir: Path,
) -> None:
    """Save frame captions to chunk's intermediate directory.

    Args:
        video_id: Video ID
        chunk_id: Chunk index
        frame_captions: List of frame descriptions
        frame_timestamps: Corresponding frame timestamps
        speech_text: Associated speech/ASR text
        out_dir: Base output directory
    """
    intermediate_dir = out_dir / video_id / f"chunk{chunk_id:03d}" / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    caption_data = {
        "unit_id": f"{video_id}_unit_{chunk_id:04d}",
        "descs": frame_captions,
        "frame_timestamps": frame_timestamps,
        "speech_text": speech_text,
    }

    caption_path = intermediate_dir / "captions.json"
    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(caption_data, f, ensure_ascii=False, indent=2)


def process_video(
    video_path: str,
    video_id: str,
    out_dir: Path,
    vlm: VLMClient,
    siglip: SigLIPEmbedder,
    embedder: BGEM3Embedder,
    fps: float = 1.0,
    theta: float = 0.80,
    n_max: int = 30,
    max_frames: Optional[int] = None,
    resolution: Optional[int] = None,
    subtitle_entries: Optional[List[Tuple[float, float, str]]] = None,
) -> Optional[MemoryBank]:
    """Process a single video: full pipeline from frames to MemoryBank.

    Args:
        video_path: Path to video file
        video_id: Video identifier (used in chunk IDs)
        out_dir: Output directory for bank JSON + keyframes
        vlm: VLMClient for captions + summaries
        siglip: SigLIPEmbedder for visual grouping
        embedder: BGEM3Embedder for semantic vectors
        fps: Frame sampling rate (default 1.0)
        theta: Cosine similarity threshold for grouping (default 0.80)
        n_max: Max frames per group (default 30)
        max_frames: Max total frames to sample (None=unlimited, default None)
        resolution: Frame resolution (None=original, default None)
        subtitle_entries: Parsed SRT entries [(t_start, t_end, text), ...]

    Returns:
        MemoryBank if successful, None if failed
    """
    try:
        log.info("[%s] Step 1: Extracting frames (FFmpeg)...", video_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract frames using FFmpeg (matches old library behavior)
            frame_paths = extract_frames(str(video_path), tmpdir_path, fps=fps)

            # Calculate timestamps and duration (same as old library)
            T = len(frame_paths)
            timestamps = [i / fps for i in range(T)]
            duration = timestamps[-1] + 1.0 if timestamps else 0.0

            # Filter out black/invalid frames (skip first frame + very small JPEG files)
            valid_indices = []
            black_count = 0
            for i, (frame_path, t) in enumerate(zip(frame_paths, timestamps)):
                # Skip first frame (t <= 0.1)
                if t <= 0.1:
                    black_count += 1
                    continue
                # Skip very small files (typically all-black frames compress to <5KB)
                file_size = Path(frame_path).stat().st_size
                if file_size < 5000:  # 5KB threshold
                    black_count += 1
                    continue
                valid_indices.append(i)

            if valid_indices:
                frame_paths = [frame_paths[i] for i in valid_indices]
                timestamps = [timestamps[i] for i in valid_indices]
                log.info("[%s] Filtered out %d black/invalid frames", video_id, black_count)

            log.info(
                "[%s] Extracted %d frames (%.1f fps target, %.1fs duration)",
                video_id,
                len(frame_paths),
                fps,
                duration,
            )

            # ─────────────────────────────────────────────────────────────
            # Step 2: SigLIP encoding
            # ─────────────────────────────────────────────────────────────
            log.info("[%s] Step 2: SigLIP encoding...", video_id)
            v_frames = siglip.encode_images([str(p) for p in frame_paths])
            log.info("[%s] SigLIP vectors: %s", video_id, v_frames.shape)

            # Load frames as numpy arrays for later processing
            frames = [np.array(Image.open(str(p)).convert("RGB")) for p in frame_paths]

            log.info("[%s] Step 3: Dynamic grouping (theta=%.2f, n_max=%d)...", video_id, theta, n_max)
            groups = group_frames(v_frames, timestamps, theta=theta, n_max=n_max)
            log.info("[%s] Grouped into %d segments", video_id, len(groups))

            # ─────────────────────────────────────────────────────────────
            # Initialize quality evaluator
            # ─────────────────────────────────────────────────────────────
            rubric = ChunkQualityRubric()

            # ─────────────────────────────────────────────────────────────
            # Step 4-8: Process each group
            # ─────────────────────────────────────────────────────────────
            chunks = []
            quality_stats = {"total": 0, "passed": 0, "low_quality": []}

            for group_idx, group in enumerate(groups):
                if group_idx % 50 == 0:
                    log.info("[%s] Processing chunk %d/%d", video_id, group_idx, len(groups))

                frame_indices = group.frame_indices
                t_start = group.t_start
                t_end = group.t_end
                center_idx = group.center_idx

                # ─────────────────────────────────────────────────────────────
                # Common input: frame captions + speech text
                # ─────────────────────────────────────────────────────────────
                group_frames_list = [frames[i] for i in frame_indices]

                captions = _caption_frames(vlm, group_frames_list, stride=1)  # 和老库一致：所有帧都有 caption

                speech_text = ""
                if subtitle_entries:
                    speech_text = align_subtitles(subtitle_entries, t_start, t_end)

                # Save caption file with frame descriptions and timestamps
                caption_timestamps = [timestamps[i] for i in frame_indices]
                _save_captions(video_id, group_idx, captions, caption_timestamps, speech_text, out_dir)

                # ──────────────────────────────────────────────���──────────────
                # Branch 1: Generate dynamic layer (independent)
                # ─────────────────────────────────────────────────────────────
                summary_data = summarize_group(captions, vlm, speech_text, max_caps=15)

                # ─────────────────────────────────────────────────────────────
                # Branch 2: Generate static layer (independent, multi-frame)
                # ─────────────────────────────────────────────────────────────
                static_attributes_list = []
                static_index_text = ""
                keyframe_path = ""  # Save the first frame as primary keyframe
                keyframe_ts = 0.0

                try:
                    # Select multiple representative frames (first, middle, last)
                    num_frames = len(frame_indices)
                    representative_indices = sorted(set([
                        0,
                        num_frames // 2,
                        num_frames - 1
                    ]))
                    representative_indices = [i for i in representative_indices if i < num_frames]

                    # Save keyframes and prepare for concurrent static attribute extraction
                    keyframe_info_list = []
                    for frame_pos, frame_idx_in_group in enumerate(representative_indices):
                        frame_idx_global = frame_indices[frame_idx_in_group]
                        keyframe = Image.fromarray(frames[frame_idx_global])
                        frame_ts = timestamps[frame_idx_global]

                        saved_keyframe_path = _save_keyframe(keyframe, video_id, group_idx, frame_pos, out_dir)
                        full_keyframe_path = out_dir / video_id / f"chunk{group_idx:03d}" / saved_keyframe_path

                        # Capture first keyframe as the primary one
                        if frame_pos == 0:
                            keyframe_path = saved_keyframe_path
                            keyframe_ts = frame_ts

                        frame_id = f"{video_id}_chunk{group_idx:03d}_f{frame_pos}"
                        keyframe_info_list.append((str(full_keyframe_path), frame_id, frame_ts))

                    # Extract static attributes concurrently (max 4 parallel threads)
                    attr_dicts = _extract_static_attributes_concurrent(keyframe_info_list, vlm, max_workers=4)
                    for attr_dict in attr_dicts:
                        if attr_dict:
                            static_attributes_list.append(StaticAttribute(**attr_dict))

                    # Merge attributes from all frames
                    if static_attributes_list:
                        merged_attrs = {
                            "ocr_text": [],
                            "numbers": [],
                            "colors": [],
                            "objects": [],
                            "object_attributes": [],
                            "people_appearance": [],
                            "clothing": [],
                            "spatial_layout": [],
                            "textures": [],
                            "scene_attributes": [],
                        }
                        for frame in static_attributes_list:
                            for key in merged_attrs:
                                val = getattr(frame, key, [])
                                if isinstance(val, list):
                                    merged_attrs[key].extend(val)

                        # Remove duplicates while preserving order
                        for key in merged_attrs:
                            unique_vals = []
                            seen = set()
                            for v in merged_attrs[key]:
                                v_str = str(v) if not isinstance(v, str) else v
                                if v_str and v_str not in seen:
                                    unique_vals.append(v_str)
                                    seen.add(v_str)
                            merged_attrs[key] = unique_vals

                        static_index_text = build_static_index_text(merged_attrs)

                except Exception as e:
                    log.debug("[%s] Static attribute extraction failed: %s", video_id, e)

                # ─────────────────────────────────────────────────────────────
                # Merge: BGE vector computation for both layers
                # ─────────────────────────────────────────────────────────────
                # v_dynamic: encode dynamic layer (summary + speech text)
                embed_text = summary_data["summary"]
                if speech_text.strip():
                    embed_text = f"{embed_text}\n\n{speech_text}"
                v_dynamic = embedder.encode([embed_text if embed_text else " "])[0].tolist()

                # v_static: encode static layer (visual attributes)
                v_static = []
                if static_index_text:
                    v_static = embedder.encode([static_index_text])[0].tolist()

                # Construct MemoryChunk
                chunk = MemoryChunk(
                    video_id=video_id,
                    chunk_id=group_idx,
                    start_time=t_start,
                    end_time=t_end,
                    memory_text=summary_data["summary"],
                    visual_caption=" | ".join(captions),
                    key_events=summary_data.get("key_events", []),
                    actors=summary_data.get("actors", []),
                    state_changes=summary_data.get("state_changes", []),
                    temporal_relations=summary_data.get("temporal_relations", []),
                    causal_clues=summary_data.get("causal_clues", []),
                    static_attributes=static_attributes_list,  # Multi-frame attributes
                    static_index_text=static_index_text,
                    v_static=v_static,
                    v_dynamic=v_dynamic,
                    v_visual=v_frames[center_idx].tolist(),
                    keyframe_path=keyframe_path,
                    keyframe_ts=keyframe_ts,
                    sampled_frames=sorted([timestamps[i] for i in frame_indices]),
                    asr=speech_text,
                    memory_kind="similarity_group",
                )

                # Evaluate chunk quality
                quality = rubric.evaluate(chunk)
                quality_stats["total"] += 1

                # Filter low-quality chunks
                duration = chunk.end_time - chunk.start_time
                valid_captions = [c for c in (chunk.visual_caption or "").split("|") if c.strip() and c.strip() != "[frame]"]
                has_valid_memory = chunk.memory_text and len(chunk.memory_text.strip()) > 10

                is_valid = (
                    duration >= 1.0 and  # 至少1秒
                    len(valid_captions) > 0 and  # 有有效的caption
                    has_valid_memory  # 有有效的memory_text
                )

                if not is_valid:
                    quality_stats["low_quality"].append({
                        "chunk_id": group_idx,
                        "reason": f"duration={duration:.1f}s, valid_captions={len(valid_captions)}, has_memory={has_valid_memory}",
                        "feedback": quality["feedback"],
                    })
                    log.debug("[%s] Chunk %d skipped: %s", video_id, group_idx,
                              f"duration={duration:.1f}s, captions={len(valid_captions)}, memory={has_valid_memory}")
                    continue

                if quality["passed"]:
                    quality_stats["passed"] += 1
                else:
                    quality_stats["low_quality"].append({
                        "chunk_id": group_idx,
                        "feedback": quality["feedback"],
                        "score": quality["score"],
                    })
                    log.debug("[%s] Chunk %d: %s", video_id, group_idx, quality["feedback"])

                chunks.append(chunk)

        # ─────────────────────────────────────────────────────────────
        # Save metadata.json and evidence.json
        # ─────────────────────────────────────────────────────────────
        video_out_dir = out_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata.json (vector index for fast retrieval)
        metadata = {
            "video_id": video_id,
            "duration": duration,
            "fps": fps,
            "max_frames": max_frames,
            "resolution": resolution,
            "memory_kind": "similarity_group",
            "siglip_model": getattr(siglip, "model_id", "siglip"),
            "bge_model": getattr(embedder, "model_path", "bge-m3"),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "v_dynamic": chunk.v_dynamic,
                    "v_static": chunk.v_static,
                }
                for chunk in chunks
            ],
        }
        metadata_path = video_out_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        log.info("[%s] Saved metadata: %s", video_id, metadata_path)

        # Save evidence.json (detailed info for all chunks)
        evidence = {}
        for chunk in chunks:
            evidence[str(chunk.chunk_id)] = {
                "chunk_id": chunk.chunk_id,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "memory_text": chunk.memory_text,
                "visual_caption": chunk.visual_caption,
                "key_events": chunk.key_events,
                "actors": chunk.actors,
                "state_changes": chunk.state_changes,
                "temporal_relations": chunk.temporal_relations,
                "causal_clues": chunk.causal_clues,
                "static_attributes": [attr.dict() for attr in chunk.static_attributes] if chunk.static_attributes else [],
                "static_index_text": chunk.static_index_text,
                "keyframe_path": chunk.keyframe_path,
                "keyframe_ts": chunk.keyframe_ts,
                "sampled_frames": chunk.sampled_frames,
                "asr": chunk.asr,
            }
        evidence_path = video_out_dir / "evidence.json"
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)
        log.info("[%s] Saved evidence: %s (%d chunks)", video_id, evidence_path, len(chunks))

        # Build MemoryBank for compatibility (not saved to disk)
        bank = MemoryBank(
            video_id=video_id,
            duration=duration,
            chunks=chunks,
            chunk_size=None,
            stride=None,
            fps=fps,
            max_frames=max_frames,
            resolution=resolution,
            memory_kind="similarity_group",
            siglip_model=getattr(siglip, "model_id", "siglip"),
            bge_model=getattr(embedder, "model_path", "bge-m3"),
        )

        return bank

    except Exception as e:
        log.error("[%s] Failed: %s", video_id, e, exc_info=True)
        return None


def main():
    ap = argparse.ArgumentParser(
        description="从零构建两层记忆库：帧采样 → SigLIP分组 → 动态层生成 + 静态层生成 → BGE向量编码"
    )

    # Input
    ap.add_argument("--parquet-path", default="/home2/ycj/Datas/VideoMME/videomme/test-00000-of-00001.parquet",
                    help="Path to VideoMME parquet file (default: VideoMME L)")
    ap.add_argument("--video-dir", default="/home2/ycj/Datas/VideoMME/videos",
                    help="Video directory (default: VideoMME videos)")
    ap.add_argument("--video-file", help="Single video file (optional, overrides --video-dir)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--subtitle-dir", help="Directory containing SRT files (optional)")
    ap.add_argument("--duration-group", default="long", help="VideoMME duration group: short/medium/long (default: long)")

    # Models
    ap.add_argument("--vlm-model", help="Path to local VLM (or set --vlm-api-url)")
    ap.add_argument("--vlm-api-url", help="VLM API URL(s) for load balancing. Format: 'url1' or 'url1,url2,url3' (comma-separated, vLLM/OpenAI compatible)")
    ap.add_argument("--siglip-model", required=True, help="Path to SigLIP model")
    ap.add_argument("--bge-model", required=True, help="Path to BGE-M3 model")

    # Hyperparameters
    ap.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    ap.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate (default: 1.0)")
    ap.add_argument("--theta", type=float, default=0.80, help="Grouping threshold (default: 0.80)")
    ap.add_argument("--n-max", type=int, default=30, help="Max frames per group (default: 30)")
    ap.add_argument("--max-frames", type=int, default=None, help="Max total frames (default: None=unlimited)")
    ap.add_argument("--resolution", type=int, default=None, help="Frame resolution (default: None=original)")

    # Sharding (multi-GPU)
    ap.add_argument("--shard-id", type=int, default=0, help="Shard ID (default: 0)")
    ap.add_argument("--num-shards", type=int, default=1, help="Total shards (default: 1)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing banks")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Validate
    if not args.video_dir and not args.video_file:
        ap.error("Either --video-dir or --video-file is required")
    if not args.vlm_model and not args.vlm_api_url:
        ap.error("Either --vlm-model or --vlm-api-url is required")

    out_dir = Path(args.out_dir)

    # Initialize models
    log.info("Loading VLM...")
    # Support multiple API URLs (comma-separated for load balancing)
    # VLMClient will use them in round-robin fashion
    vlm = VLMClient(
        model_path=args.vlm_model or "qwen-vl",
        api_url=args.vlm_api_url,  # Can be "url1,url2,url3" for load balancing
        device=args.device,
        max_new_tokens=512,
    )
    if args.vlm_api_url:
        log.info("VLM API endpoints: %s", args.vlm_api_url)

    log.info("Loading SigLIP...")
    siglip = SigLIPEmbedder(model_path=args.siglip_model, device=args.device)

    log.info("Loading BGE-M3...")
    embedder = BGEM3Embedder(model_path=args.bge_model, device=args.device)

    # Parse subtitles (optional, for speech_text in dynamic layer)
    subtitle_map = {}
    if args.subtitle_dir:
        subtitle_dir = Path(args.subtitle_dir)
        for srt_file in subtitle_dir.glob("*.srt"):
            video_id = srt_file.stem
            try:
                subtitle_map[video_id] = parse_srt(srt_file)
                log.info("Loaded subtitles: %s", video_id)
            except Exception as e:
                log.warning("Failed to parse %s: %s", srt_file, e)

    # Collect video files & video_id mapping
    video_samples = []

    if args.video_file:
        # Single file: use parquet if available, else use filename as video_id
        video_path = Path(args.video_file)
        if args.parquet_path:
            log.info("Using parquet to find video_id for %s", video_path.name)
            samples = load_videomme(
                parquet_path=args.parquet_path,
                video_dir=str(video_path.parent),
                duration_groups=[args.duration_group],
            )
            found = False
            for s in samples:
                if s.video_path == str(video_path):
                    video_samples.append((s.video_id, str(video_path)))
                    found = True
                    break
            if not found:
                log.warning("Video not found in parquet, using filename as video_id")
                video_samples.append((video_path.stem, str(video_path)))
        else:
            video_samples.append((video_path.stem, str(video_path)))
    elif args.video_dir:
        if args.parquet_path:
            log.info("Loading videos from parquet...")
            samples = load_videomme(
                parquet_path=args.parquet_path,
                video_dir=args.video_dir,
                duration_groups=[args.duration_group],
            )
            # Deduplicate while preserving parquet order
            seen = set()
            video_samples = []
            for s in samples:
                if s.video_id not in seen:
                    video_samples.append((s.video_id, s.video_path))
                    seen.add(s.video_id)
            log.info("Loaded %d unique videos from parquet", len(video_samples))
        else:
            log.warning("--parquet-path not provided; using video filenames as video_id (may not match old library)")
            video_dir = Path(args.video_dir)
            for ext in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
                for p in video_dir.glob(f"*{ext}"):
                    video_samples.append((p.stem, str(p)))
            video_samples.sort()

    if not video_samples:
        log.error("No video files found")
        return

    # Apply sharding
    if args.num_shards > 1:
        shard_size = (len(video_samples) + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min((args.shard_id + 1) * shard_size, len(video_samples))
        video_samples = video_samples[start_idx:end_idx]
        log.info("Shard %d/%d: processing %d videos", args.shard_id + 1, args.num_shards, len(video_samples))

    # Process
    log.info("Processing %d videos...", len(video_samples))
    success_count = 0

    for idx, (video_id, video_path) in enumerate(video_samples):
        bank_path = out_dir / video_id

        # Check if already processed
        metadata_file = bank_path / "metadata.json"
        if metadata_file.exists() and not args.overwrite:
            log.info("[%d/%d] Skip (exists): %s", idx + 1, len(video_samples), video_id)
            success_count += 1
            continue

        log.info("[%d/%d] Processing: %s (%s)", idx + 1, len(video_samples), video_id, video_path)

        subtitle_entries = subtitle_map.get(video_id)
        bank = process_video(
            str(video_path),
            video_id,
            out_dir,
            vlm,
            siglip,
            embedder,
            fps=args.fps,
            theta=args.theta,
            n_max=args.n_max,
            max_frames=args.max_frames,
            resolution=args.resolution,
            subtitle_entries=subtitle_entries,
        )

        if bank:
            success_count += 1

    # Write summary (quality stats will be filled by process_video if available)
    readme_lines = [
        f"Total videos: {len(video_samples)}",
        f"Successful: {success_count}",
        f"Failed: {len(video_samples) - success_count}",
        "",
        "Parameters:",
        f"  fps={args.fps}",
        f"  theta={args.theta}",
        f"  n_max={args.n_max}",
        f"  max_frames={args.max_frames}",
        f"  resolution={args.resolution}",
        "",
        "Quality Rubric:",
        "  (Chunk quality evaluation enabled)",
        "  (See logs for low-quality chunks requiring attention)",
    ]

    try:
        write_memory_build_readme(out_dir, title="MemoryBank Build Summary", lines=readme_lines)
    except Exception as e:
        log.warning("Failed to write README: %s", e)

    log.info("Done! %d/%d videos successful", success_count, len(video_samples))


if __name__ == "__main__":
    main()
