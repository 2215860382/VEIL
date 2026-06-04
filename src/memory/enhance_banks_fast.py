"""快速库增强：只在关键帧上做VLM增强，不提取额外帧。

用法:
    python -m src.memory.enhance_banks_fast \
        --bank-dir /home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b_two_layer \
        --video-dir /home2/ycj/Datas/VideoMME/videos \
        --vlm-model /home2/ycj/Models/Qwen/Qwen2.5-VL-3B-Instruct \
        --bge-model /home2/ycj/Models/BAAI/bge-m3 \
        --device cuda:4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.core.schema import MemoryBank
from src.memory.similarity import (
    DYNAMIC_SYS,
    STATIC_ATTR_SYS,
    extract_static_attributes,
    build_static_index_text,
)
from src.models.embedder import BGEM3Embedder
from src.models.vlm_client import VLMClient
from src.utils.logging import get_logger

log = get_logger("enhance_banks_fast")


def _enhance_chunk_keyframe_only(
    chunk,
    vlm,
    embedder,
) -> None:
    """对单个chunk只在关键帧上做VLM增强。不提取额外帧，速度快。

    Args:
        chunk: MemoryChunk to enhance
        vlm: VLMClient for enhancement
        embedder: BGE embedder for v_static recomputation
    """
    # 1. 动态层增强：从现有信息重新生成narrative字段
    if chunk.event_summary or chunk.asr:
        narrative_input = f"Event summary: {chunk.event_summary}\n\nOverall summary: {chunk.memory_text}"
        if chunk.asr:
            narrative_input += f"\n\nSpoken text: {chunk.asr}"

        messages = [
            {"role": "system", "content": DYNAMIC_SYS},
            {"role": "user", "content": narrative_input},
        ]

        try:
            raw = vlm._generate(messages, max_new_tokens=256).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                # Update narrative fields
                chunk.key_events = d.get("key_events", []) if isinstance(d.get("key_events"), list) else []
                chunk.actors = d.get("actors", []) if isinstance(d.get("actors"), list) else []
                chunk.state_changes = d.get("state_changes", []) if isinstance(d.get("state_changes"), list) else []
                chunk.temporal_relations = d.get("temporal_relations", []) if isinstance(d.get("temporal_relations"), list) else []
                chunk.causal_clues = d.get("causal_clues", []) if isinstance(d.get("causal_clues"), list) else []
        except Exception as e:
            log.debug("Narrative enhancement failed: %s", e)

    # 2. 静态层增强：只在关键帧上提取属性
    if chunk.keyframe_path and Path(chunk.keyframe_path).exists():
        try:
            static_attrs = extract_static_attributes(
                chunk.keyframe_path,
                frame_id=f"{chunk.video_id}_chunk{chunk.chunk_id:03d}_f0",
                timestamp=chunk.keyframe_ts,
                vlm=vlm,
            )
            if static_attrs:
                # 替换或补充existing static_frames
                chunk.static_frames = [static_attrs]
                chunk.static_index_text = build_static_index_text(static_attrs)

                # 重新计算v_static
                text_to_encode = chunk.static_index_text if chunk.static_index_text else " "
                v_static = embedder.encode([text_to_encode])[0].tolist()
                chunk.v_static = v_static
        except Exception as e:
            log.debug("Keyframe static attribute extraction failed: %s", e)


def enhance_bank_file(
    input_path: Path,
    vlm,
    embedder,
    output_path: Optional[Path] = None,
) -> Path:
    """增强单个bank文件（只在关键帧上）。

    Args:
        input_path: Path to bank JSON
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

    log.info("Enhancing %d chunks (keyframe only)...", len(bank.chunks))

    for i, chunk in enumerate(bank.chunks):
        if i % 100 == 0:
            log.info("  [%d/%d]", i, len(bank.chunks))
        try:
            _enhance_chunk_keyframe_only(chunk, vlm, embedder)
        except Exception as e:
            log.debug("Chunk %d enhancement failed: %s", i, e)

    log.info("Saving enhanced bank: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="快速库增强：只在关键帧上做VLM增强（无多帧提取）"
    )
    ap.add_argument(
        "--bank-dir",
        required=True,
        help="Directory containing bank JSON files",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as bank-dir, in-place upgrade)",
    )
    ap.add_argument(
        "--video-dir",
        default=None,
        help="Video directory (optional, not used in keyframe-only mode)",
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
    args = ap.parse_args()

    bank_dir = Path(args.bank_dir)
    output_dir = Path(args.output_dir) if args.output_dir else bank_dir

    if not bank_dir.exists():
        ap.error(f"Bank directory not found: {bank_dir}")

    if not args.vlm_model and not args.vlm_api_url:
        ap.error("Either --vlm-model or --vlm-api-url must be provided")

    log.info("快速库增强（关键帧only）")
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
        from src.memory.core import specs
        cfg = specs.cfg_for_similarity_build("videomme")
        embedder = BGEM3Embedder(
            model_path=cfg["models"]["embedder"]["model_path"],
            device=args.device,
        )

    # Process all banks
    bank_files = sorted(bank_dir.glob("*.json"))
    log.info("Found %d bank files to enhance", len(bank_files))

    for idx, bank_file in enumerate(bank_files):
        if idx % 50 == 0:
            log.info("Progress: [%d/%d]", idx, len(bank_files))

        output_file = output_dir / bank_file.name if output_dir != bank_dir else bank_file
        try:
            enhance_bank_file(bank_file, vlm, embedder, output_file)
        except Exception as e:
            log.error("Failed to enhance %s: %s", bank_file.name, e)

    log.info("Enhancement complete! %d banks enhanced", len(bank_files))


if __name__ == "__main__":
    main()
