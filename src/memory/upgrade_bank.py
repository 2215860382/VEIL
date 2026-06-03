"""Upgrade legacy similarity_group banks to two-layer format.

Converts old VideoMME-L banks (single narrative layer) to the two-layer format:
- Dynamic Narrative Layer: key_events, actors, state_changes, temporal_relations, causal_clues
- Static Attribute Layer: static_frames, static_index_text, v_static vectors

Usage:
    PYTHONPATH=. python -m src.memory.upgrade_bank \\
        --input  outputs/memory/videomme_L_v1/video_id.json \\
        --video-dir /path/to/videos \\
        --vlm-model /path/to/Qwen-VL \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.core.schema import MemoryBank
from src.memory.similarity import (
    DYNAMIC_SYS,
    extract_static_attributes,
    build_static_index_text,
)
from src.models.embedder import BGEM3Embedder
from src.models.vlm_client import VLMClient
from src.utils.logging import get_logger

log = get_logger("upgrade_bank")


def _parse_dynamic_narrative(raw_text: str, fallback_summary: str = "") -> dict:
    """Parse DYNAMIC_SYS JSON output or return fallback."""
    import json as _json
    import re

    try:
        m = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if m:
            d = _json.loads(m.group())
            return {
                "summary": str(d.get("summary", fallback_summary)),
                "key_events": d.get("key_events", []) if isinstance(d.get("key_events"), list) else [],
                "actors": d.get("actors", []) if isinstance(d.get("actors"), list) else [],
                "state_changes": d.get("state_changes", []) if isinstance(d.get("state_changes"), list) else [],
                "temporal_relations": d.get("temporal_relations", []) if isinstance(d.get("temporal_relations"), list) else [],
                "causal_clues": d.get("causal_clues", []) if isinstance(d.get("causal_clues"), list) else [],
            }
    except Exception as e:
        log.warning("  JSON parse error: %s", e)

    return {
        "summary": fallback_summary,
        "key_events": [],
        "actors": [],
        "state_changes": [],
        "temporal_relations": [],
        "causal_clues": [],
    }


def _extract_narrative_from_event_summary(raw_text: str) -> list:
    """Extract list of events from event_summary text (separated by semicolons or newlines)."""
    if not raw_text:
        return []
    # Split by common separators: 。；, or newlines
    import re
    events = re.split(r'[。；,\n]+', raw_text.strip())
    return [e.strip() for e in events if e.strip()]


def _upgrade_chunk(
    chunk,
    vlm,
    embedder,
    keyframe_dir: Path,
) -> None:
    """In-place upgrade of a single chunk to two-layer format."""
    # ─────────────────────────────────────────────────────────────
    # 1. Narrative Layer: VLM 从 event_summary + memory_text 重新生成
    # ─────────────────────────────────────────────────────────────

    # 构造输入：事件摘要 + 现有总结
    narrative_input = f"Event summary: {chunk.event_summary}\n\nOverall summary: {chunk.memory_text}"
    if chunk.asr:
        narrative_input += f"\n\nSpoken text: {chunk.asr}"

    messages = [
        {"role": "system", "content": DYNAMIC_SYS},
        {"role": "user", "content": narrative_input},
    ]

    try:
        raw = vlm._generate(messages, max_new_tokens=256).strip()
        narrative = _parse_dynamic_narrative(raw, chunk.memory_text)
    except Exception as e:
        log.warning("[%s] narrative generation failed: %s", chunk.video_id, e)
        # Fallback：从现有字段提取
        narrative = {
            "summary": chunk.memory_text,
            "key_events": _extract_narrative_from_event_summary(chunk.event_summary),
            "actors": chunk.persons if chunk.persons else [],
            "actions": chunk.actions if chunk.actions else [],
            "state_changes": [],  # 原库没有，VLM 生成失败时留空
            "temporal_relations": [],
            "causal_clues": [],
        }

    chunk.key_events = narrative.get("key_events", [])
    chunk.actors = narrative.get("actors", [])
    chunk.state_changes = narrative.get("state_changes", [])
    chunk.temporal_relations = narrative.get("temporal_relations", [])
    chunk.causal_clues = narrative.get("causal_clues", [])
    # memory_text 保持不变（已有高质量摘要）

    # ─────────────────────────────────────────────────────────────
    # 2. Static Attribute Layer: 从原库字段直接拼接（无需VLM）
    # ─────────────────────────────────────────────────────────────

    if chunk.ocr or chunk.objects or chunk.persons:
        # 构建静态属性帧
        static_attrs = {
            "frame_id": f"{chunk.video_id}_chunk{chunk.chunk_id:03d}",
            "timestamp": chunk.keyframe_ts,
            "image_path": chunk.keyframe_path,
            "ocr_text": chunk.ocr.split() if chunk.ocr else [],
            "numbers": [s for s in chunk.ocr.split() if s.isdigit()] if chunk.ocr else [],
            "objects": chunk.objects if chunk.objects else [],
            "people_appearance": chunk.persons if chunk.persons else [],
        }

        # 可选：如果有 keyframe，用 VLM 补充 colors/spatial_layout
        if chunk.keyframe_path and Path(chunk.keyframe_path).exists():
            try:
                static_frame = extract_static_attributes(
                    chunk.keyframe_path,
                    frame_id=static_attrs["frame_id"],
                    timestamp=chunk.keyframe_ts,
                    vlm=vlm,
                )
                if static_frame:
                    # 合并：VLM 的结果覆盖，保留原库数据作为 fallback
                    static_attrs.update({
                        "colors": static_frame.get("colors", []),
                        "clothing": static_frame.get("clothing", []),
                        "spatial_layout": static_frame.get("spatial_layout", []),
                        "scene_attributes": static_frame.get("scene_attributes", []),
                    })
            except Exception as e:
                log.warning("[%s] static attribute extraction failed: %s", chunk.video_id, e)
                # Fallback：只用原库数据，新字段保留空列表
                pass

        chunk.static_frames = [static_attrs]
        chunk.static_index_text = build_static_index_text(static_attrs)

        # 3. 计算属性层向量
        text_to_encode = chunk.static_index_text if chunk.static_index_text else " "
        v_static = embedder.encode([text_to_encode])[0].tolist()
        chunk.v_static = v_static


def upgrade_bank_file(
    input_path: Path,
    vlm,
    embedder,
    output_path: Optional[Path] = None,
) -> Path:
    """Upgrade a single bank file to two-layer format.

    Args:
        input_path: Path to legacy bank JSON
        vlm: VLMClient for narrative and static extraction
        embedder: BGE embedder for v_static vectors
        output_path: Where to save upgraded bank (default: same as input, in-place)

    Returns:
        Path to upgraded bank file
    """
    if output_path is None:
        output_path = input_path

    log.info("Loading bank: %s", input_path)
    bank = MemoryBank.load(input_path)

    # Skip if already upgraded
    if bank.chunks and bank.chunks[0].v_static:
        log.info("Bank already has v_static, skipping")
        return output_path

    log.info("Upgrading %d chunks...", len(bank.chunks))
    keyframe_dir = input_path.parent / bank.video_id / "keyframes"

    for i, chunk in enumerate(bank.chunks):
        if i % 10 == 0:
            log.info("  [%d/%d]", i, len(bank.chunks))
        _upgrade_chunk(chunk, vlm, embedder, keyframe_dir)

    log.info("Saving upgraded bank: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Upgrade legacy similarity_group banks to two-layer format"
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to legacy bank JSON file",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output path (default: same as input, in-place upgrade)",
    )
    ap.add_argument(
        "--vlm-model",
        required=True,
        help="Path to VLM checkpoint for narrative extraction",
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

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        ap.error(f"Input file not found: {input_path}")

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

    upgrade_bank_file(input_path, vlm, embedder, output_path)
    log.info("Done!")


if __name__ == "__main__":
    main()
