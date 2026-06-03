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
    embedder,
) -> None:
    """Phase 1: In-place quick conversion to two-layer format (no VLM calls).

    Narrative layer: extract from existing fields (will be enhanced in Phase 2)
    Static layer: concatenate from ocr + objects + persons (will be multi-frame in Phase 2)
    """
    # ─────────────────────────────────────────────────────────────
    # 1. Narrative Layer: 从现有字段快速提取（Phase 2 补充）
    # ─────────────────────────────────────────────────────────────

    chunk.key_events = _extract_narrative_from_event_summary(chunk.event_summary)
    chunk.actors = chunk.persons if chunk.persons else []
    chunk.state_changes = []  # Phase 2 补充
    chunk.temporal_relations = []  # Phase 2 补充
    chunk.causal_clues = []  # Phase 2 补充
    # memory_text 保持不变（已有高质量摘要）

    # ─────────────────────────────────────────────────────────────
    # 2. Static Attribute Layer: 从原库字段拼接（无VLM，Phase 2 扩展多帧）
    # ─────────────────────────────────────────────────────────────

    # 总是创建 static_frames，即使字段为空（Phase 2 会补充）
    static_attrs = {
        "frame_id": f"{chunk.video_id}_chunk{chunk.chunk_id:03d}_f0",
        "timestamp": chunk.keyframe_ts,
        "image_path": chunk.keyframe_path,
        "ocr_text": chunk.ocr.split() if chunk.ocr else [],
        "numbers": [s for s in chunk.ocr.split() if s.isdigit()] if chunk.ocr else [],
        "colors": [],  # Phase 2 VLM 补充
        "objects": chunk.objects if chunk.objects else [],
        "object_attributes": [],  # Phase 2 补充
        "people_appearance": chunk.persons if chunk.persons else [],
        "clothing": [],  # Phase 2 VLM 补充
        "spatial_layout": [],  # Phase 2 VLM 补充
        "textures": [],  # Phase 2 VLM 补充
        "scene_attributes": [],  # Phase 2 VLM 补充
    }

    chunk.static_frames = [static_attrs]
    chunk.static_index_text = build_static_index_text(static_attrs)

    # 3. 计算属性层向量（即使为空也要创建向量，用于 Phase 2 增强）
    text_to_encode = chunk.static_index_text if chunk.static_index_text else " "
    v_static = embedder.encode([text_to_encode])[0].tolist()
    chunk.v_static = v_static


def upgrade_bank_file(
    input_path: Path,
    embedder,
    output_path: Optional[Path] = None,
) -> Path:
    """Phase 1: Quick conversion to two-layer format (no VLM calls).

    Args:
        input_path: Path to legacy bank JSON
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

    log.info("Phase 1: Quick converting %d chunks (no VLM)...", len(bank.chunks))

    for i, chunk in enumerate(bank.chunks):
        if i % 500 == 0:
            log.info("  [%d/%d]", i, len(bank.chunks))
        _upgrade_chunk(chunk, embedder)

    log.info("Saving upgraded bank: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Phase 1: Quick conversion of legacy similarity_group banks to two-layer format (no VLM)"
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing legacy bank JSON files",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input, in-place upgrade)",
    )
    ap.add_argument(
        "--bge-model",
        default=None,
        help="Path to BGE model (default: from config)",
    )
    ap.add_argument(
        "--device",
        default="cuda:4",
        help="Device for BGE (default: cuda:4)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        ap.error(f"Input directory not found: {input_dir}")

    log.info("Phase 1: Quick conversion (no VLM)")
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
    log.info("  BGE ready")

    # 批量升级库文件
    bank_files = sorted(input_dir.glob("*.json"))
    log.info("Found %d bank files to convert", len(bank_files))

    for idx, bank_file in enumerate(bank_files):
        if idx % 50 == 0:
            log.info("Progress: [%d/%d]", idx, len(bank_files))
        output_file = output_dir / bank_file.name if output_dir != input_dir else bank_file
        try:
            upgrade_bank_file(bank_file, embedder, output_file)
        except Exception as e:
            log.error("Failed to upgrade %s: %s", bank_file.name, e)

    log.info("Phase 1 complete! %d banks converted", len(bank_files))


if __name__ == "__main__":
    main()
