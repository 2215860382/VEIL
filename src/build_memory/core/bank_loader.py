"""Load a MemoryBank from the new on-disk structure:

    videomme_L_27B/{video_id}/
        narrative.json       chunks[].narrative / caption / speech_text / keyframe_path
        attributes.json      chunks[].static_index_text / static_attributes
        vectors.npz          narrative_vecs / attribute_vecs / chunk_ids
        frames/{cid:04d}_{i}.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import MemoryBank, MemoryChunk


def load_bank(video_dir: str | Path) -> MemoryBank:
    """Build an in-memory MemoryBank from a video folder under videomme_L_27B/."""
    vd = Path(video_dir)
    video_id = vd.name
    narr = json.loads((vd / "narrative.json").read_text())
    attrs = json.loads((vd / "attributes.json").read_text())
    vecs = np.load(vd / "vectors.npz")

    attr_map = {c["chunk_id"]: c for c in attrs.get("chunks", [])}
    n_vecs = vecs["narrative_vecs"]
    a_vecs = vecs["attribute_vecs"]
    chunk_ids = vecs["chunk_ids"]
    cid_to_vidx = {int(cid): i for i, cid in enumerate(chunk_ids)}

    chunks = []
    for nc in narr.get("chunks", []):
        cid = nc["chunk_id"]
        ac = attr_map.get(cid, {})
        vi = cid_to_vidx.get(cid)
        v_dynamic = n_vecs[vi].tolist() if vi is not None else []
        v_static = a_vecs[vi].tolist() if vi is not None else []

        # Representative keyframe = first frame of the attribute layer
        # (fall back to chunk-id-based convention if attributes layer is missing)
        fps = ac.get("frame_paths") or []
        kf_rel = fps[0] if fps else f"frames/{cid:04d}_0.jpg"
        kf_abs = str(vd / kf_rel)

        chunks.append(MemoryChunk(
            video_id=video_id,
            chunk_id=cid,
            start_time=nc.get("start_time", 0.0),
            end_time=nc.get("end_time", 0.0),
            memory_text=nc.get("narrative", "") or "",
            visual_caption=" | ".join(nc.get("caption", []) or []),
            asr=nc.get("speech_text", "") or "",
            sampled_frames=nc.get("sampled_frames", []) or [],
            keyframe_path=kf_abs,
            keyframe_ts=nc.get("keyframe_ts", 0.0),
            v_dynamic=v_dynamic,
            v_static=v_static,
            static_index_text=ac.get("static_index_text", "") or "",
            static_attributes=[],
        ))

    return MemoryBank(
        video_id=video_id,
        duration=narr.get("duration", 0.0),
        chunks=chunks,
        memory_kind="similarity_group",
    )
