"""Polymorphic ``MemoryBank`` loader. Accepts any of three layouts:

* **New single-file bank**:
    ``{root}/{video_id}.json``  — written by ``MemoryBank.save()``.

* **Legacy multi-file bank**:
    ``{root}/{video_id}/``
        narrative.json   chunks[].narrative / caption / speech_text / keyframe_ts
        vectors.npz      narrative_vecs / chunk_ids
        frames/{cid:04d}_{i}.jpg

* **Pyramid bank** (4-layer fixed-time, built by ``build_pyramid.py``):
    ``{root}/{video_id}/``
        L1.jsonl         one row per 10 s chunk: idx/t_start/t_end/text/frame_paths/visual_offsets
        L1_text.npz      BGE-M3 vectors, shape (N_L1, 1024)
        L1_visual.npz    SigLIP vectors, shape (sum_k, D_siglip)
        frames/c{idx:05d}_f{local:02d}.jpg
        meta.json        {video_id, duration, status, ...}

The legacy ``attributes.json`` / ``vectors.npz["attribute_vecs"]`` are ignored —
the static attribute layer is no longer in the schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import MemoryBank, MemoryChunk


def _load_legacy_dir(vd: Path) -> MemoryBank:
    """Load a legacy {video_id}/ directory bank (narrative + vectors only).

    visual_vecs in vectors.npz is optional (pre-2026-06-19 banks stored
    v_visual inside narrative.json instead). Fall back to the per-chunk JSON
    field when npz visual_vecs is absent, then leave v_visual=[] if neither
    source has it — downstream visual rerank gates on truthy v_visual.
    """
    video_id = vd.name
    narr = json.loads((vd / "narrative.json").read_text())
    vecs = np.load(vd / "vectors.npz")

    n_vecs = vecs["narrative_vecs"]
    v_vecs = vecs["visual_vecs"] if "visual_vecs" in vecs.files else None
    chunk_ids = vecs["chunk_ids"]
    cid_to_vidx = {int(cid): i for i, cid in enumerate(chunk_ids)}

    chunks = []
    for nc in narr.get("chunks", []):
        cid = nc["chunk_id"]
        vi = cid_to_vidx.get(cid)
        v_dynamic = n_vecs[vi].tolist() if vi is not None else []
        if v_vecs is not None and vi is not None:
            v_visual = v_vecs[vi].tolist()
        else:
            v_visual = nc.get("v_visual", []) or []

        chunks.append(MemoryChunk(
            video_id=video_id,
            chunk_id=cid,
            start_time=nc.get("start_time", 0.0),
            end_time=nc.get("end_time", 0.0),
            memory_text=nc.get("narrative", "") or "",
            visual_caption=" | ".join(nc.get("caption", []) or []),
            asr=nc.get("speech_text", "") or "",
            sampled_frames=nc.get("sampled_frames", []) or [],
            keyframe_ts=nc.get("keyframe_ts", 0.0),
            keyframe_path=nc.get("keyframe_path", "") or "",
            v_dynamic=v_dynamic,
            v_visual=v_visual,
        ))

    return MemoryBank(
        video_id=video_id,
        duration=narr.get("duration", 0.0),
        chunks=chunks,
        memory_kind="similarity_group",
    )


def _load_pyramid_upper(vd: Path, layer: int, video_id: str, id_offset: int) -> list:
    """Load one upper pyramid layer (L2/L3/L4) into MemoryChunk list.

    chunk_id = id_offset + row["idx"] to keep ids globally unique across layers.
    These chunks carry no visual info (keyframe_path="", v_visual=[]).
    memory_text = row["text"] (the summary; timeline/causality prepended for display).
    """
    jsonl = vd / f"L{layer}.jsonl"
    npz   = vd / f"L{layer}.npz"
    if not jsonl.exists():
        return []
    rows = [json.loads(l) for l in jsonl.open()]
    vecs = np.load(npz)["vectors"] if npz.exists() else None  # (N, 1024)

    chunks = []
    for row in rows:
        idx = row["idx"]
        v_dyn = vecs[idx].tolist() if (vecs is not None and idx < len(vecs)) else []

        # Build a richer display text: summary + timeline bullets
        text = row.get("text", "")
        timeline = row.get("timeline", [])
        if timeline:
            tl_str = " → ".join(timeline)
            display = f"[L{layer}][{row['t_start']:.0f}s-{row['t_end']:.0f}s] {text}\nTimeline: {tl_str}"
        else:
            display = f"[L{layer}][{row['t_start']:.0f}s-{row['t_end']:.0f}s] {text}"

        chunks.append(MemoryChunk(
            video_id=video_id,
            chunk_id=id_offset + idx,
            start_time=row["t_start"],
            end_time=row["t_end"],
            memory_text=display,
            v_dynamic=v_dyn,
            layer=layer,
        ))
    return chunks


def _load_pyramid_dir(vd: Path) -> MemoryBank:
    """Load a pyramid-format {video_id}/ directory (build_pyramid.py output).

    L1 chunks go into bank.chunks (fine-grained, with visual frames).
    L2/L3/L4 chunks go into bank.l2/l3/l4_chunks (coarser, text-only).
    chunk_id offsets: L1=raw idx, L2=100000+idx, L3=200000+idx, L4=300000+idx.
    """
    video_id = vd.name
    meta = json.loads((vd / "meta.json").read_text())
    duration = meta.get("duration", 0.0)

    rows = [json.loads(l) for l in (vd / "L1.jsonl").open()]
    text_vecs = np.load(vd / "L1_text.npz")["vectors"]   # (N, 1024)
    vis_vecs  = np.load(vd / "L1_visual.npz")["vectors"] if (vd / "L1_visual.npz").exists() else None

    chunks = []
    for row in rows:
        idx = row["idx"]
        v_dyn = text_vecs[idx].tolist() if idx < len(text_vecs) else []

        # mean SigLIP over this chunk's frames for cross-chunk visual dedup
        offsets = row.get("visual_offsets", [])
        if vis_vecs is not None and offsets:
            v_vis = vis_vecs[offsets].mean(axis=0).tolist()
        elif vis_vecs is not None:
            v_vis = [0.0] * vis_vecs.shape[1]  # no frames: neutral zero vector
        else:
            v_vis = []

        # absolute path to sharpest (first) representative frame
        fps = row.get("frame_paths", [])
        kf_path = str(vd / fps[0]) if fps else ""

        chunks.append(MemoryChunk(
            video_id=video_id,
            chunk_id=idx,
            start_time=row["t_start"],
            end_time=row["t_end"],
            memory_text=row.get("text", ""),
            v_dynamic=v_dyn,
            v_visual=v_vis,
            keyframe_path=kf_path,
            keyframe_ts=(row.get("frame_ts") or [row["t_start"]])[0],
            layer=1,
        ))

    l2 = _load_pyramid_upper(vd, 2, video_id, id_offset=100_000)
    l3 = _load_pyramid_upper(vd, 3, video_id, id_offset=200_000)
    l4 = _load_pyramid_upper(vd, 4, video_id, id_offset=300_000)

    return MemoryBank(
        video_id=video_id,
        duration=duration,
        chunks=chunks,
        l2_chunks=l2,
        l3_chunks=l3,
        l4_chunks=l4,
        memory_kind="pyramid_L1",
    )


def load_bank(path: str | Path) -> MemoryBank:
    """Load a MemoryBank from either layout.

    ``path`` may point to either:
      * a ``{video_id}.json`` file (new single-file bank), or
      * a ``{video_id}/`` directory (legacy multi-file bank).

    Anything else raises ``FileNotFoundError``. The caller need not know
    which format is on disk.
    """
    p = Path(path)
    if p.is_file():
        return MemoryBank.model_validate_json(p.read_text())
    if p.is_dir():
        if (p / "L1.jsonl").exists():
            return _load_pyramid_dir(p)
        return _load_legacy_dir(p)
    # Convenience: if {path}.json exists alongside, treat that as the bank.
    sibling_json = p.with_suffix(".json")
    if sibling_json.is_file():
        return MemoryBank.model_validate_json(sibling_json.read_text())
    raise FileNotFoundError(f"no bank found at {path}")
