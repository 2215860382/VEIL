"""4 层时间金字塔记忆库构建。

层级:
  L1 = 10s  → 多图 VLM caption (纯文本, ≤200 词), 字幕带时间戳融入 prompt
  L2 = 30s  → LLM 概括 L1 文本, JSON {text, timeline, causality}, ≤300 词
  L3 = 180s → LLM 概括 L2 text, JSON {...}, ≤1500 词
  L4 = 600s → LLM 概括 L3 text, JSON {...}, ≤3000 词

视觉处理(仅 L1):
  ffmpeg 1fps → 黑帧剔除 (灰度阈值) → SigLIP 编码 →
  相邻余弦 >0.95 合并组 → 每组取 Laplacian 方差最大帧 (最清晰) →
  保留 k 张代表帧 + 字幕 → VLM 多图 caption

落盘:
  outputs/memory/pyramid/<video_id>/
      L1.jsonl           {idx, t_start, t_end, text, frame_paths, frame_ts, visual_offsets}
      L1_text.npz        (N_L1, 1024)   BGE-M3
      L1_visual.npz      (sum_k, D)     SigLIP
      frames/c{idx:05d}_f{local:02d}.jpg   ↔ L1_visual.npz 行号
      L2.jsonl  L2.npz   {idx, t_start, t_end, text, timeline, causality}
      L3.jsonl  L3.npz
      L4.jsonl  L4.npz
      meta.json          {video_id, duration, layer_counts, layers_present, status}
      progress.json      逐层完成进度

用法:
    cd /home2/ycj/Project/VEIL

    # 单视频, API 模式
    PYTHONPATH=. python -m src.build_memory.build_multi_pyramid \\
        --video-file /path/to/video.mp4 \\
        --subtitle-dir /path/to/srt_dir \\
        --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \\
        --api-url http://localhost:8000 \\
        --siglip-model /path/to/siglip \\
        --bge-model /path/to/bge-m3

    # 批量 (本地 VLM)
    PYTHONPATH=. python -m src.build_memory.build_multi_pyramid \\
        --video-dir /home/videos \\
        --vlm-model /path/to/Qwen-VL \\
        --siglip-model /path/to/siglip \\
        --bge-model /path/to/bge-m3 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger
from src.build_memory.build_single_similarity import parse_srt
from src.build_memory.core.frame_pipeline import (
    extract_frames,
    group_frames,
    select_sharpest,
    select_top_k_sharpest,
    _prededup_frames,
    is_blank_pixel,
    is_blurry,
    BLUR_FAILSAFE_RATIO,
)

log = get_logger("build_pyramid")


# ── Constants ──────────────────────────────────────────────────────────────────
L1_DUR = 10
L2_DUR = 30
L3_DUR = 180
L4_DUR = 600
L1_FPS = 1.0
BUDGETS = {1: 200, 2: 300, 3: 1500, 4: 3000}

MAX_VLM_RETRIES = 3

# SigLIP dedup
DEDUP_COS_THRESH = 0.95

# Stored frame size
FRAME_RESIZE = 448
JPEG_QUALITY = 85


# ── Prompts ────────────────────────────────────────────────────────────────────
L1_PROMPT_TEMPLATE = """You are given {n_frames} key frames extracted from a 10-second video window \
[t={t_start:.1f}s, {t_end:.1f}s) (frames at t = {timestamps}, in chronological order). \
Any speech subtitles aligned to absolute time within the window are listed below.

{subtitle_block}

Write ONE plain-text paragraph (NOT JSON, NOT a list) describing what happens \
in this window. The caption MUST cover every applicable item below; omit a \
category only if it truly does not appear:

  1. SUBJECTS: every person/animal/object that acts or is acted on. Use proper \
names if they appear on-screen or are said in the subtitles; otherwise use \
short descriptors ("the woman in the red jacket").
  2. ACTIONS: what each subject does, in temporal order (use words like \
"first", "then", "finally" when motion changes).
  3. APPEARANCE: clothing, color, hair, body posture, salient props.
  4. SPATIAL POSITION: where subjects are in frame (left/right/center/foreground/\
background) and their relative position to each other or to landmarks.
  5. ON-SCREEN NUMBERS: any digits visible (scoreboard, clock, jersey, price, \
timer, HUD) — quote them verbatim.
  6. OCR TEXT: any readable text/signs/captions/logos — quote them verbatim \
inside double quotes.

Hard constraints:
- Maximum 200 words.
- Single paragraph, no bullet points, no headings, no markdown.
- Do not invent details you cannot see or hear. If a subtitle contradicts the \
visual, prefer the visual and mention the discrepancy.
- Do not refer to "frame 1", "frame 2", etc.; speak in narrative time \
("at the start", "midway", "at the end") or in seconds.
"""


SUMMARY_PROMPT_TEMPLATE = """You are summarizing {n_children} consecutive sub-segments \
of a video. They cover the window [t={t_start:.1f}s, {t_end:.1f}s) and are \
listed below in time order, each prefixed with its absolute time range.

{child_blocks}

Produce a JSON object (and ONLY a JSON object, no markdown fences, no prose \
before or after) with EXACTLY these three keys:

  "text":      A coherent narrative summary of the whole window, written as \
continuous prose. MUST be ≤ {budget} words. Preserve named subjects, on-screen \
numbers, and verbatim OCR text from the children. Do not drop critical state \
changes.
  "timeline":  A JSON array of short event strings in chronological order, \
each starting with one of "first", "then", "next", "later", "finally". \
May be an empty array [] if nothing distinct happens.
  "causality": A JSON array of cause→effect strings, format \
"because X, Y so Z" or "X leads to Y". May be empty [].

Hard rules:
- Output must be valid JSON parseable by json.loads.
- "text" budget {budget} words is a HARD upper limit.
- Use information present in the child blocks only; do not invent.
- Quote OCR strings and numbers verbatim inside "text" when relevant.
"""


EMPTY_SUMMARY = {"text": "[summary unavailable]", "timeline": [], "causality": []}


# ── Subtitle filter (preserve timestamps) ──────────────────────────────────────
def format_window_subtitles(entries, t_start: float, t_end: float) -> str:
    """Filter SRT entries overlapping [t_start, t_end); return formatted block."""
    hits = [(es, ee, txt) for es, ee, txt in entries if es < t_end and ee > t_start]
    if not hits:
        return "No subtitles in this window."
    lines = [f"Subtitles in this window [t={t_start:.1f}s, {t_end:.1f}s):"]
    for es, _ee, txt in hits:
        lines.append(f"[t={es:.1f}s] {txt}")
    return "\n".join(lines)


def video_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def resize_save_jpeg(src: Path, dst: Path, short_side: int, quality: int):
    from PIL import Image
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = min(w, h)
    if s > short_side:
        if w <= h:
            new_w = short_side
            new_h = int(round(h * short_side / w))
        else:
            new_h = short_side
            new_w = int(round(w * short_side / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(dst, "JPEG", quality=quality)


# ── L1 per-chunk processing (dedup) ────────────────────────────────────────────
def process_l1_chunk(
    chunk_idx: int,
    chunk_frames: List[Path],
    chunk_ts: List[float],
    siglip,
    frames_outdir: Path,
) -> Tuple[List[str], List[float], List[Path], np.ndarray]:
    """Black-filter + Blur-filter (with failsafe) + SigLIP dedup + save kept frames.

    Returns (rel_paths, kept_ts, kept_src_paths, kept_vecs).
    """
    # 1. Pixel-level black/uniform frame filter.
    non_black = [(i, p) for i, p in enumerate(chunk_frames) if not is_blank_pixel(p)]
    if not non_black:
        return [], [], [], np.zeros((0, 0), dtype=np.float32)

    # 2. Blur filter with per-chunk failsafe: if the majority of frames in this
    #    10-second window would be dropped (low-light / motion blur / PPT), keep all.
    blurry_flags = [is_blurry(p) for _, p in non_black]
    drop_ratio = sum(blurry_flags) / len(blurry_flags)
    if drop_ratio > BLUR_FAILSAFE_RATIO:
        valid_pairs = non_black
    else:
        valid_pairs = [pair for pair, bl in zip(non_black, blurry_flags) if not bl]
    if not valid_pairs:
        return [], [], [], np.zeros((0, 0), dtype=np.float32)

    valid_idx   = [i for i, _ in valid_pairs]
    valid_paths = [chunk_frames[i] for i in valid_idx]
    valid_ts = [chunk_ts[i] for i in valid_idx]

    # Pre-dedup: remove near-identical consecutive frames before SigLIP encode.
    prededup = _prededup_frames(valid_paths)
    valid_paths = [valid_paths[i] for i in prededup]
    valid_ts    = [valid_ts[i]    for i in prededup]

    vecs = siglip.encode_images([str(p) for p in valid_paths])
    if vecs.shape[0] == 0:
        return [], [], [], np.zeros((0, 0), dtype=np.float32)

    dummy_ts = list(range(len(valid_paths)))
    groups = group_frames(
        vecs, dummy_ts,
        theta=DEDUP_COS_THRESH, n_max=20, min_size=1,
    )

    kept_local = []
    for g in groups:
        sharp_idx = select_sharpest([str(p) for p in valid_paths], g.frame_indices)
        kept_local.append(sharp_idx)

    kept_paths_src = [valid_paths[i] for i in kept_local]
    kept_ts_sec = [valid_ts[i] for i in kept_local]
    kept_vecs = vecs[kept_local]

    # Cap to top-2 sharpest; preserves temporal order.
    if len(kept_paths_src) > 2:
        top2 = select_top_k_sharpest(kept_paths_src, k=2)
        kept_paths_src = [kept_paths_src[i] for i in top2]
        kept_ts_sec    = [kept_ts_sec[i]    for i in top2]
        kept_vecs      = kept_vecs[np.array(top2)]

    rel_paths = []
    for local_i, src in enumerate(kept_paths_src):
        dst_name = f"c{chunk_idx:05d}_f{local_i:02d}.jpg"
        dst = frames_outdir / dst_name
        resize_save_jpeg(src, dst, FRAME_RESIZE, JPEG_QUALITY)
        rel_paths.append(f"frames/{dst_name}")

    return rel_paths, kept_ts_sec, kept_paths_src, kept_vecs


# ── L1 caption (multi-image VLM call) ──────────────────────────────────────────
async def l1_caption_window(
    vlm,
    src_frame_paths: List[Path],
    frame_ts: List[float],
    srt_block: str,
    t_start: float,
    t_end: float,
    sem: asyncio.Semaphore,
) -> str:
    from PIL import Image
    pil_frames = [Image.open(p).convert("RGB") for p in src_frame_paths]
    ts_str = ", ".join(f"{t:.1f}s" for t in frame_ts)
    prompt = L1_PROMPT_TEMPLATE.format(
        n_frames=len(src_frame_paths),
        t_start=t_start, t_end=t_end,
        timestamps=ts_str,
        subtitle_block=srt_block,
    )

    async with sem:
        for attempt in range(MAX_VLM_RETRIES):
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    lambda: vlm.chat_with_frames(pil_frames, prompt, max_new_tokens=320),
                )
                return text.strip()
            except Exception as e:
                if attempt == MAX_VLM_RETRIES - 1:
                    log.warning("L1 caption failed [%s, %s) after %d retries: %s",
                                t_start, t_end, MAX_VLM_RETRIES, e)
                    return f"[no visible content at t={t_start:.1f}s-{t_end:.1f}s]"
                await asyncio.sleep(2 ** attempt)
    return ""


async def build_l1(
    video_path: Path,
    srt_entries: list,
    duration: float,
    vlm,
    siglip,
    work_dir: Path,
    out_dir: Path,
    concurrency: int,
) -> Tuple[List[dict], np.ndarray]:
    """Extract frames, slice into 10s windows, dedup per window, then concurrently caption."""
    frames_dir = work_dir / "all_frames"
    log.info("[L1] extract frames @ %sfps", L1_FPS)
    all_paths = extract_frames(str(video_path), frames_dir, fps=L1_FPS)
    if not all_paths:
        return [], np.zeros((0, 0), dtype=np.float32)

    all_ts = [i / L1_FPS for i in range(len(all_paths))]

    n_l1 = max(1, int(np.ceil(duration / L1_DUR)))

    frames_outdir = out_dir / "frames"
    frames_outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_visual_vecs = []
    global_offset = 0

    log.info("[L1] dedup %d chunks", n_l1)
    for chunk_idx in range(n_l1):
        t_start = chunk_idx * L1_DUR
        t_end = min(t_start + L1_DUR, duration)
        chunk_idxs = [i for i, t in enumerate(all_ts) if t_start <= t < t_end]
        if not chunk_idxs:
            continue

        chunk_frames = [all_paths[i] for i in chunk_idxs]
        chunk_ts = [all_ts[i] for i in chunk_idxs]

        rel_paths, kept_ts, kept_src, kept_vecs = process_l1_chunk(
            chunk_idx, chunk_frames, chunk_ts, siglip, frames_outdir,
        )

        if not rel_paths:
            rows.append({
                "idx": chunk_idx,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "text": f"[no visible content at t={t_start:.1f}s-{t_end:.1f}s]",
                "frame_paths": [],
                "frame_ts": [],
                "visual_offsets": [],
            })
            continue

        n_kept = len(rel_paths)
        offsets = list(range(global_offset, global_offset + n_kept))
        global_offset += n_kept
        all_visual_vecs.append(kept_vecs)

        rows.append({
            "idx": chunk_idx,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "text": None,
            "frame_paths": rel_paths,
            "frame_ts": [float(t) for t in kept_ts],
            "visual_offsets": offsets,
            "_src_paths": kept_src,
        })

    # Now caption pending rows concurrently
    pending = [r for r in rows if r["text"] is None]
    log.info("[L1] caption %d windows (concurrency=%d)", len(pending), concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def caption_one(row):
        srt_block = format_window_subtitles(srt_entries, row["t_start"], row["t_end"])
        text = await l1_caption_window(
            vlm, row["_src_paths"], row["frame_ts"], srt_block,
            row["t_start"], row["t_end"], sem,
        )
        row["text"] = text

    await asyncio.gather(*[caption_one(r) for r in pending])
    for row in rows:
        row.pop("_src_paths", None)

    visual_vecs = (np.concatenate(all_visual_vecs, axis=0)
                   if all_visual_vecs
                   else np.zeros((0, 0), dtype=np.float32))
    return rows, visual_vecs


# ── L2/L3/L4 summarization ─────────────────────────────────────────────────────
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json_with_repair(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return None


async def summarize_text_window(
    llm,
    child_blocks: List[str],
    t_start: float,
    t_end: float,
    budget: int,
    sem: asyncio.Semaphore,
) -> dict:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        n_children=len(child_blocks),
        t_start=t_start, t_end=t_end,
        child_blocks="\n\n".join(child_blocks),
        budget=budget,
    )
    messages = [{"role": "user", "content": prompt}]
    max_new = int(budget * 2.2)

    async with sem:
        for attempt in range(MAX_VLM_RETRIES):
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    lambda: llm.chat(messages, max_new_tokens=max_new),
                )
                obj = _parse_json_with_repair(text)
                if obj is not None and all(k in obj for k in ("text", "timeline", "causality")):
                    obj["text"] = str(obj["text"])
                    obj["timeline"] = list(obj["timeline"]) if obj["timeline"] else []
                    obj["causality"] = list(obj["causality"]) if obj["causality"] else []
                    return obj
                log.warning("summarize JSON parse miss attempt=%d at [%s,%s)",
                            attempt, t_start, t_end)
            except Exception as e:
                log.warning("summarize call failed attempt=%d at [%s,%s): %s",
                            attempt, t_start, t_end, e)
            if attempt < MAX_VLM_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    return dict(EMPTY_SUMMARY)


async def build_upper_layer(
    child_rows: List[dict],
    layer_id: int,
    span: int,
    budget: int,
    llm,
    concurrency: int,
) -> List[dict]:
    """L2/L3/L4 builder. Children are slotted by floor(t_start / span)."""
    if not child_rows:
        return []

    total_end = max(r["t_end"] for r in child_rows)
    n_chunks = max(1, int(np.ceil(total_end / span)))
    buckets: dict[int, list] = {i: [] for i in range(n_chunks)}
    for r in child_rows:
        idx = int(r["t_start"] // span)
        if idx in buckets:
            buckets[idx].append(r)

    sem = asyncio.Semaphore(concurrency)

    async def build_one(chunk_idx: int, children: list):
        if not children:
            return None
        children = sorted(children, key=lambda c: c["t_start"])
        t_start = float(chunk_idx * span)
        t_end = float(max(c["t_end"] for c in children))
        child_blocks = [
            f"[{c['t_start']:.1f}s-{c['t_end']:.1f}s] {c['text']}"
            for c in children
        ]
        result = await summarize_text_window(llm, child_blocks, t_start, t_end, budget, sem)
        return {
            "idx": chunk_idx,
            "t_start": t_start,
            "t_end": t_end,
            "text": result["text"],
            "timeline": result["timeline"],
            "causality": result["causality"],
        }

    log.info("[L%d] summarize %d chunks (concurrency=%d)",
             layer_id, sum(1 for v in buckets.values() if v), concurrency)
    tasks = [build_one(i, ch) for i, ch in buckets.items() if ch]
    results = await asyncio.gather(*tasks)
    rows = [r for r in results if r is not None]
    rows.sort(key=lambda r: r["idx"])
    return rows


# ── Embedding + dumping ────────────────────────────────────────────────────────
def _write_jsonl(rows: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def embed_and_dump_text(rows: List[dict], layer_id: int, embedder, out_dir: Path) -> None:
    jsonl_path = out_dir / f"L{layer_id}.jsonl"
    npz_name = "L1_text.npz" if layer_id == 1 else f"L{layer_id}.npz"
    npz_path = out_dir / npz_name

    _write_jsonl(rows, jsonl_path)

    if not rows:
        np.savez(npz_path, vectors=np.zeros((0, 1024), dtype=np.float32))
        return

    texts = [r["text"] for r in rows]
    vecs = embedder.encode(texts).astype(np.float32)
    np.savez(npz_path, vectors=vecs)


def dump_visual(visual_vecs: np.ndarray, out_dir: Path) -> None:
    np.savez(out_dir / "L1_visual.npz", vectors=visual_vecs)


def write_meta(out_dir: Path, video_id: str, duration: float,
               layer_counts: dict, status: str, **extra) -> None:
    meta = {
        "video_id": video_id,
        "duration": duration,
        "layer_counts": layer_counts,
        "layers_present": [l for l, n in layer_counts.items() if n > 0],
        "status": status,
        **extra,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def write_progress(out_dir: Path, progress: dict) -> None:
    (out_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def load_progress(out_dir: Path) -> dict:
    p = out_dir / "progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Driver ─────────────────────────────────────────────────────────────────────
def process_video(video_path: Path, srt_path: Optional[Path],
                  vlm, llm, embedder, siglip,
                  out_root: Path, args) -> Tuple[str, dict]:
    video_id = video_path.stem
    out_dir = out_root / video_id

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "complete":
                log.info("[%s] skip (already complete)", video_id)
                return "skipped", meta.get("layer_counts", {})
        except json.JSONDecodeError:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration = video_duration(video_path)
    except Exception as e:
        log.error("[%s] ffprobe failed: %s", video_id, e)
        write_meta(out_dir, video_id, 0.0,
                   {1: 0, 2: 0, 3: 0, 4: 0}, "ffprobe_error", error=str(e))
        return "error", {}

    log.info("[%s] duration=%.1fs", video_id, duration)

    srt_entries: list = []
    if srt_path and srt_path.exists():
        try:
            srt_entries = parse_srt(srt_path)
            log.info("[%s] SRT %d entries", video_id, len(srt_entries))
        except Exception as e:
            log.warning("[%s] SRT parse failed: %s", video_id, e)

    progress = load_progress(out_dir)
    if args.overwrite:
        progress = {}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with tempfile.TemporaryDirectory(prefix=f"pyramid_{video_id}_") as work_dir_s:
            work_dir = Path(work_dir_s)

            # ── L1 ────────────────────────────────────────────
            if progress.get("L1", {}).get("done"):
                log.info("[%s] L1 resume from disk", video_id)
                l1_rows = _read_jsonl(out_dir / "L1.jsonl")
            else:
                l1_rows, visual_vecs = loop.run_until_complete(
                    build_l1(video_path, srt_entries, duration, vlm, siglip,
                             work_dir, out_dir, args.l1_concurrency)
                )
                embed_and_dump_text(l1_rows, 1, embedder, out_dir)
                dump_visual(visual_vecs, out_dir)
                progress["L1"] = {"n": len(l1_rows), "done": True}
                write_progress(out_dir, progress)

            # ── L2 ────────────────────────────────────────────
            if progress.get("L2", {}).get("done"):
                log.info("[%s] L2 resume from disk", video_id)
                l2_rows = _read_jsonl(out_dir / "L2.jsonl")
            else:
                l2_rows = loop.run_until_complete(
                    build_upper_layer(l1_rows, 2, L2_DUR, BUDGETS[2],
                                      llm, args.text_concurrency)
                )
                embed_and_dump_text(l2_rows, 2, embedder, out_dir)
                progress["L2"] = {"n": len(l2_rows), "done": True}
                write_progress(out_dir, progress)

            # ── L3 ────────────────────────────────────────────
            if progress.get("L3", {}).get("done"):
                log.info("[%s] L3 resume from disk", video_id)
                l3_rows = _read_jsonl(out_dir / "L3.jsonl")
            else:
                l3_rows = loop.run_until_complete(
                    build_upper_layer(l2_rows, 3, L3_DUR, BUDGETS[3],
                                      llm, args.text_concurrency)
                )
                embed_and_dump_text(l3_rows, 3, embedder, out_dir)
                progress["L3"] = {"n": len(l3_rows), "done": True}
                write_progress(out_dir, progress)

            # ── L4 ────────────────────────────────────────────
            if progress.get("L4", {}).get("done"):
                log.info("[%s] L4 resume from disk", video_id)
                l4_rows = _read_jsonl(out_dir / "L4.jsonl")
            else:
                l4_rows = loop.run_until_complete(
                    build_upper_layer(l3_rows, 4, L4_DUR, BUDGETS[4],
                                      llm, args.text_concurrency)
                )
                embed_and_dump_text(l4_rows, 4, embedder, out_dir)
                progress["L4"] = {"n": len(l4_rows), "done": True}
                write_progress(out_dir, progress)
    finally:
        loop.close()

    layer_counts = {1: len(l1_rows), 2: len(l2_rows), 3: len(l3_rows), 4: len(l4_rows)}
    write_meta(out_dir, video_id, duration, layer_counts, "complete")
    log.info("[%s] done. counts=%s", video_id, layer_counts)
    return "complete", layer_counts


def discover_videos(
    video_file: Optional[Path],
    video_dir: Optional[Path],
    video_list: Optional[Path] = None,
) -> List[Path]:
    if video_file:
        return [Path(video_file)]
    if video_list:
        return [Path(line.strip()) for line in video_list.read_text().splitlines() if line.strip()]
    if video_dir:
        exts = ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov")
        paths: List[Path] = []
        for ext in exts:
            paths.extend(Path(video_dir).rglob(ext))
        return sorted(paths)
    return []


def main():
    global DEDUP_COS_THRESH
    global MAX_VLM_RETRIES, FRAME_RESIZE, JPEG_QUALITY

    ap = argparse.ArgumentParser(description="Build 4-layer pyramid memory from videos")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video-file", type=Path)
    grp.add_argument("--video-dir", type=Path)
    grp.add_argument("--video-list", type=Path,
                     help="Text file with one video path per line")
    ap.add_argument("--subtitle-dir", type=Path, default=None,
                    help="Directory containing <stem>.srt files")

    ap.add_argument("--out-root", type=Path,
                    default=Path("outputs/memory/pyramid"))
    ap.add_argument("--overwrite", action="store_true")

    # VLM (L1 multi-image)
    ap.add_argument("--vlm-model", type=str, required=True)
    ap.add_argument("--api-url", type=str, default=None,
                    help="vLLM/OpenAI-compatible server URL. None → local model.")
    ap.add_argument("--api-model", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16")

    # LLM (L2-4 text-only)
    ap.add_argument("--llm-api-url", type=str, default=None,
                    help="defaults to --api-url")
    ap.add_argument("--llm-model", type=str, default=None,
                    help="defaults to --vlm-model")
    ap.add_argument("--llm-device", type=str, default=None,
                    help="defaults to --device")

    # SigLIP
    ap.add_argument("--siglip-model", type=str, required=True)
    ap.add_argument("--siglip-device", type=str, default="cuda:0")
    ap.add_argument("--siglip-api-url", type=str, default=None)
    ap.add_argument("--dedup-cos-thresh", type=float, default=DEDUP_COS_THRESH)

    # BGE
    ap.add_argument("--bge-model", type=str, required=True)
    ap.add_argument("--embedder-device", type=str, default="cuda:0")
    ap.add_argument("--embedder-batch", type=int, default=32)
    ap.add_argument("--bge-api-url", type=str, default=None)

    # Concurrency / retries
    ap.add_argument("--l1-concurrency", type=int, default=4)
    ap.add_argument("--text-concurrency", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=MAX_VLM_RETRIES)

    # Frame storage
    ap.add_argument("--frame-size", type=int, default=FRAME_RESIZE)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)

    # Misc
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N videos (dev)")

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    DEDUP_COS_THRESH = args.dedup_cos_thresh
    MAX_VLM_RETRIES = args.max_retries
    FRAME_RESIZE = args.frame_size
    JPEG_QUALITY = args.jpeg_quality

    # ── Clients ──────────────────────────────────
    from src.clients.vlm_client import VLMClient
    from src.clients.llm_client import LLMClient
    from src.clients.embedder import BGEM3Embedder
    from src.clients.siglip_embedder import SigLIPEmbedder

    log.info("Loading VLM (%s)...", "API" if args.api_url else "local")
    vlm = VLMClient(
        model_path=args.vlm_model,
        dtype=args.dtype,
        device=args.device,
        api_url=args.api_url,
        api_model=args.api_model,
    )

    log.info("Loading LLM (for L2-4)...")
    llm = LLMClient(
        model_path=args.llm_model or args.vlm_model,
        dtype=args.dtype,
        device=args.llm_device or args.device,
        api_url=args.llm_api_url or args.api_url,
        api_model=args.api_model,
    )

    log.info("Loading SigLIP on %s...",
             args.siglip_device if not args.siglip_api_url else "API")
    siglip = SigLIPEmbedder(
        args.siglip_model,
        device=args.siglip_device,
        api_url=args.siglip_api_url,
    )

    log.info("Loading BGE on %s...",
             args.embedder_device if not args.bge_api_url else "API")
    embedder = BGEM3Embedder(
        model_path=args.bge_model,
        device=args.embedder_device,
        batch_size=args.embedder_batch,
        api_url=args.bge_api_url,
    )

    videos = discover_videos(args.video_file, args.video_dir, args.video_list)
    if args.limit:
        videos = videos[:args.limit]

    log.info("Processing %d videos", len(videos))
    args.out_root.mkdir(parents=True, exist_ok=True)

    for i, v in enumerate(videos):
        srt = None
        if args.subtitle_dir:
            cand = args.subtitle_dir / f"{v.stem}.srt"
            if cand.exists():
                srt = cand
        log.info("[%d/%d] %s", i + 1, len(videos), v.name)
        try:
            process_video(v, srt, vlm, llm, embedder, siglip, args.out_root, args)
        except Exception:
            log.exception("Failed on %s", v.name)


if __name__ == "__main__":
    main()
