"""Build single-granularity similarity-grouped memory banks — NEW-PROMPT variant.

Uses the structured "Memory: <paragraph>\\nKey details: <comma list>" two-line
output format (CHUNK_NARRATIVE_PROMPT / CAPTION_SUMMARY_PROMPT below).

For the baseline plain-paragraph variant that produced the videomme_sim50_cap
bank (89.33% reference), see ``build_single_similarity.py``.

Pipeline per video:
  1. ffmpeg 1fps → JPEG frames
  2. Blank + blur filter (with failsafe)
  3. Pre-dedup (dHash, window=10) — removes near-identical consecutive frames
  4. SigLIP encode survivors → cosine-similarity grouping (θ≥0.8)
  5. Per-chunk dHash+SigLIP dedup → kept_frame_paths (visual representative subset)
     + top-2 sharpest cap on kept set → keyframe / v_visual
     all_frame_paths (post pre-dedup, no chunk-cap) is preserved for narrative VLM.
  6. SRT → speech_text per chunk (if subtitle_dir supplied)
  7. Narrative generation (two modes):
     • multi_image       : single VLM call with up to N frames → "Memory:/Key details:"
     • caption_summary   : per-frame caption (N calls) → LLM "Memory:/Key details:"
     Frames input to narrative = all_frame_paths down-sampled to
     --narrative-max-frames (default 12) to bound VLM context cost.
  8. BGE-M3 → semantic embedding of (narrative + speech)
  9. Write legacy directory bank:
        {out_dir}/{video_id}/narrative.json   chunks[].{narrative, caption,
                                              speech_text, sampled_frames,
                                              keyframe_ts, keyframe_path, v_visual}
        {out_dir}/{video_id}/vectors.npz      narrative_vecs + chunk_ids
        {out_dir}/{video_id}/keyframes_origin/    dedup kept frames at original res
        {out_dir}/{video_id}/keyframes_resized/   same files resized to ≤ frame_max_dim
        {out_dir}/{video_id}/frames_raw/          all post-cleanup frames per chunk (original res)

Usage:
    cd /home2/ycj/Project/VEIL
    PYTHONPATH=. python -m src.build_memory.build_single_similarity_newprompt \\
        --benchmark videomme \\
        --siglip-model /path/to/siglip \\
        --api-url http://localhost:8001 \\
        --api-model Qwen2.5-VL-7B-Instruct \\
        --siglip-gpu cuda:0 --bge-gpu cuda:0

    # Local VLM (no API):
    PYTHONPATH=. python -m src.build_memory.build_single_similarity_newprompt \\
        --benchmark videomme \\
        --siglip-model /path/to/siglip \\
        --vlm-model /path/to/Qwen-VL \\
        --vlm-gpu cuda:0 --siglip-gpu cuda:0 --bge-gpu cuda:0
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger

log = get_logger("build_single_similarity_newprompt")

# ── Narrative prompt ───────────────────────────────────────────────────────────

CHUNK_NARRATIVE_PROMPT = """\
You are building a searchable memory for long-video question answering.

Given {n_frames} key frame(s) and aligned speech/subtitles from one video chunk
[t={t_start:.0f}s–{t_end:.0f}s) (frames at t = {timestamps}, in chronological
order), write a compact memory entry that preserves the evidence needed to
answer future questions.

{subtitle_block}

Requirements:
1. Summarize the main event or scene progression in temporal order. Do not
   refer to "frame 1/2/3"; use narrative time ("at the start", "then", "finally").
2. Preserve concrete visual details: people (count, clothing, posture, position
   in frame), objects, scene, actions, locations, colors, numbers, text/OCR,
   spatial relations. Prefer specific descriptors ("woman in red jacket")
   over generic ones ("a person").
3. If speech/subtitles are present, preserve their important content,
   especially names, facts, decisions, explanations, questions, answers,
   and numbers. Quote OCR/text verbatim.
4. Only connect speech to visible events when the link is clear; otherwise
   list them separately.
5. Do not add facts not supported by the frames or speech.
6. If evidence is ambiguous, state the ambiguity instead of resolving it.

Output format (exactly these two lines, nothing else):
Memory: <one English paragraph, ≤ 100 words, describing what happens in this chunk>
Key details: <≤ 20 comma-separated searchable items: OCR strings, numbers,
named people, distinctive objects, key actions>

Example of "Key details":
Key details: woman in red jacket, man "John Smith", OCR "EXIT 47",
3 children, kitchen, hands shuffling cards, "we need to leave by 5"
"""

PER_FRAME_CAPTION_PROMPT = """\
Describe this video frame for later long-video question answering.

Focus on concrete, searchable visual evidence:
- people (count, clothing, posture, position in frame), objects, scene, actions
- spatial relations (left/right, foreground/background, who relates to whom)
- visible text/OCR, numbers, logos, signs, brands, colors, distinctive attributes
- states or attributes directly visible in this single frame

Rules:
- Use only what is visible. Do not infer identities, intentions, causes,
  or events happening outside or before/after this frame.
- Prefer specific descriptors ("woman in red jacket holding a microphone")
  over generic ones ("a person"). Avoid vague placeholders.
- Quote on-screen text inside double quotes, exactly as written.
- If something is unclear, hedge with "appears to be" or "possibly".
  Do not omit it silently.
- Output: one dense paragraph (1–2 short sentences), ≤ 50 words.
  No bullet points, no preamble, no closing remarks."""

CAPTION_SUMMARY_PROMPT = """\
You are building a searchable memory for long-video question answering.

Given frame-level captions and aligned speech/subtitles from one video chunk
[t={t_start:.0f}s–{t_end:.0f}s), write a compact memory entry that preserves
the evidence needed to answer future questions.

{subtitle_block}
Frame captions (in chronological order):
{cap_block}

Requirements:
1. Summarize the main event or scene progression in temporal order. Do not
   refer to "frame 1/2/3"; use narrative time ("at the start", "then", "finally").
2. Preserve concrete visual details: people, objects, actions, locations,
   colors, clothing, numbers, text/OCR, and spatial relations.
3. Preserve important speech/subtitle content, especially names, facts,
   decisions, explanations, questions, answers, and numbers. Quote OCR/text verbatim.
4. Only connect speech to visible events when the link is clear; otherwise
   list them separately.
5. Do not add facts not supported by the captions or speech.
6. If evidence is ambiguous, state the ambiguity instead of resolving it.

Output format (exactly these two lines, nothing else):
Memory: <one English paragraph, ≤ 100 words, describing what happens in this chunk>
Key details: <≤ 20 comma-separated searchable items: OCR strings, numbers,
named people, distinctive objects, key actions>

Example of "Key details":
Key details: woman in red jacket, man "John Smith", OCR "EXIT 47",
3 children, kitchen, hands shuffling cards, "we need to leave by 5"
"""


# ── SRT parsing & subtitle alignment ──────────────────────────────────────────

_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _srt_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: str | Path) -> List[Tuple[float, float, str]]:
    """Parse SRT file → list of (start_sec, end_sec, text)."""
    entries: List[Tuple[float, float, str]] = []
    text_lines: List[str] = []
    t_start = t_end = 0.0
    in_block = False

    for raw in Path(srt_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = _SRT_TIME_RE.match(line)
        if m:
            if in_block and text_lines:
                txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
                if txt:
                    entries.append((t_start, t_end, txt))
            t_start = _srt_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            t_end   = _srt_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
            text_lines = []
            in_block = True
        elif in_block:
            if line and not line.isdigit():
                text_lines.append(line)
            elif not line and text_lines:
                txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
                if txt:
                    entries.append((t_start, t_end, txt))
                text_lines = []
                in_block = False

    if in_block and text_lines:
        txt = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
        if txt:
            entries.append((t_start, t_end, txt))
    return entries


def align_subtitles(
    entries: List[Tuple[float, float, str]],
    t_start: float,
    t_end: float,
    max_chars: int = 500,
) -> str:
    """Return subtitle text overlapping with [t_start, t_end]."""
    parts = [txt for es, ee, txt in entries if ee >= t_start and es <= t_end]
    speech = " ".join(parts)
    if len(speech) > max_chars:
        speech = speech[:max_chars].rsplit(" ", 1)[0] + "…"
    return speech


def _select_narrative_frames(
    paths: List[Path], timestamps: List[float], n_max: int,
) -> Tuple[List[Path], List[float]]:
    """Evenly down-sample (paths, timestamps) to at most ``n_max`` items.

    Used by both narrative modes so the VLM input stays bounded regardless of
    how many post-cleanup frames the chunk has. Preserves temporal order.
    """
    n = len(paths)
    if n <= n_max:
        return list(paths), list(timestamps)
    step = n / n_max
    idxs = [int(i * step) for i in range(n_max)]
    return [paths[i] for i in idxs], [timestamps[i] for i in idxs]


# ── Narrative generation ───────────────────────────────────────────────────────

# Round-robin endpoint dispatch (supports comma-separated --api-url)
_endpoint_state: dict = {"endpoints": [], "rr": 0}


def _parse_endpoints(api_url: str) -> list[str]:
    return [u.strip().rstrip("/") for u in api_url.split(",") if u.strip()]


def _init_endpoints(api_url: str) -> None:
    _endpoint_state["endpoints"] = _parse_endpoints(api_url)
    _endpoint_state["rr"] = 0


async def _next_chat_url() -> str:
    eps = _endpoint_state["endpoints"]
    if not eps:
        raise RuntimeError("endpoints not initialized")
    ep = eps[_endpoint_state["rr"] % len(eps)]
    _endpoint_state["rr"] += 1
    return f"{ep}/v1/chat/completions"


def _build_narrative_prompt(
    n_frames: int,
    t_start: float,
    t_end: float,
    timestamps: List[float],
    speech: str,
) -> str:
    ts_str = ", ".join(f"{t:.0f}s" for t in timestamps)
    if speech.strip():
        subtitle_block = f"Speech/subtitles in this segment:\n{speech.strip()}"
    else:
        subtitle_block = "No subtitles in this segment."
    return CHUNK_NARRATIVE_PROMPT.format(
        n_frames=n_frames,
        t_start=t_start,
        t_end=t_end,
        timestamps=ts_str,
        subtitle_block=subtitle_block,
    )


async def _narrate_one_api(
    session,
    sem: asyncio.Semaphore,
    frame_paths: List[Path],
    timestamps: List[float],
    speech: str,
    t_start: float,
    t_end: float,
    api_url: str,
    api_model: str,
) -> str:
    prompt = _build_narrative_prompt(len(frame_paths), t_start, t_end, timestamps, speech)
    content: list = []
    for p in frame_paths:
        b64 = base64.b64encode(p.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 350,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    import aiohttp
    async with sem:
        try:
            url = await _next_chat_url()
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                data = await r.json()
                text = data["choices"][0]["message"]["content"].strip()
                if "</think>" in text:
                    text = text.split("</think>", 1)[1].strip()
                return text
        except Exception as e:
            log.warning("  API narrate [%.0f-%.0f s]: %s", t_start, t_end, e)
            return f"[narration unavailable at t={t_start:.0f}s–{t_end:.0f}s]"


async def _narrate_all_api(
    chunks_info: List[dict],
    api_url: str,
    api_model: str,
    concurrency: int,
) -> List[str]:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _narrate_one_api(
                session, sem,
                c["kept_frame_paths"], c["kept_timestamps"],
                c["speech"], c["t_start"], c["t_end"],
                api_url, api_model,
            )
            for c in chunks_info
        ]
        return list(await asyncio.gather(*tasks))


def narrate_chunks_api(
    chunks_info: List[dict],
    api_url: str,
    api_model: str,
    concurrency: int = 16,
) -> List[str]:
    """Async-batch multi-image narrative for all chunks of one video (API mode)."""
    _init_endpoints(api_url)
    return asyncio.run(_narrate_all_api(chunks_info, api_url, api_model, concurrency))


# ── Caption + Summary narrative (per-frame caption → LLM summary) ──────────────

async def _caption_one_api(session, sem, fp: Path,
                           api_url: str, api_model: str) -> str:
    import aiohttp
    b64 = base64.b64encode(Path(fp).read_bytes()).decode()
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": PER_FRAME_CAPTION_PROMPT},
    ]
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 80, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        try:
            url = await _next_chat_url()
            async with session.post(
                url,
                json=payload, timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                data = await r.json()
            t = data["choices"][0]["message"]["content"].strip()
            if "</think>" in t:
                t = t.split("</think>", 1)[1].strip()
            return t
        except Exception:
            return ""


async def _summarize_one_api(session, sem,
                             captions: List[str], timestamps: List[float],
                             speech: str, t_start: float, t_end: float,
                             api_url: str, api_model: str) -> str:
    import aiohttp
    labeled = [f"[t={t:.0f}s] {c}" for t, c in zip(timestamps, captions) if c.strip()]
    if not labeled:
        return ""
    sub = f"Speech/subtitles: {speech.strip()}\n" if speech.strip() else ""
    prompt = CAPTION_SUMMARY_PROMPT.format(
        t_start=t_start, t_end=t_end,
        subtitle_block=sub, cap_block="\n".join(labeled),
    )
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        try:
            url = await _next_chat_url()
            async with session.post(
                url,
                json=payload, timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                data = await r.json()
            t = data["choices"][0]["message"]["content"].strip()
            if "</think>" in t:
                t = t.split("</think>", 1)[1].strip()
            return t
        except Exception:
            return labeled[0].split("] ", 1)[-1]


async def _narrate_caption_summary(chunks_info: List[dict],
                                   api_url: str, api_model: str,
                                   concurrency: int) -> List[str]:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: per-frame captions (intermediate, not persisted)
        async def caption_chunk(c):
            tasks = [_caption_one_api(session, sem, fp, api_url, api_model)
                     for fp in c["narrative_frame_paths"]]
            return await asyncio.gather(*tasks)
        all_caps = await asyncio.gather(*[caption_chunk(c) for c in chunks_info])
        # Phase 2: LLM summary (the only output we keep)
        sum_tasks = [
            _summarize_one_api(session, sem,
                               caps, c["narrative_timestamps"],
                               c["speech"], c["t_start"], c["t_end"],
                               api_url, api_model)
            for caps, c in zip(all_caps, chunks_info)
        ]
        return list(await asyncio.gather(*sum_tasks))


def narrate_chunks_caption_summary(chunks_info: List[dict], api_url: str,
                                   api_model: str, concurrency: int = 16) -> List[str]:
    """Per-frame caption (intermediate) + LLM summary; one video at a time."""
    _init_endpoints(api_url)
    return asyncio.run(_narrate_caption_summary(chunks_info, api_url, api_model, concurrency))


def narrate_chunk_local(
    frame_paths: List[Path],
    timestamps: List[float],
    speech: str,
    t_start: float,
    t_end: float,
    vlm,
) -> str:
    """Single-chunk multi-image narrative (local VLM mode)."""
    from PIL import Image
    prompt = _build_narrative_prompt(len(frame_paths), t_start, t_end, timestamps, speech)
    pil_frames = [Image.open(str(p)).convert("RGB") for p in frame_paths]
    try:
        return vlm.chat_with_frames(pil_frames, prompt, max_new_tokens=200).strip()
    except Exception as e:
        log.warning("  local narrate [%.0f-%.0f s]: %s", t_start, t_end, e)
        return f"[narration unavailable at t={t_start:.0f}s–{t_end:.0f}s]"


# ── Video paths ────────────────────────────────────────────────────────────────

def _get_video_paths(cfg: dict) -> dict[str, str]:
    bench = cfg["benchmark"]["name"]
    if bench == "mlvu":
        json_dir  = Path(cfg["benchmark"]["json_dir"])
        video_dir = Path(cfg["benchmark"]["video_dir"])
        paths = {}
        for jf in cfg["benchmark"]["json_files"].values():
            p = json_dir / jf
            if not p.exists():
                continue
            subdir = video_dir / Path(jf).stem
            for item in json.loads(p.read_text()):
                vid = item.get("video", item.get("video_id", ""))
                if vid:
                    k = Path(vid).stem
                    if k not in paths:
                        paths[k] = str(subdir / vid)
        return paths
    elif bench == "videomme":
        from src.dataloader.videomme import load_videomme
        b = cfg["benchmark"]
        samples = load_videomme(
            parquet_path=b["parquet_path"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
        return {s.video_id: s.video_path for s in samples}
    else:
        raise ValueError(f"Unsupported benchmark: {bench}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benchmark", choices=["mlvu", "videomme"])
    src.add_argument("--config", help="Legacy YAML path")
    ap.add_argument("--vlm-model",       default=None,
                    help="Local VLM checkpoint (required without --api-url)")
    ap.add_argument("--vlm-gpu",         default="cuda:0")
    ap.add_argument("--siglip-model",    required=True)
    ap.add_argument("--siglip-gpu",      default="cuda:0")
    ap.add_argument("--bge-gpu",         default="cuda:0")
    ap.add_argument("--subtitle-dir",    default=None)
    ap.add_argument("--filter-from",     default=None)
    ap.add_argument("--fps",             type=float, default=1.0)
    ap.add_argument("--theta",           type=float, default=0.80)
    ap.add_argument("--n-max",           type=int,   default=30)
    ap.add_argument("--api-url",         default=None,
                    help="vLLM OpenAI-compatible base URL")
    ap.add_argument("--api-model",       default=None,
                    help="Required with --api-url")
    ap.add_argument("--api-concurrency", type=int, default=16)
    ap.add_argument("--out-dir",         default=None)
    ap.add_argument("--narrative-mode",  choices=["multi_image", "caption_summary"],
                    default="multi_image",
                    help="multi_image: send N frames to VLM in one call → paragraph; "
                         "caption_summary: per-frame caption + LLM summary")
    ap.add_argument("--narrative-max-frames", type=int, default=12,
                    help="Cap frames per chunk fed to narrative (both modes); "
                         "evenly down-sampled from g.all_frame_paths when exceeded")
    args = ap.parse_args()

    use_api = bool(args.api_url)
    if use_api and not args.api_model:
        ap.error("--api-model is required when --api-url is set.")
    if not use_api and not args.vlm_model:
        ap.error("--vlm-model is required when --api-url is not set.")

    if args.benchmark is not None:
        from src.build_memory.core import specs as build_specs
        cfg = build_specs.cfg_for_similarity_build(args.benchmark)
    else:
        cfg = load_config(args.config)
        bn = cfg.get("benchmark", {}).get("name")
        if bn in ("mlvu", "videomme") and not (cfg.get("memory") or {}).get("cache_dir"):
            from src.build_memory.core import specs as build_specs
            cfg.setdefault("memory", {})["cache_dir"] = str(
                build_specs.similarity_memory_cache_dir(bn)
            )

    mem_cfg = cfg.get("memory") or {}
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif mem_cfg.get("cache_dir"):
        out_dir = Path(mem_cfg["cache_dir"])
    else:
        ap.error("Pass --out-dir, or use --benchmark / YAML with memory.cache_dir set.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir.resolve())

    if args.subtitle_dir:
        subtitle_dir = Path(args.subtitle_dir)
    else:
        sub_yaml = cfg.get("benchmark", {}).get("subtitle_dir")
        subtitle_dir = Path(sub_yaml) if sub_yaml else None
    if subtitle_dir:
        log.info("Subtitle dir: %s", subtitle_dir)

    # ── Models ────────────────────────────────────────────────────────────────
    log.info("Loading SigLIP on %s …", args.siglip_gpu)
    t0 = time.time()
    from src.clients.siglip_embedder import SigLIPEmbedder
    siglip = SigLIPEmbedder(args.siglip_model, device=args.siglip_gpu)
    log.info("  SigLIP ready (%.1fs)", time.time() - t0)

    vlm = None
    if not use_api:
        log.info("Loading VLM %s on %s …", args.vlm_model, args.vlm_gpu)
        t0 = time.time()
        from src.clients.vlm_client import VLMClient
        vlm = VLMClient(model_path=args.vlm_model, device=args.vlm_gpu, max_new_tokens=64)
        log.info("  VLM ready (%.1fs)", time.time() - t0)
    else:
        log.info("API mode — VLM at %s (%s)", args.api_url, args.api_model)

    log.info("Loading BGE-M3 on %s …", args.bge_gpu)
    t0 = time.time()
    from src.clients.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(
        model_path=cfg["models"]["embedder"]["model_path"],
        use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
        device=args.bge_gpu,
    )
    log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    # ── Video list ─────────────────────────────────────────────────────────────
    video_paths = _get_video_paths(cfg)
    log.info("Found %d videos", len(video_paths))

    if args.filter_from:
        keep = set()
        for line in Path(args.filter_from).open():
            try:
                keep.add(json.loads(line)["video_id"])
            except Exception:
                pass
        video_paths = {k: v for k, v in video_paths.items() if k in keep}
        log.info("Filtered to %d videos", len(video_paths))

    from src.build_memory.core.frame_pipeline import clean_and_group_frames

    for video_id, video_path in sorted(video_paths.items()):
        video_dir = out_dir / video_id
        narr_path = video_dir / "narrative.json"
        vec_path  = video_dir / "vectors.npz"
        if narr_path.exists() and vec_path.exists():
            log.info("[%s] already built, skipping", video_id)
            continue
        if not Path(video_path).exists():
            log.warning("[%s] not found: %s", video_id, video_path)
            continue

        srt_entries: List[Tuple[float, float, str]] = []
        if subtitle_dir is not None:
            srt_path = subtitle_dir / f"{video_id}.srt"
            if srt_path.exists():
                try:
                    srt_entries = parse_srt(srt_path)
                    log.info("[%s] %d subtitle entries", video_id, len(srt_entries))
                except Exception as e:
                    log.warning("[%s] SRT parse failed: %s", video_id, e)
            else:
                log.warning("[%s] no SRT at %s", video_id, srt_path)

        log.info("[%s] starting …", video_id)
        t_video = time.time()

        # 1–5: frame pipeline (extract → filter → pre-dedup → SigLIP group →
        #      per-chunk dedup → top-2 sharpest)
        cleaned = clean_and_group_frames(
            video_path=video_path,
            out_dir=out_dir / video_id,
            siglip=siglip,
            fps=args.fps,
            theta=args.theta,
            n_max=args.n_max,
        )
        duration = max((g.t_end for g in cleaned), default=0.0)
        log.info("  %d chunks, %d kept frames total",
                 len(cleaned), sum(len(g.kept_frame_paths) for g in cleaned))

        # 6: subtitle alignment
        speech_texts = [
            align_subtitles(srt_entries, g.t_start, g.t_end) if srt_entries else ""
            for g in cleaned
        ]

        # 7: narrative generation
        # Narrative VLM sees the full post-cleanup frame set (g.all_frame_paths,
        # post blank+blur+pre-dedup, no per-chunk top-2 cap) down-sampled to at
        # most --narrative-max-frames.  g.kept_* (the top-2 dedup subset) is
        # only used for the visual layer (keyframe + v_visual).
        picked = [
            _select_narrative_frames(g.all_frame_paths, g.all_timestamps,
                                     args.narrative_max_frames)
            for g in cleaned
        ]
        chunks_info = [
            {
                "narrative_frame_paths": fps,
                "narrative_timestamps":  ts,
                "kept_frame_paths":      fps,    # alias for narrate_chunks_api
                "kept_timestamps":       ts,
                "speech":  sp,
                "t_start": g.t_start,
                "t_end":   g.t_end,
            }
            for g, sp, (fps, ts) in zip(cleaned, speech_texts, picked)
        ]

        t0 = time.time()
        if args.narrative_mode == "caption_summary":
            if not use_api:
                raise NotImplementedError("caption_summary mode requires --api-url")
            narratives = narrate_chunks_caption_summary(
                chunks_info, args.api_url, args.api_model,
                concurrency=args.api_concurrency,
            )
        elif use_api:
            narratives = narrate_chunks_api(
                chunks_info, args.api_url, args.api_model,
                concurrency=args.api_concurrency,
            )
        else:
            narratives = [
                narrate_chunk_local(
                    info["narrative_frame_paths"], info["narrative_timestamps"],
                    info["speech"], info["t_start"], info["t_end"], vlm,
                )
                for info in chunks_info
            ]
        log.info("  narratives done (%.1fs)", time.time() - t0)

        # 8: BGE-M3 encode (narrative + speech)
        embed_texts = [
            f"{narr}\n\n{sp}" if sp.strip() else narr
            for narr, sp in zip(narratives, speech_texts)
        ]
        sem_vecs = embedder.encode(embed_texts)

        # 9: write legacy directory bank (narrative.json + vectors.npz)
        narr_chunks = []
        n_vecs = []
        chunk_ids = []
        for g, narr, speech in zip(cleaned, narratives, speech_texts):
            center = g.center_idx
            kf_path = g.kept_frame_paths[center]
            try:
                kf_rel = str(kf_path.relative_to(video_dir))
            except ValueError:
                kf_rel = str(kf_path)
            narr_chunks.append({
                "chunk_id":      g.chunk_id,
                "start_time":    g.t_start,
                "end_time":      g.t_end,
                "narrative":     narr,
                "caption":       [],
                "speech_text":   speech,
                "sampled_frames": g.all_timestamps,
                "keyframe_ts":   g.kept_timestamps[center],
                "keyframe_path": kf_rel,
                "v_visual":      g.kept_v_visual[center].tolist(),
            })
            n_vecs.append(sem_vecs[g.chunk_id])
            chunk_ids.append(g.chunk_id)

        narr_json = {
            "video_id":   video_id,
            "duration":   duration,
            "num_chunks": len(narr_chunks),
            "chunks":     narr_chunks,
        }
        narr_path.write_text(json.dumps(narr_json, ensure_ascii=False))
        np.savez(
            vec_path,
            narrative_vecs=np.array(n_vecs, dtype=np.float32),
            chunk_ids=np.array(chunk_ids, dtype=np.int64),
        )
        log.info("[%s] saved %d chunks (%.1fs total)",
                 video_id, len(narr_chunks), time.time() - t_video)

    log.info("Done — output in %s", out_dir)


if __name__ == "__main__":
    main()
