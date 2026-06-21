"""Build single-granularity similarity-grouped memory banks.

Narration method: one multi-image VLM call per chunk — frames are interleaved
with timestamp labels, the overlapping subtitle lines are injected with their
time ranges, and the model writes a single chunk-level caption. This replaces
the earlier per-frame caption + LLM summary baseline (which lost frame-to-frame
continuity and laundered facts through a second LLM hop).

For the alternative structured "Memory: <paragraph>\\nKey details: <comma list>"
two-line variant, see ``build_single_similarity_newprompt.py``.

Pipeline per video:
  1. ffmpeg 1fps → JPEG frames
  2. Blank + blur filter (with failsafe)
  3. Pre-dedup (dHash, window=10) — removes near-identical consecutive frames
  4. SigLIP encode survivors → cosine-similarity grouping (θ≥0.8)
  5. Per-chunk dHash+SigLIP dedup → kept_frame_paths (visual representative subset)
     + top-2 sharpest cap on kept set → keyframe / v_visual
     all_frame_paths (post pre-dedup, no chunk-cap) is preserved for narrative VLM.
  6. SRT → speech_text + speech_lines per chunk (if subtitle_dir supplied)
  7. Narrative — single multi-image VLM call per chunk:
     - Down-sample g.all_frame_paths evenly to --narrative-max-frames (default 30)
     - One call: system prompt + segment window + transcript lines + interleaved
       (frame timestamp, image) pairs → single paragraph caption
  8. BGE-M3 encode (narrative + speech), max_length=4096
  9. Write legacy directory bank:
        {out_dir}/{video_id}/narrative.json       chunks[].{narrative, caption,
                                                  speech_text, sampled_frames,
                                                  keyframe_ts, keyframe_path, v_visual}
        {out_dir}/{video_id}/vectors.npz          narrative_vecs + chunk_ids
        {out_dir}/{video_id}/keyframes_origin/    dedup kept frames at original res
        {out_dir}/{video_id}/keyframes_resized/   same files resized to ≤ frame_max_dim
        {out_dir}/{video_id}/frames_raw/          all post-cleanup frames per chunk (original res)

Usage:
    cd /home2/ycj/Project/VEIL
    PYTHONPATH=. python -m src.build_memory.build_single_similarity \\
        --benchmark videomme \\
        --siglip-model /home2/ycj/Models/google/siglip-large-patch16-384 \\
        --siglip-gpu cuda:1 --bge-gpu cuda:1 \\
        --api-url http://10.82.1.145:8001,http://10.82.1.145:8002 \\
        --api-model Qwen3.5-27B
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger

log = get_logger("build_single_similarity")


# ── Prompts (Binlu-style segment captioning, one multi-image call per chunk) ──

SYSTEM_PROMPT = (
    "You are an expert video captioner.\n"
    "\n"
    "You will receive a short video segment represented by ordered frames, "
    "each labeled with its timestamp in the video, and optionally the "
    "transcript lines of the speech with their time ranges on the same "
    "timeline.\n"
    "Write a single paragraph caption describing the visual content and, when "
    "transcript lines are provided, the spoken content.\n"
    "\n"
    "Guidelines:\n"
    "- Describe visible people, actions, objects, and the setting.\n"
    "- Present actions and events in their temporal order; use the timestamps "
    "to align speech with what is shown.\n"
    "- When transcript lines are provided, weave what is said into the "
    "caption; do not describe sounds you cannot know from the frames or "
    "transcript.\n"
    "- Preserve names, numbers, and on-screen text (OCR) verbatim. Do not "
    "invent details.\n"
    "- Keep the caption factual and neutral.\n"
    "- Do not mention frames, transcripts, timestamps, or how the input was "
    "produced.\n"
    "- Avoid speculation about emotions or intentions unless clearly visible "
    "or stated in the transcript.\n"
    "\n"
    "Output only the final caption text."
)

USER_HEADER_TEMPLATE = (
    "Segment window: {t0} to {t1}\n"
    "{transcript_block}"
    "The following frames are ordered chronologically within the segment."
)
TRANSCRIPT_BLOCK_TEMPLATE = "Transcript lines:\n{lines}\n"
NO_TRANSCRIPT_LINE = "- No transcript lines overlap this segment."
FRAME_TS_TEMPLATE = "Frame timestamp: {ts}"
USER_TRAILER = "Write the caption for this segment."


def _hms(seconds: float) -> str:
    # Round (not truncate) — matches Binlu multiscale_segment.hms() so the
    # timestamps in the VLM prompt line up with where speech/actions actually
    # happen (truncation drifts by up to a second per chunk boundary).
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


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


def collect_speech_lines(
    entries: List[Tuple[float, float, str]],
    t_start: float,
    t_end: float,
) -> List[Tuple[float, float, str]]:
    """Return SRT entries overlapping [t_start, t_end], preserving timestamps."""
    return [(es, ee, txt) for es, ee, txt in entries if ee >= t_start and es <= t_end]


# ── Narrative-input frame down-sampler ────────────────────────────────────────

def _select_narrative_frames(
    paths: List[Path], timestamps: List[float], n_max: int,
) -> Tuple[List[Path], List[float]]:
    """Evenly down-sample (paths, timestamps) to at most ``n_max`` items.

    Bounds VLM input cost regardless of how many post-cleanup frames the
    chunk has. Preserves temporal order.
    """
    n = len(paths)
    if n <= n_max:
        return list(paths), list(timestamps)
    step = n / n_max
    idxs = [int(i * step) for i in range(n_max)]
    return [paths[i] for i in idxs], [timestamps[i] for i in idxs]


# ── API endpoint round-robin ──────────────────────────────────────────────────

_endpoint_state: dict = {"endpoints": [], "rr": 0, "lock": None}


def _init_endpoints(api_url: str) -> None:
    _endpoint_state["endpoints"] = [
        u.strip().rstrip("/") for u in api_url.split(",") if u.strip()
    ]
    _endpoint_state["rr"] = 0
    _endpoint_state["lock"] = asyncio.Lock()


async def _pick_endpoint() -> str:
    async with _endpoint_state["lock"]:
        eps = _endpoint_state["endpoints"]
        ep = eps[_endpoint_state["rr"] % len(eps)]
        _endpoint_state["rr"] += 1
    return ep


async def _call_api(
    session, messages, api_model: str, max_tokens: int = 150,
    max_retries: int = 5, retry_base_delay: float = 2.0,
) -> str:
    """Exponential-backoff retry on connection drops / 5xx (matches Binlu
    api_backend.py:79-101). Only retries network-level failures; 4xx and JSON
    parse errors still propagate."""
    import aiohttp
    payload_base = {
        "model": api_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        ep = await _pick_endpoint()
        try:
            async with session.post(
                f"{ep}/v1/chat/completions", json=payload_base,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as r:
                data = await r.json()
            text = data["choices"][0]["message"]["content"]
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()
        except (aiohttp.ClientConnectionError,
                aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            await asyncio.sleep(delay)
    raise last_err  # unreachable


def _img_block(path: Path, max_side: int = 448) -> dict:
    """Long-edge resize → JPEG q=85 → base64 (mirrors Binlu api_backend.py:104-119).
    Matters most for the multi-image narrative call: raw 1fps frames are 1280+
    on the long edge; sending 30 of them un-scaled triples the prompt-token
    budget and bloats attention noise, dragging caption quality down."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


# ── Narrative (single multi-image VLM call per chunk) ─────────────────────────

def _format_transcript_block(
    speech_lines: List[Tuple[float, float, str]],
    video_has_speech: bool,
) -> str:
    """Three-way subtitle handling matching Binlu segment_caption.py:155-165:
    - video has no SRT at all → block omitted entirely
    - video has SRT but this chunk has no overlap → explicit NO_TRANSCRIPT marker
    - chunk has overlapping lines → list with time ranges
    """
    if not video_has_speech:
        return ""
    if speech_lines:
        lines = "\n".join(
            f"- [{_hms(es)} --> {_hms(ee)}] {txt}"
            for es, ee, txt in speech_lines
        )
    else:
        lines = NO_TRANSCRIPT_LINE
    return TRANSCRIPT_BLOCK_TEMPLATE.format(lines=lines)


async def _narrate_chunk(
    session, sem, info: dict, api_model: str, video_has_speech: bool,
) -> str:
    fps = info["narrative_frame_paths"]
    ts = info["narrative_timestamps"]
    speech_lines = info.get("speech_lines", [])
    t_start = info["t_start"]
    t_end = info["t_end"]

    if not fps:
        return ""

    transcript_block = _format_transcript_block(speech_lines, video_has_speech)
    header = USER_HEADER_TEMPLATE.format(
        t0=_hms(t_start), t1=_hms(t_end),
        transcript_block=transcript_block,
    )
    content: list = [{"type": "text", "text": header}]
    for tt, fp in zip(ts, fps):
        content.append({"type": "text", "text": FRAME_TS_TEMPLATE.format(ts=_hms(tt))})
        content.append(_img_block(fp))
    content.append({"type": "text", "text": USER_TRAILER})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    async with sem:
        try:
            return await _call_api(
                session, messages, api_model=api_model, max_tokens=400,
            )
        except Exception as e:
            return f"[narrative failed: {e}]"


async def _narrate_all(chunks_info: List[dict], api_model: str, concurrency: int) -> List[str]:
    import aiohttp
    # Per-video flag: if any chunk has SRT overlap, we treat the video as having
    # speech; chunks without overlap then get the NO_TRANSCRIPT marker rather
    # than dropping the transcript block silently.
    video_has_speech = any(
        c.get("speech_lines") or c.get("speech", "").strip()
        for c in chunks_info
    )
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _narrate_chunk(session, sem, c, api_model, video_has_speech)
            for c in chunks_info
        ]
        return list(await asyncio.gather(*tasks))


def narrate_chunks(
    chunks_info: List[dict], api_url: str, api_model: str, concurrency: int = 16,
) -> List[str]:
    """One multi-image VLM call per chunk → single paragraph caption."""
    _init_endpoints(api_url)
    return asyncio.run(_narrate_all(chunks_info, api_model, concurrency))


# ── Video paths ───────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benchmark", choices=["mlvu", "videomme"])
    src.add_argument("--config", help="Legacy YAML path")
    ap.add_argument("--siglip-model",     required=True)
    ap.add_argument("--siglip-gpu",       default="cuda:0")
    ap.add_argument("--bge-gpu",          default="cuda:0")
    ap.add_argument("--subtitle-dir",     default=None)
    ap.add_argument("--filter-from",      default=None)
    ap.add_argument("--fps",              type=float, default=1.0)
    ap.add_argument("--theta",            type=float, default=0.80)
    ap.add_argument("--n-max",            type=int,   default=30)
    ap.add_argument("--api-url",          required=True,
                    help="vLLM OpenAI-compatible base URL(s); comma-separated"
                         " = round-robin")
    ap.add_argument("--api-model",        required=True)
    ap.add_argument("--api-concurrency",  type=int, default=16)
    ap.add_argument("--out-dir",          default=None)
    ap.add_argument("--narrative-max-frames", type=int, default=30,
                    help="Cap frames per chunk fed to the single multi-image VLM"
                         " call; evenly down-sampled from g.all_frame_paths when"
                         " exceeded")
    ap.add_argument("--frame-max-dim", type=int, default=448,
                    help="Longer-side target for keyframes_resized/. Origin"
                         " keyframes and frames_raw are always full res."
                         " Default 448 matches Binlu save_keyframe (build_index.py).")
    args = ap.parse_args()

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

    log.info("API mode — VLM/LLM at %s (%s)", args.api_url, args.api_model)

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
    from queue import Queue
    from threading import Thread

    # ── Pipeline: prepare next video on a worker thread while main runs VLM ──
    # The prepare stage is ffmpeg-extract + SigLIP-encode/group + SRT parse,
    # which runs on CPU + GPU 1 (SigLIP). The build stage is the VLM caption
    # (vLLM 8000+8001 on GPU 1/3) + BGE encode + disk write. By the time the
    # main thread finishes captioning video N (~150 s), the worker has already
    # finished preparing video N+1 (~40-70 s), so the next caption can start
    # immediately. Queue size 1 caps the worker at one video ahead of main.

    def _prepare_video(video_id: str, video_path: str):
        """Pre-caption stage. Returns dict for build, or None to skip."""
        video_dir = out_dir / video_id
        narr_path = video_dir / "narrative.json"
        vec_path  = video_dir / "vectors.npz"
        if narr_path.exists() and vec_path.exists():
            log.info("[%s] already built, skipping", video_id)
            return None
        if not Path(video_path).exists():
            log.warning("[%s] not found: %s", video_id, video_path)
            return None

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

        cleaned = clean_and_group_frames(
            video_path=video_path,
            out_dir=video_dir,
            siglip=siglip,
            fps=args.fps,
            theta=args.theta,
            n_max=args.n_max,
            frame_max_dim=args.frame_max_dim,
        )
        duration = max((g.t_end for g in cleaned), default=0.0)
        log.info("  [%s] %d chunks, %d kept frames total",
                 video_id, len(cleaned),
                 sum(len(g.kept_frame_paths) for g in cleaned))

        speech_texts = [
            align_subtitles(srt_entries, g.t_start, g.t_end) if srt_entries else ""
            for g in cleaned
        ]
        speech_lines_per_chunk = [
            collect_speech_lines(srt_entries, g.t_start, g.t_end) if srt_entries else []
            for g in cleaned
        ]
        picked = [
            _select_narrative_frames(g.all_frame_paths, g.all_timestamps,
                                     args.narrative_max_frames)
            for g in cleaned
        ]
        chunks_info = [
            {
                "narrative_frame_paths": fps,
                "narrative_timestamps":  ts,
                "speech":        sp,
                "speech_lines":  sl,
                "t_start": g.t_start,
                "t_end":   g.t_end,
            }
            for g, sp, sl, (fps, ts) in zip(
                cleaned, speech_texts, speech_lines_per_chunk, picked,
            )
        ]
        return {
            "video_id":     video_id,
            "video_dir":    video_dir,
            "narr_path":    narr_path,
            "vec_path":     vec_path,
            "cleaned":      cleaned,
            "duration":     duration,
            "speech_texts": speech_texts,
            "chunks_info":  chunks_info,
            "t_video":      t_video,
        }

    # Iterate videos in the order they came in (videomme loader returns samples
    # by sample_idx, and dict preserves first-seen insertion order). For
    # --filter-from, the filtered dict also preserves the same order.
    SENTINEL: object = object()
    prep_queue: Queue = Queue(maxsize=1)

    def _producer():
        try:
            for video_id, video_path in video_paths.items():
                try:
                    item = _prepare_video(video_id, video_path)
                except Exception as e:
                    log.exception("[%s] prepare failed: %s", video_id, e)
                    item = None
                prep_queue.put(item)
        finally:
            prep_queue.put(SENTINEL)

    prod_thread = Thread(target=_producer, name="prepare", daemon=True)
    prod_thread.start()

    while True:
        item = prep_queue.get()
        if item is SENTINEL:
            break
        if item is None:
            continue
        video_id     = item["video_id"]
        video_dir    = item["video_dir"]
        narr_path    = item["narr_path"]
        vec_path     = item["vec_path"]
        cleaned      = item["cleaned"]
        duration     = item["duration"]
        speech_texts = item["speech_texts"]
        chunks_info  = item["chunks_info"]
        t_video      = item["t_video"]

        t0 = time.time()
        narratives = narrate_chunks(
            chunks_info, args.api_url, args.api_model,
            concurrency=args.api_concurrency,
        )
        log.info("  [%s] narratives done (%.1fs)", video_id, time.time() - t0)

        embed_texts = [
            f"{narr}\n\n{sp}" if sp.strip() else narr
            for narr, sp in zip(narratives, speech_texts)
        ]
        sem_vecs = embedder.encode(embed_texts)

        # v_visual lives in vectors.npz["visual_vecs"] (one row per chunk in the
        # same order as narrative_vecs / chunk_ids), NOT inside narrative.json.
        narr_chunks = []
        n_vecs = []
        v_vecs = []
        chunk_ids = []
        for g, narr, speech in zip(cleaned, narratives, speech_texts):
            center = g.center_idx
            kf_path = g.kept_frame_paths[center]
            try:
                kf_rel = str(kf_path.relative_to(video_dir))
            except ValueError:
                kf_rel = str(kf_path)
            narr_chunks.append({
                "chunk_id":       g.chunk_id,
                "start_time":     g.t_start,
                "end_time":       g.t_end,
                "narrative":      narr,
                "caption":        [],
                "speech_text":    speech,
                "sampled_frames": g.all_timestamps,
                "keyframe_ts":    g.kept_timestamps[center],
                "keyframe_path":  kf_rel,
            })
            n_vecs.append(sem_vecs[g.chunk_id])
            v_vecs.append(g.kept_v_visual[center])
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
            visual_vecs=np.array(v_vecs, dtype=np.float32),
            chunk_ids=np.array(chunk_ids, dtype=np.int64),
        )
        log.info("[%s] saved %d chunks (%.1fs total)",
                 video_id, len(narr_chunks), time.time() - t_video)

    prod_thread.join()
    log.info("Done — output in %s", out_dir)


if __name__ == "__main__":
    main()
