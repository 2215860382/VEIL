"""Build multi-layer memory banks (episodic captions + semantic summaries + visual SigLIP): SigLIP grouping + VLM/API captions + summaries + BGE-M3.

Pipeline per video:
  1. ffmpeg 1fps → JPEG frames
  2. SigLIP  → frame visual vectors
  3. Cosine-similarity grouping  (θ≥0.8, n_max=30)
  4. VLM/API → per-frame caption (≤20 words; empty API reply → “[frame at t=…s]”; --caption-stride subsamples)
  5. SRT → whenever {video_id}.srt exists under subtitle_dir, always parse + interval-overlap → speech_text per group (no opt-out)
  6. VLM/API → group summary (frame lines + “Spoken words:” whenever that speech_text is non-empty; >15 caps subsampled)
  7. BGE-M3       → semantic embedding on summary + aligned speech_text whenever non-empty (concat for the vector)
  8. Save MemoryBank  (memory_text = summary, memory_kind = similarity_group)

Fixed wall-clock chunking (no SigLIP grouping): ``python -m src.memory.fixedframe --benchmark mlvu``.

Output dir: ``--out-dir``, else ``memory.cache_dir`` from ``--benchmark`` (via ``src/memory/core/specs.py``) or legacy ``--config`` YAML.

Usage:
    cd /home2/ycj/Project/VEIL
    # Pass explicit checkpoints (no baked-in defaults). Dataset + BGE from memory/specs or --config.
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.memory.similarity \\
        --benchmark mlvu \\
        --vlm-model /path/to/Qwen3-VL \\
        --siglip-model /path/to/siglip \\
        --vlm-gpu cuda:0 --bge-gpu cuda:0 --siglip-gpu cuda:0

    # API captions/summaries: --vlm-model omitted only in single-pass when both use API (--api-url + --api-model).
    # Legacy YAML:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.memory.similarity \\
        --config configs/mlvu_memory_bank.yaml \\
        --vlm-model /path/to/Qwen3-VL --siglip-model /path/to/siglip \\
        --vlm-gpu cuda:0 --bge-gpu cuda:0 --siglip-gpu cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger

log = get_logger("build_similarity_memory")

# SigLIP cosine-similarity grouping (dynamic segments), not fixed frame windows.
MEMORY_KIND = "similarity_group"

FRAME_CAPTION_PROMPT = (
    "Describe this video frame in ≤50 words. Be specific and factual. Cover ALL that apply:\n"
    "- SUBJECTS: name every person, animal, or object precisely "
    "(e.g. 'golden retriever', 'red Tesla Model S', not 'dog' or 'car'); "
    "give exact counts when multiple (e.g. '3 penguins', '~20 spectators').\n"
    "- PRESENTER: if a person is speaking directly to camera as a host, narrator, or presenter, "
    "explicitly identify them — use their name if visible or stated (e.g. 'host John'), "
    "otherwise a consistent descriptor (e.g. 'the male presenter', 'the female host'). "
    "Never write just 'a man' or 'a person' for a direct-to-camera speaker.\n"
    "- ACTION: describe what each subject is doing with specific verbs and manner.\n"
    "- ATTRIBUTES: note key colors, clothing, materials, or states relevant to the scene.\n"
    "- SPATIAL: mention positions if relevant (e.g. 'left foreground', 'behind the counter').\n"
    "- ON-SCREEN TEXT: if ANY text, number, score, label, sign, or subtitle is visible, "
    "transcribe it EXACTLY (e.g. '23–17', 'EXIT 42', 'Step 3: Add salt').\n"
    "Example: 'The male presenter speaks to camera explaining Japan's economy; "
    "scoreboard reads \"HOME 2 – AWAY 1\"; stadium crowd in background.'"
)

DYNAMIC_SYS = (
    "You are a video event analyst. Analyze this video segment and return a JSON object.\n"
    "Priority rules:\n"
    "1. OPEN WITH NUMBERS/CHANGES: if any numbers, scores, counts, or state changes exist, "
    "state them first. Use delta notation 'from X to Y' for changes.\n"
    "2. SPEAKER FIRST: if a host/narrator is speaking, make them the subject.\n"
    "3. STATE the main topic/activity/event. PRESERVE all specifics: numbers, names, scores, text.\n"
    "4. SEQUENCE events with temporal markers (first/then/after/finally) if the frames show process/steps.\n"
    "5. INCLUDE causal language (because, in order to, which causes, as a result).\n"
    "6. INCLUDE key attributes (color, appearance, location) that distinguish subjects.\n"
    "Return JSON with these keys:\n"
    "{\n"
    '  "summary": "2-4 sentence narrative with numbers/changes in sentence 1",\n'
    '  "key_events": ["ordered", "list", "of", "key", "events"],\n'
    '  "actors": ["named persons or consistent descriptors"],\n'
    '  "state_changes": ["observable changes: \'X changed from A to B\'", "empty list if none"],\n'
    '  "temporal_relations": ["sequence markers: \'first...\', \'then...\', \'finally...\'"],\n'
    '  "causal_clues": ["causal statements: \'because X, Y happened\'"]\n'
    "}\n"
    "Be factual. Do not infer beyond frame descriptions."
)

STATIC_ATTR_SYS = (
    "You are a visual attribute extractor. Analyze this video frame and return a JSON object.\n"
    "Extract ONLY static visible attributes — do NOT describe actions or events.\n"
    "Return JSON with these keys:\n"
    "{\n"
    '  "ocr_text": ["exact on-screen text/numbers/scores — transcribe verbatim"],\n'
    '  "numbers": ["every visible number"],\n'
    '  "colors": ["dominant colors of main objects/persons"],\n'
    '  "objects": ["specific object names with precise names"],\n'
    '  "object_attributes": [{"object": "name", "attributes": ["color", "shape", "material", "texture"]}],\n'
    '  "people_appearance": ["physical description of each person"],\n'
    '  "clothing": ["clothing items with colors/styles"],\n'
    '  "spatial_layout": ["relative positions: \'person on left\', \'text in top-right\'"],\n'
    '  "textures": ["notable textures or materials"],\n'
    '  "scene_attributes": ["location type, lighting, background details"]\n'
    "}\n"
    "If a category has no items, use an empty list."
)

SUMMARY_SYS = DYNAMIC_SYS  # For backward compatibility


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
                txt = " ".join(text_lines).strip()
                txt = _HTML_TAG_RE.sub("", txt).strip()
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
                txt = " ".join(text_lines).strip()
                txt = _HTML_TAG_RE.sub("", txt).strip()
                if txt:
                    entries.append((t_start, t_end, txt))
                text_lines = []
                in_block = False

    if in_block and text_lines:
        txt = " ".join(text_lines).strip()
        txt = _HTML_TAG_RE.sub("", txt).strip()
        if txt:
            entries.append((t_start, t_end, txt))
    return entries


def _subtitle_bank_fields(
    subtitle_dir: Optional[Path],
    video_id: str,
    srt_entries: List[Tuple[float, float, str]],
) -> dict:
    """MemoryBank JSON fields: caption/summary models are separate; this marks subtitle availability."""
    if subtitle_dir is None:
        return dict(subtitle_dir=None, subtitle_file_present=None, subtitle_cue_count=None)
    p = subtitle_dir / f"{video_id}.srt"
    return dict(
        subtitle_dir=str(subtitle_dir.resolve()),
        subtitle_file_present=p.exists(),
        subtitle_cue_count=len(srt_entries),
    )


def align_subtitles(
    entries: List[Tuple[float, float, str]],
    t_start: float,
    t_end: float,
    max_chars: int = 500,
) -> str:
    """Return subtitle text overlapping with [t_start, t_end]."""
    parts = []
    for es, ee, txt in entries:
        if ee < t_start:
            continue
        if es > t_end:
            break
        parts.append(txt)
    speech = " ".join(parts)
    if len(speech) > max_chars:
        speech = speech[:max_chars].rsplit(" ", 1)[0] + "…"
    return speech


def _empty_caption_placeholder(timestamp_sec: float) -> str:
    """Quality gate: replace empty VLM/API captions (spec: ≤20 words line still applies after trim)."""
    return f"[frame at t={timestamp_sec}s]"


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: Path, fps: float = 1.0) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "3",
        str(out_dir / "frame_%06d.jpg"),
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


# ── Per-frame caption (local VLM) ─────────────────────────────────────────────

def caption_frames(
    frame_paths: List[Path],
    vlm,
    stride: int = 2,
    timestamps: Optional[List[float]] = None,
) -> List[str]:
    """Return one caption per frame; skipped indices stay \"\". Empty model output → placeholder."""
    from PIL import Image
    captions: List[str] = [""] * len(frame_paths)
    for i in range(0, len(frame_paths), stride):
        try:
            img = Image.open(str(frame_paths[i])).convert("RGB")
            cap = vlm.chat_with_frames([img], FRAME_CAPTION_PROMPT,
                                       max_new_tokens=40).strip()
            captions[i] = " ".join(cap.split()[:25])
        except Exception as e:
            log.warning("  caption frame %d: %s", i, e)
        if not captions[i].strip():
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else float(i)
            captions[i] = _empty_caption_placeholder(ts)
    return captions


# ── Per-frame caption (vLLM API, async) ───────────────────────────────────────

def caption_frames_api(
    frame_paths: List[Path],
    api_url: str,
    api_model: str,
    stride: int = 2,
    concurrency: int = 16,
    timestamps: Optional[List[float]] = None,
) -> List[str]:
    """Caption frames via OpenAI-compatible vLLM API with async concurrency."""
    import asyncio, base64
    import aiohttp

    async def _caption_one(session, idx: int, path: Path) -> tuple[int, str]:
        b64 = base64.b64encode(path.read_bytes()).decode()
        payload = {
            "model": api_model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": FRAME_CAPTION_PROMPT},
            ]}],
            "max_tokens": 60,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            async with session.post(f"{api_url}/v1/chat/completions",
                                    json=payload, timeout=aiohttp.ClientTimeout(total=60)) as r:
                data = await r.json()
                text = data["choices"][0]["message"]["content"].strip()
                return idx, " ".join(text.split()[:25])
        except Exception as e:
            log.warning("  API caption frame %d: %s", idx, e)
            return idx, ""

    async def _run_all():
        indices = list(range(0, len(frame_paths), stride))
        captions: List[str] = [""] * len(frame_paths)
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(idx):
            async with sem:
                return await _caption_one(session, idx, frame_paths[idx])

        connector = aiohttp.TCPConnector(limit=concurrency + 4)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.gather(*[_bounded(i) for i in indices])
        for idx, cap in results:
            if cap.strip():
                captions[idx] = cap
            else:
                ts = timestamps[idx] if timestamps is not None and idx < len(timestamps) else float(idx)
                captions[idx] = _empty_caption_placeholder(ts)
        return captions

    return asyncio.run(_run_all())


# ── Group summary (local VLM, text-only) ──────────────────────────────────────

def summarize_group(
    frame_captions: List[str],
    vlm,
    speech_text: str = "",
    max_caps: int = 15,
) -> dict:
    """Generate structured narrative (summary, key_events, actors, state_changes, etc.) from frame captions.
    Returns dict with keys: summary, key_events, actors, state_changes, temporal_relations, causal_clues."""
    caps = [c for c in frame_captions if c.strip()]
    if not caps and not speech_text.strip():
        return {
            "summary": "",
            "key_events": [],
            "actors": [],
            "state_changes": [],
            "temporal_relations": [],
            "causal_clues": [],
        }
    if len(caps) > max_caps:
        step = len(caps) / max_caps
        caps = [caps[int(i * step)] for i in range(max_caps)]

    desc = "\n".join(f"- {c}" for c in caps) if caps else "(no frame captions)"
    user_msg = f"Frame descriptions:\n{desc}\n"
    if speech_text.strip():
        user_msg += f"\nSpoken words:\n{speech_text.strip()}\n"

    messages = [
        {"role": "system", "content": DYNAMIC_SYS},
        {"role": "user",   "content": user_msg},
    ]
    try:
        raw = vlm._generate(messages, max_new_tokens=256).strip()
        # Extract JSON from response
        import json as _json
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            d = _json.loads(m.group())
            return {
                "summary": str(d.get("summary", "")),
                "key_events": d.get("key_events", []) if isinstance(d.get("key_events"), list) else [],
                "actors": d.get("actors", []) if isinstance(d.get("actors"), list) else [],
                "state_changes": d.get("state_changes", []) if isinstance(d.get("state_changes"), list) else [],
                "temporal_relations": d.get("temporal_relations", []) if isinstance(d.get("temporal_relations"), list) else [],
                "causal_clues": d.get("causal_clues", []) if isinstance(d.get("causal_clues"), list) else [],
            }
    except Exception as e:
        log.warning("  summary error: %s", e)

    # Fallback to simple summary
    return {
        "summary": caps[0] if caps else speech_text[:200],
        "key_events": [],
        "actors": [],
        "state_changes": [],
        "temporal_relations": [],
        "causal_clues": [],
    }


# ── Static Attribute Extraction (local VLM) ───────────────────────────────────

def extract_static_attributes(
    frame_path: str,
    frame_id: str,
    timestamp: float,
    vlm,
) -> Optional[dict]:
    """Extract static visual attributes from a single frame. Returns dict for StaticAttributeFrame."""
    try:
        from PIL import Image
        img = Image.open(frame_path).convert("RGB")

        raw = vlm.chat_with_frames([img], STATIC_ATTR_SYS, max_new_tokens=512).strip()

        import json as _json
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            d = _json.loads(m.group())
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": frame_path,
                "ocr_text": d.get("ocr_text", []) if isinstance(d.get("ocr_text"), list) else [],
                "numbers": d.get("numbers", []) if isinstance(d.get("numbers"), list) else [],
                "colors": d.get("colors", []) if isinstance(d.get("colors"), list) else [],
                "objects": d.get("objects", []) if isinstance(d.get("objects"), list) else [],
                "object_attributes": d.get("object_attributes", []) if isinstance(d.get("object_attributes"), list) else [],
                "people_appearance": d.get("people_appearance", []) if isinstance(d.get("people_appearance"), list) else [],
                "clothing": d.get("clothing", []) if isinstance(d.get("clothing"), list) else [],
                "spatial_layout": d.get("spatial_layout", []) if isinstance(d.get("spatial_layout"), list) else [],
                "textures": d.get("textures", []) if isinstance(d.get("textures"), list) else [],
                "scene_attributes": d.get("scene_attributes", []) if isinstance(d.get("scene_attributes"), list) else [],
            }
    except Exception as e:
        log.warning("  static attr error: %s", e)
        return None


def build_static_index_text(static_frame: dict) -> str:
    """Build searchable text from static attributes for BGE encoding."""
    parts = []
    if static_frame.get("ocr_text"):
        parts.append(f"OCR: {' '.join(static_frame['ocr_text'])}")
    if static_frame.get("numbers"):
        parts.append(f"Numbers: {' '.join(static_frame['numbers'])}")
    if static_frame.get("colors"):
        parts.append(f"Colors: {' '.join(static_frame['colors'])}")
    if static_frame.get("objects"):
        parts.append(f"Objects: {' '.join(static_frame['objects'])}")
    if static_frame.get("people_appearance"):
        parts.append(f"People: {' '.join(static_frame['people_appearance'])}")
    if static_frame.get("clothing"):
        parts.append(f"Clothing: {' '.join(static_frame['clothing'])}")
    if static_frame.get("spatial_layout"):
        parts.append(f"Layout: {' '.join(static_frame['spatial_layout'])}")
    if static_frame.get("textures"):
        parts.append(f"Textures: {' '.join(static_frame['textures'])}")
    if static_frame.get("scene_attributes"):
        parts.append(f"Scene: {' '.join(static_frame['scene_attributes'])}")
    return " | ".join(parts) if parts else ""


# ── Group summary (vLLM API) ───────────────────────────────────────────────────

def _build_summary_payload(
    frame_captions: List[str],
    api_model: str,
    speech_text: str = "",
    max_caps: int = 15,
) -> dict:
    caps = [c for c in frame_captions if c.strip()]
    if len(caps) > max_caps:
        step = len(caps) / max_caps
        caps = [caps[int(i * step)] for i in range(max_caps)]
    desc = "\n".join(f"- {c}" for c in caps) if caps else "(no frame captions)"
    user_msg = f"Frame descriptions:\n{desc}\n"
    if speech_text.strip():
        user_msg += f"\nSpoken words:\n{speech_text.strip()}\n"
    return {
        "model": api_model,
        "messages": [
            {"role": "system", "content": DYNAMIC_SYS},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 256,
        "temperature": 0,
        "repetition_penalty": 1.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def summarize_groups_api(
    groups_captions: List[List[str]],
    groups_speech: List[str],
    api_url: str,
    api_model: str,
    concurrency: int = 16,
) -> List[dict]:
    """Async-batch summarize all groups for one video. Returns list of dicts with 'summary' key."""
    import asyncio
    import aiohttp

    async def _one(session, sem, idx, caps, speech):
        payload = _build_summary_payload(caps, api_model, speech_text=speech)
        async with sem:
            try:
                async with session.post(
                    f"{api_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as r:
                    data = await r.json()
                    summary_text = data["choices"][0]["message"]["content"].strip()
                    return idx, {
                        "summary": summary_text,
                        "key_events": [],
                        "actors": [],
                        "state_changes": [],
                        "temporal_relations": [],
                        "causal_clues": [],
                    }
            except Exception as e:
                log.warning("  API summary group %d: %s", idx, e)
                fallback = " ".join(c for c in caps if c)[:200] or speech[:200]
                return idx, {
                    "summary": fallback,
                    "key_events": [],
                    "actors": [],
                    "state_changes": [],
                    "temporal_relations": [],
                    "causal_clues": [],
                }

    async def _run():
        sem = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=concurrency + 4)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_one(session, sem, i, caps, sp)
                     for i, (caps, sp) in enumerate(zip(groups_captions, groups_speech))]
            results = await asyncio.gather(*tasks)
        out = [{}] * len(groups_captions)
        for idx, summary_dict in results:
            out[idx] = summary_dict
        return out

    return asyncio.run(_run())


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

def _select_center(frame_paths, g):
    """Blur-check group center frame; fallback to sharpest."""
    import cv2
    from src.memory.core.dynamic_grouper import select_sharpest
    center = g.center_idx
    try:
        img = cv2.imread(str(frame_paths[center]), cv2.IMREAD_GRAYSCALE)
        if img is not None and cv2.Laplacian(img, cv2.CV_64F).var() < 100:
            center = select_sharpest([str(p) for p in frame_paths], g.frame_indices)
    except Exception:
        pass
    return center


def _save_keyframe(frame_paths, center, kf_path):
    from PIL import Image as PILImage
    try:
        img = PILImage.open(str(frame_paths[center])).convert("RGB")
        img = img.resize((448, 448), PILImage.LANCZOS)
        img.save(str(kf_path), "JPEG", quality=85)
        return str(kf_path)
    except Exception as e:
        log.warning("  keyframe save failed: %s", e)
        return str(frame_paths[center])


def _memory_bank_provenance(args: argparse.Namespace, cfg: dict, *, two_pass: bool) -> dict:
    """Fields persisted on MemoryBank for caption/summary/SigLIP/BGE provenance."""
    bge_m = None
    emb = cfg.get("models", {}).get("embedder")
    if isinstance(emb, dict):
        bge_m = emb.get("model_path")

    if two_pass:
        if args.api_url:
            prov = dict(
                vlm_caption_backend="api",
                vlm_summary_backend="api",
                vlm_caption_model=args.api_model,
                vlm_summary_model=args.api_model,
                vlm_api_base_url=args.api_url,
            )
        else:
            prov = dict(
                vlm_caption_backend="local",
                vlm_summary_backend="local",
                vlm_caption_model=args.caption_model,
                vlm_summary_model=args.vlm_model,
                vlm_api_base_url=None,
            )
    elif args.api_url:
        prov = dict(
            vlm_caption_backend="api",
            vlm_summary_backend="api",
            vlm_caption_model=args.api_model,
            vlm_summary_model=args.api_model,
            vlm_api_base_url=args.api_url,
        )
    else:
        prov = dict(
            vlm_caption_backend="local",
            vlm_summary_backend="local",
            vlm_caption_model=args.vlm_model,
            vlm_summary_model=args.vlm_model,
            vlm_api_base_url=None,
        )
    prov["siglip_model"] = args.siglip_model
    prov["bge_model"] = bge_m
    return prov


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--benchmark",
        choices=["mlvu", "videomme"],
        help="Build spec from src/memory/core/specs.py (recommended; eval YAMLs live under configs/)",
    )
    src.add_argument(
        "--config",
        help="Legacy: path to eval YAML (uses benchmark + embedder + memory.cache_dir from file)",
    )
    ap.add_argument("--vlm-gpu",         default="cuda:0")
    ap.add_argument("--siglip-gpu",      default="cuda:0")
    ap.add_argument("--bge-gpu",         default="cuda:0")
    ap.add_argument("--vlm-model",       default=None,
                    help="Local VLM checkpoint for summaries (and captions unless --api-url). "
                         "Required for two-pass, and for single-pass without --api-url.")
    ap.add_argument("--caption-model",   default=None,
                    help="Smaller VLM for per-frame captions (two-pass: must differ from --vlm-model). "
                         "If omitted, single-pass uses --vlm-model for both.")
    ap.add_argument("--siglip-model",      required=True,
                    help="SigLIP checkpoint directory (no default; caller supplies path).")
    ap.add_argument("--subtitle-dir",    default=None,
                    help="Directory with per-video .srt files (named {video_id}.srt). "
                         "Any file that exists here is always loaded and aligned into groups (summary + BGE + chunk ASR). "
                         "If omitted, uses benchmark.subtitle_dir from the YAML when set "
                         "(videomme_memory_bank.yaml / benchmark.subtitle_dir). Missing files yield empty speech for that video only.")
    ap.add_argument("--filter-from",     default=None)
    ap.add_argument("--fps",             type=float, default=1.0)
    ap.add_argument("--theta",           type=float, default=0.80)
    ap.add_argument("--n-max",           type=int,   default=30)
    ap.add_argument("--caption-stride",  type=int,   default=1,
                    help="Caption every N-th sampled frame (1 = all frames). Default 1.")
    ap.add_argument("--api-url",         default=None,
                    help="vLLM OpenAI-compatible base URL (e.g. http://localhost:8082). "
                         "If set, captions and summaries are sent to this API instead of local VLM.")
    ap.add_argument("--api-model",       default=None,
                    help="Required with --api-url: served model name for vLLM OpenAI-compatible API.")
    ap.add_argument("--api-concurrency", type=int, default=16,
                    help="Max parallel caption requests to vLLM API")
    ap.add_argument("--out-dir",         default=None,
                    help="Output directory (default: memory.cache_dir from --benchmark/--config)")
    ap.add_argument("--caps-dir",        default=None,
                    help="Override caption cache directory (default: sibling …/{out_dir.name}_caps)")
    args = ap.parse_args()

    if args.api_url and not args.api_model:
        ap.error("--api-model is required when --api-url is set.")

    two_pass_early = bool(args.caption_model and args.caption_model != args.vlm_model)
    use_api = bool(args.api_url)
    # Local VLM weights are loaded whenever not pure single-pass API, or always in two-pass (current code paths).
    need_vlm_weights = two_pass_early or not use_api
    if need_vlm_weights and not args.vlm_model:
        ap.error("--vlm-model is required for local VLM inference (two-pass, or single-pass without --api-url).")

    if args.benchmark is not None:
        from memory import specs as build_specs
        cfg = build_specs.cfg_for_similarity_build(args.benchmark)
    else:
        cfg = load_config(args.config)
        # Eval YAMLs no longer carry memory.cache_dir; fill default so legacy --config scripts keep working.
        bn = cfg.get("benchmark", {}).get("name")
        if bn in ("mlvu", "videomme") and not (cfg.get("memory") or {}).get("cache_dir"):
            from memory import specs as build_specs
            cfg.setdefault("memory", {})["cache_dir"] = str(
                build_specs.similarity_memory_cache_dir(bn)
            )
    bench    = cfg["benchmark"]["name"]
    mem_cfg  = cfg.get("memory") or {}
    cache_dir_cfg = mem_cfg.get("cache_dir")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif cache_dir_cfg:
        out_dir = Path(cache_dir_cfg)
    else:
        ap.error("Pass --out-dir, or use --benchmark / YAML with memory.cache_dir set.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Memory bank output dir: %s", out_dir.resolve())

    if args.subtitle_dir:
        subtitle_dir = Path(args.subtitle_dir)
    else:
        sub_yaml = cfg.get("benchmark", {}).get("subtitle_dir")
        subtitle_dir = Path(sub_yaml) if sub_yaml else None
    if subtitle_dir is not None:
        log.info("Subtitle dir: %s", subtitle_dir)
    two_pass = bool(args.caption_model and args.caption_model != args.vlm_model)
    if args.caps_dir:
        caps_dir = Path(args.caps_dir)
    else:
        caps_dir = out_dir.parent / f"{out_dir.name}_caps"
    log.info("Caption cache dir (two-pass): %s", caps_dir.resolve())

    # ── SigLIP (used in both modes) ────────────────────────────────────────────
    log.info("Loading SigLIP on %s ...", args.siglip_gpu)
    t0 = time.time()
    from src.models.siglip_embedder import SigLIPEmbedder
    siglip = SigLIPEmbedder(args.siglip_model, device=args.siglip_gpu)
    log.info("  SigLIP ready (%.1fs)", time.time() - t0)

    # ── Video paths ────────────────────────────────────────────────────────────
    video_paths = _get_video_paths(cfg)
    log.info("Found %d videos for %s", len(video_paths), bench)

    if args.filter_from:
        keep = set()
        for line in Path(args.filter_from).open():
            try: keep.add(json.loads(line)["video_id"])
            except: pass
        video_paths = {k: v for k, v in video_paths.items() if k in keep}
        log.info("Filtered to %d videos", len(video_paths))

    from src.memory.core.dynamic_grouper import group_frames
    from src.memory.core.schema import MemoryBank, MemoryChunk
    from src.models.vlm_client import VLMClient

    # ══════════════════════════════════════════════════════════════════════════
    # TWO-PASS MODE
    # ══════════════════════════════════════════════════════════════════════════
    if two_pass:
        bank_prov = _memory_bank_provenance(args, cfg, two_pass=True)
        caps_dir.mkdir(parents=True, exist_ok=True)
        log.info("Two-pass mode: caption=%s  summary=%s",
                 args.caption_model, args.vlm_model)

        # ── Phase 1: captions with small model ────────────────────────────────
        pending_cap = [vid for vid in sorted(video_paths)
                       if not (out_dir / f"{vid}.json").exists()
                       and not (caps_dir / f"{vid}.json").exists()]
        log.info("Phase 1 — captions: %d videos to process", len(pending_cap))

        # Load both models upfront to reserve GPU memory
        log.info("Loading caption VLM %s ...", args.caption_model)
        t0 = time.time()
        cap_vlm = VLMClient(model_path=args.caption_model,
                            device=args.vlm_gpu, max_new_tokens=64)
        log.info("  caption VLM ready (%.1fs)", time.time() - t0)

        log.info("Loading summary VLM %s ...", args.vlm_model)
        t0 = time.time()
        sum_vlm = VLMClient(model_path=args.vlm_model,
                            device=args.vlm_gpu, max_new_tokens=128)
        log.info("  summary VLM ready (%.1fs)", time.time() - t0)

        if pending_cap:

            for video_id in pending_cap:
                video_path = video_paths[video_id]
                if not Path(video_path).exists():
                    log.warning("[%s] not found, skipping", video_id)
                    continue
                log.info("[%s] phase1 caption ...", video_id)
                t_vid = time.time()
                with tempfile.TemporaryDirectory() as tmpdir:
                    frame_dir = Path(tmpdir) / "frames"
                    frame_paths = extract_frames(video_path, frame_dir, fps=args.fps)
                    T = len(frame_paths)
                    timestamps = [i / args.fps for i in range(T)]
                    duration = timestamps[-1] + 1.0 if timestamps else 0.0
                    log.info("  %d frames", T)

                    v_frames = siglip.encode_images([str(p) for p in frame_paths])
                    groups = group_frames(v_frames, timestamps,
                                         theta=args.theta, n_max=args.n_max)
                    log.info("  %d groups", len(groups))

                    t0 = time.time()
                    if args.api_url:
                        all_captions = caption_frames_api(
                            frame_paths, args.api_url, args.api_model,
                            stride=args.caption_stride,
                            concurrency=args.api_concurrency,
                            timestamps=timestamps)
                    else:
                        all_captions = caption_frames(frame_paths, cap_vlm,
                                                      stride=args.caption_stride,
                                                      timestamps=timestamps)
                    log.info("  captions done (%.1fs)", time.time() - t0)

                    # Blur-check centers, save keyframes, collect group data
                    video_kf_dir = out_dir / video_id / "keyframes"
                    video_kf_dir.mkdir(parents=True, exist_ok=True)
                    groups_data = []
                    for chunk_id, g in enumerate(groups):
                        center = _select_center(frame_paths, g)
                        kf_path = _save_keyframe(
                            frame_paths, center,
                            video_kf_dir / f"{chunk_id:04d}.jpg")
                        groups_data.append({
                            "frame_indices": g.frame_indices,
                            "t_start": g.t_start,
                            "t_end": g.t_end,
                            "center_idx": center,
                            "size": g.size,
                            "keyframe_path": kf_path,
                            "keyframe_ts": timestamps[center],
                            "v_visual": v_frames[center].tolist(),
                        })

                    cap_data = {
                        "video_id": video_id,
                        "duration": duration,
                        "fps": args.fps,
                        "timestamps": timestamps,
                        "groups": groups_data,
                        "captions": all_captions,
                    }
                    (caps_dir / f"{video_id}.json").write_text(
                        json.dumps(cap_data, ensure_ascii=False))
                    log.info("[%s] phase1 done (%.1fs)", video_id, time.time() - t_vid)

        # ── Phase 2: summaries + bank ──────────────────────────────────────────
        vlm = sum_vlm  # already loaded
        pending_sum = [vid for vid in sorted(video_paths)
                       if not (out_dir / f"{vid}.json").exists()
                       and (caps_dir / f"{vid}.json").exists()]
        log.info("Phase 2 — summaries: %d videos to process", len(pending_sum))

        if pending_sum:
            log.info("Loading BGE-M3 on %s ...", args.bge_gpu)
            t0 = time.time()
            from src.models.embedder import BGEM3Embedder
            embedder = BGEM3Embedder(
                model_path=cfg["models"]["embedder"]["model_path"],
                use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
                device=args.bge_gpu,
            )
            log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

            for video_id in pending_sum:
                cap_data = json.loads((caps_dir / f"{video_id}.json").read_text())
                timestamps = cap_data["timestamps"]
                duration = cap_data["duration"]
                all_captions = cap_data["captions"]
                groups_data = cap_data["groups"]

                # Load SRT
                srt_entries: List[Tuple[float, float, str]] = []
                if subtitle_dir is not None:
                    srt_path = subtitle_dir / f"{video_id}.srt"
                    if srt_path.exists():
                        try:
                            srt_entries = parse_srt(srt_path)
                        except Exception as e:
                            log.warning("[%s] SRT parse failed: %s", video_id, e)
                    else:
                        log.warning("[%s] no SRT at %s — phase2 chunks will have empty speech/asr",
                                  video_id, srt_path)

                log.info("[%s] phase2 summary (%d groups) ...",
                         video_id, len(groups_data))
                t_vid = time.time()

                speech_texts = [
                    align_subtitles(srt_entries, gd["t_start"], gd["t_end"]) if srt_entries else ""
                    for gd in groups_data
                ]
                if args.api_url:
                    groups_caps = [[all_captions[fi] for fi in gd["frame_indices"]] for gd in groups_data]
                    summaries = summarize_groups_api(
                        groups_caps, speech_texts,
                        args.api_url, args.api_model,
                        concurrency=args.api_concurrency)
                else:
                    summaries = []
                    for gd, speech in zip(groups_data, speech_texts):
                        caps = [all_captions[fi] for fi in gd["frame_indices"]]
                        summary = summarize_group(caps, vlm, speech_text=speech)
                        summaries.append(summary)

                embed_texts = [
                    f"{s}\n\n{sp}" if sp.strip() else s
                    for s, sp in zip(summaries, speech_texts)
                ]
                sem_vecs = embedder.encode(embed_texts)

                video_ep_dir = out_dir / video_id / "episodic"
                video_ep_dir.mkdir(parents=True, exist_ok=True)

                chunks = []
                for chunk_id, (gd, summary_dict, speech) in enumerate(
                        zip(groups_data, summaries, speech_texts)):
                    summary_text = summary_dict.get("summary", "") if isinstance(summary_dict, dict) else summary_dict
                    ep_data = {
                        "unit_id": f"{video_id}_unit_{chunk_id:04d}",
                        "episodic_descs": [all_captions[fi] for fi in gd["frame_indices"]
                                           if all_captions[fi].strip()],
                        "frame_timestamps": [timestamps[fi] for fi in gd["frame_indices"]],
                        "speech_text": speech,
                    }
                    (video_ep_dir / f"{chunk_id:04d}.json").write_text(
                        json.dumps(ep_data, ensure_ascii=False))

                    # Extract static attributes from keyframe
                    static_frame = extract_static_attributes(
                        gd["keyframe_path"],
                        frame_id=f"{video_id}_chunk{chunk_id:03d}",
                        timestamp=gd["keyframe_ts"],
                        vlm=vlm,
                    ) if Path(gd["keyframe_path"]).exists() else None

                    static_index_text = build_static_index_text(static_frame) if static_frame else ""
                    text_to_encode = static_index_text if static_index_text else " "
                    static_vecs = embedder.encode([text_to_encode])

                    chunk_kwargs = {
                        "video_id": video_id,
                        "chunk_id": chunk_id,
                        "start_time": gd["t_start"],
                        "end_time": gd["t_end"],
                        "memory_text": summary_text,
                        "asr": speech,
                        "sampled_frames": [timestamps[fi] for fi in gd["frame_indices"]],
                        "keyframe_path": gd["keyframe_path"],
                        "keyframe_ts": gd["keyframe_ts"],
                        "v_semantic": sem_vecs[chunk_id].tolist(),
                        "v_visual": gd["v_visual"],
                    }
                    if isinstance(summary_dict, dict):
                        chunk_kwargs.update({
                            "key_events": summary_dict.get("key_events", []),
                            "actors": summary_dict.get("actors", []),
                            "state_changes": summary_dict.get("state_changes", []),
                            "temporal_relations": summary_dict.get("temporal_relations", []),
                            "causal_clues": summary_dict.get("causal_clues", []),
                        })
                    if static_frame:
                        chunk_kwargs.update({
                            "static_frames": [static_frame],
                            "static_index_text": static_index_text,
                            "v_static": static_vecs[0].tolist(),
                        })

                    chunks.append(MemoryChunk(**chunk_kwargs))

                bank = MemoryBank(
                    video_id=video_id,
                    duration=duration,
                    chunks=chunks,
                    fps=args.fps,
                    memory_kind=MEMORY_KIND,
                    **_subtitle_bank_fields(subtitle_dir, video_id, srt_entries),
                    **bank_prov,
                )
                bank.save(out_dir / f"{video_id}.json")
                log.info("[%s] saved %d chunks (%.1fs)", video_id,
                         len(chunks), time.time() - t_vid)

        log.info("Done (two-pass) — output in %s", out_dir)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-PASS MODE (caption model == summary model, or API mode)
    # ══════════════════════════════════════════════════════════════════════════
    bank_prov = _memory_bank_provenance(args, cfg, two_pass=False)
    vlm = None
    if not args.api_url:
        log.info("Loading VLM %s on %s ...", args.vlm_model, args.vlm_gpu)
        t0 = time.time()
        vlm = VLMClient(model_path=args.vlm_model, device=args.vlm_gpu, max_new_tokens=64)
        log.info("  VLM ready (%.1fs)", time.time() - t0)
    else:
        log.info("API mode — skipping local VLM load (using %s)", args.api_url)

    log.info("Loading BGE-M3 on %s ...", args.bge_gpu)
    t0 = time.time()
    from src.models.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(
        model_path=cfg["models"]["embedder"]["model_path"],
        use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
        device=args.bge_gpu,
    )
    log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    from src.utils.gpu_lock import lock_gpu
    # Skip gpu_lock in API mode — vLLM already occupies the GPU; multiple workers share SigLIP/BGE GPU
    _gpu_lock = lock_gpu(args.bge_gpu) if not args.api_url else None

    for video_id, video_path in sorted(video_paths.items()):
        out_path = out_dir / f"{video_id}.json"
        if out_path.exists():
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
                    log.info("[%s] loaded %d subtitle entries", video_id, len(srt_entries))
                except Exception as e:
                    log.warning("[%s] SRT parse failed: %s", video_id, e)
            else:
                log.warning("[%s] no SRT at %s — chunks will have empty speech/asr", video_id, srt_path)

        log.info("[%s] starting ...", video_id)
        t_video = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"

            frame_paths = extract_frames(video_path, frame_dir, fps=args.fps)
            T = len(frame_paths)
            timestamps = [i / args.fps for i in range(T)]
            duration   = timestamps[-1] + 1.0 if timestamps else 0.0
            log.info("  %d frames", T)

            v_frames = siglip.encode_images([str(p) for p in frame_paths])
            groups = group_frames(v_frames, timestamps,
                                  theta=args.theta, n_max=args.n_max)
            log.info("  %d groups", len(groups))

            t0 = time.time()
            if args.api_url:
                all_captions = caption_frames_api(
                    frame_paths, args.api_url, args.api_model,
                    stride=args.caption_stride, concurrency=args.api_concurrency,
                    timestamps=timestamps)
            else:
                all_captions = caption_frames(frame_paths, vlm, stride=args.caption_stride,
                                              timestamps=timestamps)
            log.info("  captions done (%.1fs)", time.time() - t0)

            t0 = time.time()
            speech_texts = [
                align_subtitles(srt_entries, g.t_start, g.t_end) if srt_entries else ""
                for g in groups
            ]
            if args.api_url:
                groups_caps = [[all_captions[fi] for fi in g.frame_indices] for g in groups]
                summaries = summarize_groups_api(
                    groups_caps, speech_texts,
                    args.api_url, args.api_model,
                    concurrency=args.api_concurrency)
            else:
                summaries = []
                for g, speech in zip(groups, speech_texts):
                    caps = [all_captions[fi] for fi in g.frame_indices]
                    summary = summarize_group(caps, vlm, speech_text=speech)
                    summaries.append(summary)
            log.info("  summaries done (%.1fs)", time.time() - t0)

            embed_texts = [
                f"{s}\n\n{sp}" if sp.strip() else s
                for s, sp in zip(summaries, speech_texts)
            ]
            sem_vecs = embedder.encode(embed_texts)

            video_kf_dir = out_dir / video_id / "keyframes"
            video_ep_dir = out_dir / video_id / "episodic"
            video_kf_dir.mkdir(parents=True, exist_ok=True)
            video_ep_dir.mkdir(parents=True, exist_ok=True)

            chunks = []
            for chunk_id, (g, summary_dict, speech) in enumerate(
                    zip(groups, summaries, speech_texts)):
                center = _select_center(frame_paths, g)
                kf_path = _save_keyframe(frame_paths, center,
                                         video_kf_dir / f"{chunk_id:04d}.jpg")
                summary_text = summary_dict.get("summary", "") if isinstance(summary_dict, dict) else summary_dict
                ep_data = {
                    "unit_id": f"{video_id}_unit_{chunk_id:04d}",
                    "episodic_descs": [all_captions[fi] for fi in g.frame_indices
                                       if all_captions[fi].strip()],
                    "frame_timestamps": [timestamps[fi] for fi in g.frame_indices],
                    "speech_text": speech,
                }
                (video_ep_dir / f"{chunk_id:04d}.json").write_text(
                    json.dumps(ep_data, ensure_ascii=False))

                # Extract static attributes from keyframe
                static_frame = extract_static_attributes(
                    kf_path,
                    frame_id=f"{video_id}_chunk{chunk_id:03d}",
                    timestamp=timestamps[center],
                    vlm=vlm,
                ) if not args.api_url and vlm and Path(kf_path).exists() else None

                static_index_text = build_static_index_text(static_frame) if static_frame else ""
                text_to_encode = static_index_text if static_index_text else " "
                static_vecs = embedder.encode([text_to_encode])

                chunk_kwargs = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "start_time": g.t_start,
                    "end_time": g.t_end,
                    "memory_text": summary_text,
                    "asr": speech,
                    "sampled_frames": [timestamps[fi] for fi in g.frame_indices],
                    "keyframe_path": kf_path,
                    "keyframe_ts": timestamps[center],
                    "v_semantic": sem_vecs[chunk_id].tolist(),
                    "v_visual": v_frames[center].tolist(),
                }
                if isinstance(summary_dict, dict):
                    chunk_kwargs.update({
                        "key_events": summary_dict.get("key_events", []),
                        "actors": summary_dict.get("actors", []),
                        "state_changes": summary_dict.get("state_changes", []),
                        "temporal_relations": summary_dict.get("temporal_relations", []),
                        "causal_clues": summary_dict.get("causal_clues", []),
                    })
                if static_frame:
                    chunk_kwargs.update({
                        "static_frames": [static_frame],
                        "static_index_text": static_index_text,
                        "v_static": static_vecs[0].tolist(),
                    })

                chunks.append(MemoryChunk(**chunk_kwargs))

            bank = MemoryBank(
                video_id=video_id,
                duration=duration,
                chunks=chunks,
                fps=args.fps,
                memory_kind=MEMORY_KIND,
                **_subtitle_bank_fields(subtitle_dir, video_id, srt_entries),
                **bank_prov,
            )
            bank.save(out_path)
            log.info("[%s] saved %d chunks (%.1fs total)",
                     video_id, len(chunks), time.time() - t_video)

    log.info("Done — output in %s", out_dir)


if __name__ == "__main__":
    main()
