"""Fixed wall-clock window memory: time-based chunks + in-window frame sampling + VLM JSON.

Library API
-----------
    chunks, duration, actual_sec = sample_event_chunks(video_path, ...)
    bank = build_fixedframe_memory_bank(chunks, video_id, duration, actual_sec, vlm, ...)

CLI (batch build for a whole benchmark)
-----------------------------------------
    PYTHONPATH=. python -m src.memory.fixedframe --benchmark mlvu --modes event dense

Dataset paths and chunk defaults live in ``memory.specs`` (not under ``configs/``).

For SigLIP / cosine-similarity grouping (similarity_group banks), run
``python -m src.memory.similarity`` (see ``models/siglip_embedder.py`` + ``memory/dynamic_grouper.py``).

Modes
-----
event  (default): chunk_size_sec=4, frames_per_chunk=3, at 20%/50/80%
dense            : chunk_size_sec=1, frames_per_chunk=1, at 50%

Both modes apply an adaptive cap: if the video would produce > max_chunks chunks,
chunk_size_sec is increased to ceil(duration / max_chunks).
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from src.memory.core.schema import MemoryBank, MemoryChunk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── VLM prompts ───────────────────────────────────────────────────────────────

EVENT_PROMPT = """\
You are a meticulous video analyst. You are shown {n_frames} frame(s) sampled \
from {t0:.1f}s to {t1:.1f}s of a video.

Describe ONLY what is directly observable. Return a strict JSON object with exactly \
these keys (no extras, no markdown fences):

{{
  "scene":             "1 sentence: location/setting/environment/lighting.",
  "objects":           ["list", "of", "salient", "objects", "≤8 items"],
  "actions":           ["list", "of", "observed", "actions/movements", "≤6 items"],
  "state_change":      "1 sentence: what changed between start and end of this clip (or 'static').",
  "temporal_relation": "short phrase relating this clip to the surrounding content, e.g. 'new scene begins', 'continues previous action', 'camera pans to'.",
  "asr":               "any audible speech or dialogue visible as subtitles (or empty string).",
  "ocr":               "any on-screen text: signs, UI, captions, labels (or empty string).",
  "evidence_caption":  "2–3 sentences that combine the above into a retrieval-optimised description of this clip, mentioning who, what, where, and any text/numbers visible."
}}
"""

DENSE_PROMPT = """\
You are a video analyst. You are shown 1 frame sampled at {t0:.1f}s of a video.

Describe ONLY what is visible. Return a strict JSON object (no markdown fences):

{{
  "scene":             "1 sentence: setting/environment.",
  "objects":           ["list", "of", "salient", "objects", "≤6 items"],
  "actions":           ["list", "of", "visible", "actions", "≤4 items"],
  "state_change":      "",
  "temporal_relation": "",
  "asr":               "",
  "ocr":               "any on-screen text (or empty string).",
  "evidence_caption":  "1–2 sentences: who/what/where and any text or numbers."
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(str(x).strip() for x in v if x and str(x).strip())
    return str(v).strip()


def _as_list(v) -> list:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x and str(x).strip()]
    s = _as_str(v)
    return [s] if s else []


def _safe_parse(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(l for l in lines if not l.strip().startswith("```"))
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i:j+1]
    try:
        raw = json.loads(s)
        if not isinstance(raw, dict):
            raise ValueError
    except Exception:
        raw = {"evidence_caption": text.strip()[:400]}
    return {
        "scene":             _as_str(raw.get("scene", "")),
        "objects":           _as_list(raw.get("objects", [])),
        "actions":           _as_list(raw.get("actions", [])),
        "state_change":      _as_str(raw.get("state_change", "")),
        "temporal_relation": _as_str(raw.get("temporal_relation", "")),
        "asr":               _as_str(raw.get("asr", "")),
        "ocr":               _as_str(raw.get("ocr", "")),
        "evidence_caption":  _as_str(raw.get("evidence_caption", "")),
    }


def _make_memory_text(t0: float, t1: float, p: dict) -> str:
    """Build the canonical retrieval string from parsed fields."""
    cap = p.get("evidence_caption") or ""
    if not cap:
        # Fallback: assemble from parts
        parts = [p.get("scene",""), _as_str(p.get("actions",[])), p.get("ocr","")]
        cap = " ".join(x for x in parts if x)
    prefix = f"[{t0:.0f}s-{t1:.0f}s]"
    extra = []
    if p.get("ocr"):
        extra.append(f"OCR: {p['ocr']}")
    if p.get("asr"):
        extra.append(f"Speech: {p['asr']}")
    tail = "  " + "  ".join(extra) if extra else ""
    return f"{prefix} {cap}{tail}".strip()


# ── Frame extraction ──────────────────────────────────────────────────────────

@dataclass
class TimedChunk:
    chunk_id:   int
    start_time: float
    end_time:   float
    frames:     List[np.ndarray]      # RGB uint8, already resized
    frame_times: List[float] = field(default_factory=list)


def _resize_and_crop(frame_rgb: np.ndarray, resolution: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(frame_rgb)
    w, h = img.size
    if w <= h:
        nw, nh = resolution, max(resolution, int(h * resolution / w))
    else:
        nw, nh = max(resolution, int(w * resolution / h)), resolution
    img = img.resize((nw, nh), Image.BILINEAR)
    l = (nw - resolution) // 2
    t = (nh - resolution) // 2
    img = img.crop((l, t, l + resolution, t + resolution))
    return np.array(img)


def sample_event_chunks(
    video_path: str,
    chunk_size_sec: float = 4.0,
    max_chunks: int = 900,
    frames_per_chunk: int = 3,
    resolution: int = 448,
) -> tuple:
    """Sample time-based chunks from a video.

    Returns (chunks: List[TimedChunk], duration: float, actual_chunk_sec: float).
    """
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=2)
    fps_native = float(vr.get_avg_fps() or 25.0)
    n_total    = len(vr)
    duration   = n_total / fps_native

    # Adaptive chunk size
    n_natural = math.ceil(duration / chunk_size_sec)
    actual_sec = chunk_size_sec if n_natural <= max_chunks else math.ceil(duration / max_chunks)

    # Frame positions within each chunk (relative, 0–1)
    if frames_per_chunk == 1:
        positions = [0.5]
    elif frames_per_chunk == 2:
        positions = [0.25, 0.75]
    else:  # 3+
        positions = [0.2 + i * 0.3 for i in range(frames_per_chunk)]  # 0.2, 0.5, 0.8 for 3 frames

    chunks: List[TimedChunk] = []
    t = 0.0
    chunk_id = 0
    while t < duration - 1e-6:
        t_end = min(t + actual_sec, duration)

        # Compute frame indices
        idxs = []
        for pos in positions:
            t_sample = t + pos * (t_end - t)
            idx = min(int(round(t_sample * fps_native)), n_total - 1)
            idxs.append(idx)
        idxs = sorted(set(idxs))
        frame_times = [i / fps_native for i in idxs]

        raw = vr.get_batch(idxs).asnumpy()
        frames = [_resize_and_crop(f, resolution) for f in raw]

        chunks.append(TimedChunk(
            chunk_id=chunk_id,
            start_time=t,
            end_time=t_end,
            frames=frames,
            frame_times=frame_times,
        ))
        chunk_id += 1
        t = t_end

    return chunks, duration, float(actual_sec)


# ── Memory bank builder ───────────────────────────────────────────────────────

def build_fixedframe_memory_bank(
    chunks: List[TimedChunk],
    video_id: str,
    duration: float,
    actual_chunk_sec: float,
    vlm,
    frames_per_chunk: int = 3,
    progress: bool = True,
    mode: str = "event",       # "event" | "dense"
) -> MemoryBank:
    """Caption each TimedChunk with the VLM and return a MemoryBank.

    Args:
        mode: "event" uses EVENT_PROMPT (3 frames, structured);
              "dense" uses DENSE_PROMPT (1 frame, lighter).
    """
    from tqdm import tqdm as _tqdm

    use_dense  = (mode == "dense")
    prompt_tpl = DENSE_PROMPT if use_dense else EVENT_PROMPT

    memory_chunks: List[MemoryChunk] = []
    iterator = _tqdm(chunks, desc=f"fixedframe[{video_id}]") if progress else chunks

    for tc in iterator:
        t0, t1 = tc.start_time, tc.end_time
        if use_dense:
            prompt = prompt_tpl.format(t0=t0)
        else:
            prompt = prompt_tpl.format(n_frames=len(tc.frames), t0=t0, t1=t1)

        raw    = vlm.chat_with_frames(tc.frames, prompt, max_new_tokens=512)
        parsed = _safe_parse(raw)

        memory_chunks.append(MemoryChunk(
            video_id=video_id,
            chunk_id=tc.chunk_id,
            start_time=t0,
            end_time=t1,
            scene=parsed["scene"],
            objects=parsed["objects"],
            actions=parsed["actions"],
            state_change=parsed["state_change"],
            temporal_relation=parsed["temporal_relation"],
            asr=parsed["asr"],
            ocr=parsed["ocr"],
            evidence_caption=parsed["evidence_caption"],
            sampled_frames=tc.frame_times,
            memory_text=_make_memory_text(t0, t1, parsed),
        ))

    kind = "dense_1fps_v1" if use_dense else "event_v1"

    if frames_per_chunk == 1:
        pos_list = [0.5]
    elif frames_per_chunk == 2:
        pos_list = [0.25, 0.75]
    else:
        pos_list = [0.2 + i * 0.3 for i in range(frames_per_chunk)]

    if use_dense:
        chunking_readme_zh = (
            f"密集分块（dense）：每块约 {actual_chunk_sec:.2f} 秒；每块采样 1 帧（块内相对位置 50%）。"
        )
        chunking_readme_en = (
            f"Dense baseline: ~{actual_chunk_sec:.2f}s wall time per chunk; "
            "1 frame per chunk sampled at 50% position within the bin."
        )
    else:
        pos_desc = "、".join(f"{p * 100:.0f}%" for p in pos_list)
        chunking_readme_zh = (
            f"固定时间分块：每块约 {actual_chunk_sec:.2f} 秒（视频过长时单块时长会自动加大，使总块数不超过构建时的 max_chunks）；"
            f"每块内采样 {frames_per_chunk} 帧，帧在块内的相对时刻：{pos_desc}。"
        )
        pos_en = ", ".join(f"{p * 100:.0f}%" for p in pos_list)
        chunking_readme_en = (
            f"Fixed-time chunks: ~{actual_chunk_sec:.2f}s wall time per chunk "
            "(chunk duration may increase when the adaptive max_chunks cap applies); "
            f"{frames_per_chunk} frame(s) per chunk at relative positions within each chunk: {pos_en}."
        )

    return MemoryBank(
        video_id=video_id,
        duration=duration,
        chunks=memory_chunks,
        chunk_sec=actual_chunk_sec,
        stride_sec=actual_chunk_sec,
        memory_kind=kind,
        chunking_readme_zh=chunking_readme_zh,
        chunking_readme_en=chunking_readme_en,
        frames_per_chunk=frames_per_chunk,
        frame_positions_within_chunk=list(pos_list),
    )


# ── CLI: batch fixed-frame banks (paths from ``memory.specs``) ────────────────

def _cli_get_video_paths(cfg: dict) -> dict[str, str]:
    """Return {video_id: video_path} for the benchmark."""
    bench = cfg["benchmark"]["name"]
    if bench == "mlvu":
        json_dir  = Path(cfg["benchmark"]["json_dir"])
        video_dir = Path(cfg["benchmark"]["video_dir"])
        paths: dict[str, str] = {}
        for jf in cfg["benchmark"]["json_files"].values():
            p = json_dir / jf
            if not p.exists():
                continue
            subdir = video_dir / Path(jf).stem
            for item in json.loads(p.read_text()):
                vid = item.get("video", item.get("video_id", ""))
                if vid:
                    vid_key = Path(vid).stem
                    if vid_key not in paths:
                        paths[vid_key] = str(subdir / vid)
        return paths
    if bench == "videomme":
        from src.dataloader.videomme import load_videomme
        b = cfg["benchmark"]
        samples = load_videomme(
            parquet_path=b["parquet_path"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
        return {s.video_id: s.video_path for s in samples}
    raise ValueError(f"Unsupported benchmark: {bench}")


def _cli_main() -> None:
    from src.utils.logging import get_logger
    from .readme import write_memory_build_readme

    from memory import specs as build_specs

    log = get_logger("build_fixedframe_memory")

    ap = argparse.ArgumentParser(
        description="Build fixed wall-clock memory banks (event + dense). "
                    "Dataset locations: src/memory/core/specs.py",
    )
    ap.add_argument(
        "--benchmark",
        required=True,
        choices=["mlvu", "videomme"],
        help="Which benchmark spec from src/memory/core/specs.py to use",
    )
    ap.add_argument("--vlm-gpu", default="cuda:0")
    ap.add_argument("--vlm-model", default=build_specs.FIXEDFRAME_DEFAULT_VLM)
    ap.add_argument("--modes", nargs="+", choices=["event", "dense"], default=["event", "dense"])
    ap.add_argument("--max-chunks", type=int, default=build_specs.FIXEDFRAME_DEFAULT_MAX_CHUNKS)
    ap.add_argument("--resolution", type=int, default=build_specs.FIXEDFRAME_DEFAULT_RESOLUTION)
    ap.add_argument(
        "--filter-from",
        default=None,
        help="JSONL with video_id lines; only build banks for those ids",
    )
    args = ap.parse_args()

    cfg = build_specs.cfg_for_fixedframe_build(args.benchmark)
    bench = cfg["benchmark"]["name"]
    out_root = Path(cfg["paths"]["outputs_root"])

    event_dir = out_root / "memory" / f"{bench}_fixed"
    out_dirs = {
        "event": event_dir,
        "dense": out_root / "memory" / f"{bench}_dense",
    }
    for mode in args.modes:
        out_dirs[mode].mkdir(parents=True, exist_ok=True)

    log.info("loading VLM %s on %s ...", args.vlm_model, args.vlm_gpu)
    from src.models.vlm_client import VLMClient
    t0 = time.time()
    vlm = VLMClient(model_path=args.vlm_model, device=args.vlm_gpu, max_new_tokens=512)
    log.info("  VLM ready (%.1fs)", time.time() - t0)

    video_paths = _cli_get_video_paths(cfg)
    log.info("found %d videos for %s", len(video_paths), bench)

    if args.filter_from:
        import json as _json
        keep = set()
        for line in Path(args.filter_from).open():
            try:
                keep.add(_json.loads(line)["video_id"])
            except Exception:
                pass
        before = len(video_paths)
        video_paths = {k: v for k, v in video_paths.items() if k in keep}
        log.info("filtered to %d/%d videos from %s", len(video_paths), before, args.filter_from)

    mode_params = {k: dict(v) for k, v in build_specs.FIXEDFRAME_MODE_PARAMS.items()}

    for video_id, video_path in sorted(video_paths.items()):
        vp = Path(video_path)
        if not vp.exists():
            log.warning("video not found: %s", video_path)
            continue

        for mode in args.modes:
            out_path = out_dirs[mode] / f"{video_id}.json"
            if out_path.exists():
                log.info("[%s/%s] already built, skipping", video_id, mode)
                continue

            params = mode_params[mode]
            log.info("[%s/%s] sampling chunks (chunk_sec=%.0f, fpc=%d) ...",
                     video_id, mode, params["chunk_size_sec"], params["frames_per_chunk"])
            try:
                t0 = time.time()
                chunks, duration, actual_sec = sample_event_chunks(
                    str(vp),
                    chunk_size_sec=float(params["chunk_size_sec"]),
                    max_chunks=args.max_chunks,
                    frames_per_chunk=int(params["frames_per_chunk"]),
                    resolution=args.resolution,
                )
                log.info("  %d chunks (%.0fs video, actual_sec=%.0f, %.1fs sampling)",
                         len(chunks), duration, actual_sec, time.time() - t0)

                t0 = time.time()
                bank = build_fixedframe_memory_bank(
                    chunks, video_id, duration, actual_sec, vlm,
                    frames_per_chunk=int(params["frames_per_chunk"]),
                    mode=mode,
                )
                log.info("  built %d memory chunks (%.1fs)", len(bank.chunks), time.time() - t0)
                bank.save(out_path)
            except Exception as e:
                log.error("  FAILED: %s", e, exc_info=True)

    log.info("done")

    dir_modes: dict[Path, list[str]] = defaultdict(list)
    for mode, d in out_dirs.items():
        dir_modes[Path(d).resolve()].append(mode)
    argv_q = " ".join(shlex.quote(a) for a in sys.argv)
    for out_p, modes in sorted(dir_modes.items()):
        lines = [
            "build: fixed-time-window memory (python -m src.memory.fixedframe)",
            f"benchmark: {bench}",
            f"modes: {', '.join(sorted(modes))}",
            f"specs: src/memory/core/specs.py",
            f"argv: {argv_q}",
            f"vlm_model: {args.vlm_model}",
            f"vlm_gpu: {args.vlm_gpu}",
            f"max_chunks: {args.max_chunks}",
            f"resolution: {args.resolution}",
            f"filter_from: {args.filter_from or '(none)'}",
            "",
            "Mode params from src.memory.core.specs.FIXEDFRAME_MODE_PARAMS:",
            "  event: ~4s bins, 3 frames @ 20%/50%/80% (adaptive max_chunks stretches bin size).",
            "  dense: 1s bins, 1 frame @ 50%.",
        ]
        write_memory_build_readme(
            out_p,
            title="VEIL memory bank build record (fixed-window)",
            lines=lines,
        )
        log.info("wrote %s", out_p / "MEMORY_BUILD_README.txt")


if __name__ == "__main__":
    _cli_main()
