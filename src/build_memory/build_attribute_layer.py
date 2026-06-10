"""属性层一站式生成：VLM 抽 entities + 后处理截断 + BGE 编码 → vectors.npz。

合并自旧的四个脚本：
  generate_attributes.py    → VLM 调用、merge_attrs、build_index_text
  fix_missing_chunks.py     → skip 非空 chunk（已含在 process_video 内）+ 多 URL round-robin
  postprocess_cap_entities.py → entities 截断（已含在 merge_attrs 内）
  encode_vectors.py         → BGE-M3 编码 narrative + attribute

输入（每个 video 目录必须已经有这两个文件）：
    {video_id}/narrative.json   叙述层 + frame_timestamps
    {video_id}/attributes.json  v2 skeleton 或部分完成

输出：
    {video_id}/frames/{cid:04d}_{i}.jpg   每个 chunk 抽 k=6 帧
    {video_id}/attributes.json            填 static_attributes + static_index_text + frame_paths
    {video_id}/vectors.npz                narrative_vecs + attribute_vecs + chunk_ids

关键改动：
  1. extract_one 失败时不再静默写 _empty_attr() 落盘。若某 chunk 所有帧都失败，
     该 chunk 保持原状态（空 static_index_text），下次 pass 会自然重试。
  2. --vlm-api-url 支持逗号分隔多 URL，按帧 round-robin 分发请求。
  3. --video-ids-file 允许从外部 txt 文件指定视频列表（取代字母序遍历）。
  4. 跑完 VLM 立刻 BGE 编码，每个视频一次性产出干净的目标态。

用法：
    PYTHONPATH=. python src/build_memory/build_attribute_layer.py \\
        --vlm-api-url http://127.0.0.1:8001,http://127.0.0.1:8003 \\
        --bge-device cuda:6 \\
        --video-ids-file /tmp/build_todo.txt
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
from PIL import Image


BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")
VIDEO_DIR = Path("/home2/ycj/Datas/VideoMME/videos")
PARQUET_PATH = "/home2/ycj/Datas/VideoMME/videomme/test-00000-of-00001.parquet"
SAMPLER_LOG_PATH = BANK_DIR.parent / ".frame_sampling_log.jsonl"


STATIC_ATTR_PROMPT = (
    "You are a visual attribute extractor building a SEARCHABLE INDEX for this frame. "
    "The original image is preserved and can be re-examined later for details — your job is to "
    "produce concise, retrievable key signals, NOT an exhaustive description.\n\n"
    "Return ONLY a JSON object — no prose, no markdown decorations beyond fenced code:\n"
    "{\n"
    '  "entities": [\n'
    '    ["entity_name_or_role", "concise key attributes — 1 phrase"],\n'
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Entity examples:\n"
    "- A person → entity = role/position; attrs = identifying features\n"
    '  ["man on left", "glasses, dark gray blazer, blue shirt"]\n'
    '  ["woman in center", "long black hair, blue top"]\n'
    "- An object → entity = specific name; attrs = key visible details\n"
    '  ["light bulb", "vintage yellow glowing"]\n'
    '  ["scoreboard", "G2 0 - 2 Team Liquid"]\n'
    "- Text/numbers → entity = label of what; attrs = exact text (short)\n"
    '  ["logo", "PBS NOVA"]\n'
    '  ["stock price", "$32"]\n'
    "- Scene → entity = \"scene\"; attrs = location/lighting in 1 phrase\n"
    '  ["scene", "indoor office, soft lighting"]\n\n'
    "DENSE-CONTENT HANDLING (tables, lists, menus, long text blocks, charts):\n"
    "Do NOT enumerate every item. Describe the TYPE, scope, and 2-3 salient examples only. "
    "Specific items can be looked up from the original frame when needed.\n"
    "  GOOD: [\"comparison table\", \"Samsung Galaxy specs across ~15 models, columns include S7, Note 7, S5\"]\n"
    "  BAD:  [\"table header\", \"Galaxy S7, S7 edge, Note 7, S6, S6 edge, S6 edge+, Note 5, S6 Active, Note 4, S5, S5 Active, ... (every model)\"]\n"
    "  GOOD: [\"menu list\", \"~12 dishes, includes pasta, salad, steak; prices visible\"]\n"
    "  BAD:  [\"menu\", \"Caesar Salad $8, Caprese $10, Spaghetti $14, Carbonara $15, Lasagna $16, ...\"]\n"
    "  GOOD: [\"document\", \"academic paper, English, references section visible\"]\n"
    "  BAD:  [\"document text\", \"Smith et al. 2018, Jones 2019, Wang 2020, Lee 2021, ...\"]\n\n"
    "Rules:\n"
    "- Each pair is 1 entity + 1 short attribute phrase\n"
    "- Each attrs ≤ 25 words; if you need more, you're enumerating — summarize instead\n"
    "- Skip if no notable entities — return \"entities\": []\n"
    "- Be factual: list only what's clearly visible\n"
    "- Prefer specific names: 'golden retriever' not 'dog'\n"
    "- Group person attributes (clothing/hair/glasses) into ONE phrase per person\n"
    "- Group object details (color/shape/material) into ONE phrase per object\n"
    "- Transcribe SHORT OCR/numbers verbatim; for long text, describe its kind + a few keywords\n"
    "- Aim for 5-12 entities max — only what matters for retrieval"
)


def load_video_ids_from_file(path: str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def load_long_video_ids(shard_id: int = 0, num_shards: int = 1,
                       start: int = 0, end: int | None = None) -> list[str]:
    df = pd.read_parquet(PARQUET_PATH)
    df_long = df[df["duration"] == "long"]
    all_ids = sorted(df_long["videoID"].unique().tolist())
    if end is None:
        end = len(all_ids)
    all_ids = all_ids[start:end]
    return [vid for i, vid in enumerate(all_ids) if i % num_shards == shard_id]


def select_frame_positions(
    frame_timestamps: list[float],
    fps: float = 0.5,
) -> list[tuple[int, float]]:
    """Pick frames at fixed sampling rate (~fps per second of chunk).

    No artificial cap: the chunk is already SigLIP-similarity-grouped, so within-
    chunk variation is low. Blank-frame skip + dHash dedup downstream collapse
    redundant samples before VLM. Floor is 1 frame so any non-empty chunk gets
    at least one sample.
    """
    n = len(frame_timestamps)
    if n == 0:
        return []
    duration = frame_timestamps[-1] - frame_timestamps[0]
    k = max(1, round(duration * fps) + 1)
    k = min(k, n)
    if k == 1:
        return [(0, frame_timestamps[0])]
    idx_seq = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    seen = set()
    out = []
    for i in idx_seq:
        if i not in seen:
            seen.add(i)
            out.append((i, frame_timestamps[i]))
    return out


SAMPLER_TAG = "0.5fps_no_cap_dhash4"

# Two-stage in-chunk dedup:
#   1) dHash Hamming ≤ 4: cheap pixel-level near-identical filter (no model load)
#   2) SigLIP cos_sim ≥ 0.95: semantic-level filter (catches "same scene + texture noise")
DEDUP_HAMMING_THRESHOLD = 4
SIGLIP_COS_THRESHOLD = 0.95


def compute_dhash(path: Path, size: int = 8) -> Optional[str]:
    """8x8 difference hash. Returns 64-char binary string or None on failure."""
    try:
        img = Image.open(path).convert("L").resize((size + 1, size))
        arr = np.array(img)
        bits = []
        for i in range(size):
            for j in range(size):
                bits.append("1" if arr[i, j + 1] > arr[i, j] else "0")
        return "".join(bits)
    except Exception:
        return None


def dhash_hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def extract_frame_from_video(video_path: Path, timestamp: float, out_path: Path) -> str:
    """Extract one frame. Returns "ok" / "blank" / "failed":
        ok     — written, size ≥ 1000 bytes (has visible content)
        blank  — written, 200 ≤ size < 1000 bytes (uniform/black frame — skip VLM)
        failed — ffmpeg error, missing file, or < 200 bytes
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        sz = out_path.stat().st_size
        if sz >= 1000:
            return "ok"
        if sz >= 200:
            return "blank"
    cmd = [
        "ffmpeg", "-ss", f"{timestamp:.2f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3",
        "-vf", "scale='min(448,iw)':'min(448,ih)':force_original_aspect_ratio=decrease",
        str(out_path),
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        if not out_path.exists():
            return "failed"
        sz = out_path.stat().st_size
        if sz >= 1000:
            return "ok"
        if sz >= 200:
            return "blank"
        return "failed"
    except Exception:
        return "failed"


def _coerce_entity_pair(e) -> Optional[list]:
    if isinstance(e, (list, tuple)) and len(e) >= 2:
        name, attrs = str(e[0]).strip(), str(e[1]).strip()
    elif isinstance(e, dict):
        name = str(e.get("entity") or e.get("name") or e.get("label") or "").strip()
        attrs = str(e.get("attrs") or e.get("attributes") or e.get("description") or "").strip()
    else:
        return None
    if not name or not attrs:
        return None
    return [name, attrs]


def _dedupe_entities(entities: list[list]) -> list[list]:
    seen = set()
    out = []
    for pair in entities:
        if not pair:
            continue
        key = pair[0].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


_ENTITY_PAIR_RE = re.compile(
    r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]',
    re.DOTALL,
)


def _salvage_entity_pairs(raw: str) -> Optional[dict]:
    """Last-ditch parser: regex-extract complete ["name", "attrs"] pairs.

    Useful when VLM output is truncated by max_tokens (no closing `}`) — most
    entities are already emitted; we just keep the complete pairs.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s).strip()
    pairs = _ENTITY_PAIR_RE.findall(s)
    entities = []
    for name, attrs in pairs:
        name, attrs = name.strip(), attrs.strip()
        if name and attrs:
            entities.append([name, attrs])
    if not entities:
        return None
    return {"entities": _dedupe_entities(entities)}


def parse_attr_json(raw: str) -> Optional[dict]:
    """Parse v2 format. Falls back to regex salvage on truncated/malformed output."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?|```$", "", s, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        d = json.loads(s)
        if not isinstance(d, dict):
            return _salvage_entity_pairs(raw)
        raw_ents = d.get("entities", None)
        if not isinstance(raw_ents, list):
            return _salvage_entity_pairs(raw)
        entities = []
        for e in raw_ents:
            pair = _coerce_entity_pair(e)
            if pair:
                entities.append(pair)
        if not entities:
            return _salvage_entity_pairs(raw)
        return {"entities": _dedupe_entities(entities)}
    except Exception:
        return _salvage_entity_pairs(raw)


def merge_attrs(attr_list: list[dict], max_entities: int = 10) -> dict:
    """Merge entities across frames; dedupe; rank; cap.

    Priority: entities with numbers/OCR > with quoted text > longer attrs > scene last.
    """
    if not attr_list:
        return {"entities": []}
    all_ents = []
    for a in attr_list:
        for e in a.get("entities", []) or []:
            if isinstance(e, list) and len(e) >= 2:
                all_ents.append(e)
    deduped = _dedupe_entities(all_ents)

    def priority_key(e):
        name, attrs = e[0], e[1]
        has_numbers = bool(re.search(r"\d", attrs))
        has_quote = bool(re.search(r'["\'].*?["\']', attrs))
        score = 0
        if has_numbers: score += 100
        if has_quote: score += 50
        score += min(len(attrs), 200) // 10
        if name.lower() == "scene": score -= 30
        return -score

    deduped.sort(key=priority_key)
    return {"entities": deduped[:max_entities]}


def build_index_text(attrs: dict) -> str:
    parts = []
    for e in attrs.get("entities", []) or []:
        if isinstance(e, list) and len(e) >= 2:
            name, a = str(e[0]).strip(), str(e[1]).strip()
            if name and a:
                parts.append(f"{name}: {a}")
    return " | ".join(parts)


class URLPool:
    """Round-robin pool over multiple vLLM URLs."""
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._i = 0
    def next(self) -> str:
        url = self.urls[self._i % len(self.urls)]
        self._i += 1
        return url


async def extract_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    frame_path: Path,
    api_url: str,
    api_model: str,
    max_retries: int = 2,
) -> Optional[dict]:
    """Call VLM on frame. Retry on transient errors only; parse-fail is terminal
    (temperature=0 makes retries useless — salvage path inside parse_attr_json
    already handles truncated JSON)."""
    payload = {
        "model": api_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": ""}},
                {"type": "text", "text": STATIC_ATTR_PROMPT},
            ],
        }],
        "max_tokens": 2048,
        "temperature": 0,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        payload["messages"][0]["content"][0]["image_url"]["url"] = f"data:image/jpeg;base64,{b64}"
        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    f"{api_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as r:
                    data = await r.json()
                    raw = data["choices"][0]["message"]["content"]
                    if "</think>" in raw:
                        raw = raw.split("</think>", 1)[1]
                    parsed = parse_attr_json(raw.strip())
                    if parsed is not None:
                        return parsed
                    # Parse failed even after salvage — give up (no retry: temperature=0
                    # makes same prompt deterministic, so retry can't help).
                    print(f"  [parse-fail] {frame_path.name} @ {api_url}: "
                          f"{raw.strip()[:200]!r}", flush=True)
                    return None
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                print(f"  [error] {frame_path.name} @ {api_url} after {max_retries+1} tries: "
                      f"[{type(e).__name__}] {e}", flush=True)
                return None


async def process_video_vlm(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    video_id: str,
    urls: URLPool,
    api_model: str,
    overwrite: bool,
    k_frames: int,
    ffmpeg_executor,
    siglip=None,
) -> tuple[int, int, int]:
    """Run VLM extraction on chunks needing it. Returns (chunks_done, total_chunks, vlm_calls)."""
    video_dir = BANK_DIR / video_id
    attr_path = video_dir / "attributes.json"
    narr_path = video_dir / "narrative.json"
    if not attr_path.exists() or not narr_path.exists():
        return 0, 0, 0

    narrative = json.loads(narr_path.read_text())
    attr_data = json.loads(attr_path.read_text())
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        print(f"  [error] video not found: {video_path}", flush=True)
        return 0, 0, 0

    chunks = attr_data.get("chunks", [])
    narrative_chunks = {c["chunk_id"]: c for c in narrative.get("chunks", [])}
    total = len(chunks)
    frames_dir = video_dir / "frames"

    # Pick chunks needing processing
    frames_to_extract: list[tuple[int, int, int, Path, float]] = []
    for chunk_idx, chunk in enumerate(chunks):
        if not overwrite and chunk.get("static_index_text"):
            continue
        cid = chunk["chunk_id"]
        nchunk = narrative_chunks.get(cid)
        if not nchunk:
            continue
        positions = select_frame_positions(nchunk.get("frame_timestamps", []))
        if not positions:
            continue
        for i, (local_idx, ts) in enumerate(positions):
            fpath = frames_dir / f"{cid:04d}_{i}.jpg"
            frames_to_extract.append((chunk_idx, i, local_idx, fpath, ts))

    if not frames_to_extract:
        done = sum(1 for c in chunks if c.get("static_index_text"))
        return done, total, 0

    # Expected frame count per chunk
    expected_per_chunk: dict[int, int] = {}
    for chunk_idx, _, _, _, _ in frames_to_extract:
        expected_per_chunk[chunk_idx] = expected_per_chunk.get(chunk_idx, 0) + 1

    # Per-chunk dedup state: each chunk gets a lock + (dHashes, SigLIP vecs) of frames kept so far.
    # First frame past the lock wins; later near-identical / semantically similar frames are dropped before VLM.
    chunk_hash_state: dict[int, dict] = {
        ci: {"lock": asyncio.Lock(), "hashes": [], "embs": []} for ci in expected_per_chunk
    }

    # Pipeline: ffmpeg in thread pool, VLM kicks off per-frame as soon as extraction finishes
    loop = asyncio.get_running_loop()

    def _siglip_encode_sync(p: Path):
        if siglip is None:
            return None
        try:
            vecs = siglip.encode_images([str(p)])
            return vecs[0]  # (D,) L2-normalized
        except Exception:
            return None

    async def extract_then_vlm(meta):
        chunk_idx, i, _local_idx, fpath, ts = meta
        try:
            status = await loop.run_in_executor(
                ffmpeg_executor, extract_frame_from_video, video_path, ts, fpath
            )
            if status == "failed":
                return chunk_idx, i, None
            if status == "blank":
                try:
                    fpath.unlink(missing_ok=True)
                except Exception:
                    pass
                return chunk_idx, i, "BLANK"
            # Compute dHash (cheap, CPU)
            h = await loop.run_in_executor(ffmpeg_executor, compute_dhash, fpath)
            st = chunk_hash_state[chunk_idx]
            # Stage 1: dHash check under lock. If passes, also SigLIP check before releasing.
            async with st["lock"]:
                if h is not None and any(
                    dhash_hamming(h, prev) <= DEDUP_HAMMING_THRESHOLD
                    for prev in st["hashes"]
                ):
                    try:
                        fpath.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return chunk_idx, i, "DUP"
                # Stage 2: SigLIP cos-sim check (only if model loaded). Computed under lock —
                # per-chunk serialization is fine since avg chunk has only ~5 frames.
                emb = None
                if siglip is not None:
                    emb = await loop.run_in_executor(ffmpeg_executor, _siglip_encode_sync, fpath)
                    if emb is not None and st["embs"]:
                        # All vectors are L2-normalized → cos = dot product
                        prev_mat = np.stack(st["embs"])  # (k, D)
                        sims = prev_mat @ emb  # (k,)
                        if float(sims.max()) >= SIGLIP_COS_THRESHOLD:
                            try:
                                fpath.unlink(missing_ok=True)
                            except Exception:
                                pass
                            return chunk_idx, i, "DUP"
                if h is not None:
                    st["hashes"].append(h)
                if emb is not None:
                    st["embs"].append(emb)
            attrs = await extract_one(session, sem, fpath, urls.next(), api_model)
            return chunk_idx, i, attrs
        except asyncio.CancelledError:
            raise  # propagate true cancellations (Ctrl-C, shutdown)
        except Exception as e:
            print(f"  [skip frame] c{chunk_idx} f{i}: [{type(e).__name__}] {e}", flush=True)
            return chunk_idx, i, None

    tasks = [extract_then_vlm(meta) for meta in frames_to_extract]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    vlm_calls = sum(1 for r in results if not isinstance(r, BaseException) and r[2] not in (None, "BLANK", "DUP"))

    # Group results by chunk:
    #   - "ok" frames contribute attrs + raw paths (in arrival order — renumbered later)
    #   - "BLANK" / "DUP" / None counted separately but contribute nothing to attrs/paths
    chunk_to_attrs: dict[int, list[dict]] = {}
    chunk_to_paths: dict[int, list[str]] = {}
    chunk_to_blank: dict[int, int] = {}
    chunk_to_dup: dict[int, int] = {}
    for r in results:
        if isinstance(r, BaseException):
            print(f"  [skip frame] exception in gather: [{type(r).__name__}] {r}", flush=True)
            continue
        chunk_idx, i, payload = r
        if payload is None:
            continue  # truly failed — chunk will stay partial
        cid = chunks[chunk_idx]["chunk_id"]
        if payload == "BLANK":
            chunk_to_blank[chunk_idx] = chunk_to_blank.get(chunk_idx, 0) + 1
            continue
        if payload == "DUP":
            chunk_to_dup[chunk_idx] = chunk_to_dup.get(chunk_idx, 0) + 1
            continue
        chunk_to_attrs.setdefault(chunk_idx, []).append(payload)
        chunk_to_paths.setdefault(chunk_idx, []).append(f"frames/{cid:04d}_{i}.jpg")

    # A chunk is "complete" when n_ok + n_blank + n_dup == expected.
    # Renumber kept frame files to contiguous 0..k-1 (DUP holes filled).
    def _renumber_chunk_files(cid: int, raw_paths: list[str]) -> list[str]:
        new_paths = []
        for new_i, old_rel in enumerate(raw_paths):
            new_rel = f"frames/{cid:04d}_{new_i}.jpg"
            if old_rel == new_rel:
                new_paths.append(new_rel)
                continue
            old_abs = video_dir / old_rel
            new_abs = video_dir / new_rel
            if old_abs.exists():
                try:
                    if new_abs.exists():
                        new_abs.unlink()
                    old_abs.rename(new_abs)
                except Exception:
                    new_paths.append(old_rel)
                    continue
            new_paths.append(new_rel)
        return new_paths

    updated = 0
    partial_skipped = 0
    blank_chunks = 0
    dedup_total = 0
    ledger_lines = []
    ts_now = int(time.time())
    touched = set(chunk_to_attrs.keys()) | set(chunk_to_blank.keys()) | set(chunk_to_dup.keys())
    for chunk_idx in touched:
        n_attrs = len(chunk_to_attrs.get(chunk_idx, []))
        n_blank = chunk_to_blank.get(chunk_idx, 0)
        n_dup = chunk_to_dup.get(chunk_idx, 0)
        if n_attrs + n_blank + n_dup < expected_per_chunk.get(chunk_idx, 0):
            partial_skipped += 1
            continue
        cid = chunks[chunk_idx]["chunk_id"]
        if n_attrs > 0:
            merged = merge_attrs(chunk_to_attrs[chunk_idx])
            idx_text = build_index_text(merged) or "[blank frames]"
            kept_paths = _renumber_chunk_files(cid, chunk_to_paths.get(chunk_idx, []))
        else:
            merged = {"entities": []}
            idx_text = "[blank frames]"
            kept_paths = []
            blank_chunks += 1
        chunks[chunk_idx]["static_attributes"] = merged
        chunks[chunk_idx]["static_index_text"] = idx_text
        chunks[chunk_idx]["frame_paths"] = kept_paths
        dedup_total += n_dup
        updated += 1
        ledger_lines.append(json.dumps({
            "video_id": video_id,
            "chunk_id": cid,
            "sampler": SAMPLER_TAG,
            "n_frames": len(kept_paths) + n_blank,
            "n_dup": n_dup,
            "ts": ts_now,
        }))

    if updated > 0:
        attr_path.write_text(json.dumps(attr_data, ensure_ascii=False, indent=2))
        SAMPLER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLER_LOG_PATH.open("a") as f:
            f.write("\n".join(ledger_lines) + "\n")

    if partial_skipped > 0:
        print(f"  [partial] {video_id}: {partial_skipped} chunks had missing frames → retry next pass",
              flush=True)
    if blank_chunks > 0:
        print(f"  [blank] {video_id}: {blank_chunks} chunks all-blank → marked done with sentinel",
              flush=True)
    if dedup_total > 0:
        print(f"  [dedup] {video_id}: dropped {dedup_total} duplicate frames (dHash≤{DEDUP_HAMMING_THRESHOLD})",
              flush=True)

    done = sum(1 for c in chunks if c.get("static_index_text"))
    return done, total, vlm_calls


def encode_vectors_for_video(video_id: str, embedder, overwrite: bool) -> tuple[bool, str]:
    """BGE-M3 encode narrative + attribute texts → vectors.npz."""
    video_dir = BANK_DIR / video_id
    narr_path = video_dir / "narrative.json"
    attr_path = video_dir / "attributes.json"
    out_path = video_dir / "vectors.npz"
    if not narr_path.exists() or not attr_path.exists():
        return False, "missing source"
    if not overwrite and out_path.exists():
        return True, "skip (vectors.npz exists)"

    narr = json.loads(narr_path.read_text())
    attr = json.loads(attr_path.read_text())
    attr_map = {c["chunk_id"]: c for c in attr.get("chunks", [])}
    narr_chunks = narr.get("chunks", [])
    if not narr_chunks:
        return False, "no chunks"

    narrative_texts, attribute_texts, chunk_ids = [], [], []
    for nc in narr_chunks:
        cid = nc["chunk_id"]
        narrative = (nc.get("narrative", "") or "").strip()
        speech = (nc.get("speech_text", "") or "").strip()
        ac = attr_map.get(cid, {})
        attr_text = (ac.get("static_index_text", "") or "").strip()
        embed_text = f"{narrative}\n\n{speech}" if speech else narrative
        narrative_texts.append(embed_text or "[no narrative]")
        attribute_texts.append(attr_text or "[no attributes]")
        chunk_ids.append(cid)

    n_vecs = embedder.encode(narrative_texts)
    a_vecs = embedder.encode(attribute_texts)
    np.savez_compressed(
        out_path,
        narrative_vecs=n_vecs.astype(np.float32),
        attribute_vecs=a_vecs.astype(np.float32),
        chunk_ids=np.array(chunk_ids, dtype=np.int32),
    )
    return True, f"{len(chunk_ids)} chunks"


async def main_async(args):
    if args.video_ids_file:
        video_ids = load_video_ids_from_file(args.video_ids_file)
    else:
        video_ids = load_long_video_ids(args.shard_id, args.num_shards, args.start, args.end)

    urls = URLPool([u.strip() for u in args.vlm_api_url.split(",") if u.strip()])
    print(f"vLLM URLs ({len(urls.urls)}): {urls.urls}")
    print(f"BGE device: {args.bge_device}")
    print(f"frames/chunk: {args.k_frames}    concurrency: {args.concurrency}")
    print(f"target videos: {len(video_ids)}\n")

    # Lazy-load BGE only if we'll actually need it
    embedder = None
    def get_embedder():
        nonlocal embedder
        if embedder is None:
            from src.clients.embedder import BGEM3Embedder
            print(f"  loading BGE-M3 on {args.bge_device}...", flush=True)
            t0 = time.time()
            embedder = BGEM3Embedder(args.bge_model, device=args.bge_device)
            print(f"  BGE loaded ({time.time()-t0:.1f}s)", flush=True)
        return embedder

    # Eager-load SigLIP if dedup enabled (else None — pipeline skips Stage-2 check)
    siglip = None
    if not args.no_siglip_dedup:
        from src.clients.siglip_embedder import SigLIPEmbedder
        print(f"  loading SigLIP on {args.siglip_device}...", flush=True)
        t0 = time.time()
        siglip = SigLIPEmbedder(args.siglip_model, device=args.siglip_device)
        print(f"  SigLIP loaded ({time.time()-t0:.1f}s)  | cos_sim threshold: {SIGLIP_COS_THRESHOLD}",
              flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 8)
    ffmpeg_executor = ThreadPoolExecutor(max_workers=args.ffmpeg_workers)
    print(f"ffmpeg workers: {args.ffmpeg_workers}")

    t_start = time.time()
    total_vlm = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, vid in enumerate(video_ids):
            t_v = time.time()
            try:
                # Step 1: VLM fill (ffmpeg+VLM pipelined per-frame)
                done, total, calls = await process_video_vlm(
                    session, sem, vid, urls, args.api_model,
                    args.overwrite, args.k_frames, ffmpeg_executor,
                    siglip=siglip,
                )
                total_vlm += calls

                # Step 2: BGE encode (only if attributes fully done or already done before)
                attr_path = BANK_DIR / vid / "attributes.json"
                vec_status = "—"
                if attr_path.exists() and total > 0 and done == total:
                    ok, msg = encode_vectors_for_video(vid, get_embedder(), args.overwrite)
                    vec_status = msg if ok else f"FAILED({msg})"
                elif total == 0:
                    vec_status = "no skeleton"
                else:
                    vec_status = f"partial {done}/{total} → skip vec"

                elapsed = time.time() - t_v
                total_elapsed = (time.time() - t_start) / 60
                print(f"[{i+1}/{len(video_ids)}] {vid}: vlm {done}/{total} ({calls} calls) "
                      f"| vec: {vec_status} | {elapsed:.1f}s, total {total_elapsed:.1f}min, "
                      f"vlm_total {total_vlm}", flush=True)
            except asyncio.CancelledError:
                raise  # propagate true cancellations
            except Exception as e:
                elapsed = time.time() - t_v
                print(f"[{i+1}/{len(video_ids)}] {vid}: VIDEO FAILED "
                      f"[{type(e).__name__}] {e}  | {elapsed:.1f}s — continuing", flush=True)
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-api-url", default="http://127.0.0.1:8001",
                    help="逗号分隔多 URL，调用时 round-robin")
    ap.add_argument("--api-model", default="Qwen3.5-27B")
    ap.add_argument("--bge-model", default="/home2/ycj/Models/BAAI/bge-m3")
    ap.add_argument("--bge-device", default="cuda:5")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="VLM 并发 inflight 请求数")
    ap.add_argument("--ffmpeg-workers", type=int, default=24,
                    help="并发 ffmpeg 进程数（CPU 抽帧线程池）")
    ap.add_argument("--k-frames", type=int, default=6,
                    help="每个 chunk 抽多少帧（短 chunk 自动减少）")
    ap.add_argument("--video-ids-file", default=None,
                    help="可选：从文件读 video ID 列表（每行一个），覆盖 shard/start/end")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="重做所有 chunk（默认只补空 static_index_text 的）")
    ap.add_argument("--siglip-model", default="/home2/ycj/Models/google/siglip-large-patch16-384",
                    help="SigLIP 模型路径（用于 chunk 内语义去重）")
    ap.add_argument("--siglip-device", default="cuda:1",
                    help="SigLIP 运行设备")
    ap.add_argument("--no-siglip-dedup", action="store_true",
                    help="禁用 SigLIP 语义去重，只用 dHash")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
