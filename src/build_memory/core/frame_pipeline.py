"""Shared frame pipeline: extract → drop blank → drop blurry → SigLIP group →
persist all post-cleanup frames + one keyframe per chunk (Binlu-aligned).

Output directories under each ``{out_dir}/``:
* ``frames_raw/{cid:04d}/{i:04d}.jpg`` — every post-cleanup frame for the chunk
  (used as narrative VLM input). Always at original resolution.
* ``keyframes_origin/{cid:04d}_0.jpg`` — exactly one keyframe per chunk at
  original resolution (canonical).
* ``keyframes_resized/{cid:04d}_0.jpg`` — same file, resized so longer
  side ≤ ``frame_max_dim`` (eval-time VLM input cost control).

Exposes both the full pipeline (``clean_and_group_frames`` — used by
``build_single_similarity.py`` for similarity-grouped single-granularity banks)
and the raw building blocks (``extract_frames``, ``group_frames``,
``select_sharpest`` — used by ``build_multi_pyramid.py`` for fixed-time pyramid
banks).  Downstream code (``experiments/core/_keyframes.py``) reads
keyframes via ``glob("{keyframes_origin|keyframes_resized}/{cid:04d}_*.jpg")``.

Decisions (Binlu-aligned, see Binlu/videolens/build/build_index_multiscale.py
:_pick_keyframe_idx):
* No dedup. Pre-dedup (dHash window=10 before SigLIP) and per-chunk dHash+SigLIP
  dedup both removed — recurring shots and near-identical frames all kept,
  matching Binlu's "every 1fps frame survives" policy.
* One keyframe per chunk: the group center frame, unless its Laplacian
  variance is below ``KEYFRAME_LAPLACIAN_THRESH`` (=100) in which case the
  sharpest frame in the chunk is used instead. No top-k retention.
* Blank detection by file size (≥3000B = ok) + pixel mean/std check (catches
  near-uniform white/black fades that the size pre-filter misses); cheap.
* Blur detection: ``cv2.Laplacian(gray).var() < blur_threshold``, with a
  fail-safe that disables the filter when it would drop the majority of frames
  (low-texture content like PPT slides or anime).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


# ── Tunables ───────────────────────────────────────────────────────────────────

DEFAULT_FPS              = 1.0
DEFAULT_BLANK_BYTES_MIN  = 3000   # < this = blank/uniform frame, drop.
                                  # Uniform-color frames JPEG-compress to ~1-3 KB
                                  # regardless of darkness; 3000B safely catches
                                  # "all-pixel-the-same" content (e.g. 1601-byte
                                  # all-black outros) while preserving low-texture
                                  # but visible content (~8-10 KB for channel
                                  # logos / fade-in / dim scenes).
DEFAULT_BLUR_THRESHOLD   = 10.0   # Laplacian variance below = blurry, drop.
                                  # Conservative: low-texture-but-sharp content
                                  # (white-background, PPT, anime) sits around
                                  # 20–30, severe motion blur is <5.
BLUR_FAILSAFE_RATIO      = 0.5    # if blur filter would drop > this fraction of
                                  # frames, the threshold is wrong for this clip
                                  # → skip the blur pass entirely.
DEFAULT_GROUP_THETA      = 0.80   # SigLIP cosine threshold for grouping
DEFAULT_GROUP_N_MAX      = 30
DEFAULT_DEDUP_HAMMING    = 4      # legacy; dedup is disabled in the current
DEFAULT_DEDUP_SIGLIP_COS = 0.95   # pipeline (Binlu-aligned). Kept for callers
DHASH_SIZE               = 8      # that re-enable dedup (e.g. multi-pyramid).
KEYFRAME_LAPLACIAN_THRESH = 100.0 # Binlu build_index_multiscale.py default:
                                  # center frame is used as the chunk's single
                                  # keyframe unless its Laplacian variance is
                                  # below this, in which case the sharpest
                                  # frame in the chunk replaces it.


# ── Output schemas ─────────────────────────────────────────────────────────────

@dataclass
class FrameGroup:
    """Raw output of ``group_frames``: one contiguous run of similar frames.

    Lighter than ``CleanedGroup`` — used by ``build_multi_pyramid.py`` and
    internally by ``clean_and_group_frames``.
    """
    frame_indices: List[int]   # indices into the input frame list
    t_start: float
    t_end: float
    center_idx: int            # index of the center/sharpest frame
    size: int


@dataclass
class CleanedGroup:
    """One chunk's worth of cleaned, grouped, persisted frames.

    Two parallel views of this chunk's frames:
      * ``all_*`` — every frame that survived blank+blur filtering. Use these
        to feed the narrative VLM caption — preserves event-sequence detail.
      * ``kept_*`` — the ``all_*`` set further reduced by chunk-local dHash +
        SigLIP dedup. Use these for keyframe selection / visual embedding /
        anywhere a sparse representative set is enough.
    """
    chunk_id: int
    t_start: float
    t_end: float
    # All cleaned (post-blank, post-blur, pre-dedup) frames for this chunk.
    all_frame_paths: List[Path]        # persisted under out_dir/frames_raw/
    all_timestamps: List[float]
    # Dedup-survived representative subset.
    # kept_frame_paths points to keyframes_origin/ (full res, canonical).
    # A parallel resized copy exists under keyframes_resized/ with identical
    # filenames; the loader picks subdir at eval time.
    kept_frame_paths: List[Path]
    kept_timestamps: List[float]
    kept_v_visual: np.ndarray          # (N_kept, D) SigLIP vecs for kept frames
    center_idx: int                    # index into kept_frame_paths


# ── Frame extraction (was similarity.extract_frames) ───────────────────────────

def extract_frames(video_path: str, out_dir: Path, fps: float = DEFAULT_FPS) -> List[Path]:
    """ffmpeg → frame_000001.jpg, frame_000002.jpg, … in out_dir.

    ``-q:v 2`` (≈ JPEG q95) matches Binlu extract_frames.py — keep narrative
    frames near-lossless so the multi-image VLM call sees full detail before
    the 448-side downscale at the API boundary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}", "-q:v", "2",
        str(out_dir / "frame_%06d.jpg"),
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


# ── Frame quality filters ──────────────────────────────────────────────────────

DEFAULT_BLACK_MEAN_THRESH = 8.0   # pixel mean  < this → black/near-black frame
DEFAULT_BLACK_STD_THRESH  = 3.0   # pixel stdev < this → near-uniform (e.g. white fade)


def is_blank(path: Path, min_bytes: int = DEFAULT_BLANK_BYTES_MIN) -> bool:
    """Fast pre-filter: uniform frames compress tiny; reject by file size, no decode."""
    try:
        return path.stat().st_size < min_bytes
    except OSError:
        return True


def is_blank_pixel(
    path: Path,
    mean_thresh: float = DEFAULT_BLACK_MEAN_THRESH,
    std_thresh: float = DEFAULT_BLACK_STD_THRESH,
) -> bool:
    """Pixel-level black/uniform frame detector (requires decode).

    Catches near-black frames that survive the file-size pre-filter (e.g. very
    dark-but-not-pure-black scenes) and near-white uniform frames (fade-to-white).
    """
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    m = float(img.mean())
    s = float(img.std())
    return m < mean_thresh or s < std_thresh


def is_blurry(path: Path, threshold: float = DEFAULT_BLUR_THRESHOLD) -> bool:
    """Laplacian variance ≥ threshold = sharp enough. Returns True on read error."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) < threshold


def select_sharpest(frame_paths: List[str], indices: List[int]) -> int:
    """Pick the frame with highest Laplacian variance (sharpest) from a subset.

    ``frame_paths`` is the full list; ``indices`` are positions into it that
    define the candidate subset. Returns one of the values from ``indices``
    (i.e. a global index into ``frame_paths``).
    """
    import cv2
    best_idx, best_score = indices[0], -1.0
    for fi in indices:
        img = cv2.imread(frame_paths[fi], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if score > best_score:
            best_score, best_idx = score, fi
    return best_idx


# ── dHash dedup ────────────────────────────────────────────────────────────────

def compute_dhash(path: Path, size: int = DHASH_SIZE) -> Optional[str]:
    """8×8 difference hash → 64-char binary string. None on decode failure."""
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


def _prededup_frames(
    paths: List[Path],
    hamming_threshold: int = DEFAULT_DEDUP_HAMMING,
    window: int = 10,
) -> List[int]:
    """Global sequential dHash pre-dedup before SigLIP encoding.

    Compares each frame against the last ``window`` kept frames.  Catches
    long static-scene runs (identical or near-identical consecutive frames)
    without accidentally removing legitimately recurring shots (e.g. anchor
    cut-backs) that are far apart in time.

    Returns indices into ``paths`` to keep.
    """
    kept: List[int] = []
    kept_hashes: List[str] = []
    for i, p in enumerate(paths):
        h = compute_dhash(p)
        if h is None:
            kept.append(i)
            continue
        recent = kept_hashes[-window:]
        if any(dhash_hamming(h, prev) <= hamming_threshold for prev in recent):
            continue
        kept.append(i)
        kept_hashes.append(h)
    return kept


def select_top_k_sharpest(paths: List[Path], k: int = 2) -> List[int]:
    """Return the indices of the k sharpest frames (Laplacian variance), in
    their original time order.

    ``paths`` is a small subset (post-dedup), so the O(N) scan is cheap.
    """
    import cv2
    scores: List[tuple] = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        score = float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else 0.0
        scores.append((score, i))
    top = sorted(scores, reverse=True)[:k]
    return sorted(idx for _, idx in top)   # restore temporal order


def _dedup_chunk(
    paths: List[Path],
    vecs: np.ndarray,                            # (N, D) L2-normed SigLIP for these paths
    hamming_threshold: int = DEFAULT_DEDUP_HAMMING,
    siglip_cos_threshold: Optional[float] = DEFAULT_DEDUP_SIGLIP_COS,
) -> List[int]:
    """Greedy two-gate chunk-local dedup. Returns indices into ``paths`` to KEEP.

    Both gates must reject a candidate vs. ALL already-kept frames for the
    candidate to be dropped. Set ``siglip_cos_threshold=None`` to use dHash only.
    """
    kept: List[int] = []
    kept_hashes: List[str] = []
    kept_vecs: List[np.ndarray] = []
    for i, p in enumerate(paths):
        h = compute_dhash(p)
        if h is None:
            continue
        if kept_hashes and any(dhash_hamming(h, prev) <= hamming_threshold for prev in kept_hashes):
            continue
        if siglip_cos_threshold is not None and kept_vecs:
            sims = np.stack(kept_vecs) @ vecs[i]
            if float(sims.max()) >= siglip_cos_threshold:
                continue
        kept.append(i)
        kept_hashes.append(h)
        kept_vecs.append(vecs[i])
    return kept


# ── SigLIP cosine grouping ─────────────────────────────────────────────────────

def group_frames(
    v_frames: np.ndarray,      # (T, D) L2-normalized SigLIP/CLIP embeddings
    timestamps: List[float],   # length T, seconds
    theta: float = DEFAULT_GROUP_THETA,
    n_max: int = DEFAULT_GROUP_N_MAX,
    min_size: int = 3,
) -> List[FrameGroup]:
    """Greedy cosine-similarity grouping of consecutive frames into "shots".

    Adjacent frames with cosine similarity ≥ ``theta`` are merged into one
    group, up to ``n_max`` frames. Short groups (< ``min_size``) are folded
    into the left neighbour.
    """
    T = len(v_frames)
    if T == 0:
        return []

    groups: List[List[int]] = []
    current: List[int] = [0]

    for t in range(1, T):
        sim = float(v_frames[t] @ v_frames[t - 1])
        if sim >= theta and len(current) < n_max:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)

    merged: List[List[int]] = []
    for g in groups:
        if len(g) < min_size and merged:
            merged[-1].extend(g)
        else:
            merged.append(g)

    result: List[FrameGroup] = []
    for g in merged:
        center = g[len(g) // 2]
        result.append(FrameGroup(
            frame_indices=g,
            t_start=timestamps[g[0]],
            t_end=timestamps[g[-1]] + 1.0,  # +1s so end > start
            center_idx=center,
            size=len(g),
        ))
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _persist_frame(src: Path, dst: Path, max_dim: Optional[int]) -> None:
    """Copy ``src`` JPEG to ``dst``. If ``max_dim`` is given, also resize so the
    longer side is at most ``max_dim`` (preserving aspect ratio; LANCZOS + q=85).
    """
    if max_dim is None:
        shutil.copyfile(src, dst)
        return
    from PIL import Image
    img = Image.open(src)
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img.convert("RGB").save(dst, format="JPEG", quality=85)


def clean_and_group_frames(
    video_path: str,
    out_dir: Path,
    siglip,                                       # SigLIPEmbedder
    fps: float = DEFAULT_FPS,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    blank_bytes_min: int = DEFAULT_BLANK_BYTES_MIN,
    theta: float = DEFAULT_GROUP_THETA,
    n_max: int = DEFAULT_GROUP_N_MAX,
    frame_max_dim: Optional[int] = 448,
    keyframe_laplacian_thresh: float = KEYFRAME_LAPLACIAN_THRESH,
) -> List[CleanedGroup]:
    """Run the shared pre-build frame pipeline for one video (Binlu-aligned).

    Steps:
        1. ffmpeg extract at ``fps`` into a tempdir (-q:v 2)
        2. drop blank frames (file-size + pixel mean/std check)
        3. drop blurry frames (Laplacian variance) — fail-safe disables on
           low-texture clips that would lose the majority of frames
        4. SigLIP-encode survivors
        5. greedy grouping (``group_frames``)
        6. persist the full post-cleanup chunk to
           ``{out_dir}/frames_raw/{cid:04d}/{i:04d}.jpg`` (narrative VLM input)
           — always at original resolution
        7. pick a single keyframe per chunk: group center frame, unless its
           Laplacian variance is below ``keyframe_laplacian_thresh`` in which
           case the sharpest frame in the chunk replaces it. Persisted to
           ``{out_dir}/keyframes_origin/{cid:04d}_0.jpg`` (full res) and
           ``{out_dir}/keyframes_resized/{cid:04d}_0.jpg`` (≤ frame_max_dim).
           ``frame_max_dim`` no longer affects ``frames_raw/``.

    Returns one ``CleanedGroup`` per chunk; chunks with zero post-cleanup
    frames are dropped, and ``chunk_id`` is reassigned to stay contiguous
    from 0.
    """
    out_kf_origin  = out_dir / "keyframes_origin"
    out_kf_resized = out_dir / "keyframes_resized"
    out_raw_dir    = out_dir / "frames_raw"
    out_kf_origin.mkdir(parents=True, exist_ok=True)
    out_kf_resized.mkdir(parents=True, exist_ok=True)
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="veil_frames_") as tmp:
        tmp_dir = Path(tmp)

        # 1. Extract
        raw_paths = extract_frames(video_path, tmp_dir, fps=fps)
        if not raw_paths:
            return []

        # 2. Drop blank: file-size pre-filter (cheap) then pixel mean/std check.
        after_size: List[tuple[int, Path]] = [
            (idx, p) for idx, p in enumerate(raw_paths) if not is_blank(p, blank_bytes_min)
        ]
        non_blank: List[tuple[int, Path]] = [
            (idx, p) for idx, p in after_size if not is_blank_pixel(p)
        ]
        if not non_blank:
            return []

        # 3. Drop blurry by Laplacian variance — with fail-safe: if the
        # threshold would drop the majority of frames, the clip is low-texture
        # (white background, PPT, anime); abandon the blur filter.
        blurry_flags = [is_blurry(p, blur_threshold) for _, p in non_blank]
        drop_ratio = sum(blurry_flags) / len(blurry_flags)
        if drop_ratio > BLUR_FAILSAFE_RATIO:
            survived = non_blank
        else:
            survived = [pair for pair, bl in zip(non_blank, blurry_flags) if not bl]

        if not survived:
            return []

        survived_paths = [p for _, p in survived]
        survived_timestamps = [idx / fps for idx, _ in survived]

        # 4. SigLIP encode (Binlu-aligned: no pre-dedup; all post-cleanup
        #    frames go to the encoder so the chunk grouping sees the full
        #    sequence and recurring shots are not silently merged).
        v_frames = siglip.encode_images([str(p) for p in survived_paths])

        # 5. Group
        groups = group_frames(
            v_frames=np.asarray(v_frames),
            timestamps=survived_timestamps,
            theta=theta,
            n_max=n_max,
        )

        # 6–7. Persist the full cleaned chunk (frames_raw/) and one keyframe
        # per chunk (keyframes_origin / keyframes_resized), with contiguous
        # chunk_id reassignment. No per-chunk dedup, no top-k cap.
        cleaned: List[CleanedGroup] = []
        next_cid = 0
        for g in groups:
            group_paths = [survived_paths[i] for i in g.frame_indices]
            group_ts    = [survived_timestamps[i] for i in g.frame_indices]
            group_vecs  = v_frames[g.frame_indices]
            if not group_paths:
                continue

            cid = next_cid
            next_cid += 1

            # Persist the full cleaned chunk to frames_raw/{cid:04d}/{i:04d}.jpg
            # — always at original resolution (frame_max_dim is keyframe-only).
            raw_chunk_dir = out_raw_dir / f"{cid:04d}"
            raw_chunk_dir.mkdir(parents=True, exist_ok=True)
            all_paths_out: List[Path] = []
            for i, src in enumerate(group_paths):
                dst = raw_chunk_dir / f"{i:04d}.jpg"
                _persist_frame(src, dst, None)
                all_paths_out.append(dst)

            # Pick ONE keyframe: chunk center, unless its Laplacian variance
            # is below the threshold and there's more than one frame to choose
            # from — then use the sharpest frame in the chunk
            # (Binlu/videolens/build/build_index_multiscale.py:_pick_keyframe_idx).
            center_local = len(group_paths) // 2
            kf_local = center_local
            if len(group_paths) > 1:
                center_path = group_paths[center_local]
                import cv2
                cimg = cv2.imread(str(center_path), cv2.IMREAD_GRAYSCALE)
                center_lap = float(cv2.Laplacian(cimg, cv2.CV_64F).var()) if cimg is not None else 0.0
                if center_lap < keyframe_laplacian_thresh:
                    kf_local = select_sharpest(
                        [str(p) for p in group_paths],
                        list(range(len(group_paths))),
                    )

            # Persist the single keyframe under both subdirs (full res + resized).
            name = f"{cid:04d}_0.jpg"
            dst_origin  = out_kf_origin  / name
            dst_resized = out_kf_resized / name
            _persist_frame(group_paths[kf_local], dst_origin,  None)
            _persist_frame(group_paths[kf_local], dst_resized, frame_max_dim)
            kept_paths_out  = [dst_origin]
            kept_ts_out     = [group_ts[kf_local]]
            kept_vecs_out   = group_vecs[[kf_local]]

            cleaned.append(CleanedGroup(
                chunk_id=cid,
                t_start=g.t_start,
                t_end=g.t_end,
                all_frame_paths=all_paths_out,
                all_timestamps=group_ts,
                kept_frame_paths=kept_paths_out,
                kept_timestamps=kept_ts_out,
                kept_v_visual=kept_vecs_out,
                center_idx=0,
            ))

        return cleaned
