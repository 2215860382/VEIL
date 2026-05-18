"""Run benchmark evals (MLVU, Video-MME) against memory banks from the given config.

Pipelines (all use Answerer — multimodal, summary + ASR + keyframe images):
  coarse8       : BGE-M3 top-8 → Answerer (no cross-encoder)
  coarse64      : BGE-M3 top-64 → Answerer
  rerank_rag8   : BGE-M3 coarse → reranker top-8 → Answerer
  veil_coarse8  : VEIL loop, coarse top-8 per iter (no reranker)
  veil_coarse64 : VEIL loop, coarse top-64 per iter (no reranker)
  veil_rerank8  : VEIL loop, coarse pool → reranker top-8 per iter
  *_27b         : same as above but answerer bound to --vlm-api-url (Qwen3.5-27B); planner/verifier use --llm-api-url
  direct_27b    : same frames as ``direct``, but **multimodal Qwen3.5-27B over OpenAI-compatible API** (requires ``--vlm-api-url`` + ``--vlm-api-model``; compute runs on the vLLM server GPU, not ``--vlm-gpu``).

Usage:
    cd /home2/ycj/Project/VEIL
    PYTHONPATH=. python experiments/run_experiments.py \\
        --config configs/mlvu_memory_bank.yaml \\
        --pipelines coarse8 coarse64 rerank_rag8 veil_coarse8 veil_rerank8 \\
        --filter-from outputs/results/mlvu/some_subset.jsonl \\
        --vlm-gpu cuda:0 --bge-gpu cuda:3 --llm-gpu cuda:1 \\
        --out outputs/results/mlvu/experiments_run.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger

log = get_logger("run_experiments")

ALL_PIPELINES = (
    "direct",
    "coarse8", "coarse64",
    "rerank_rag8", "veil_coarse8", "veil_coarse64", "veil_rerank8",
    # ablation: swap answerer → 27B text LLM (API via --llm-api-url)
    "direct_27b", "coarse8_27b", "coarse16_27b", "coarse64_27b", "rerank_rag8_27b",
    "veil_coarse8_27b", "veil_coarse64_27b", "veil_rerank8_27b",
    # ablation: flat retrieval at matched chunk counts
    "coarse24_27b",
    # default VEIL with 27B text answerer (subq decomp + unified planner)
    "veil_27b",
    # clearer aliases for paper ablations
    "veil_rubric_repair_dynamic_27b",
    # ablation: force one option-grounded sub-question per answer option in iter0
    "veil_option4_init_27b",
    "veil_option4_subq_27b",
    # oracle upper bound (early stop when gold answerer matches)
    "veil_oracle_27b",
    # ablation: swap cross-encoder → LLM listwise reranker
    "llm_rerank8",
)

_DYNAMIC_COARSE_27B_RE = re.compile(r"^coarse(\d+)_27b(?:_rubric)?$")
_VEIL_VL_ANS      = frozenset({"veil_coarse8", "veil_coarse64", "veil_rerank8"})
_VEIL_27B_ANS     = frozenset({"veil_coarse8_27b", "veil_coarse64_27b", "veil_rerank8_27b",
                                "veil_27b",
                                "veil_rubric_repair_dynamic_27b",
                                "veil_option4_init_27b",
                                "veil_option4_subq_27b",
                                "veil_oracle_27b"})
_VEIL_PIPES       = _VEIL_VL_ANS | _VEIL_27B_ANS
# 27B answerer pipelines — use Answerer(vlm) so the 27B VLM sees keyframes too.
_27B_ANS_PIPES    = frozenset({"coarse8_27b", "coarse16_27b", "coarse24_27b", "coarse64_27b",
                               "rerank_rag8_27b"}) | _VEIL_27B_ANS
_LLM_RERANK_PIPES = frozenset({"llm_rerank8"})


def _dynamic_coarse_27b(pipeline: str) -> tuple[int, bool] | None:
    """Return (top_k, rubric_rerank) for dynamic coarse{N}_27b[_rubric] pipelines."""
    m = _DYNAMIC_COARSE_27B_RE.match(pipeline)
    if not m:
        return None
    return int(m.group(1)), pipeline.endswith("_rubric")


# ── Sample loading ─────────────────────────────────────────────────────────────

def load_samples(cfg: dict, filter_video_ids: set | None = None):
    bench = cfg["benchmark"]["name"]
    if bench == "mlvu":
        from src.dataloader.mlvu import load_mlvu
        b = cfg["benchmark"]
        samples = load_mlvu(
            json_dir=b["json_dir"],
            video_dir=b["video_dir"],
            json_files=b["json_files"],
        )
    elif bench == "videomme":
        from src.dataloader.videomme import load_videomme
        b = cfg["benchmark"]
        samples = load_videomme(
            parquet_path=b["parquet_path"],
            video_dir=b["video_dir"],
            duration_groups=b.get("duration_groups"),
        )
    else:
        raise ValueError(f"Unknown benchmark: {bench}")
    if filter_video_ids is not None:
        samples = [s for s in samples if s.video_id in filter_video_ids]
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--memory-dir",  default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--pipelines",   nargs="+", default=list(ALL_PIPELINES),
                    help="Pipeline names; also accepts dynamic coarse{N}_27b and coarse{N}_27b_rubric")
    ap.add_argument("--filter-from", default=None,
                    help="JSONL with video_id keys; only run those videos")
    ap.add_argument("--sample-start", type=int, default=None,
                    help="Optional inclusive lower bound on sample_idx")
    ap.add_argument("--sample-end",   type=int, default=None,
                    help="Optional exclusive upper bound on sample_idx")
    ap.add_argument("--vlm-gpu",        default="cuda:0")
    ap.add_argument("--bge-gpu",        default="cuda:3")
    ap.add_argument("--llm-gpu",        default="cuda:1")
    ap.add_argument("--siglip-gpu",     default=None,
                    help="GPU for SigLIP visual scoring in coarse/veil. If omitted, uses the same device as --bge-gpu "
                         "(dual-path fusion when memory chunks have v_visual).")
    ap.add_argument("--reranker-gpu",   default=None,
                    help="GPU for BGE cross-encoder reranker. If omitted, uses the same device as --bge-gpu.")
    ap.add_argument("--no-siglip", action="store_true",
                    help="Disable SigLIP (semantic-only coarse scores) even when chunks contain v_visual.")
    ap.add_argument("--no-keyframes", action="store_true",
                    help="Do not load keyframe images for the answerer (text-only evidence).")
    ap.add_argument("--text-alpha",  type=float, default=0.6,
                    help="Weight of semantic (BGE) path in dual-path fusion (1.0 = semantic only)")
    ap.add_argument("--vlm-model",      default=None)
    ap.add_argument("--vlm-api-url",   default=None,
                    help="vLLM API base URL(s) for VLM API mode; comma-separated = round-robin across workers")
    ap.add_argument("--vlm-api-model", default=None,
                    help="Model name override for --vlm-api-url")
    ap.add_argument("--llm-model",      default=None)
    ap.add_argument("--use-vllm",    action="store_true",
                    help="Use vLLM backend for text LLM (faster inference)")
    ap.add_argument("--llm-api-url",  default=None,
                    help="OpenAI-compatible API base URL(s); comma-separated = round-robin (e.g. http://a:8778,http://b:8779)")
    ap.add_argument("--llm-api-model", default=None,
                    help="Served model id for --llm-api-url (default: Qwen3.5-27B)")
    ap.add_argument("--answer-evidence-k", type=int, default=None,
                    help="Cap evidence blocks passed to the final answerer (e.g. 8 for top-8 plot layer after coarse-64 retrieval)")
    ap.add_argument("--max-frames",  type=int, default=None,
                    help="Override frame_sampling.max_frames from config (e.g. 32 or 128)")
    ap.add_argument("--direct-max-new-tokens", type=int, default=None,
                    help="Override direct_video_qa.max_new_tokens for direct / direct_27b (API multimodal often needs ≥512)")
    ap.add_argument("--wait-banks",  action="store_true")
    ap.add_argument("--wait-total",  type=int, default=None)
    ap.add_argument("--workers",     type=int, default=1,
                    help="Parallel workers for pipeline execution (safe with API-mode LLM)")
    args = ap.parse_args()
    unknown = [
        p for p in args.pipelines
        if p not in ALL_PIPELINES and _dynamic_coarse_27b(p) is None
    ]
    if unknown:
        raise ValueError(f"Unknown pipeline(s): {unknown}")

    cfg   = load_config(args.config)
    bench = cfg["benchmark"]["name"]
    out_root = Path(cfg.get("paths", {}).get("outputs_root", "outputs"))

    active_preview = set(args.pipelines)
    if "direct_27b" in active_preview and not args.vlm_api_url:
        raise ValueError(
            "direct_27b: pass --vlm-api-url (and --vlm-api-model) pointing at your multimodal "
            "Qwen3.5-27B vLLM server; frames are sent as image_url messages."
        )

    _ret = cfg.get("retrieval", {})
    coarse_top_k = int(_ret.get("coarse_top_k", 64))
    rerank_top_k = int(_ret.get("rerank_top_k", 8))
    _dvq = cfg.get("direct_video_qa", {})
    direct_max_tokens = int(
        args.direct_max_new_tokens
        if args.direct_max_new_tokens is not None
        else _dvq.get("max_new_tokens", 192)
    )
    _vl = cfg.get("veil_loop", {})
    veil_max_iter = int(_vl.get("max_iter", 3))
    veil_dedup = float(_vl.get("dedup_threshold", 0.85))

    memory_dir = Path(args.memory_dir) if args.memory_dir else \
                 out_root / "memory" / f"{bench}_fixed"
    default_out_dir = Path(cfg.get("eval", {}).get("output_dir") or (out_root / "results" / bench))
    out_path   = Path(args.out) if args.out else \
                 default_out_dir / "experiments_fixed.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter video ids
    filter_vids: set | None = None
    if args.filter_from:
        filter_vids = set()
        for line in Path(args.filter_from).open():
            try: filter_vids.add(json.loads(line)["video_id"])
            except: pass
        log.info("filter: %d video_ids from %s", len(filter_vids), args.filter_from)

    # Optionally wait for all banks
    if args.wait_banks:
        n_needed = args.wait_total or (len(filter_vids) if filter_vids else 0)
        log.info("waiting for %d banks in %s ...", n_needed, memory_dir)
        while True:
            built   = {p.stem for p in memory_dir.glob("*.json")}
            missing = (filter_vids - built) if filter_vids else set()
            n_built = len(filter_vids) - len(missing) if filter_vids else len(built)
            log.info("  %d/%d banks ready", n_built, n_needed)
            if n_built >= n_needed:
                break
            time.sleep(60)

    samples = load_samples(cfg, filter_vids)
    if args.sample_start is not None:
        samples = [s for s in samples if int(s.sample_idx) >= args.sample_start]
    if args.sample_end is not None:
        samples = [s for s in samples if int(s.sample_idx) < args.sample_end]
    log.info("loaded %d samples (%d videos)", len(samples),
             len({s.video_id for s in samples}))

    # Resume
    done_keys: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try: done_keys.add(json.loads(line)["key"])
            except: pass
    log.info("already done: %d records", len(done_keys))

    # ── Model loading ──────────────────────────────────────────────────────────
    active = set(args.pipelines)
    dynamic_coarse_27b = {p for p in active if _dynamic_coarse_27b(p) is not None}
    _RERANKER_USERS = frozenset({"rerank_rag8", "veil_rerank8", "rerank_rag8_27b", "veil_rerank8_27b"})
    needs_direct      = "direct" in active or "direct_27b" in active
    needs_vlm         = bool(active - {"direct"})
    needs_bge         = bool(active - {"direct", "direct_27b"})
    needs_reranker    = bool(_RERANKER_USERS & active)
    needs_llm         = bool(((_VEIL_PIPES | _LLM_RERANK_PIPES) & active) or any(p.endswith("_rubric") for p in dynamic_coarse_27b))
    needs_27b_ans     = bool((_27B_ANS_PIPES & active) or dynamic_coarse_27b)
    needs_siglip      = bool(needs_bge and not args.no_siglip)
    siglip_device     = (args.siglip_gpu if args.siglip_gpu is not None else args.bge_gpu) if needs_siglip else None

    vlm = embedder = reranker = llm = llm_reranker = vl_answerer = text_answerer_27b = siglip = None

    if needs_vlm:
        t0 = time.time()
        from src.models.vlm_client import VLMClient
        from src.reasoning.answerer import Answerer
        vlm_api_url = getattr(args, "vlm_api_url", None)
        vlm_model   = args.vlm_model or cfg["models"]["vlm"]["model_path"]
        if vlm_api_url:
            log.info("loading VLM via API %s ...", vlm_api_url)
            vlm = VLMClient(model_path=vlm_model, api_url=vlm_api_url,
                            api_model=getattr(args, "vlm_api_model", None))
        else:
            log.info("loading VLM %s on %s ...", vlm_model, args.vlm_gpu)
            vlm = VLMClient(model_path=vlm_model, device=args.vlm_gpu)
        vl_answerer = Answerer(vlm)
        log.info("  VLM ready (%.1fs)", time.time() - t0)

    if needs_bge:
        from src.models.embedder import BGEM3Embedder
        log.info("loading BGE-M3 on %s ...", args.bge_gpu)
        t0 = time.time()
        embedder = BGEM3Embedder(
            model_path=cfg["models"]["embedder"]["model_path"],
            use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
            device=args.bge_gpu,
        )
        log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    if needs_reranker:
        from src.models.reranker import BGEReranker
        reranker_dev = args.reranker_gpu if args.reranker_gpu is not None else args.bge_gpu
        log.info("loading BGE reranker on %s ...", reranker_dev)
        t0 = time.time()
        reranker = BGEReranker(
            model_path=cfg["models"]["reranker"]["model_path"],
            use_fp16=cfg["models"]["reranker"].get("use_fp16", True),
            device=reranker_dev,
        )
        log.info("  BGE reranker ready (%.1fs)", time.time() - t0)

    shared_api_llm = None
    if args.llm_api_url and (needs_llm or needs_27b_ans):
        from src.models.llm_client import LLMClient
        api_model = args.llm_api_model or "Qwen3.5-27B"
        log.info("loading shared API LLM %s @ %s ...", api_model, args.llm_api_url)
        t0 = time.time()
        shared_api_llm = LLMClient(
            model_path=api_model,
            api_url=args.llm_api_url,
            api_model=api_model,
        )
        log.info("  API LLM ready (%.1fs)", time.time() - t0)

    if needs_llm:
        from src.models.llm_client import LLMClient
        from src.models.reranker import LLMReranker
        if shared_api_llm is not None:
            llm = shared_api_llm
            log.info("planner / LLM-rerank use API LLM")
        else:
            llm_model = args.llm_model or cfg["models"]["llm"]["model_path"]
            use_vllm = getattr(args, "use_vllm", False)
            log.info("loading LLM %s on %s (backend=%s) ...",
                     llm_model, args.llm_gpu, "vllm" if use_vllm else "transformers")
            t0 = time.time()
            llm = LLMClient(
                model_path=llm_model,
                device=args.llm_gpu,
                dtype=cfg["models"]["llm"].get("dtype", "bfloat16"),
                attn_impl=cfg["models"]["llm"].get("attn_impl", "sdpa"),
                use_vllm=use_vllm,
            )
            log.info("  LLM ready (%.1fs)", time.time() - t0)
        llm_reranker = LLMReranker(llm)

    if needs_27b_ans:
        if vlm is None:
            raise RuntimeError("*_27b pipelines require --vlm-api-url (VLM for multimodal answering)")
        text_answerer_27b = Answerer(vlm)
        log.info("27B answerer = Answerer(vlm)")

    if needs_siglip and siglip_device:
        from src.models.siglip_embedder import SigLIPEmbedder
        siglip_model = "/home2/ycj/Models/google/siglip-large-patch16-384"
        log.info("loading SigLIP on %s (dual-path coarse/veil when v_visual present) ...", siglip_device)
        t0 = time.time()
        siglip = SigLIPEmbedder(model_path=siglip_model, device=siglip_device)
        log.info("  SigLIP ready (%.1fs)", time.time() - t0)

    # ── Thread safety (for --workers > 1) ─────────────────────────────────────
    # GPU encoder models (BGE-M3, SigLIP, cross-encoder) are NOT thread-safe.
    # Wrap them with a shared lock so only one thread runs GPU inference at a time.
    # LLMClient (HTTP API) is thread-safe and does not need a lock.
    _gpu_lock = threading.Lock()

    class _LockedModel:
        """Serializes GPU model calls across threads via a shared lock."""
        def __init__(self, model):
            self._m = model
        def __getattr__(self, name):
            attr = getattr(self._m, name)
            if callable(attr):
                def _locked(*a, **kw):
                    with _gpu_lock:
                        return attr(*a, **kw)
                return _locked
            return attr

    if embedder: embedder = _LockedModel(embedder)
    if siglip:   siglip   = _LockedModel(siglip)
    if reranker: reranker = _LockedModel(reranker)

    # ── Caches ─────────────────────────────────────────────────────────────────
    _frame_cache: dict = {}
    _bank_cache:  dict = {}
    _cache_lock = threading.Lock()

    def get_frames(video_path: str):
        with _cache_lock:
            if video_path in _frame_cache:
                return _frame_cache[video_path]
        from src.memory.core.sample_frames import sample_frames
        fs = cfg.get("frame_sampling", {})
        sv = sample_frames(video_path,
                           fps=fs.get("fps", 1),
                           max_frames=args.max_frames or fs.get("max_frames", 256),
                           resolution=fs.get("resolution", 448))
        with _cache_lock:
            _frame_cache[video_path] = sv.frames
        return sv.frames

    def get_bank(video_id: str):
        with _cache_lock:
            if video_id in _bank_cache:
                return _bank_cache[video_id]
        bp = memory_dir / f"{video_id}.json"
        if not bp.exists():
            return None
        from src.memory.core.schema import MemoryBank
        bank = MemoryBank.load(bp)
        with _cache_lock:
            _bank_cache[video_id] = bank
        return bank

    # ── Pipeline dispatch ──────────────────────────────────────────────────────
    from src.pipelines.direct_video_qa import run_direct_video_qa
    from src.pipelines.coarse_rag import run_coarse_rag
    from src.pipelines.rerank_rag import run_rerank_rag
    from src.pipelines.veil import run_veil
    from src.eval.parse_answer import parse_letter

    aek = args.answer_evidence_k

    def run_pipeline(pipeline: str, s):
        if pipeline in ("direct", "direct_27b"):
            frames = get_frames(s.video_path)
            return run_direct_video_qa(
                frames, s.question, s.candidates, vlm, max_new_tokens=direct_max_tokens
            ), None

        bank = get_bank(s.video_id)
        if bank is None:
            return None, "bank_missing"

        va = args.text_alpha
        kf_dir = None if getattr(args, "no_keyframes", False) else memory_dir
        dyn_coarse = _dynamic_coarse_27b(pipeline)
        if dyn_coarse is not None:
            top_k, use_rubric_rerank = dyn_coarse
            rubric = None
            if use_rubric_rerank:
                from src.reasoning.verifier import get_rubric_dict
                rubric = get_rubric_dict(s.question, s.question_type)
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, text_answerer_27b,
                               top_k=top_k, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek, rubric_rerank=use_rubric_rerank,
                               rubric=rubric, llm=llm)
        elif pipeline == "coarse8":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, vl_answerer,
                               top_k=8, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "coarse64":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, vl_answerer,
                               top_k=64, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "rerank_rag8":
            r = run_rerank_rag(s.question, s.candidates, bank, embedder, reranker, vl_answerer,
                               coarse_top_k=coarse_top_k, rerank_top_k=rerank_top_k, siglip=siglip, text_alpha=va,
                               keyframe_dir=kf_dir)
        elif pipeline == "veil_coarse8":
            r = run_veil(s.question, s.candidates, bank, embedder, vl_answerer, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=8, final_top_k=8,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        elif pipeline == "veil_coarse64":
            r = run_veil(s.question, s.candidates, bank, embedder, vl_answerer, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=coarse_top_k, final_top_k=rerank_top_k,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        elif pipeline == "veil_rerank8":
            r = run_veil(s.question, s.candidates, bank, embedder, vl_answerer, llm,
                         task_type=s.question_type,
                         reranker=reranker, coarse_top_k=coarse_top_k, final_top_k=rerank_top_k,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        # ── Ablation: 27B text answerer ────────────────────────────────────────
        elif pipeline == "coarse8_27b":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, text_answerer_27b,
                               top_k=8, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "coarse16_27b":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, text_answerer_27b,
                               top_k=16, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "coarse24_27b":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, text_answerer_27b,
                               top_k=24, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "coarse64_27b":
            r = run_coarse_rag(s.question, s.candidates, bank, embedder, text_answerer_27b,
                               top_k=64, siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                               answer_evidence_cap=aek)
        elif pipeline == "rerank_rag8_27b":
            r = run_rerank_rag(s.question, s.candidates, bank, embedder, reranker, text_answerer_27b,
                               coarse_top_k=coarse_top_k, rerank_top_k=rerank_top_k, siglip=siglip, text_alpha=va,
                               keyframe_dir=kf_dir)
        elif pipeline == "veil_coarse8_27b":
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=8, final_top_k=8,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        elif pipeline == "veil_coarse64_27b":
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=coarse_top_k, final_top_k=rerank_top_k,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        elif pipeline == "veil_rerank8_27b":
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=reranker, coarse_top_k=coarse_top_k, final_top_k=rerank_top_k,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek)
        elif pipeline in {"veil_27b", "veil_rubric_repair_dynamic_27b"}:
            # Default VEIL: iter0 subq decomposition + iter≥1 unified planner with
            # option_status / failed criteria signals + keyword/fallback broadcast.
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=8, final_top_k=8,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek,
                         rubric_rerank=True)
        elif pipeline in {"veil_option4_init_27b", "veil_option4_subq_27b"}:
            # Iter0 forces one deterministic option-grounded query per answer option.
            # Later repair rounds still use the rubric-aware dynamic planner.
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=8, final_top_k=8,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek,
                         rubric_rerank=True, force_option_subquestions=True)
        elif pipeline == "veil_oracle_27b":
            # Oracle upper bound: stop iterating once gold answerer matches.
            r = run_veil(s.question, s.candidates, bank, embedder, text_answerer_27b, llm,
                         task_type=s.question_type,
                         reranker=None, coarse_top_k=8, final_top_k=8,
                         max_iter=veil_max_iter, dedup_thresh=veil_dedup,
                         siglip=siglip, text_alpha=va, keyframe_dir=kf_dir,
                         answer_evidence_cap=aek,
                         rubric_rerank=True,
                         use_oracle=True, gold_answer=s.answer)
        # ── Ablation: LLM listwise reranker ───────────────────────────────────
        elif pipeline == "llm_rerank8":
            r = run_rerank_rag(s.question, s.candidates, bank, embedder, llm_reranker, vl_answerer,
                               coarse_top_k=coarse_top_k, rerank_top_k=rerank_top_k)
        else:
            raise ValueError(pipeline)
        return r, None

    # ── Run loop ───────────────────────────────────────────────────────────────
    out_fh    = out_path.open("a")
    _write_lock = threading.Lock()
    _stats_lock = threading.Lock()

    def append_record(rec: dict):
        with _write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

    total      = len(samples) * len(args.pipelines)
    done_count = [len(done_keys)]   # mutable counter shared across threads
    correct_by: dict = {}

    def process_item(pipeline: str, s) -> None:
        key = f"{bench}|{s.video_id}|{s.sample_idx}|{pipeline}"

        t0 = time.time()
        try:
            result, err = run_pipeline(pipeline, s)
        except Exception as e:
            log.error("[%s/%s/%s] ERROR: %s", s.video_id, pipeline,
                      s.question_type, e, exc_info=True)
            result, err = None, str(e)
        elapsed = time.time() - t0

        if result is None:
            pred_letter, pred_text, correct = "", "", False
        else:
            raw_ans = result.get("answer", "")
            pred_letter = raw_ans[:1].upper() if raw_ans else \
                          (parse_letter(result.get("raw", ""), len(s.candidates)) or "")
            idx = ord(pred_letter) - ord("A") if pred_letter else -1
            pred_text = s.candidates[idx] if 0 <= idx < len(s.candidates) else ""
            correct = (pred_text == s.answer)

        trace = result.get("trace_iters") if result else None
        rec = {
            "key":              key,
            "benchmark":        bench,
            "question_type":    s.question_type,
            "video_id":         s.video_id,
            "sample_idx":       s.sample_idx,
            "pipeline":         pipeline,
            "question":         s.question,
            "candidates":       s.candidates,
            "gold":             s.answer,
            "pred_letter":      pred_letter,
            "pred_text":        pred_text,
            "correct":          correct,
            "evidence_chunk_ids": (result or {}).get("evidence_chunk_ids", []),
            "trace_iters":      trace,
            "elapsed":          round(elapsed, 2),
            "error":            err,
        }
        append_record(rec)

        with _stats_lock:
            done_count[0] += 1
            k = (pipeline, s.question_type)
            correct_by.setdefault(k, []).append(int(correct))
            n_done = done_count[0]

        n_iter = len(trace) if trace else 0
        log.info("[%d/%d] %-14s | %-12s | %-16s | %s %s (iter=%d) %.1fs",
                 n_done, total, pipeline, s.question_type, s.video_id,
                 pred_letter, "✓" if correct else "✗", n_iter, elapsed)

    tasks = [
        (pipeline, s)
        for s in samples
        for pipeline in args.pipelines
        if f"{bench}|{s.video_id}|{s.sample_idx}|{pipeline}" not in done_keys
    ]
    log.info("tasks to run: %d  (workers=%d)", len(tasks), args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, p, s) for p, s in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("worker exception: %s", exc)

    out_fh.close()

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n=== Accuracy ===")
    qtypes = sorted({qt for _, qt in correct_by})
    print(f"{'Pipeline':<16}" + "".join(f"{qt[:8]:>10}" for qt in qtypes) + f"{'Overall':>10}")
    for pipeline in args.pipelines:
        row = f"{pipeline:<16}"
        all_c = []
        for qt in qtypes:
            c = correct_by.get((pipeline, qt), [])
            acc = sum(c) / len(c) if c else float("nan")
            row += f"{acc*100:>9.1f}%"
            all_c.extend(c)
        ov = sum(all_c) / len(all_c) if all_c else float("nan")
        row += f"{ov*100:>9.1f}%"
        print(row)

    log.info("done — results in %s", out_path)


if __name__ == "__main__":
    main()
