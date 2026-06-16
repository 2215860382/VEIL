"""VEIL-27B — iterative evidence retrieval with rubric judgment + rubric rerank.

Answerer / Planner / Verifier all use Qwen3.5-27B via --vlm-api-url / --llm-api-url.

Usage:
    cd /home2/ycj/Project/VEIL
    PYTHONPATH=. python experiments/veil_27b.py \\
        --config configs/mlvu_memory_bank.yaml \\
        --vlm-api-url http://localhost:8000 \\
        --llm-api-url http://localhost:8001 \\
        --bge-gpu cuda:3
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.utils.logging import get_logger
from experiments.core.veil import run_veil

PIPELINE_NAME = "veil_27b"

log = get_logger(PIPELINE_NAME)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--memory-dir",  default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--filter-from", default=None,
                    help="JSONL with video_id keys; only run those videos")
    ap.add_argument("--sample-start", type=int, default=None,
                    help="Inclusive lower bound on sample_idx")
    ap.add_argument("--sample-end",   type=int, default=None,
                    help="Exclusive upper bound on sample_idx")
    ap.add_argument("--vlm-gpu",        default="cuda:0")
    ap.add_argument("--bge-gpu",        default="cuda:3")
    ap.add_argument("--llm-gpu",        default="cuda:1")
    ap.add_argument("--siglip-gpu",     default=None,
                    help="GPU for SigLIP visual scoring; defaults to --bge-gpu")
    ap.add_argument("--embed-api-url",  default=None,
                    help="Embedding service URL (BGE+SigLIP); skips loading locally")
    ap.add_argument("--no-siglip",      action="store_true",
                    help="Disable SigLIP dual-path fusion")
    ap.add_argument("--no-keyframes",   action="store_true",
                    help="Do not load keyframe images for the answerer")
    ap.add_argument("--text-alpha",     type=float, default=0.6,
                    help="Weight of semantic (BGE) path in dual-path fusion")
    ap.add_argument("--vlm-model",      default=None)
    ap.add_argument("--vlm-api-url",    default=None,
                    help="vLLM API base URL(s); comma-separated = round-robin")
    ap.add_argument("--vlm-api-model",  default=None)
    ap.add_argument("--llm-model",      default=None)
    ap.add_argument("--use-vllm",       action="store_true",
                    help="Use vLLM backend for local LLM")
    ap.add_argument("--llm-api-url",    default=None,
                    help="OpenAI-compatible API base URL(s); comma-separated = round-robin")
    ap.add_argument("--llm-api-model",  default=None,
                    help="Served model id for --llm-api-url (default: Qwen3.5-27B)")
    ap.add_argument("--answer-evidence-k",   type=int, default=None,
                    help="Cap evidence blocks passed to the final answerer")
    ap.add_argument("--verifier-evidence-k", type=int, default=None,
                    help="Cap evidence blocks passed to verifier")
    ap.add_argument("--answer-keyframe-k",   type=int, default=32,
                    help="Cap keyframe images passed to the final answerer")
    ap.add_argument("--max-frames",          type=int, default=None,
                    help="Override frame_sampling.max_frames from config")
    ap.add_argument("--workers",             type=int, default=1,
                    help="Parallel workers (safe with API-mode LLM)")
    ap.add_argument("--pipeline-name",       default=None,
                    help="Override PIPELINE_NAME (= output jsonl filename stem)")
    ap.add_argument("--per-chunk-keyframe-cap", type=int, default=1,
                    help="Max keyframes per chunk fed to VLM (1=current behaviour)")
    ap.add_argument("--single-query-iter0",  action="store_true",
                    help="At iter 0, use the original question as a single query "
                         "instead of LLM-decomposing into per-option sub-queries")
    ap.add_argument("--pass-verifier-judgment", action="store_true",
                    help="Pass verifier's last option_judgment+scores into the final answerer prompt")
    ap.add_argument("--loose-verifier", action="store_true",
                    help="Use VERIFIER_SYS_LOOSE that accepts indirect/synthesized evidence")
    ap.add_argument("--dialogue-first", action="store_true",
                    help="Reformat evidence_texts to put ASR/Speech first with [DIALOGUE] tag")
    ap.add_argument("--asr-alpha", type=float, default=0.0,
                    help="Blend an ASR-only BGE channel into retrieval scoring (0=off, 0.4=balanced)")
    ap.add_argument("--text-first-keyframes", action="store_true",
                    help="Two-stage answerer: LLM (text-only) picks which chunks need visual "
                         "confirmation; VLM gets keyframes only for those chunks")
    ap.add_argument("--image-placement",
                    choices=["images_first", "text_first", "interleaved"],
                    default="images_first",
                    help="Image placement in answerer prompt: images_first (default), "
                         "text_first (all text then images), interleaved (text+image per segment)")
    ap.add_argument("--multi-layer-mode",
                    choices=["none", "coarse_to_fine", "multi_pool", "planner_ctx"],
                    default="none",
                    help="Multi-layer pyramid retrieval mode: "
                         "none (L1 only, default), "
                         "coarse_to_fine (L3→L1 window filter), "
                         "multi_pool (L1+L2 merged), "
                         "planner_ctx (L3 injected into planner)")
    args = ap.parse_args()
    global PIPELINE_NAME
    if args.pipeline_name:
        PIPELINE_NAME = args.pipeline_name

    cfg      = load_config(args.config)
    bench    = cfg["benchmark"]["name"]
    out_root = Path(cfg.get("paths", {}).get("outputs_root", "outputs"))

    _vl = cfg.get("veil_loop", {})
    veil_max_iter   = int(_vl.get("max_iter", 3))
    q_hist_dedup    = float(_vl.get("query_history_dedup_threshold", 0.9))
    ev_dedup        = float(_vl.get("evidence_dedup_threshold", 0.9))
    q_ev_dedup      = float(_vl.get("query_evidence_dedup_threshold", 0.9))
    kf_dedup_thresh = float(_vl.get("kf_dedup_threshold", 0.9))

    memory_dir = Path(args.memory_dir) if args.memory_dir else \
                 out_root / "memory" / f"{bench}_L_27B"
    default_out_dir = Path(cfg.get("eval", {}).get("output_dir") or (out_root / "results" / bench))
    out_path = Path(args.out) if args.out else \
               default_out_dir / "veil_27b.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter video ids
    filter_vids: set | None = None
    if args.filter_from:
        filter_vids = set()
        for line in Path(args.filter_from).open():
            try: filter_vids.add(json.loads(line)["video_id"])
            except: pass
        log.info("filter: %d video_ids from %s", len(filter_vids), args.filter_from)

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
    needs_siglip  = not args.no_siglip
    siglip_device = (args.siglip_gpu if args.siglip_gpu is not None else args.bge_gpu) if needs_siglip else None

    vlm = embedder = llm = answerer = siglip = None

    from src.clients.vlm_client import VLMClient
    from src.agents.answerer import Answerer

    vlm_model = args.vlm_model or cfg["models"]["vlm"]["model_path"]
    if args.vlm_api_url:
        log.info("loading VLM via API %s ...", args.vlm_api_url)
        vlm = VLMClient(model_path=vlm_model, api_url=args.vlm_api_url,
                        api_model=args.vlm_api_model)
    else:
        log.info("loading VLM %s on %s ...", vlm_model, args.vlm_gpu)
        vlm = VLMClient(model_path=vlm_model, device=args.vlm_gpu)
    answerer = Answerer(vlm)
    log.info("  VLM ready")

    from src.clients.embedder import BGEM3Embedder
    if args.embed_api_url:
        log.info("loading BGE-M3 via API %s ...", args.embed_api_url)
        embedder = BGEM3Embedder(api_url=args.embed_api_url)
    else:
        log.info("loading BGE-M3 on %s ...", args.bge_gpu)
        t0 = time.time()
        embedder = BGEM3Embedder(
            model_path=cfg["models"]["embedder"]["model_path"],
            use_fp16=cfg["models"]["embedder"].get("use_fp16", True),
            device=args.bge_gpu,
        )
        log.info("  BGE-M3 ready (%.1fs)", time.time() - t0)

    from src.clients.llm_client import LLMClient
    if args.llm_api_url:
        api_model = args.llm_api_model or "Qwen3.5-27B"
        log.info("loading LLM via API %s ...", args.llm_api_url)
        t0 = time.time()
        llm = LLMClient(model_path=api_model, api_url=args.llm_api_url, api_model=api_model)
        log.info("  API LLM ready (%.1fs)", time.time() - t0)
    else:
        llm_model = args.llm_model or cfg["models"]["llm"]["model_path"]
        log.info("loading LLM %s on %s ...", llm_model, args.llm_gpu)
        t0 = time.time()
        llm = LLMClient(
            model_path=llm_model,
            device=args.llm_gpu,
            dtype=cfg["models"]["llm"].get("dtype", "bfloat16"),
            attn_impl=cfg["models"]["llm"].get("attn_impl", "sdpa"),
            use_vllm=args.use_vllm,
        )
        log.info("  LLM ready (%.1fs)", time.time() - t0)

    if needs_siglip:
        from src.clients.siglip_embedder import SigLIPEmbedder
        if args.embed_api_url:
            log.info("loading SigLIP via API %s ...", args.embed_api_url)
            siglip = SigLIPEmbedder(api_url=args.embed_api_url)
        elif siglip_device:
            siglip_model = "/home2/ycj/Models/google/siglip-large-patch16-384"
            log.info("loading SigLIP on %s ...", siglip_device)
            t0 = time.time()
            siglip = SigLIPEmbedder(model_path=siglip_model, device=siglip_device)
            log.info("  SigLIP ready (%.1fs)", time.time() - t0)

    # ── Thread safety (GPU models are not thread-safe) ────────────────────────
    _gpu_lock = threading.Lock()

    class _LockedModel:
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

    # ── Bank cache ────────────────────────────────────────────────────────────
    _bank_cache:  dict = {}
    _cache_lock = threading.Lock()

    def get_bank(video_id: str):
        with _cache_lock:
            if video_id in _bank_cache:
                return _bank_cache[video_id]
        vd = memory_dir / video_id
        if not (vd / "narrative.json").exists() and not (vd / "L1.jsonl").exists():
            return None
        from src.build_memory.core.bank_loader import load_bank
        bank = load_bank(vd)
        with _cache_lock:
            _bank_cache[video_id] = bank
        return bank

    from src.eval.parse_answer import parse_letter

    aek    = args.answer_evidence_k
    vek    = args.verifier_evidence_k
    akk    = None if args.answer_keyframe_k is not None and args.answer_keyframe_k < 0 else args.answer_keyframe_k
    kf_dir = None if args.no_keyframes else memory_dir

    def run_sample(s):
        bank = get_bank(s.video_id)
        if bank is None:
            return None, "bank_missing"
        kw = dict(
            reranker=None, coarse_top_k=8, final_top_k=8,
            max_iter=veil_max_iter,
            query_history_dedup_threshold=q_hist_dedup,
            evidence_dedup_threshold=ev_dedup,
            query_evidence_dedup_threshold=q_ev_dedup,
            kf_dedup_threshold=kf_dedup_thresh,
            siglip=siglip, text_alpha=args.text_alpha, keyframe_dir=kf_dir,
            answer_evidence_cap=aek,
            answer_keyframe_cap=akk,
            verifier_evidence_cap=vek,
            rubric_rerank=True,
            per_chunk_keyframe_cap=args.per_chunk_keyframe_cap,
            single_query_iter0=args.single_query_iter0,
            pass_verifier_judgment_to_answerer=args.pass_verifier_judgment,
            loose_verifier=args.loose_verifier,
            dialogue_first=args.dialogue_first,
            asr_alpha=args.asr_alpha,
            text_first_keyframes=args.text_first_keyframes,
            image_placement=args.image_placement,
            multi_layer_mode=args.multi_layer_mode,
        )
        return run_veil(s.question, s.candidates, bank, embedder, answerer, llm,
                        task_type=s.question_type, **kw), None

    # ── Run loop ──────────────────────────────────────────────────────────────
    out_fh      = out_path.open("a")
    _write_lock = threading.Lock()
    _stats_lock = threading.Lock()

    def append_record(rec: dict):
        with _write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

    total      = len(samples)
    done_count = [len(done_keys)]
    correct_by: dict = {}

    def process_item(s) -> None:
        key = f"{bench}|{s.video_id}|{s.sample_idx}|{PIPELINE_NAME}"

        t0 = time.time()
        try:
            result, err = run_sample(s)
        except Exception as e:
            log.error("[%s/%s] ERROR: %s", s.video_id, s.question_type, e, exc_info=True)
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
            "pipeline":         PIPELINE_NAME,
            "question":         s.question,
            "candidates":       s.candidates,
            "gold_answer":      s.answer,
            "pred_letter":      pred_letter,
            "pred_text":        pred_text,
            "correct":          correct,
            "evidence_chunk_ids": (result or {}).get("evidence_chunk_ids", []),
            "n_keyframes_to_answer": (result or {}).get("n_keyframes_to_answer"),
            "n_evidence_to_answer":  (result or {}).get("n_evidence_to_answer"),
            "trace_iters":      trace,
            "elapsed":          round(elapsed, 2),
            "error":            err,
        }
        append_record(rec)

        with _stats_lock:
            done_count[0] += 1
            correct_by.setdefault(s.question_type, []).append(int(correct))
            n_done = done_count[0]

        n_iter = len(trace) if trace else 0
        log.info("[%d/%d] %-12s | %-16s | %s %s (iter=%d) %.1fs",
                 n_done, total, s.question_type, s.video_id,
                 pred_letter, "✓" if correct else "✗", n_iter, elapsed)

    tasks = [s for s in samples
             if f"{bench}|{s.video_id}|{s.sample_idx}|{PIPELINE_NAME}" not in done_keys]
    log.info("tasks to run: %d  (workers=%d)", len(tasks), args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, s) for s in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("worker exception: %s", exc)

    out_fh.close()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== Accuracy ===")
    qtypes = sorted(correct_by)
    print(f"{'Pipeline':<20}" + "".join(f"{qt[:8]:>10}" for qt in qtypes) + f"{'Overall':>10}")
    row = f"{PIPELINE_NAME:<20}"
    all_c = []
    for qt in qtypes:
        c = correct_by.get(qt, [])
        acc = sum(c) / len(c) if c else float("nan")
        row += f"{acc*100:>9.1f}%"
        all_c.extend(c)
    ov = sum(all_c) / len(all_c) if all_c else float("nan")
    row += f"{ov*100:>9.1f}%"
    print(row)
    log.info("done — results in %s", out_path)


if __name__ == "__main__":
    main()
