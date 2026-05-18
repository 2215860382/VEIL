#!/usr/bin/env bash
# VideoMME-L (long) eval on banks under outputs/memory/videomme_L_27b_27b (300 videos).
#
# - direct_27b: Qwen3.5-27B multimodal via vLLM (--vlm-api-url / --vlm-api-model).
# - coarse64/8/rerank8/veil_coarse8: 27B text answerer + planner via --llm-api-url.
# - Evidence = summary + ASR + keyframe images (no frame captions).
# - coarse64 passes all 64 text summaries; keyframes capped at 8 (visual dedup).
#
# Layout: vLLM on GPU4:8780 (``bash scripts/start_vllm_gpu4.sh``) plus existing GPU0:8777 if desired:
#   export LLM_API=http://127.0.0.1:8777,http://127.0.0.1:8780
#   bash scripts/run_videomme_L_27b_bank_eval.sh
# Encoders: BGE-M3 on GPU1; SigLIP + reranker on GPU2 (override with BGE_GPU / SIGLIP_GPU / RERANKER_GPU).

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

FILTER="${FILTER:-$ROOT/dataloader/filters/videomme_L_300_banks.jsonl}"
MEM="${MEM:-$ROOT/outputs/memory/videomme_L_27b_27b}"
RUN_NAME="${RUN_NAME:-videommeL_27b300_$(date +%Y%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/results/videommeL/$RUN_NAME}"
OUT="${OUT:-$RUN_DIR/results.jsonl}"
LOG="${LOG:-$RUN_DIR/run.log}"
# After ``bash scripts/start_vllm_gpu4.sh`` is healthy, prefer:
#   export LLM_API=http://127.0.0.1:8777,http://127.0.0.1:8780
LLM_API="${LLM_API:-http://127.0.0.1:8777}"
LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}"
VLM_GPU="${VLM_GPU:-cuda:1}"
BGE_GPU="${BGE_GPU:-cuda:1}"
SIGLIP_GPU="${SIGLIP_GPU:-cuda:2}"
RERANKER_GPU="${RERANKER_GPU:-cuda:2}"
LLM_GPU="${LLM_GPU:-cuda:1}"

mkdir -p "$RUN_DIR"
{
  echo "=== $(date -Is) ==="
  echo "OUT=$OUT MEM=$MEM FILTER=$FILTER LLM_API=$LLM_API BGE=$BGE_GPU SIGLIP=$SIGLIP_GPU RERANK=$RERANKER_GPU"
  "$PYTHON" experiments/run_experiments.py \
  --config experiments/configs/videomme_memory_bank.yaml \
  --memory-dir "$MEM" \
  --filter-from "$FILTER" \
  --pipelines direct_27b coarse64_27b coarse8_27b rerank_rag8_27b veil_coarse8_27b \
  --vlm-api-url "$LLM_API" \
  --vlm-api-model "$LLM_MODEL" \
  --vlm-gpu "$VLM_GPU" \
  --bge-gpu "$BGE_GPU" \
  --siglip-gpu "$SIGLIP_GPU" \
  --reranker-gpu "$RERANKER_GPU" \
  --llm-gpu "$LLM_GPU" \
  --llm-api-url "$LLM_API" \
  --llm-api-model "$LLM_MODEL" \
  --direct-max-new-tokens 512 \
  --max-frames 64 \
  --out "$OUT"
} >>"$LOG" 2>&1
echo "Finished -> log $LOG"
