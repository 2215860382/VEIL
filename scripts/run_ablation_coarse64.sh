#!/usr/bin/env bash
# Run coarse64 ablation on GPU6 vLLM (port 8781), parallel with coarse8 ablation.
# Encoders on cuda:2 (61 GB free), LLM API on port 8781.
# Start GPU6 vLLM first: bash scripts/start_vllm_gpu6.sh

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

FILTER="${FILTER:-$ROOT/dataloader/filters/videomme_L_300_banks.jsonl}"
MEM="${MEM:-$ROOT/outputs/memory/videomme_L_27b_27b}"
OUTROOT="$ROOT/outputs/results/videommeL"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUTROOT"

LLM_API="${LLM_API:-http://127.0.0.1:8777,http://127.0.0.1:8780}"
LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}"
BGE_GPU="${BGE_GPU:-cuda:2}"
SIGLIP_GPU="${SIGLIP_GPU:-cuda:2}"

BASE_ARGS=(
  --config configs/videomme_memory_bank.yaml
  --memory-dir "$MEM"
  --filter-from "$FILTER"
  --vlm-api-url "$LLM_API"
  --vlm-api-model "$LLM_MODEL"
  --bge-gpu "$BGE_GPU"
  --siglip-gpu "$SIGLIP_GPU"
  --llm-api-url "$LLM_API"
  --llm-api-model "$LLM_MODEL"
  --max-frames 64
)

run_one() {
    local tag="$1"; shift
    local run_dir="$OUTROOT/${RUN_PREFIX}_${tag}"
    local out="$run_dir/results.jsonl"
    local log="$run_dir/run.log"
    mkdir -p "$run_dir"
    echo "=== $(date -Is)  START $tag ==="
    {
      echo "=== $(date -Is) ==="
      echo "tag=$tag  run_dir=$run_dir  BGE=$BGE_GPU SIGLIP=$SIGLIP_GPU LLM=$LLM_API"
      "$PYTHON" experiments/run_experiments.py "${BASE_ARGS[@]}" --out "$out" "$@"
    } >>"$log" 2>&1
    echo "=== $(date -Is)  DONE  $tag  -> $run_dir ==="
}

run_one "coarse64_sum_asr"    --pipelines coarse64_27b --no-keyframes
run_one "coarse64_sum_asr_kf" --pipelines coarse64_27b

echo "coarse64 ablation done. Results under $OUTROOT (prefix=$RUN_PREFIX)."
