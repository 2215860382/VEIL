#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
OUT_ROOT="${OUT_ROOT:-/home2/ycj/Project/VEIL/outputs/results/intermediate_goods/overnight_20260627_full720}"
FINAL_RUBRIC="${FINAL_RUBRIC:-$OUT_ROOT/round2_type_level_rubric.yaml}"
TEMPLATE_COPY="${TEMPLATE_COPY:-/home2/ycj/Project/VEIL/outputs/rubric/round2_generated_20260627.yaml}"
RESULT_OUT="${RESULT_OUT:-outputs/results/videomme/tuning/mf_round2_generated_900.jsonl}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LLM_API_URL="${LLM_API_URL:-http://127.0.0.1:8003,http://127.0.0.1:8004}"
VLM_API_URL="${VLM_API_URL:-http://127.0.0.1:8003,http://127.0.0.1:8004}"
EMBED_API_URL="${EMBED_API_URL:-http://127.0.0.1:9000}"
MEMORY_DIR="${MEMORY_DIR:-outputs/memory/videomme_multiframe}"
PIPELINE_NAME="${PIPELINE_NAME:-mf_round2_generated_900}"

echo "[wait] round2 rubric: $FINAL_RUBRIC"
while [ ! -s "$FINAL_RUBRIC" ]; do
  sleep "$POLL_SECONDS"
done

mkdir -p "$(dirname "$TEMPLATE_COPY")" "$(dirname "$RESULT_OUT")"
cp "$FINAL_RUBRIC" "$TEMPLATE_COPY"
echo "[ready] copied final rubric to $TEMPLATE_COPY"

VEIL_RUBRIC_PATH="$TEMPLATE_COPY" \
  "$PYTHON" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml \
    --memory-dir "$MEMORY_DIR" \
    --embed-api-url "$EMBED_API_URL" \
    --vlm-api-url "$VLM_API_URL" \
    --llm-api-url "$LLM_API_URL" \
    --answer-keyframe-k 32 \
    --pipeline-name "$PIPELINE_NAME" \
    --sample-start 0 \
    --sample-end 900 \
    --out "$RESULT_OUT"
