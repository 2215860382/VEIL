#!/usr/bin/env bash
# Run 8 pipelines sequentially to full 900 questions on GPU2/8782.
# Each pipeline uses its existing output file; dedup skips the 312 already done.
set -euo pipefail

ROOT="/home2/ycj/Project/VEIL"
PY="/home2/ycj/miniconda3/envs/veil/bin/python"
API="http://127.0.0.1:8782"
CFG="configs/videomme_memory_bank.yaml"
WORKERS=4
BGE="cuda:4"

run_pipe() {
  local pipeline="$1"
  local out_file="$2"
  local log_file="${ROOT}/outputs/results/videommeL/${out_file}.log"
  echo "[$(date -Is)] START $pipeline -> $out_file"
  "$PY" experiments/run_experiments.py \
    --config "$CFG" \
    --pipelines "$pipeline" \
    --vlm-api-url "$API" \
    --llm-api-url "$API" \
    --bge-gpu "$BGE" \
    --workers "$WORKERS" \
    --out "outputs/results/videommeL/${out_file}" \
    >> "$log_file" 2>&1
  echo "[$(date -Is)] DONE  $pipeline"
}

cd "$ROOT"

# ── Queue (sequential) ─────────────────────────────────────────────────────
run_pipe veil_27b                   20260519_main
run_pipe veil_27b_no_rubric_rerank  20260519_main
run_pipe veil_27b_no_rubric_judge   20260520_ablations_strict
run_pipe veil_27b_prune_satisfied   20260520_ablations_strict
run_pipe veil_27b_llmbcast          20260519_ablations_2
run_pipe veil_27b_option4           20260519_ablations_2
run_pipe veil_27b_no_rubric_attr    20260521_no_rubric_ablations
run_pipe veil_27b_o4_lb             20260521_o4lb

echo "[$(date -Is)] ALL DONE"
