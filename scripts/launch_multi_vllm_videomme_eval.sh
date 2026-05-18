#!/usr/bin/env bash
# 1) Optional: stop your old vLLM + eval.
# 2) Start multi-GPU vLLM (see scripts/start_multi_vllm_qwen35.sh).
# 3) Build comma-separated LLM_API / pass to eval; encoders on a free card (default cuda:0).
#
# Env:
#   VLLM_GPU_PORTS   default "1:8778,2:8779,4:8780"
#   BGE_GPU          default cuda:0 (keep encoders off vLLM GPUs 1/2/4)
#   KILL_MY_VLLM     default 1 before starting servers
#
# Usage:
#   bash scripts/launch_multi_vllm_videomme_eval.sh

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
export VLLM_GPU_PORTS="${VLLM_GPU_PORTS:-1:8778,2:8779,4:8780}"
export KILL_MY_VLLM="${KILL_MY_VLLM:-1}"

bash "$ROOT/scripts/start_multi_vllm_qwen35.sh"

# Build comma-separated API list from VLLM_GPU_PORTS
LLM_API_LIST=""
IFS=',' read -r -a PAIRS <<< "$VLLM_GPU_PORTS"
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  port="${pair##*:}"
  if [ -n "$LLM_API_LIST" ]; then LLM_API_LIST+=","; fi
  LLM_API_LIST+="http://127.0.0.1:${port}"
done

export LLM_API="$LLM_API_LIST"
export LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}"
export BGE_GPU="${BGE_GPU:-cuda:0}"
export LLM_GPU="${LLM_GPU:-cuda:0}"
export VLM_GPU="${VLM_GPU:-cuda:0}"

pkill -u "$(whoami)" -f 'experiments/run_experiments.py' 2>/dev/null || true
sleep 1

RUN_NAME="${RUN_NAME:-videommeL_27b300_multi_$(date +%Y%m%d_%H%M)}"
RUN_DIR="$ROOT/outputs/results/videommeL/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOGR="$RUN_DIR/run.log"
export OUT="$RUN_DIR/results.jsonl"
export LOG="$LOGR"
export RUN_NAME
export RUN_DIR

nohup env LLM_API="$LLM_API_LIST" LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}" \
  BGE_GPU="$BGE_GPU" LLM_GPU="$LLM_GPU" VLM_GPU="$VLM_GPU" ROOT="$ROOT" \
  OUT="$OUT" LOG="$LOGR" \
  bash "$ROOT/scripts/run_videomme_L_27b_bank_eval.sh" >>"$LOGR" 2>&1 &
echo "[$(date -Is)] eval nohup PID=$!  log=$LOGR  OUT=$OUT"
