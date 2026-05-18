#!/usr/bin/env bash
# Start Qwen3.5-27B vLLM on **physical GPU 4**, port **8780** (does not kill other vLLM).
# Pair with GPU0:8777 for eval:
#   export LLM_API="http://127.0.0.1:8777,http://127.0.0.1:8780"

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PY="${PY:-/home2/ycj/miniconda3/envs/veil/bin/python}"
PORT="${PORT:-8780}"
GPU="${GPU:-4}"
mkdir -p "$ROOT/outputs/logs"
LOGV="$ROOT/outputs/logs/vllm_gpu${GPU}_p${PORT}_$(date +%Y%m%d_%H%M).log"
echo "[$(date -Is)] vLLM GPU $GPU port $PORT -> $LOGV"
nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  >>"$LOGV" 2>&1 &
echo "pid=$!  (wait for curl http://127.0.0.1:${PORT}/v1/models)"
