#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_PORT="${VLLM_PORT:-8003}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
exec /home2/ycj/miniconda3/envs/veil/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --enable-prefix-caching \
  > "outputs/logs/vllm_gpu${CUDA_VISIBLE_DEVICES}_p${VLLM_PORT}_${MAX_MODEL_LEN}.log" 2>&1
