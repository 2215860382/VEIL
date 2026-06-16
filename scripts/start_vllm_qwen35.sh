#!/bin/bash
# 启动 Qwen3.5-27B vLLM 服务
# 用法：bash scripts/start_vllm_qwen35.sh [GPU_ID] [PORT]
# 例：bash scripts/start_vllm_qwen35.sh 1 8000

GPU_ID=${1:-1}
PORT=${2:-8000}

echo "启动 Qwen3.5-27B vLLM on GPU $GPU_ID, port $PORT"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup /home2/ycj/miniconda3/envs/veil/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --host 127.0.0.1 \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    > /tmp/vllm_qwen35_gpu${GPU_ID}_p${PORT}.log 2>&1 &

PID=$!
echo "vLLM PID=$PID, log=/tmp/vllm_qwen35_gpu${GPU_ID}_p${PORT}.log"
sleep 60
curl -s http://127.0.0.1:$PORT/v1/models | grep -q "Qwen3.5-27B" && echo "✓ vLLM ready" || echo "⚠ vLLM still initializing"
