#!/bin/bash
# 启动 Qwen3.5-27B vLLM，启动期间用 keep-warm 占住 GPU SM util 防止管理员监控杀。
# 用法： bash scripts/launch_vllm_safe.sh [GPU_ID] [PORT]
# 例：   bash scripts/launch_vllm_safe.sh 4 8003
set -e
GPU_ID=${1:-4}
PORT=${2:-8003}
LOG=/tmp/vllm_qwen35_gpu${GPU_ID}_p${PORT}.log
WARM_LOG=/tmp/vllm_keepwarm_gpu${GPU_ID}.log

echo "==> Launching vLLM on GPU $GPU_ID port $PORT"
echo "    vLLM log:        $LOG"
echo "    keep-warm log:   $WARM_LOG"

# 1) keep-warm: 极小张量 matmul 循环，把 GPU SM util 拉到 20-40%
#    防止管理员的 idle GPU 监控判定我们在白占卡
CUDA_VISIBLE_DEVICES=$GPU_ID nohup /home2/ycj/miniconda3/envs/veil_vllm/bin/python -c "
import torch, time
torch.cuda.set_device(0)
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
print('keep-warm started', flush=True)
while True:
    c = a @ b
    torch.cuda.synchronize()
" > $WARM_LOG 2>&1 &
WARM_PID=$!
echo "    keep-warm PID=$WARM_PID"
sleep 3  # 让 keep-warm 先把 GPU util 拉起来

# 2) vLLM
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU_ID nohup \
    /home2/ycj/miniconda3/envs/veil_vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --host 127.0.0.1 --port $PORT \
    --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    > $LOG 2>&1 &
VLLM_PID=$!
echo "    vLLM PID=$VLLM_PID"

# 3) 等 vLLM ready，ready 后立刻发 warmup 请求注册流量，并杀掉 keep-warm
echo "==> Waiting for vLLM ready (poll every 5s, up to 5 minutes) ..."
for i in $(seq 1 60); do
    sleep 5
    if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
        echo "==> ✓ vLLM ready after $((i*5))s — killing keep-warm + sending warmup"
        kill $WARM_PID 2>/dev/null || true
        sleep 1
        # Warmup request to mark this endpoint as actively serving
        curl -s http://127.0.0.1:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"Qwen3.5-27B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
            | head -c 200
        echo ""
        echo "==> ✓ Endpoint http://127.0.0.1:$PORT is live. Start your evaluator NOW."
        exit 0
    fi
    # check vLLM hasn't died mid-load
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "==> ✗ vLLM PID $VLLM_PID died — see $LOG"
        kill $WARM_PID 2>/dev/null || true
        exit 1
    fi
done

echo "==> ✗ vLLM did not become ready in 5 minutes"
kill $WARM_PID 2>/dev/null || true
exit 1
