#!/usr/bin/env bash
# Start extra Qwen3.5-27B vLLM instances for VEIL eval (does NOT kill existing servers).
# Default: GPU 1 → 8778, GPU 3 → 8779 (GPU 0 reserved for BGE/SigLIP).
# Also appends 8780/8787/8788 if already healthy.
#
#   bash scripts/start_extra_vllm_for_veil.sh
#   export LLM_API="http://127.0.0.1:8778,http://127.0.0.1:8779,http://127.0.0.1:8780,..."

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PY="${PY:-/home2/ycj/miniconda3/envs/veil/bin/python}"
ME="$(whoami)"
VLLM_GPU_PORTS="${VLLM_GPU_PORTS:-1:8778,3:8779}"
KILL_MY_VLLM=0

mkdir -p "$ROOT/outputs/logs"
TS="$(date +%Y%m%d_%H%M)"

IFS=',' read -r -a PAIRS <<< "$VLLM_GPU_PORTS"
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  gpu="${pair%%:*}"
  port="${pair##*:}"
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "[$(date -Is)] port $port already up — skip GPU $gpu"
    continue
  fi
  LOGV="$ROOT/outputs/logs/vllm_gpu${gpu}_p${port}_${TS}.log"
  echo "[$(date -Is)] starting vLLM GPU $gpu port $port -> $LOGV"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    >>"$LOGV" 2>&1 &
  echo "  pid=$!"
done

echo "[$(date -Is)] waiting for new ports ..."
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  port="${pair##*:}"
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && continue
  ok=0
  for _ in $(seq 1 200); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ok=1; echo "  port $port ready"; break; fi
    sleep 3
  done
  if [ "$ok" != "1" ]; then
    echo "ERROR: port $port not ready"
    exit 1
  fi
done

LLM_API_LIST=""
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  port="${pair##*:}"
  [ -n "$LLM_API_LIST" ] && LLM_API_LIST+=","
  LLM_API_LIST+="http://127.0.0.1:${port}"
done
for extra_port in 8780 8787 8788; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${extra_port}/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    LLM_API_LIST="${LLM_API_LIST},http://127.0.0.1:${extra_port}"
  fi
done
echo "export LLM_API=\"$LLM_API_LIST\""
echo "export WORKERS=\"\${WORKERS:-8}\""
