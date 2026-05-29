#!/usr/bin/env bash
# Start multiple Qwen3.5-27B vLLM OpenAI servers — one process per GPU, each with its own port.
#
# Env:
#   VLLM_GPU_PORTS  comma-separated "physical_gpu:port" (default uses idle-looking cards 1,2,4)
#   KILL_MY_VLLM    if 1 (default), pkill your vllm.entrypoints processes first
#   VLLM_LIMIT_MM   optional, e.g. '{"image":128}' passed as --limit-mm-per-prompt (multimodal direct)
#   VLLM_EXTRA      optional extra args (single string, split on spaces — use carefully)
#
# Example:
#   VLLM_GPU_PORTS="1:8778,2:8779,4:8780" bash scripts/start_multi_vllm_qwen35.sh
#
# Then point eval at all backends (round-robin in LLMClient / VLMClient):
#   export LLM_API="http://127.0.0.1:8778,http://127.0.0.1:8779,http://127.0.0.1:8780"

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PY="${PY:-/home2/ycj/miniconda3/envs/veil/bin/python}"
ME="$(whoami)"
KILL_MY_VLLM="${KILL_MY_VLLM:-1}"
# Default: GPUs 1,2,4 were often free; skip 3 if shared; skip 5–6 when busy — override VLLM_GPU_PORTS.
VLLM_GPU_PORTS="${VLLM_GPU_PORTS:-1:8778,2:8779,4:8780}"

if [ "$KILL_MY_VLLM" = "1" ]; then
  pkill -u "$ME" -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 2
fi

mkdir -p "$ROOT/outputs/logs"
TS="$(date +%Y%m%d_%H%M)"

LIMIT_ARGS=()
if [ -n "${VLLM_LIMIT_MM:-}" ]; then
  LIMIT_ARGS=(--limit-mm-per-prompt "$VLLM_LIMIT_MM")
fi
EXTRA_ARGS=()
if [ -n "${VLLM_EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($VLLM_EXTRA)
fi

IFS=',' read -r -a PAIRS <<< "$VLLM_GPU_PORTS"
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  gpu="${pair%%:*}"
  port="${pair##*:}"
  LOGV="$ROOT/outputs/logs/vllm_gpu${gpu}_p${port}_${TS}.log"
  echo "[$(date -Is)] vLLM GPU $gpu port $port -> $LOGV"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    "${LIMIT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    >>"$LOGV" 2>&1 &
  echo "  started pid=$!"
done

echo "[$(date -Is)] waiting for /v1/models on each port ..."
for pair in "${PAIRS[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  port="${pair##*:}"
  ok=0
  for _ in $(seq 1 200); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ok=1; echo "  port $port ready"; break; fi
    sleep 3
  done
  if [ "$ok" != "1" ]; then
    echo "ERROR: port $port not ready — see outputs/logs/vllm_gpu*_p${port}_*.log"
    exit 1
  fi
done

echo "[$(date -Is)] all vLLM endpoints up."
LLM_API_LIST=""
IFS=',' read -r -a PAIRS2 <<< "$VLLM_GPU_PORTS"
for pair in "${PAIRS2[@]}"; do
  pair="${pair// /}"
  [ -n "$pair" ] || continue
  port="${pair##*:}"
  [ -n "$LLM_API_LIST" ] && LLM_API_LIST+=","
  LLM_API_LIST+="http://127.0.0.1:${port}"
done
echo "For eval (round-robin across vLLM workers):"
echo "  export LLM_API=\"$LLM_API_LIST\""
echo "  export VLM_API=\"$LLM_API_LIST\"   # same string for --vlm-api-url / --llm-api-url"
