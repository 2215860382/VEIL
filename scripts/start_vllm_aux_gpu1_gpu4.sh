#!/usr/bin/env bash
# Start **two** extra Qwen3.5-27B vLLM servers on physical GPU **1** and **4** (ports 8778 / 8780).
# Intended layout with GPU **0** already serving 8777:
#   - 3× vLLM: GPU0:8777 + GPU1:8778 + GPU4:8780
#   - GPU **2** left for BGE / SigLIP / rerank → eval:  BGE_GPU=cuda:2  (and same for --llm-gpu if local)
#
# Does **not** kill existing vLLM (set KILL_MY_VLLM=1 only if you want to stop *your* ycj api_server first).
#
# Env:
#   MIN_FREE_MIB   default 75000  — skip a GPU if less free VRAM (MiB)
#   KILL_MY_VLLM   default 0
#
# After this + GPU0 listener:
#   export LLM_API="http://127.0.0.1:8777,http://127.0.0.1:8778,http://127.0.0.1:8780"

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PY="${PY:-/home2/ycj/miniconda3/envs/veil/bin/python}"
ME="$(whoami)"
MIN_FREE_MIB="${MIN_FREE_MIB:-75000}"
KILL_MY_VLLM="${KILL_MY_VLLM:-0}"

if [ "$KILL_MY_VLLM" = "1" ]; then
  pkill -u "$ME" -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 2
fi

free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1
}

start_one() {
  local gpu="$1" port="$2"
  local fm
  fm="$(free_mib "$gpu")"
  if [ -z "${fm:-}" ] || ! [[ "$fm" =~ ^[0-9]+$ ]]; then
    echo "[skip] GPU $gpu: could not read free memory"
    return 1
  fi
  if [ "$fm" -lt "$MIN_FREE_MIB" ]; then
    echo "[skip] GPU $gpu: only ${fm} MiB free (< $MIN_FREE_MIB MiB) — not starting vLLM on port $port"
    return 1
  fi
  local LOGV="$ROOT/outputs/logs/vllm_gpu${gpu}_p${port}_$(date +%Y%m%d_%H%M).log"
  mkdir -p "$ROOT/outputs/logs"
  echo "[$(date -Is)] starting vLLM GPU $gpu port $port -> $LOGV"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m vllm.entrypoints.openai.api_server \
    --model /home2/ycj/Models/Qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    >>"$LOGV" 2>&1 &
  echo "  pid=$!"
}

start_one 1 8778 || true
start_one 4 8780 || true

echo "[$(date -Is)] waiting for new ports (8778, 8780) ..."
for port in 8778 8780; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    ok=0
    for _ in $(seq 1 200); do
      code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
      if [ "$code" = "200" ]; then ok=1; echo "  port $port ready"; break; fi
      sleep 3
    done
    if [ "$ok" != "1" ]; then
      echo "WARN: port $port bound but /v1/models not 200 — check outputs/logs/vllm_gpu*_p${port}_*.log"
    fi
  else
    echo "  (port $port not listening — likely skipped due to VRAM)"
  fi
done

echo "Suggested eval:"
echo "  export LLM_API=\"http://127.0.0.1:8777,http://127.0.0.1:8778,http://127.0.0.1:8780\"   # drop dead ports if skipped"
echo "  export BGE_GPU=cuda:2 LLM_GPU=cuda:2 VLM_GPU=cuda:2"
