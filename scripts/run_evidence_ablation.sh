#!/usr/bin/env bash
# Evidence ablation: compare what the answerer sees.
#
# Naming convention:  <pipeline>_<evidence>
#   caption_nokf  : frame captions, no keyframes  (OLD code, already done)
#   caption_kf    : frame captions + keyframes     (OLD code, already done)
#   sum_asr       : summary + ASR, no keyframes    (NEW)
#   sum_asr_kf    : summary + ASR + keyframes      (NEW)
#
# Runs sequentially to avoid BGE GPU conflicts.

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

FILTER="${FILTER:-$ROOT/dataloader/filters/videomme_L_300_banks.jsonl}"
MEM="${MEM:-$ROOT/outputs/memory/videomme_L_27b_27b}"
OUTROOT="$ROOT/outputs/results/videommeL"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M)}"
export RUN_PREFIX
mkdir -p "$OUTROOT"

LLM_API="${LLM_API:-http://127.0.0.1:8777,http://127.0.0.1:8780}"
LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}"
BGE_GPU="${BGE_GPU:-cuda:1}"
SIGLIP_GPU="${SIGLIP_GPU:-cuda:2}"

BASE_ARGS=(
  --config configs/videomme_memory_bank.yaml
  --memory-dir "$MEM"
  --filter-from "$FILTER"
  --vlm-api-url "$LLM_API"
  --vlm-api-model "$LLM_MODEL"
  --bge-gpu "$BGE_GPU"
  --siglip-gpu "$SIGLIP_GPU"
  --llm-api-url "$LLM_API"
  --llm-api-model "$LLM_MODEL"
  --max-frames 64
)

run_one() {
    local tag="$1"; shift
    local run_dir="$OUTROOT/${RUN_PREFIX}_${tag}"
    local out="$run_dir/results.jsonl"
    local log="$run_dir/run.log"
    mkdir -p "$run_dir"
    echo "=== $(date -Is)  START $tag ==="
    {
      echo "=== $(date -Is) ==="
      echo "tag=$tag  run_dir=$run_dir"
      "$PYTHON" experiments/run_experiments.py "${BASE_ARGS[@]}" --out "$out" "$@"
    } >>"$log" 2>&1
    echo "=== $(date -Is)  DONE  $tag  -> $run_dir ==="
}

# ── coarse8 ────────────────────────────────────────────────────────────────────
run_one "coarse8_sum_asr"    --pipelines coarse8_27b --no-keyframes
run_one "coarse8_sum_asr_kf" --pipelines coarse8_27b

# ── coarse64 ───────────────────────────────────────────────────────────────────
run_one "coarse64_sum_asr"    --pipelines coarse64_27b --no-keyframes
run_one "coarse64_sum_asr_kf" --pipelines coarse64_27b

echo ""
echo "All done. Results under $OUTROOT (prefix=$RUN_PREFIX)."
echo ""
python3 - <<'PYEOF'
import json, glob
from pathlib import Path

def acc(path):
    correct = total = 0
    seen = set()
    for line in open(path):
        try: r = json.loads(line)
        except: continue
        key = r.get('key','')
        if key in seen: continue
        seen.add(key)
        if 'correct' not in r: continue
        total += 1
        if r['correct']: correct += 1
    return correct, total, correct/total*100 if total else 0

run_prefix = __import__("os").environ.get("RUN_PREFIX", "")
files = {
    'caption_nokf  (old)': 'outputs/results/videommeL/videommeL_coarse64_27b_nokf.jsonl',
    'caption_kf    (old)': 'outputs/results/videommeL/videommeL_coarse64_27b_kf.jsonl',
    'sum_asr       (new)': f'outputs/results/videommeL/{run_prefix}_coarse64_sum_asr/results.jsonl',
    'sum_asr_kf    (new)': f'outputs/results/videommeL/{run_prefix}_coarse64_sum_asr_kf/results.jsonl',
}
print("\n=== coarse64 evidence ablation ===")
for label, path in files.items():
    if Path(path).exists():
        c,t,a = acc(path)
        print(f"  {label}: {c}/{t}  {a:.1f}%")
    else:
        print(f"  {label}: (not found)")

files8 = {
    'caption_kf    (old)': 'outputs/results/videommeL/videommeL_coarse8_27b_kf.jsonl',
    'sum_asr       (new)': f'outputs/results/videommeL/{run_prefix}_coarse8_sum_asr/results.jsonl',
    'sum_asr_kf    (new)': f'outputs/results/videommeL/{run_prefix}_coarse8_sum_asr_kf/results.jsonl',
}
print("\n=== coarse8 evidence ablation ===")
for label, path in files8.items():
    if Path(path).exists():
        c,t,a = acc(path)
        print(f"  {label}: {c}/{t}  {a:.1f}%")
    else:
        print(f"  {label}: (not found)")
PYEOF
