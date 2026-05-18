#!/usr/bin/env bash
# Re-run coarse64 and coarse8 summary+kf with fixed config:
#   - max_evidence_chars=80000 (no truncation)
#   - vLLM max-model-len=32768 (32K context)
#
# vLLM endpoints: GPU4:8780 + GPU2:8783 (both 32768 context)
# BGE/SigLIP: cuda:0 (63 GB free)

set -euo pipefail
ROOT="${ROOT:-/home2/ycj/Project/VEIL}"
PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

FILTER="${FILTER:-$ROOT/dataloader/filters/videomme_L_300_banks_ordered.jsonl}"
MEM="${MEM:-$ROOT/outputs/memory/videomme_L_27b_27b}"
OUTROOT="$ROOT/outputs/results/videommeL"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M)}"
export RUN_PREFIX
mkdir -p "$OUTROOT"

LLM_API="${LLM_API:-http://127.0.0.1:8780,http://127.0.0.1:8783}"
LLM_MODEL="${LLM_MODEL:-Qwen3.5-27B}"
BGE_GPU="${BGE_GPU:-cuda:0}"
SIGLIP_GPU="${SIGLIP_GPU:-cuda:0}"

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
      echo "tag=$tag  LLM=$LLM_API  BGE=$BGE_GPU SIGLIP=$SIGLIP_GPU"
      "$PYTHON" experiments/run_experiments.py "${BASE_ARGS[@]}" --out "$out" "$@"
    } >>"$log" 2>&1
    echo "=== $(date -Is)  DONE  $tag  -> $run_dir ==="
}

run_one "coarse64_27b_summary_kf" --pipelines coarse64_27b
run_one "coarse8_27b_summary_kf"  --pipelines coarse8_27b

echo ""
echo "All done. Results under $OUTROOT (prefix=$RUN_PREFIX)."

python3 - <<'PYEOF'
import json
from pathlib import Path

RESULT_DIR = Path("outputs/results/videommeL")
ORDERED_IDS_FILE = Path("dataloader/filters/videomme_L_300_banks_ordered.jsonl")
ordered_ids = [json.loads(l)["video_id"] for l in open(ORDERED_IDS_FILE) if l.strip()]
classmate_ids = set(ordered_ids[:102])

def acc(path, subset_ids=None):
    correct = total = 0
    seen = set()
    for line in open(path):
        try: r = json.loads(line)
        except: continue
        key = r.get('key','')
        if key in seen: continue
        seen.add(key)
        if 'correct' not in r: continue
        vid = r.get('video_id', key.split('/')[0] if '/' in key else '')
        if subset_ids and vid not in subset_ids: continue
        total += 1
        if r['correct']: correct += 1
    return correct, total, correct/total*100 if total else 0

files = [
    ("coarse8  caption_kf  (old)",  "videommeL_coarse8_27b_caption_kf.jsonl"),
    ("coarse8  summary_kf  (new)",  f"{__import__('os').environ.get('RUN_PREFIX', '')}_coarse8_27b_summary_kf/results.jsonl"),
    ("coarse64 caption_kf  (old)",  "videommeL_coarse64_27b_caption_kf.jsonl"),
    ("coarse64 summary_kf  (new)",  f"{__import__('os').environ.get('RUN_PREFIX', '')}_coarse64_27b_summary_kf/results.jsonl"),
    ("coarse64 sum_kf trunc(bad)",  "videommeL_coarse64_27b_summary_kf_ctx8k_truncated.jsonl"),
]
print(f"\n{'Method':<32} {'All 900':>9}  {'601-702 (306)':>14}")
print("-" * 60)
for label, fname in files:
    p = RESULT_DIR / fname
    if not p.exists(): print(f"  {label:<30}  (not found)"); continue
    c, t, a = acc(p)
    c2, t2, a2 = acc(p, classmate_ids)
    flag = " *partial*" if t < 890 else ""
    print(f"  {label:<30}  {c:>4}/{t:<4} {a:5.1f}%   {c2:>3}/{t2:<3} {a2:5.1f}%{flag}")
PYEOF
