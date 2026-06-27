#!/usr/bin/env bash
# Step 6: compare 3 rubric versions on the held-out val-90 split (no dev leakage).
# Runs each rubric through the veil pipeline on the SAME 90 val indices, then
# prints overall accuracy. Sequential per rubric; vLLM(8003/8004)=LLM+VLM, embed=9000.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
D=outputs/results/intermediate_goods/overnight_20260627_full720
SPLIT="${SPLIT:-val}"                       # val | test
IDX="$D/splits/${SPLIT}_indices.txt"
OUTDIR="outputs/results/videomme/${SPLIT}_compare"
WORKERS="${WORKERS:-24}"
EMBED="http://127.0.0.1:9000"
SRV="http://127.0.0.1:8003,http://127.0.0.1:8004"
export PYTHONPATH=.
mkdir -p "$OUTDIR"

# name -> rubric path  (override via NAMES_CSV / PATHS_CSV env, comma-separated)
if [ -n "${NAMES_CSV:-}" ]; then
  IFS=',' read -r -a NAMES <<< "$NAMES_CSV"
  IFS=',' read -r -a PATHS <<< "$PATHS_CSV"
else
  NAMES=(v2 round1 round2_merged)
  PATHS=(
    "outputs/rubric/direct_answer_generated_v2.yaml"
    "$D/round1_type_level_rubric.yaml"
    "$D/round2_merged_runtime.yaml"
  )
fi

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; rubric="${PATHS[$i]}"
  OUT="$OUTDIR/${SPLIT}_${name}.jsonl"
  echo "========================================================"
  echo "[$SPLIT] rubric=$name  ->  $OUT"
  echo "========================================================"
  VEIL_RUBRIC_PATH="$rubric" "$PY" experiments/tuning/veil_27b.py \
    --config configs/veil.yaml \
    --memory-dir outputs/memory/videomme_multiframe \
    --embed-api-url "$EMBED" --vlm-api-url "$SRV" --llm-api-url "$SRV" \
    --answer-keyframe-k 32 \
    --pipeline-name "${SPLIT}_${name}" \
    --sample-indices-file "$IDX" \
    --workers "$WORKERS" \
    --out "$OUT"
  "$PY" - "$OUT" "$name" <<'PYEOF'
import json, sys
path, name = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
c = sum(1 for r in rows if r.get("correct"))
print(f">>> [{name}] val accuracy = {c}/{len(rows)} = {c/len(rows)*100:.1f}%")
PYEOF
done
echo "ALL DONE: $OUTDIR"
