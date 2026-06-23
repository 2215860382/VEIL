#!/usr/bin/env bash
# Supervisor: wait for instance rubric generation, then run distillation.
# Used by: nohup ./run_pipeline.sh <PHASE1_PID> &
set -u
cd "$(dirname "$0")/../../.."   # → VEIL/
PHASE1_PID="${1:-}"
TS="$(date +%Y%m%d_%H%M%S)"
INSTANCE_OUT="src/rubric/artifacts/instances"
LOG_OUT="src/rubric/artifacts/logs"
mkdir -p "$INSTANCE_OUT" "$LOG_OUT"

echo "[$(date)] supervisor start; phase1_pid=$PHASE1_PID"

if [ -n "$PHASE1_PID" ]; then
  while kill -0 "$PHASE1_PID" 2>/dev/null; do
    n=$(wc -l < "$INSTANCE_OUT/instance_qwen.jsonl" 2>/dev/null || echo 0)
    echo "[$(date)] phase1 still running (qwen jsonl=$n), sleep 60"
    sleep 60
  done
fi
echo "[$(date)] phase1 done (qwen jsonl=$(wc -l < $INSTANCE_OUT/instance_qwen.jsonl 2>/dev/null || echo 0))"

echo "[$(date)] launching src.rubric.generation.distill"
python3 -m src.rubric.generation.distill --sample-per-qtype 40 \
  > "$LOG_OUT/distill_${TS}.log" 2>&1
ec=$?
echo "[$(date)] 02b exit code=$ec"

if [ $ec -eq 0 ] && [ -f "src/rubric/templates/generated_v2.yaml" ]; then
  echo "[$(date)] SUCCESS — generated rubric ready at src/rubric/templates/generated_v2.yaml"
else
  echo "[$(date)] FAILED — see $LOG_OUT/distill_${TS}.log"
fi
