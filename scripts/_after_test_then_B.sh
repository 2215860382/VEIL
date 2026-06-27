#!/usr/bin/env bash
# Wait for the test-compare job to finish, print clean test accuracy (denominator
# excludes the header row), then immediately run the Step-B answerer probe:
# r2_merged on val-90 with --answer-keyframe-k 64 (vs the existing k=32 run).
# Chains B right after A so GPU3/4 never idles.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
D=outputs/results/intermediate_goods/overnight_20260627_full720
TDIR=outputs/results/videomme/test_compare
export PYTHONPATH=.

echo "[wait] test-compare (v2/r2_merged/r2_weighted) to reach 90 rows each ..."
for _ in $(seq 1 480); do
  ok=1
  for n in v2 r2_merged r2_weighted; do
    c=$(grep -c '"sample_idx"' "$TDIR/test_${n}.jsonl" 2>/dev/null || echo 0)
    [ "$c" -ge 90 ] || ok=0
  done
  # also ensure the test runner's veil_27b has exited
  if [ "$ok" = 1 ] && ! pgrep -f veil_27b >/dev/null; then break; fi
  sleep 30
done

echo "==================== TEST-90 (clean /90) ===================="
$PY - <<'PYEOF'
import json
D="outputs/results/intermediate_goods/overnight_20260627_full720"
test=sorted(int(x) for x in open(f"{D}/splits/test_indices.txt") if x.strip())
def load(f):
    o={}
    for l in open(f):
        r=json.loads(l)
        if r.get("record_type"): continue
        o[int(r["sample_idx"])]=r
    return o
runs={
 "v2":"outputs/results/videomme/test_compare/test_v2.jsonl",
 "r2_merged":"outputs/results/videomme/test_compare/test_r2_merged.jsonl",
 "r2_weighted":"outputs/results/videomme/test_compare/test_r2_weighted.jsonl",
 "tsqf":"outputs/results/videomme/tuning/mf_tsqf.jsonl",
}
for k,f in runs.items():
    try: d=load(f)
    except FileNotFoundError: print(f"  {k}: missing"); continue
    ids=[i for i in test if i in d]; c=sum(1 for i in ids if d[i].get("correct"))
    print(f"  test  {k:<12}: {c}/{len(ids)} = {c/len(ids)*100:.1f}%")
PYEOF

echo "==================== STEP B: answerer probe keyframe-k=64 ===================="
VEIL_RUBRIC_PATH="$D/round2_merged_runtime.yaml" "$PY" experiments/tuning/veil_27b.py \
  --config configs/veil.yaml \
  --memory-dir outputs/memory/videomme_multiframe \
  --embed-api-url http://127.0.0.1:9000 \
  --vlm-api-url http://127.0.0.1:8003,http://127.0.0.1:8004 \
  --llm-api-url http://127.0.0.1:8003,http://127.0.0.1:8004 \
  --answer-keyframe-k 64 --pipeline-name val_r2_merged_kf64 \
  --sample-indices-file "$D/splits/val_indices.txt" --workers 24 \
  --out outputs/results/videomme/val_compare/val_r2_merged_kf64.jsonl

echo "==================== B RESULT: keyframe k=32 vs k=64 on val-90 ===================="
$PY - <<'PYEOF'
import json
D="outputs/results/intermediate_goods/overnight_20260627_full720"
val=sorted(int(x) for x in open(f"{D}/splits/val_indices.txt") if x.strip())
def load(f):
    o={}
    for l in open(f):
        r=json.loads(l)
        if r.get("record_type"): continue
        o[int(r["sample_idx"])]=r
    return o
k32=load("outputs/results/videomme/val_compare/val_round2_merged.jsonl")
k64=load("outputs/results/videomme/val_compare/val_r2_merged_kf64.jsonl")
for name,d in [("r2_merged k=32",k32),("r2_merged k=64",k64)]:
    ids=[i for i in val if i in d]; c=sum(1 for i in ids if d[i].get("correct"))
    print(f"  {name:<18}: {c}/{len(ids)} = {c/len(ids)*100:.1f}%")
# 那15道"oracle也救不了"的题在 k=64 下是否被救
hard=[224,229,279,305,372,471,490,513,564,606,712,768,784,793,804]
saved=[i for i in hard if k64.get(i,{}).get("correct")]
print(f"  15 道答题瓶颈题中, k=64 救回: {len(saved)} -> {saved}")
PYEOF
echo "DONE"
