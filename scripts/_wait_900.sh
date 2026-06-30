#!/usr/bin/env bash
# Wait for dev-720 r2_merged to finish, then stitch dev(720)+val(90)+test(90)
# into the full 900 and report overall + per-type accuracy, comparing r2_merged
# against the handwritten TSQF baseline and the oracle (perfect-planner) run on
# the SAME 900.
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
DEV=outputs/results/videomme/tuning/dev_r2_merged.jsonl

echo "[wait] dev-720 r2_merged ..."
for _ in $(seq 1 600); do
  c=$(grep -c '"sample_idx"' "$DEV" 2>/dev/null) || c=0
  if [ "${c:-0}" -ge 720 ] && ! pgrep -f "pipeline-name dev_r2_merged" >/dev/null; then break; fi
  sleep 30
done
echo "dev rows: $(grep -c '"sample_idx"' "$DEV" 2>/dev/null)"

$PY - <<'PYEOF'
import json
def load(f):
    o={}
    for l in open(f):
        r=json.loads(l)
        if r.get("record_type"): continue
        o[int(r["sample_idx"])]=r
    return o
# stitch r2_merged over the full 900
r2={}
for f in ["outputs/results/videomme/tuning/dev_r2_merged.jsonl",
          "outputs/results/videomme/tuning/val_round2_merged.jsonl",
          "outputs/results/videomme/tuning/test_r2_merged.jsonl"]:
    r2.update(load(f))
tsqf=load("outputs/results/videomme/tuning/mf_tsqf.jsonl")
orc=load("outputs/results/videomme/tuning/mf_qf_oracle.jsonl")
ids=sorted(set(r2)&set(tsqf)&set(orc))
print(f"全 900 对比集: {len(ids)} 题")
def acc(d):
    c=sum(1 for i in ids if d[i].get("correct")); return c, c/len(ids)*100
for name,d in [("r2_merged(新)",r2),("tsqf(手写)",tsqf),("oracle(完美检索)",orc)]:
    c,a=acc(d); print(f"  {name:<18}: {c}/{len(ids)} = {a:.1f}%")
# per-type for r2_merged
import collections
bytype=collections.defaultdict(lambda:[0,0])
for i in ids:
    t=r2[i].get("question_type","?"); bytype[t][1]+=1; bytype[t][0]+=int(bool(r2[i].get("correct")))
print("\nr2_merged 按题型(900):")
for t in sorted(bytype):
    c,n=bytype[t]; print(f"  {t:<22} {c}/{n} = {c/n*100:.1f}%")
PYEOF
echo "DONE-900"
