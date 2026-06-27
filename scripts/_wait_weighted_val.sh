#!/usr/bin/env bash
set -uo pipefail
cd /home2/ycj/Project/VEIL
PY=/home2/ycj/miniconda3/envs/veil/bin/python
F=outputs/results/videomme/tuning/val_round2_weighted.jsonl
for _ in $(seq 1 240); do
  n=$(grep -c '"sample_idx"' "$F" 2>/dev/null || echo 0)
  [ "$n" -ge 90 ] && break
  sleep 30
done
echo "weighted val done: $(grep -c '"sample_idx"' "$F") rows"
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
runs={n:load(f) for n,f in [
 ("v2","outputs/results/videomme/tuning/val_v2.jsonl"),
 ("round1","outputs/results/videomme/tuning/val_round1.jsonl"),
 ("r2_merged","outputs/results/videomme/tuning/val_round2_merged.jsonl"),
 ("r2_weighted","outputs/results/videomme/tuning/val_round2_weighted.jsonl"),
 ("tsqf","outputs/results/videomme/tuning/mf_tsqf.jsonl"),
]}
ref=runs["v2"]; qt={i:ref[i]["question_type"] for i in val if i in ref}
print(f"{'type':<20} {'n':>2} |"+ "".join(f"{k:>11}" for k in runs))
print("-"*78)
for t in sorted(set(qt.values())):
    ids=[i for i in val if qt.get(i)==t]; row=f"{t:<20} {len(ids):>2} |"
    for k,d in runs.items():
        c=sum(1 for i in ids if i in d and d[i].get("correct")); row+=f"{c}/{len(ids):>9}"
    print(row)
print("-"*78)
row=f"{'TOTAL':<20} {90:>2} |"
for k,d in runs.items():
    c=sum(1 for i in val if i in d and d[i].get("correct")); row+=f"{c/90*100:>10.1f}%"
print(row)
PYEOF
