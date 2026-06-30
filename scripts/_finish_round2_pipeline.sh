#!/usr/bin/env bash
# One-off orchestrator: finish round2 extraction (resume to 720), then merge
# round2 delta onto the runtime round1 rubric, then validate via the verifier.
# Idempotent: round2 extraction resumes from its existing jsonl; merge uses a
# fresh state file (round2_merged_runtime.yaml.state.json).
set -uo pipefail
cd /home2/ycj/Project/VEIL
D=outputs/results/intermediate_goods/overnight_20260627_full720
PY=/home2/ycj/miniconda3/envs/veil/bin/python
URL="http://127.0.0.1:8003,http://127.0.0.1:8004"
WORKERS="${WORKERS:-32}"
AGG_WORKERS="${AGG_WORKERS:-8}"
export PYTHONPATH=.

echo "[1/3] resume round2 extraction -> 720"
"$PY" -m src.generate_rubric.construct.extract_pair_criteria \
  --dev "$D/splits/dev_questions.jsonl" \
  --chains "$D/round1_chains.jsonl" \
  --pairs "$D/round1_pairs.jsonl" \
  --rubric-path "$D/round1_type_level_rubric.yaml" \
  --llm-api-url "$URL" --workers "$WORKERS" \
  --out "$D/round2_pair_criteria.jsonl"
N=$(wc -l < "$D/round2_pair_criteria.jsonl")
echo "round2 pair_criteria now: $N/720"

echo "[2/3] merge round2 delta onto round1 base -> round2_merged_runtime.yaml"
rm -f "$D/round2_merged_runtime.yaml.state.json"
"$PY" -m src.generate_rubric.construct.aggregate_type_rubric \
  --criteria "$D/round2_pair_criteria.jsonl" \
  --base-rubric "$D/round1_type_level_rubric.yaml" \
  --llm-api-url "$URL" --agg-workers "$AGG_WORKERS" \
  --out "$D/round2_merged_runtime.yaml"

echo "[3/3] validate merged rubric via verifier.get_rubric_dict"
VEIL_RUBRIC_PATH="$D/round2_merged_runtime.yaml" "$PY" - <<'PYEOF'
from src.agents import verifier
verifier._rubric_config.cache_clear()
g, t, al, _ = verifier._rubric_config()
print("general:", len(g["rubric_criteria"]), "| templates:", len(t), "| aliases:", len(al))
issues = 0
for qtype, key in al.items():
    d = verifier.get_rubric_dict("", qtype)
    n = len(d.get("rubric_criteria") or [])
    if key not in t or n <= len(g["rubric_criteria"]):
        issues += 1
        print("  WARN", qtype, "n=", n, "routed=", key in t)
print("RESULT:", "all 12 types = general + type criteria OK" if issues == 0 else f"{issues} type(s) with issue")
verifier._rubric_config.cache_clear()
PYEOF
echo "DONE: $D/round2_merged_runtime.yaml"
