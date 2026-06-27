"""Step 4 — weight rubric criteria (1-5) and inject anti-gaming groundedness guards.

Takes a verifier-loadable rubric (general + type_aliases + templates) and:
  1. Asks an LLM to rate each criterion's importance (1-5) for deciding whether
     retrieved evidence is sufficient to verify/exclude the options, per question
     type and for the general block.
  2. Injects positively-framed "groundedness" guard criteria into ``general`` so
     every type penalizes a sufficiency judgment that leans on facts NOT actually
     present in the retrieved evidence (anti-gaming). Positive framing keeps the
     verifier's weighted-average in [0,1] — no negative weights, no math change.

Output is validated by loading it through the verifier rubric loader.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from src.generate_rubric.construct.llm import LLMRouter


# Anti-gaming guard criteria added to `general` (apply to every question type).
# Positive framing: 1.0 == judgment rests on real retrieved evidence; 0.0 == it
# leans on world knowledge / option-text echo / invented visual detail.
GUARD_CRITERIA = [
    {
        "name": "evidence_groundedness",
        "description": (
            "The verify/exclude decision for this option rests on facts ACTUALLY present in the "
            "retrieved evidence chunks, not on world knowledge, the wording of the option, or "
            "invented visual/audio details."
        ),
        "score_1": "Every fact used to confirm or rule out this option is explicitly stated in the retrieved evidence.",
        "score_half": "The decision partly relies on plausible inference that goes beyond what the evidence states.",
        "score_0": "The decision depends on details not present in the evidence (outside knowledge, option-text echo, or invented specifics).",
        "failure_repair_action": "refine_query",
    },
]


WEIGHT_SYSTEM = """\
You assign importance weights to evidence-sufficiency rubric criteria for long-video multiple-choice QA.
Each criterion checks whether the retrieved video evidence is sufficient to VERIFY or EXCLUDE the answer options
(not whether an answer is correct). Rate how essential each criterion is for THAT sufficiency decision:
  5 = indispensable; without it the sufficiency judgment is unreliable
  4 = very important
  3 = moderately important
  2 = minor or situational
  1 = nice-to-have
Output ONE strict JSON object mapping each criterion name to an integer 1-5. No prose, no markdown.
"""

WEIGHT_USER = """\
Question type: {qtype}

Criteria:
{criteria_block}

Return JSON mapping EVERY criterion name to an integer weight 1-5:
{{"<criterion_name>": <1-5>, ...}}
"""


def _norm(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _parse_weights(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        raw = json.loads(m.group())
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        try:
            out[_norm(k)] = max(1.0, min(5.0, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def _weight_group(llm: LLMRouter, qtype: str, criteria: list[dict]) -> None:
    """LLM-assign weights in place for one criterion list."""
    if not criteria:
        return
    block = "\n".join(f"- {c.get('name')}: {c.get('description', '')}" for c in criteria)
    raw = llm.chat(WEIGHT_SYSTEM, WEIGHT_USER.format(qtype=qtype, criteria_block=block), max_tokens=600)
    weights = _parse_weights(raw)
    for c in criteria:
        w = weights.get(_norm(c.get("name")))
        c["weight"] = round(w, 1) if w is not None else float(c.get("weight", 3.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rubric", required=True, help="Input verifier-loadable rubric YAML.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-api-url", default="http://127.0.0.1:8003,http://127.0.0.1:8004")
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--guard-weight", type=float, default=3.0,
                    help="Fixed weight for the injected groundedness guard criteria.")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.rubric).read_text(encoding="utf-8")) or {}
    general = data.get("general") or {}
    templates = data.get("templates") or {}
    if not general.get("rubric_criteria"):
        raise SystemExit(f"{args.rubric} has no general.rubric_criteria")

    llm = LLMRouter.from_urls(args.llm_api_url, model=args.model)

    # 1) weight general criteria
    _weight_group(llm, "GENERAL (applies to every question type)", general["rubric_criteria"])

    # 2) inject groundedness guard(s) into general with a fixed weight
    existing = {_norm(c.get("name")) for c in general["rubric_criteria"]}
    for guard in GUARD_CRITERIA:
        if _norm(guard["name"]) in existing:
            continue
        g = dict(guard)
        g["weight"] = float(args.guard_weight)
        general["rubric_criteria"].append(g)

    # 3) weight each type's criteria
    for key, tpl in templates.items():
        _weight_group(llm, key, tpl.get("rubric_criteria") or [])
        print(f"[weight] {key}: {len(tpl.get('rubric_criteria') or [])} criteria")

    out = {"general": general, "type_aliases": data.get("type_aliases") or {}, "templates": templates}
    header = (
        "# Weighted rubric (step 4): criteria importance 1-5 + positive-framed\n"
        "# groundedness guard in `general` (anti-gaming, no negative weights).\n"
        f"# source: {args.rubric}\n\n"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        header + yaml.safe_dump(out, sort_keys=False, allow_unicode=False, width=4096, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[weight] wrote {args.out}")

    if not args.no_validate:
        import importlib, os
        os.environ["VEIL_RUBRIC_PATH"] = str(Path(args.out))
        verifier = importlib.import_module("src.agents.verifier")
        verifier._rubric_config.cache_clear()
        g, t, al, _ = verifier._rubric_config()
        ws = [c.get("weight") for c in g["rubric_criteria"]]
        print(f"[weight] validated: general={len(g['rubric_criteria'])} (weights {ws}), templates={len(t)}, aliases={len(al)}")
        verifier._rubric_config.cache_clear()


if __name__ == "__main__":
    main()
