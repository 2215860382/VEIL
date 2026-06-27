"""Aggregate pairwise criterion candidates into compact type-level rubrics."""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import yaml

from src.generate_rubric.construct.io import read_jsonl
from src.generate_rubric.construct.llm import LLMRouter
from src.generate_rubric.construct.prompts import (
    AGGREGATE_CHUNK_USER_TEMPLATE,
    AGGREGATE_MERGE_USER_TEMPLATE,
    AGGREGATE_SYSTEM,
    format_initial_rubric,
)

GENERAL_CRITERIA = [
    {
        "name": "evidence_coverage",
        "description": (
            "Evidence covers the key event, time window, state change, comparison target, "
            "or causal chain needed to verify or exclude each option."
        ),
        "score_1": "All evidence needed to verify or exclude the option is covered.",
        "score_half": "Relevant evidence is present, but a key event, time window, state, comparison, or causal link is missing.",
        "score_0": "The evidence does not cover the facts needed to judge the option.",
        "weight": 1.0,
        "failure_repair_action": "dense_sample",
    },
    {
        "name": "evidence_specificity",
        "description": (
            "Evidence explicitly grounds the relevant person, object, action, scene, speaker, "
            "text, or spatial relation instead of only giving a generic related scene."
        ),
        "score_1": "The required entities, actions, attributes, or relations are unambiguously grounded.",
        "score_half": "The evidence is relevant but leaves an entity, action, attribute, or relation ambiguous.",
        "score_0": "The evidence is generic or concerns a different entity, action, attribute, or relation.",
        "weight": 1.0,
        "failure_repair_action": "refine_query",
    },
    {
        "name": "evidence_consistency",
        "description": (
            "Evidence is internally consistent and contains enough contrary evidence to exclude "
            "options that conflict with the video."
        ),
        "score_1": "The evidence consistently establishes or rules out the option without unresolved conflict.",
        "score_half": "The evidence points in one direction but leaves a conflict or distractor unresolved.",
        "score_0": "The evidence conflicts internally or cannot establish or rule out the option.",
        "weight": 1.0,
        "failure_repair_action": "refine_query",
    },
]


def _chunk(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _clean_yaml_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    return text.strip()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"types": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _qkey(qtype: str) -> str:
    return qtype.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


_SCALAR_FIELDS = ("name", "description", "score_1", "score_half", "score_0",
                  "failure_repair_action", "scoring_rule")


def _quote_scalars(block: str) -> str:
    """Wrap unquoted scalar values in double quotes.

    LLM-emitted YAML frequently leaves plain scalars containing ``: `` (e.g.
    ``score_1: At time A: state X``) which is invalid YAML. Re-quoting the value
    of every known scalar field makes the block parseable.
    """
    out = []
    pat = re.compile(r'^(\s*)(- )?(' + "|".join(_SCALAR_FIELDS) + r'):\s+(.*)$')
    for line in block.splitlines():
        m = pat.match(line)
        if not m:
            out.append(line)
            continue
        indent, dash, key, val = m.groups()
        val = val.strip()
        if val and val[0] not in "\"'[{" and not re.fullmatch(r'-?\d+(\.\d+)?', val):
            val = '"' + val.replace("\\", "\\\\").replace('"', '\\"') + '"'
        out.append(f"{indent}{dash or ''}{key}: {val}")
    return "\n".join(out)


def _safe_parse_block(block: str) -> dict | None:
    """Parse one LLM-emitted type rubric block, repairing unquoted scalars."""
    block = block.strip()
    for attempt in (block, _quote_scalars(block)):
        try:
            data = yaml.safe_load(attempt)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("rubric_criteria"):
            return data
    return None


def _render_final_yaml(state: dict) -> str:
    """Assemble a verifier-loadable rubric (general + type_aliases + templates).

    Each per-type ``final_yaml`` block is parsed and re-dumped (not string
    spliced), so the output is guaranteed to be valid YAML; unparseable type
    blocks are skipped with a warning rather than corrupting the whole file.
    """
    templates: dict = {}
    for qtype in sorted(state["types"]):
        block = (state["types"][qtype].get("final_yaml") or "").strip()
        if not block:
            continue
        parsed = _safe_parse_block(block)
        if not parsed:
            print(f"[render] WARN: unparseable rubric block for {qtype!r}; skipped")
            continue
        tpl: dict = {"rubric_criteria": parsed.get("rubric_criteria") or []}
        if "scoring_rule" in parsed:
            tpl["scoring_rule"] = parsed["scoring_rule"]
        if "sufficient_threshold" in parsed:
            tpl["sufficient_threshold"] = parsed["sufficient_threshold"]
        templates[_qkey(qtype)] = tpl

    document = {
        "general": {
            "rubric_criteria": GENERAL_CRITERIA,
            "scoring_rule": "average",
            "sufficient_threshold": 0.80,
        },
        "type_aliases": {
            qtype: _qkey(qtype)
            for qtype in sorted(state["types"])
            if _qkey(qtype) in templates
        },
        "templates": templates,
    }
    header = (
        "# Auto-generated evidence-sufficiency rubric.\n"
        "# Criteria evaluate whether evidence can verify or exclude options, not answer correctness.\n\n"
    )
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=False, width=4096, default_flow_style=False)
    return header + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", default=None)
    ap.add_argument("--out", default="outputs/rubric/type_level_rubric.yaml")
    ap.add_argument("--llm-api-url", default="http://127.0.0.1:8003,http://127.0.0.1:8004")
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--max-criteria", type=int, default=7)
    ap.add_argument("--chunk-size", type=int, default=24)
    ap.add_argument(
        "--base-rubric",
        default=None,
        help="Optional runtime rubric whose type criteria are merged with the new criterion delta.",
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render the final YAML from an existing complete .state.json without calling any LLM.",
    )
    ap.add_argument(
        "--agg-workers",
        type=int,
        default=8,
        help="Number of question types aggregated concurrently (client-side threads; "
        "GPU memory is unaffected — vLLM queues beyond max-num-seqs).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    state_path = Path(str(out_path) + ".state.json")

    # ── Render-only: reuse a complete state, never touch the LLM endpoints ──────
    if args.render_only:
        state = _load_state(state_path)
        types = state.get("types") or {}
        if not types:
            raise SystemExit(f"--render-only: no usable state at {state_path}")
        missing = [t for t, v in types.items() if not v.get("final_yaml")]
        if missing:
            raise SystemExit(
                f"--render-only: state has {len(missing)} type(s) without final_yaml: {sorted(missing)}"
            )
        out_path.write_text(_render_final_yaml(state), encoding="utf-8")
        print(f"[render-only] wrote {args.out} ({len(types)} types, no LLM calls)")
        return

    if not args.criteria:
        raise SystemExit("--criteria is required unless --render-only is set")

    by_type = defaultdict(list)
    for row in read_jsonl(args.criteria):
        for c in row.get("new_criteria") or []:
            c = dict(c)
            c["source_pair_id"] = row["pair_id"]
            by_type[row["question_type"]].append(c)

    if args.base_rubric:
        base = yaml.safe_load(Path(args.base_rubric).read_text(encoding="utf-8")) or {}
        aliases = base.get("type_aliases") or {}
        reverse_aliases = {str(v): str(k) for k, v in aliases.items()}
        for key, template in (base.get("templates") or {}).items():
            qtype = reverse_aliases.get(str(key), str(key).replace("_", " ").title())
            for criterion in template.get("rubric_criteria") or []:
                criterion = dict(criterion)
                criterion["source_pair_id"] = f"base:{criterion.get('name', 'criterion')}"
                by_type[qtype].append(criterion)

    # Lazy LLM init: only dial the endpoints when a type actually needs work,
    # so a fully cached state can be re-rendered (or a no-op run) without hanging.
    _llm_box: dict = {}

    def get_llm() -> LLMRouter:
        if "llm" not in _llm_box:
            _llm_box["llm"] = LLMRouter.from_urls(args.llm_api_url, model=args.model)
        return _llm_box["llm"]

    state = _load_state(state_path)
    state.setdefault("types", {})

    # Pre-create every type entry BEFORE threading so no thread inserts a new key
    # into state["types"] concurrently (keeps _save_state's json.dumps consistent).
    pending: list[tuple[str, list]] = []
    for qtype, criteria in sorted(by_type.items()):
        type_state = state["types"].setdefault(qtype, {"chunks": [], "final_yaml": ""})
        if not type_state.get("final_yaml"):
            pending.append((qtype, criteria))

    state_lock = threading.Lock()

    def process_type(qtype: str, criteria: list) -> str:
        type_state = state["types"][qtype]
        chunks = _chunk(criteria, max(1, args.chunk_size))
        chunk_summaries: list[dict] = type_state.get("chunks", [])

        for idx, chunk in enumerate(chunks, 1):
            if len(chunk_summaries) >= idx and chunk_summaries[idx - 1].get("criteria_ids") == [c["source_pair_id"] for c in chunk]:
                continue
            user = AGGREGATE_CHUNK_USER_TEMPLATE.format(
                initial_rubric=format_initial_rubric(),
                question_type=qtype,
                criteria_json=json.dumps(chunk, ensure_ascii=False, indent=2),
                max_criteria=max(2, min(args.max_criteria, 4)),
            )
            yaml_block = _clean_yaml_block(get_llm().chat(AGGREGATE_SYSTEM, user, max_tokens=2200))
            entry = {
                "chunk_index": idx,
                "criteria_ids": [c["source_pair_id"] for c in chunk],
                "yaml_block": yaml_block,
            }
            with state_lock:
                if len(chunk_summaries) >= idx:
                    chunk_summaries[idx - 1] = entry
                else:
                    chunk_summaries.append(entry)
                type_state["chunks"] = chunk_summaries
                _save_state(state_path, state)

        merge_input = "\n\n".join(
            f"[chunk {item['chunk_index']}]\n{item['yaml_block']}" for item in chunk_summaries
        )
        merge_user = AGGREGATE_MERGE_USER_TEMPLATE.format(
            initial_rubric=format_initial_rubric(),
            question_type=qtype,
            chunk_summaries=merge_input,
            max_criteria=args.max_criteria,
        )
        final_yaml = _clean_yaml_block(get_llm().chat(AGGREGATE_SYSTEM, merge_user, max_tokens=2600))
        with state_lock:
            type_state["final_yaml"] = final_yaml
            _save_state(state_path, state)
        return qtype

    if pending:
        get_llm()  # init the router once, before threads, to avoid a lazy-init race
        workers = max(1, min(args.agg_workers, len(pending)))
        print(f"[agg] aggregating {len(pending)} type(s) with {workers} concurrent worker(s)")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_type, q, c): q for q, c in pending}
            for fut in as_completed(futs):
                q = futs[fut]
                try:
                    fut.result()
                    print(f"[agg] done: {q}")
                except Exception as exc:  # noqa: BLE001 — surface, keep other types
                    print(f"[agg] FAILED: {q}: {exc}")

    out_path.write_text(_render_final_yaml(state), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
