"""
Analyze VEIL verifier failures to generate improved rubric rules.

Two failure modes:
  A. sufficient + wrong (46 cases): rubric too lenient → generate stricter gate/criterion
  B. insufficient + correct (37 cases): rubric too strict → find what was already satisfied

Outputs a JSON file with LLM-generated rule suggestions per case,
plus a summary aggregated by question_type.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import requests

# ── Config ─────────────────────────────────────────────────────────────────────
RESULT_FILE  = Path("/home2/ycj/Project/VEIL/outputs/results/videommeL/20260515_veil-subq/results.jsonl")
BANK_DIR     = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b")
OUT_FILE     = Path("/home2/ycj/Project/VEIL/outputs/rubric_analysis/failure_rules.jsonl")
SUMMARY_FILE = Path("/home2/ycj/Project/VEIL/outputs/rubric_analysis/rule_summary.json")

API_ENDPOINTS = [
    "http://127.0.0.1:8780",
    "http://127.0.0.1:8782",
    "http://127.0.0.1:8783",
]
MODEL_NAME   = "Qwen3.5-27B"
MAX_TOKENS   = 400

# ── LLM call ───────────────────────────────────────────────────────────────────
_ep_idx = 0

def llm_chat(messages: List[Dict], max_tokens: int = MAX_TOKENS) -> str:
    global _ep_idx
    ep = API_ENDPOINTS[_ep_idx % len(API_ENDPOINTS)]
    _ep_idx += 1
    resp = requests.post(
        f"{ep}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Prompts ────────────────────────────────────────────────────────────────────
SYS_TOO_LENIENT = """\
You are a rubric designer for a video-QA verifier.
The verifier judged the evidence as SUFFICIENT, but the model answered INCORRECTLY.
Your job: identify exactly what critical information is MISSING from the evidence
that caused the wrong answer, and write ONE binary gate condition that would catch this.

Output ONLY a JSON object with these fields:
{
  "missing_info": "one sentence: what specific fact is absent from the evidence",
  "gate_name": "snake_case_name (≤5 words)",
  "gate_condition": "one sentence: the condition that must be TRUE for evidence to be sufficient",
  "fail_reason": "one sentence: why failure here means the question cannot be answered"
}"""

SYS_TOO_STRICT = """\
You are a rubric designer for a video-QA verifier.
The verifier judged the evidence as INSUFFICIENT and triggered extra retrieval,
but the model actually answered CORRECTLY from the existing evidence.
Your job: identify what criterion WAS already satisfied that the verifier missed,
and write ONE rubric criterion that would correctly rate this evidence as sufficient.

Output ONLY a JSON object with these fields:
{
  "satisfied_info": "one sentence: what specific fact IS present in the evidence",
  "criterion_name": "snake_case_name (≤5 words)",
  "criterion_description": "one sentence: what this criterion checks for",
  "score_1": "what makes it fully satisfied (score=1.0)",
  "score_half": "what makes it partially satisfied (score=0.5)",
  "score_0": "what makes it unsatisfied (score=0.0)"
}"""


def build_user_lenient(q: str, candidates: List[str], evidence: List[str],
                        gold: str, pred: str) -> str:
    opts = "\n".join(f"  ({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev   = "\n".join(f"  [E{i+1}] {t[:300]}" for i, t in enumerate(evidence))
    return (
        f"Question: {q}\n"
        f"Options:\n{opts}\n\n"
        f"Evidence (judged sufficient by verifier):\n{ev}\n\n"
        f"Correct answer: ({gold})  |  Model answered: ({pred})\n\n"
        "What gate would catch this as insufficient?"
    )


def build_user_strict(q: str, candidates: List[str], evidence: List[str],
                       gold: str) -> str:
    opts = "\n".join(f"  ({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev   = "\n".join(f"  [E{i+1}] {t[:300]}" for i, t in enumerate(evidence))
    return (
        f"Question: {q}\n"
        f"Options:\n{opts}\n\n"
        f"Evidence (judged insufficient by verifier, but actually sufficient):\n{ev}\n\n"
        f"Correct answer: ({gold})\n\n"
        "What criterion IS already satisfied that the verifier missed?"
    )


# ── Data loading ───────────────────────────────────────────────────────────────
def load_bank(video_id: str) -> Dict[int, Dict]:
    """Return {chunk_id: chunk_dict}."""
    p = BANK_DIR / f"{video_id}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {c["chunk_id"]: c for c in data.get("chunks", [])}


def reconstruct_evidence(video_id: str, chunk_ids: List[int]) -> List[str]:
    bank = load_bank(video_id)
    texts = []
    for cid in chunk_ids:
        c = bank.get(cid)
        if c:
            t = f"[{c['start_time']:.0f}s-{c['end_time']:.0f}s] {c['memory_text']}"
            if c.get("asr", "").strip():
                t += f"\nSpeech: {c['asr']}"
            texts.append(t)
    return texts


def load_results(path: Path, pipeline: str) -> List[Dict]:
    results = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("pipeline") == pipeline:
                results.append(d)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(RESULT_FILE, "veil_coarse8_27b")

    # Classify
    too_lenient = []  # sufficient + wrong
    too_strict  = []  # insufficient + correct

    for r in results:
        trace = r.get("trace_iters", [])
        if not trace:
            continue
        final_verdict = trace[-1]["verdict"]
        correct = r["correct"]
        if final_verdict == "sufficient" and not correct:
            too_lenient.append(r)
        elif final_verdict == "insufficient" and correct:
            too_strict.append(r)

    print(f"Too lenient (sufficient+wrong): {len(too_lenient)}")
    print(f"Too strict  (insufficient+correct): {len(too_strict)}")

    already_done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                d = json.loads(line)
                already_done.add(d["key"])
        print(f"Already processed: {len(already_done)}, resuming...")

    out = open(OUT_FILE, "a")

    def process_batch(samples, failure_type):
        for i, r in enumerate(samples):
            key = r["key"]
            if key in already_done:
                continue

            video_id   = r["video_id"]
            question   = r["question"]
            candidates = r["candidates"]
            gold       = r["gold"]
            pred       = r.get("pred_letter", "?")
            chunk_ids  = r.get("evidence_chunk_ids", [])
            qtype      = r.get("question_type", "")
            trace      = r.get("trace_iters", [])
            verifier_reasoning = trace[-1].get("reasoning", "") if trace else ""

            evidence = reconstruct_evidence(video_id, chunk_ids)
            if not evidence:
                print(f"  [{i+1}] skip {key[:40]} — no evidence")
                continue

            if failure_type == "too_lenient":
                sys_p  = SYS_TOO_LENIENT
                user_p = build_user_lenient(question, candidates, evidence, gold, pred)
            else:
                sys_p  = SYS_TOO_STRICT
                user_p = build_user_strict(question, candidates, evidence, gold)

            try:
                raw = llm_chat([
                    {"role": "system", "content": sys_p},
                    {"role": "user",   "content": user_p},
                ])
                # parse JSON
                import re
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                parsed = json.loads(m.group()) if m else {"raw": raw}
            except Exception as e:
                parsed = {"error": str(e), "raw": raw if 'raw' in dir() else ""}

            record = {
                "key":          key,
                "failure_type": failure_type,
                "question_type": qtype,
                "question":     question,
                "gold":         gold,
                "pred":         pred if failure_type == "too_lenient" else None,
                "verifier_reasoning": verifier_reasoning,
                "suggested_rule": parsed,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            already_done.add(key)

            tag = "LENIENT" if failure_type == "too_lenient" else "STRICT"
            print(f"  [{i+1}/{len(samples)}] {tag} {qtype:30s} -> {list(parsed.keys())[:3]}")

    print("\n=== Processing too_lenient (sufficient+wrong) ===")
    process_batch(too_lenient, "too_lenient")

    print("\n=== Processing too_strict (insufficient+correct) ===")
    process_batch(too_strict, "too_strict")

    out.close()
    print(f"\nDone. Results -> {OUT_FILE}")

    # ── Aggregate summary ──────────────────────────────────────────────────────
    print("Generating summary...")
    records = []
    with open(OUT_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    summary = {"too_lenient": {}, "too_strict": {}}
    for r in records:
        ft   = r["failure_type"]
        qt   = r.get("question_type", "unknown")
        rule = r.get("suggested_rule", {})
        if qt not in summary[ft]:
            summary[ft][qt] = []
        summary[ft][qt].append(rule)

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary -> {SUMMARY_FILE}")

    # Print quick overview
    print("\n=== Too Lenient: suggested gates by question_type ===")
    for qt, rules in sorted(summary["too_lenient"].items()):
        print(f"  {qt} ({len(rules)}개):")
        for rule in rules[:2]:
            print(f"    gate: {rule.get('gate_name','?')} — {rule.get('gate_condition','')[:80]}")

    print("\n=== Too Strict: suggested criteria by question_type ===")
    for qt, rules in sorted(summary["too_strict"].items()):
        print(f"  {qt} ({len(rules)}개):")
        for rule in rules[:2]:
            print(f"    crit: {rule.get('criterion_name','?')} — {rule.get('criterion_description','')[:80]}")


if __name__ == "__main__":
    main()
