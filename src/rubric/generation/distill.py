"""Phase 2: Distill per-qtype + general rubric from instance rubrics.

Reads:
  src/rubric/artifacts/instances/instance_qwen.jsonl
  src/rubric/artifacts/instances/instance_opus.jsonl   (optional, mixed if present)

Process:
  - Group by question_type (12 VideoMME types). For each type, sample N instance
    rubrics (mixed across models), feed to Qwen3.5-27B → per-qtype YAML.
  - Feed 12 per-qtype YAMLs to Qwen3.5-27B → general YAML.
  - Merge all into rubric_templates_v2.yaml.

Output:
  src/rubric/artifacts/distilled/qtype_{qkey}.raw.txt + .yaml
  src/rubric/artifacts/distilled/general.raw.txt + .yaml
  src/rubric/artifacts/distilled/rubric_templates_v2.yaml
  src/rubric/templates/generated_v2.yaml                     (runtime template)

Usage:
  python -m src.rubric.generation.distill [--sample-per-qtype 40] [--qwen-urls URL1,URL2,...] [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

GENERATION_DIR = Path(__file__).resolve().parent
RUBRIC_DIR = GENERATION_DIR.parent
INSTANCE_DIR = RUBRIC_DIR / 'artifacts/instances'
OUT_DIR = RUBRIC_DIR / 'artifacts/distilled'
PROMPT_DIR = GENERATION_DIR / 'prompts'
RUNTIME_TEMPLATE = RUBRIC_DIR / 'templates/generated_v2.yaml'

QWEN_MODEL = 'Qwen3.5-27B'
QWEN_URLS_DEFAULT = (
    'http://127.0.0.1:8000/v1/chat/completions,'
    'http://127.0.0.1:8001/v1/chat/completions,'
    'http://127.0.0.1:8003/v1/chat/completions,'
    'http://10.82.1.145:8003/v1/chat/completions'
)

# VideoMME question types → snake_case keys for YAML
QTYPE_KEYS = {
    "Action Reasoning":      "action_reasoning",
    "Action Recognition":    "action_recognition",
    "Attribute Perception":  "attribute_perception",
    "Counting Problem":      "counting_problem",
    "Information Synopsis":  "information_synopsis",
    "OCR Problems":          "ocr_problems",
    "Object Reasoning":      "object_reasoning",
    "Object Recognition":    "object_recognition",
    "Spatial Perception":    "spatial_perception",
    "Spatial Reasoning":     "spatial_reasoning",
    "Temporal Perception":   "temporal_perception",
    "Temporal Reasoning":    "temporal_reasoning",
}


def load_instance_rubrics() -> dict[str, list[dict]]:
    """Return qtype → list of {model, sample_idx, question, gold, rubric_yaml}."""
    by_qtype: dict[str, list[dict]] = defaultdict(list)
    for model in ['opus', 'qwen']:
        p = INSTANCE_DIR / f'instance_{model}.jsonl'
        if not p.exists():
            print(f'  warn: {p} missing')
            continue
        for line in p.open():
            r = json.loads(line)
            if r.get('error') or not r.get('rubric_yaml'):
                continue
            by_qtype[r['question_type']].append({
                'model':       model,
                'sample_idx':  r['sample_idx'],
                'question':    r['question'],
                'gold':        r['gold'],
                'rubric_yaml': r['rubric_yaml'],
            })
    return by_qtype


def format_qtype_user_msg(qtype: str, items: list[dict]) -> str:
    """Render instance rubrics for one qtype into a single user message."""
    parts = [f'question_type: {qtype}\n',
             f'Number of instance rubrics shown below: {len(items)}\n']
    for i, it in enumerate(items, 1):
        parts.append(f'\n--- instance {i} (model={it["model"]}, sample_idx={it["sample_idx"]}) ---')
        parts.append(f'question: {it["question"]}')
        parts.append(f'gold: {it["gold"]}')
        parts.append(f'rubric_yaml:\n{it["rubric_yaml"]}')
    return '\n'.join(parts)


def format_general_user_msg(qtype_yamls: dict[str, str]) -> str:
    parts = ['You have 11 question-type rubrics below. Produce the general rubric.\n']
    for qtype, yaml_str in qtype_yamls.items():
        parts.append(f'\n--- question_type: {qtype} ---\n{yaml_str}')
    return '\n'.join(parts)


def extract_yaml(raw: str) -> str:
    m = re.search(r'```ya?ml\s*\n(.*?)\n```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)\n```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def call_qwen(session, url: str, system_prompt: str, user_msg: str,
              max_tokens: int = 6000) -> str:
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = session.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-per-qtype', type=int, default=40,
                    help='Max instance rubrics per qtype to feed LLM.')
    ap.add_argument('--qwen-urls', type=str, default=QWEN_URLS_DEFAULT,
                    help='Comma-separated Qwen vLLM endpoints; round-robin across them.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import requests
    import itertools
    import threading as _th
    urls = [u.strip() for u in args.qwen_urls.split(',') if u.strip()]
    print(f'qwen endpoints ({len(urls)}): ' + ', '.join(urls))
    session = requests.Session()
    url_cycle = itertools.cycle(urls)
    _url_lock = _th.Lock()
    def _pick_url():
        with _url_lock:
            return next(url_cycle)

    print('=== loading instance rubrics ===')
    by_qtype = load_instance_rubrics()
    for qt, items in sorted(by_qtype.items()):
        n_opus = sum(1 for it in items if it['model'] == 'opus')
        n_qwen = sum(1 for it in items if it['model'] == 'qwen')
        print(f'  {qt:<22} opus={n_opus:>3}  qwen={n_qwen:>3}  total={len(items):>4}')

    if args.dry_run:
        print('dry-run, exiting')
        return

    # ── Phase 2a: per-qtype distillation ───────────────────────────────────
    print('\n=== Phase 2a: per-qtype distillation ===')
    qtype_system = (PROMPT_DIR / 'distill_qtype.md').read_text()
    qtype_yamls: dict[str, str] = {}
    for qtype, qkey in QTYPE_KEYS.items():
        items = by_qtype.get(qtype, [])
        if not items:
            print(f'  {qtype}: SKIP (no instances)')
            continue
        # Balanced sample: half opus, half qwen
        cap = args.sample_per_qtype
        opus_items = [it for it in items if it['model'] == 'opus']
        qwen_items = [it for it in items if it['model'] == 'qwen']
        random.shuffle(opus_items); random.shuffle(qwen_items)
        sampled = opus_items[:cap // 2] + qwen_items[:cap // 2]
        if len(sampled) < cap:
            extra = [it for it in items if it not in sampled]
            random.shuffle(extra)
            sampled += extra[:cap - len(sampled)]
        random.shuffle(sampled)

        print(f'  {qtype}: distilling from {len(sampled)} instance rubrics...')
        user_msg = format_qtype_user_msg(qtype, sampled)
        raw = call_qwen(session, _pick_url(), qtype_system, user_msg, max_tokens=6000)
        yaml_part = extract_yaml(raw)
        (OUT_DIR / f'qtype_{qkey}.raw.txt').write_text(raw)
        (OUT_DIR / f'qtype_{qkey}.yaml').write_text(yaml_part)
        qtype_yamls[qtype] = yaml_part
        print(f'    -> wrote qtype_{qkey}.yaml ({len(yaml_part)} chars)')

    # ── Phase 2b: cross-qtype general distillation ─────────────────────────
    print('\n=== Phase 2b: general distillation ===')
    general_system = (PROMPT_DIR / 'distill_general.md').read_text()
    user_msg = format_general_user_msg(qtype_yamls)
    raw = call_qwen(session, _pick_url(), general_system, user_msg, max_tokens=4000)
    yaml_part = extract_yaml(raw)
    (OUT_DIR / 'general.raw.txt').write_text(raw)
    (OUT_DIR / 'general.yaml').write_text(yaml_part)
    print(f'  wrote general.yaml ({len(yaml_part)} chars)')

    # ── Phase 2c: combine into rubric_templates_v2.yaml ────────────────────
    print('\n=== Phase 2c: combine ===')
    combined = [
        '# Auto-generated by src.rubric.generation.distill — DO NOT EDIT BY HAND.',
        '# Distilled by Qwen3.5-27B from per-question rubrics generated by Qwen3.5-27B (+ optional claude-opus-4-7).',
        '',
        yaml_part,
        '',
        '# Map VideoMME question_type → snake_case template key.',
        'type_aliases:',
    ]
    for qtype, qkey in QTYPE_KEYS.items():
        if qtype in qtype_yamls:
            combined.append(f'  {qtype!r}: {qkey}')
    combined.append('')
    combined.append('# Per-qtype templates')
    combined.append('templates:')
    for qtype, qkey in QTYPE_KEYS.items():
        if qtype not in qtype_yamls:
            continue
        block = qtype_yamls[qtype]
        # Indent the block under 'templates:'
        indented = '\n'.join('  ' + line for line in block.split('\n'))
        combined.append(indented)
        combined.append('')
    combined_text = '\n'.join(combined)
    (OUT_DIR / 'rubric_templates_v2.yaml').write_text(combined_text)
    RUNTIME_TEMPLATE.write_text(combined_text)
    print(f'  wrote rubric_templates_v2.yaml and {RUNTIME_TEMPLATE}')


if __name__ == '__main__':
    main()
