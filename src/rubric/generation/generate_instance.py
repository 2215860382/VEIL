"""Phase 1: Generate per-question instance rubrics with two models.

Reads questions from one of the result jsonls (any of them — they share Q/gold).
For each question, calls Opus 4.7 (via Anthropic proxy) and Qwen3.5-27B (via
local vLLM at 127.0.0.1:8003), saves raw responses + parsed YAML.

Output:
  src/rubric/artifacts/instances/instance_opus.jsonl    — one row per question
  src/rubric/artifacts/instances/instance_qwen.jsonl

Each row: {sample_idx, video_id, question_type, question, candidates, gold,
          rubric_yaml, raw_response, model, error}

Usage:
  python -m src.rubric.generation.generate_instance --model opus [--limit N] [--resume]
  python -m src.rubric.generation.generate_instance --model qwen [--limit N] [--resume]
  python -m src.rubric.generation.generate_instance --model both
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GENERATION_DIR = Path(__file__).resolve().parent
RUBRIC_DIR = GENERATION_DIR.parent
ROOT = RUBRIC_DIR.parents[1]
OUT_DIR = RUBRIC_DIR / 'artifacts/instances'
PROMPT_PATH = GENERATION_DIR / 'prompts/instance_gen.md'
SOURCE_JSONL = ROOT / 'outputs/results/videomme/tuning/mf_qf.jsonl'

OPUS_MODEL = 'claude-opus-4-7'
ANTHROPIC_BASE = 'https://crs.codechildczj.cn/api'
QWEN_URLS_DEFAULT = 'http://127.0.0.1:8003/v1/chat/completions'
QWEN_MODEL = 'Qwen3.5-27B'


def load_questions() -> list[dict]:
    rows = []
    with SOURCE_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                'sample_idx':    int(r['sample_idx']),
                'video_id':      r['video_id'],
                'question_type': r['question_type'],
                'question':      r['question'],
                'candidates':    r['candidates'],
                'gold':          r['gold_answer'],
            })
    return rows


def format_user_msg(q: dict) -> str:
    opts = '\n'.join(f'  ({chr(ord("A")+i)}) {c}' for i, c in enumerate(q['candidates']))
    return (
        f"question_type: {q['question_type']}\n\n"
        f"question: {q['question']}\n\n"
        f"candidates:\n{opts}\n\n"
        f"gold answer (verbatim text of the correct option): {q['gold']}"
    )


# ── Opus 4.7 via Anthropic proxy ──────────────────────────────────────────

def call_opus(client, system_prompt: str, user_msg: str) -> str:
    """Returns raw response text. System prompt is cached."""
    resp = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=4000,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
    )
    return next((b.text for b in resp.content if b.type == 'text'), '')


# ── Qwen3.5-27B via local vLLM ────────────────────────────────────────────

def call_qwen(session, url: str, system_prompt: str, user_msg: str) -> str:
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 4000,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = session.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


# ── Parse YAML from response ──────────────────────────────────────────────

def extract_yaml(raw: str) -> str:
    """Extract YAML between ```yaml ... ``` or return raw stripped."""
    import re
    m = re.search(r'```ya?ml\s*\n(.*?)\n```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)\n```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


# ── Main run loop ─────────────────────────────────────────────────────────

def run_model(model_name: str, questions: list[dict], system_prompt: str,
              resume: bool, workers: int, qwen_urls: list[str] | None = None):
    out_path = OUT_DIR / f'instance_{model_name}.jsonl'
    done_idx = set()
    if resume and out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                if not r.get('error'):
                    done_idx.add(int(r['sample_idx']))
            except Exception:
                pass
        print(f'  resume: skipping {len(done_idx)} already-done')

    todo = [q for q in questions if int(q['sample_idx']) not in done_idx]
    if not todo:
        print(f'  all done, skipping')
        return

    # Init client
    if model_name == 'opus':
        os.environ['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
        from anthropic import Anthropic
        client = Anthropic(base_url=ANTHROPIC_BASE)
        caller = lambda u: call_opus(client, system_prompt, u)
    elif model_name == 'qwen':
        import requests
        import itertools
        import threading as _th
        session = requests.Session()
        urls = qwen_urls or [QWEN_URLS_DEFAULT]
        print(f'  qwen endpoints ({len(urls)}): ' + ', '.join(urls))
        url_cycle = itertools.cycle(urls)
        _url_lock = _th.Lock()
        def _pick_url():
            with _url_lock:
                return next(url_cycle)
        caller = lambda u: call_qwen(session, _pick_url(), system_prompt, u)
    else:
        raise ValueError(model_name)

    fh = out_path.open('a')
    import threading
    write_lock = threading.Lock()
    n_done = [0]
    n_err = [0]
    t0 = time.time()

    def process(q):
        user_msg = format_user_msg(q)
        try:
            raw = caller(user_msg)
            rubric_yaml = extract_yaml(raw)
            rec = {**q, 'rubric_yaml': rubric_yaml, 'raw_response': raw,
                   'model': model_name, 'error': None}
        except Exception as e:
            rec = {**q, 'rubric_yaml': None, 'raw_response': None,
                   'model': model_name, 'error': str(e)[:300]}
            with write_lock:
                n_err[0] += 1
        with write_lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            n_done[0] += 1
            if n_done[0] % 10 == 0:
                rate = n_done[0] / max(time.time() - t0, 1)
                eta = (len(todo) - n_done[0]) / max(rate, 1e-6)
                print(f'  [{n_done[0]}/{len(todo)}] err={n_err[0]} '
                      f'rate={rate:.2f}/s eta={eta/60:.1f}min')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process, q) for q in todo]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f'  worker exception: {exc}')

    fh.close()
    print(f'  done: {n_done[0]} new records, {n_err[0]} errors, '
          f'total time {(time.time()-t0)/60:.1f}min')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['opus', 'qwen', 'both'], required=True)
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap number of questions (for smoke test)')
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel workers. Opus proxy: try 8. Qwen local: 4.')
    ap.add_argument('--qwen-urls', type=str, default=QWEN_URLS_DEFAULT,
                    help='Comma-separated Qwen vLLM chat-completions URLs; round-robin across them.')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPT_PATH.read_text()
    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    print(f'loaded {len(questions)} questions')

    qwen_urls = [u.strip() for u in args.qwen_urls.split(',') if u.strip()]

    targets = ['opus', 'qwen'] if args.model == 'both' else [args.model]
    for m in targets:
        print(f'\n=== {m} ===')
        run_model(m, questions, system_prompt, args.resume, args.workers,
                  qwen_urls=qwen_urls)


if __name__ == '__main__':
    main()
