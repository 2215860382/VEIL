import json, sys, time, types
sys.path.insert(0, '/home2/ycj/Project/VEIL')
from reasoning.answerer import ANSWERER_SYS, _normalize, _format_options, _format_evidence
from utils.jsonx import extract_json
from models.llm_client import LLMClient

INPUT = '/home2/ycj/Project/VedioSifter/data/coarse_retrieval_bge.jsonl'
questions = [json.loads(l) for l in open(INPUT)][:15]

llm = LLMClient(model_path='Qwen3.5-27B',
                api_url='http://127.0.0.1:8780,http://127.0.0.1:8783',
                api_model='Qwen3.5-27B')

def _chat_api_patched(self, messages, n_tokens, temperature, enable_thinking):
    import random
    ep = random.choice(self._api_endpoints)
    payload = {"model": self._api_model, "messages": messages,
               "max_tokens": n_tokens, "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    resp = self._requests.post(ep, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

llm._chat_api = types.MethodType(_chat_api_patched, llm)

def fmt_evidence(chunk):
    t0, t1 = chunk.get('t_start', 0), chunk.get('t_end', 0)
    summary = (chunk.get('semantic_summary') or '').strip()
    speech  = (chunk.get('speech_text') or '').strip()
    parts = [f"[{t0:.0f}s-{t1:.0f}s] {summary}"]
    if speech: parts.append(f"Speech: {speech}")
    return "\n".join(parts)

correct = total = 0
for q in questions:
    candidates = [c.split('. ', 1)[-1] if '. ' in c else c for c in q.get('choices', [])]
    gold = q.get('answer', '').strip().upper()
    evidence_texts = [fmt_evidence(c) for c in q.get('candidates', [])]
    per = 80000 // len(evidence_texts)
    evidence_texts = [t[:per] for t in evidence_texts]

    user = (f"Question: {q['question']}\n"
            f"Options:\n{_format_options(candidates)}\n\n"
            f"Evidence:\n{_format_evidence(evidence_texts)}\n\nReturn the JSON now.")
    messages = [{"role": "system", "content": ANSWERER_SYS},
                {"role": "user",   "content": user}]

    t0 = time.time()
    try:
        raw = llm.chat(messages, max_new_tokens=4096, enable_thinking=True)
        elapsed = time.time() - t0
        result = _normalize(extract_json(raw))
        pred = result.get('answer', '')
    except Exception as e:
        elapsed = time.time() - t0
        pred = ''
        print(f"  ERROR {q['question_id']}: {e}", flush=True)

    ok = pred == gold
    if ok: correct += 1
    total += 1
    print(f"  {q['question_id']}  gold={gold}  pred={pred}  {'✓' if ok else '✗'}  {elapsed:.0f}s", flush=True)

print(f"\nthinking: {correct}/{total} = {correct/total*100:.1f}%")
print(f"no-think: 10/15 = 66.7%")
