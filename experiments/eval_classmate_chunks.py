"""
用同学的 top-64 pre-retrieved chunks + 我们的 Qwen3.5-27B 答题模型，
测量 306 题准确率，定位差距来源是检索还是答题。
"""
import json, sys, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reasoning.answerer import Answerer, ANSWERER_SYS
from src.utils.jsonx import extract_json, as_str, as_list
from src.models.llm_client import LLMClient


def fmt_evidence(chunk: dict) -> str:
    t0, t1 = chunk.get('t_start', 0), chunk.get('t_end', 0)
    summary = (chunk.get('semantic_summary') or chunk.get('text') or '').strip()
    speech  = (chunk.get('speech_text') or '').strip()
    parts = [f"[{t0:.0f}s-{t1:.0f}s] {summary}"]
    if speech:
        parts.append(f"Speech: {speech}")
    return "\n".join(parts)


def answer_one(llm, q: dict) -> dict:
    candidates_raw = q.get('choices', [])
    # strip "A. " prefix → plain text
    candidates = [c.split('. ', 1)[-1] if '. ' in c else c for c in candidates_raw]
    gold_letter = (q.get('answer') or '').strip().upper()

    chunks = q.get('candidates', [])
    evidence_texts = [fmt_evidence(c) for c in chunks]

    answerer = Answerer(llm)
    result = answerer.answer(q['question'], candidates, evidence_texts,
                             max_evidence_chars=80000)
    pred = result.get('answer', '').upper()
    correct = (pred == gold_letter)
    return {
        'qa_id':     q.get('question_id', ''),
        'video_id':  q.get('video_id', ''),
        'question':  q['question'],
        'gold':      gold_letter,
        'pred':      pred,
        'correct':   correct,
        'rationale': result.get('rationale', ''),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default='/home2/ycj/Project/VedioSifter/data/coarse_retrieval_bge.jsonl')
    ap.add_argument('--out',    default='/home2/ycj/Project/VEIL/outputs/results/videommeL/eval_classmate_chunks_our_llm.jsonl')
    ap.add_argument('--llm-api-url',   default='http://127.0.0.1:8780,http://127.0.0.1:8783')
    ap.add_argument('--llm-api-model', default='Qwen3.5-27B')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(questions)} questions")

    # skip already done
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try: done.add(json.loads(line)['qa_id'])
            except: pass
    todo = [q for q in questions if q.get('question_id', '') not in done]
    print(f"todo: {len(todo)}  (already done: {len(done)})")

    llm = LLMClient(
        model_path=args.llm_api_model,
        api_url=args.llm_api_url,
        api_model=args.llm_api_model,
    )

    correct = sum(1 for l in open(out_path) if json.loads(l).get('correct')) if out_path.exists() and done else 0
    total   = len(done)

    with open(out_path, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(answer_one, llm, q): q for q in todo}
            for i, fut in enumerate(as_completed(futures)):
                try:
                    r = fut.result()
                    fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                    fout.flush()
                    total += 1
                    if r['correct']: correct += 1
                    if (i+1) % 20 == 0 or i+1 == len(todo):
                        print(f"  {total}/306  acc={correct/total*100:.1f}%  "
                              f"q={r['qa_id']}  {'✓' if r['correct'] else '✗'}")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\n=== Final: {correct}/{total}  {correct/total*100:.1f}% ===")


if __name__ == '__main__':
    main()
