"""
用同学 top-64 pre-retrieved chunks + Answerer（Qwen3.5-27B 多模态）+ 我们的 keyframe 答题。
对比 eval_classmate_chunks.py（TextAnswerer，无图），验证加了 keyframe 是否有提升。
"""
import json, sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reasoning.answerer import Answerer
from src.models.vlm_client import VLMClient
from src.pipelines._keyframes import load_keyframe_pil, keyframe_path

MEM_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b")
CLASSMATE_INPUT = "/home2/ycj/Project/VedioSifter/data/coarse_retrieval_bge.jsonl"


def build_numid_to_ytid(classmate_rows, our_result_path):
    """通过 question 文本匹配，建立数字 video_id → YouTube video_id 映射。"""
    our_q2vid = {}
    for l in open(our_result_path):
        r = json.loads(l)
        our_q2vid[r['question'].strip()] = r['video_id']
    mapping = {}
    for row in classmate_rows:
        yt = our_q2vid.get(row['question'].strip())
        if yt:
            mapping[row['video_id']] = yt
    return mapping


def load_bank_chunks(yt_vid):
    """读取我们 memory bank 的 chunk 列表（含 chunk_id 和时间范围）。"""
    p = MEM_DIR / f"{yt_vid}.json"
    if not p.exists():
        return []
    data = json.load(open(p))
    return data.get('chunks', [])


def get_keyframes_for_chunks(classmate_chunks, yt_vid, cap=8):
    """
    为同学的 top-N chunks 加载 keyframe：
    按 t_start 找我们 bank 中时间最近的 chunk，取其 keyframe 文件。
    """
    bank_chunks = load_bank_chunks(yt_vid)
    if not bank_chunks:
        return []

    imgs = []
    seen_chunk_ids = set()
    for cc in classmate_chunks:
        t = (cc.get('t_start', 0) + cc.get('t_end', 0)) / 2
        best = min(bank_chunks, key=lambda c: abs((c['start_time'] + c['end_time']) / 2 - t))
        cid = best['chunk_id']
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        kp = keyframe_path(str(MEM_DIR), yt_vid, cid)
        img = load_keyframe_pil(kp)
        if img is not None:
            imgs.append(img)
        if len(imgs) >= cap:
            break
    return imgs


def fmt_evidence(chunk):
    t0, t1 = chunk.get('t_start', 0), chunk.get('t_end', 0)
    summary = (chunk.get('semantic_summary') or chunk.get('text') or '').strip()
    speech = (chunk.get('speech_text') or '').strip()
    parts = [f"[{t0:.0f}s-{t1:.0f}s] {summary}"]
    if speech:
        parts.append(f"Speech: {speech}")
    return "\n".join(parts)


def answer_one(answerer, q, numid_to_ytid):
    candidates_raw = q.get('choices', [])
    candidates = [c.split('. ', 1)[-1] if '. ' in c else c for c in candidates_raw]
    gold_letter = (q.get('answer') or '').strip().upper()

    chunks = q.get('candidates', [])
    evidence_texts = [fmt_evidence(c) for c in chunks]

    yt_vid = numid_to_ytid.get(q['video_id'], '')
    keyframes = get_keyframes_for_chunks(chunks[:16], yt_vid, cap=8) if yt_vid else []

    result = answerer.answer(q['question'], candidates, evidence_texts,
                             keyframe_images=keyframes, max_evidence_chars=80000)
    pred = result.get('answer', '').upper()
    return {
        'qa_id':    q.get('question_id', ''),
        'video_id': q.get('video_id', ''),
        'question': q['question'],
        'gold':     gold_letter,
        'pred':     pred,
        'correct':  pred == gold_letter,
        'n_kf':     len(keyframes),
        'rationale': result.get('rationale', ''),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default=CLASSMATE_INPUT)
    ap.add_argument('--out',    default='/home2/ycj/Project/VEIL/outputs/results/videommeL/eval_classmate_chunks_vlm_kf.jsonl')
    ap.add_argument('--ref-result', default='/home2/ycj/Project/VEIL/outputs/results/videommeL/videommeL_coarse64_27b_summary_kf.jsonl')
    ap.add_argument('--vlm-api-url',   default='http://127.0.0.1:8780,http://127.0.0.1:8783')
    ap.add_argument('--vlm-api-model', default='Qwen3.5-27B')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(questions)} questions")

    numid_to_ytid = build_numid_to_ytid(questions, args.ref_result)
    print(f"video_id mapping: {len(numid_to_ytid)} videos")

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try: done.add(json.loads(line)['qa_id'])
            except: pass
    todo = [q for q in questions if q.get('question_id', '') not in done]
    print(f"todo: {len(todo)}  (already done: {len(done)})")

    vlm = VLMClient(model_path=args.vlm_api_model, api_url=args.vlm_api_url,
                    api_model=args.vlm_api_model)
    answerer = Answerer(vlm)

    correct = sum(1 for l in open(out_path) if json.loads(l).get('correct')) if out_path.exists() and done else 0
    total = len(done)

    with open(out_path, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(answer_one, answerer, q, numid_to_ytid): q for q in todo}
            for i, fut in enumerate(as_completed(futures)):
                try:
                    r = fut.result()
                    fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                    fout.flush()
                    total += 1
                    if r['correct']: correct += 1
                    if (i + 1) % 20 == 0 or i + 1 == len(todo):
                        print(f"  {total}/306  acc={correct/total*100:.1f}%  "
                              f"kf={r['n_kf']}  {'✓' if r['correct'] else '✗'}  {r['qa_id']}")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\n=== Final: {correct}/{total}  {correct/total*100:.1f}% ===")


if __name__ == '__main__':
    main()
