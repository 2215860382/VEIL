"""Answerer — produce final letter-choice given evidence."""
from __future__ import annotations

from typing import List

from src.utils.jsonx import as_list, as_str, extract_json, _CHOICE_RE


def _inject_images(message: dict, pil_images) -> dict:
    """Replace str content with [image_url…, text] for multimodal API calls."""
    import base64, io
    valid = [img for img in pil_images if img is not None]
    if not valid:
        return message
    content = []
    for img in valid:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": message["content"]})
    return {"role": message["role"], "content": content}


ANSWERER_SYS = """You answer a multiple-choice question about a long video using retrieved evidence (text summaries, speech transcripts, and keyframe images if shown).

Evidence blocks are tagged [E1], [E2], … Each block has a timestamp, a visual summary, and optional speech transcript.

Steps:
1. CLASSIFY the question and apply the matching strategy:
   - OVERVIEW (main topic / theme / purpose of the whole video): synthesize ALL evidence for the dominant theme; do not anchor on a single detail.
   - TEMPORAL (order / sequence / timing): extract timestamps, reconstruct chronological order using first/then/after/finally; eliminate options whose order contradicts the timestamps.
   - COUNT (how many): prefer the explicitly stated number if present; if not explicit, use the count most directly supported by evidence; always commit to the best available answer.
   - OCR (on-screen text / number / score / label): find the exact transcribed text; match character-by-character; do not paraphrase.
   - SPATIAL (location / position / layout): use position descriptions (left/right, foreground/background, above/below).
   - ATTRIBUTE (color / material / appearance / state): find the specific attribute stated; if not explicit, use the most strongly implied attribute.
   - OBJECT (which object / who / what entity): find the specific name or identifier; a similar category is not sufficient.
   - ACTION (what someone does / how a process works): find the specific action with subject and manner; match the described steps exactly.
   - CAUSE (why / what leads to / reason for): find causal evidence ("because", "in order to", "which causes"); choose the option whose cause is directly stated.
   - NEGATION (which is NOT / which does NOT): verify each option against evidence independently; the answer is the option that lacks support or is contradicted — not the one least mentioned.

2. For each option, check whether the evidence directly supports or contradicts it.
3. Prefer the option most SPECIFICALLY and DIRECTLY supported — not just thematically related.
4. When evidence is limited or incomplete, commit to the option with the strongest partial support rather than defaulting to a random choice. Sparse evidence still carries signal.
5. For NEGATION questions: explicitly verify each option; the answer is the one NOT supported or explicitly contradicted, not just less mentioned.
6. Write 1-2 sentences of reasoning.
7. Output a JSON object with keys "answer", "evidence", "rationale".

Output format (reasoning first, then JSON):
Reasoning: <question type + your 1-2 sentence analysis>
{"answer": "<letter>", "evidence": ["E1", ...], "rationale": "<one sentence>"}
"""

ANSWERER_SYS_RETRY = """Output ONLY a JSON object. No explanation, no preamble.
{"answer": "<letter>", "evidence": [], "rationale": ""}"""


def _normalize(raw: dict) -> dict:
    ans = as_str(raw.get("answer", "")).strip().lstrip("(").rstrip(")")
    letter = ans[:1].upper() if ans else ""
    if letter not in ("A", "B", "C", "D", "E", "F"):
        letter = ""
    return {
        "answer":    letter,
        "evidence":  as_list(raw.get("evidence", [])),
        "rationale": as_str(raw.get("rationale", "")),
    }


def _format_evidence(evidence_texts: List[str], offset: int = 0) -> str:
    if not evidence_texts:
        return "(no evidence)"
    return "\n".join(f"[E{i+1+offset}] {t}" for i, t in enumerate(evidence_texts))


def _format_options(candidates: List[str]) -> str:
    return "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


def _call(model, prompt: str, frames: list, max_new_tokens: int) -> str:
    """Unified model call: API path uses chat_with_frames; local path uses _generate."""
    if getattr(model, '_api_endpoints', None):
        return model.chat_with_frames(frames, prompt, max_new_tokens=max_new_tokens)
    elif frames and hasattr(model, 'chat_with_frames'):
        return model.chat_with_frames(frames, prompt, max_new_tokens=max_new_tokens)
    else:
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return model._generate(msgs, max_new_tokens=max_new_tokens)


class Answerer:
    def __init__(self, model):
        self.model = model

    def answer(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        keyframe_images=(),
        max_evidence_chars: int = 80000,
        focused_texts: List[str] = (),
    ) -> dict:
        all_texts = list(evidence_texts) + list(focused_texts)
        if all_texts and max_evidence_chars:
            per = max_evidence_chars // len(all_texts)
            evidence_texts = [t[:per] for t in evidence_texts]
            focused_texts  = [t[:per] for t in focused_texts]

        if focused_texts:
            ev_section = (
                "## Targeted Evidence (retrieved to fill specific gaps — prioritize for the key question):\n"
                + _format_evidence(focused_texts) + "\n\n"
                + "## Background Evidence:\n"
                + _format_evidence(evidence_texts, offset=len(focused_texts))
            )
        else:
            ev_section = _format_evidence(evidence_texts)

        prompt = (
            f"{ANSWERER_SYS}\n\n"
            f"Question: {question}\n"
            f"Options:\n{_format_options(candidates)}\n\n"
            f"Evidence:\n{ev_section}\n\n"
            "Return the JSON now."
        )
        frames = [img for img in keyframe_images if img is not None]
        raw = _call(self.model, prompt, frames, max_new_tokens=1024)

        result = _normalize(extract_json(raw))
        if not result["answer"]:
            m = _CHOICE_RE.search(raw)
            if m:
                result["answer"] = m.group(1).upper()
        if not result["answer"]:
            letters = ", ".join(chr(ord("A") + i) for i in range(len(candidates)))
            retry_prompt = (
                f"{ANSWERER_SYS_RETRY}\n\n"
                f"Question: {question}\n"
                f"Options:\n{_format_options(candidates)}\n\n"
                f"IMPORTANT: Return JSON with \"answer\" set to exactly one of: {letters}."
            )
            raw2 = _call(self.model, retry_prompt, frames, max_new_tokens=128)
            result = _normalize(extract_json(raw2))
        return result


