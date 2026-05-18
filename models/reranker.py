"""Rerankers for RAG pipelines.

BGEReranker  : bge-reranker-v2-m3 cross-encoder (transformers)
LLMReranker  : listwise LLM reranker — asks an LLM to select the top-k most
               relevant chunks from the coarse candidate set.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class BGEReranker:
    def __init__(
        self,
        model_path: str,
        use_fp16: bool = True,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        dtype = torch.float16 if use_fp16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=dtype
        ).to(device).eval()

    @torch.inference_mode()
    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]
        out: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            qb = queries[i : i + self.batch_size]
            pb = passages[i : i + self.batch_size]
            enc = self.tokenizer(
                qb, pb,
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc, return_dict=True).logits.view(-1).float()
            # bge-reranker-v2-m3 outputs a single logit per pair; sigmoid → [0,1].
            scores = torch.sigmoid(logits).cpu().tolist()
            out.extend(scores)
        return out

    def rerank(self, query: str, candidates: Sequence[str], top_k: int = 10):
        """Return list of (idx, score) sorted by score desc, truncated to top_k."""
        pairs = [(query, c) for c in candidates]
        scores = self.score(pairs)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(i, scores[i]) for i in order]


_LLM_RERANK_SYS = """You are a relevance ranker for a video QA retrieval system.
Given a question and numbered evidence chunks, return the indices of the most relevant chunks in order of relevance (most relevant first).
Return ONLY a JSON object: {"selected": [i, j, k, ...]}
Use only the indices shown. Do not explain."""


class LLMReranker:
    """Listwise reranker: asks an LLM to select the top-k most relevant chunks."""

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, candidates: Sequence[str], top_k: int = 8):
        """Return list of (idx, score) same interface as BGEReranker.rerank."""
        import json, re
        n = len(candidates)
        formatted = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        user = (
            f"Question: {query}\n\n"
            f"Evidence chunks:\n{formatted}\n\n"
            f"Select the {min(top_k, n)} most relevant chunk indices."
        )
        messages = [
            {"role": "system", "content": _LLM_RERANK_SYS},
            {"role": "user",   "content": user},
        ]
        raw = self.llm.chat(messages, max_new_tokens=128, enable_thinking=False)
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            try:
                selected = json.loads(m.group()).get("selected", [])
                selected = [int(i) for i in selected if 0 <= int(i) < n][:top_k]
                if selected:
                    return [(i, 1.0 - rank * 0.01) for rank, i in enumerate(selected)]
            except Exception:
                pass
        # Fallback: return first top_k
        return [(i, 1.0) for i in range(min(top_k, n))]
