"""bge-reranker-v2-m3 cross-encoder reranker.

Implemented directly on top of transformers (AutoModelForSequenceClassification) to avoid
FlagEmbedding's dependency on the legacy `tokenizer.prepare_for_model` API, which was
removed in transformers 5.x.
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
