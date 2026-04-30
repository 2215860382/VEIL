"""Trivial in-memory vector store with cosine similarity.

For prototype scale (one video → ~tens to hundreds of chunks), brute force is fine.
Swap for FAISS/Chroma later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class SearchHit:
    index: int
    score: float
    payload: dict


class VectorStore:
    def __init__(self):
        self._vecs: np.ndarray | None = None
        self._payloads: List[dict] = []

    def build(self, vectors: np.ndarray, payloads: Sequence[dict]) -> None:
        assert vectors.shape[0] == len(payloads), "vectors / payloads count mismatch"
        self._vecs = vectors.astype(np.float32, copy=False)
        self._payloads = list(payloads)

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> List[SearchHit]:
        if self._vecs is None or len(self._payloads) == 0:
            return []
        q = query_vec.astype(np.float32, copy=False).reshape(-1)
        # both are L2-normalized → cosine == dot product
        scores = self._vecs @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [SearchHit(index=int(i), score=float(scores[i]), payload=self._payloads[i]) for i in idx]
