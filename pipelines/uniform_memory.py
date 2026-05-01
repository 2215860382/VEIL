"""Baseline: feed the entire memory bank (no retrieval) to Qwen3-8B.

All chunk memory_texts are concatenated and passed as evidence. This tests
whether the memory representation itself is informative without any retrieval.
Long videos may produce 60+ chunks; we cap at `max_chunks` to stay within
the LLM context window.
"""
from __future__ import annotations

from typing import List

from memory.schema import MemoryBank
from reasoning.answerer import TextAnswerer


def run_uniform_memory(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    llm_answerer: TextAnswerer,
    max_chunks: int = 80,
) -> dict:
    """Answer using all (capped) memory chunks as evidence — no retrieval.

    Args:
        question:      Question text.
        candidates:    Answer option strings.
        bank:          Memory bank (pre-built, question-agnostic).
        llm_answerer:  TextAnswerer backed by Qwen3-8B.
        max_chunks:    Cap on number of chunks to keep within LLM context.
    """
    chunks = bank.chunks
    if len(chunks) > max_chunks:
        # Uniform subsample to preserve temporal spread.
        import numpy as np
        sel = np.round(np.linspace(0, len(chunks) - 1, max_chunks)).astype(int)
        chunks = [chunks[i] for i in sel]

    evidence_texts = [c.memory_text for c in chunks]
    evidence_chunk_ids = [c.chunk_id for c in chunks]

    result = llm_answerer.answer(question, candidates, evidence_texts)
    result["evidence_texts"] = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    return result
