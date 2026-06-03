"""Online evidence补充 for INSUFFICIENT verdicts.

When verifier signals INSUFFICIENT (evidence doesn't support any answer option),
this module generates targeted temporary evidence by:
1. Analyzing failed criteria from verifier output
2. Identifying relevant time windows / chunks
3. Using VLM to answer criterion-specific queries on keyframes
4. Returning formatted evidence without modifying the offline bank
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set

from src.memory.core.schema import MemoryBank
from src.utils.logging import get_logger

log = get_logger("online_refresh")


def _extract_criterion_topics(failed_criteria: List[dict]) -> List[str]:
    """Extract key topics/entities from failed criteria for targeted retrieval."""
    topics = []
    for crit in failed_criteria:
        desc = str(crit.get("description", "")).lower()
        # Extract nouns and entity mentions
        # Simple heuristic: split by common prepositions and punctuation
        parts = re.split(r'\b(?:the|of|in|on|at|to|from|by)\b', desc)
        for part in parts:
            part = part.strip()
            if len(part) > 3 and len(part) < 100:
                topics.append(part)
    return topics


def _select_relevant_chunks(
    bank: MemoryBank,
    failed_criteria: List[dict],
    top_k: int = 5,
) -> List[int]:
    """Select chunks most relevant to failed criteria.

    Uses heuristic matching: chunks with high keyword overlap or
    chunks in time ranges mentioned in criterion descriptions.
    """
    topics = _extract_criterion_topics(failed_criteria)
    if not topics:
        # Fallback: select chunks uniformly across timeline
        step = max(1, len(bank.chunks) // top_k)
        return [i for i in range(0, len(bank.chunks), step)][:top_k]

    # Score chunks based on keyword overlap with criterion topics
    scores = [0.0] * len(bank.chunks)
    for i, chunk in enumerate(bank.chunks):
        # Check text fields for keyword matches
        text_to_search = (chunk.memory_text + " " + " ".join(chunk.key_events) +
                         " " + " ".join(chunk.actors) + " " + chunk.static_index_text).lower()
        for topic in topics:
            if topic in text_to_search:
                scores[i] += 1.0

    # Select top-k chunks by score, breaking ties by chunk_id
    indexed_scores = [(i, scores[i]) for i in range(len(scores))]
    indexed_scores.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in indexed_scores[:top_k]]


def _generate_criterion_answer(
    criterion_desc: str,
    chunk,
    keyframe_path: str,
    vlm,
) -> Optional[str]:
    """Use VLM to answer a specific criterion using keyframe + chunk context.

    Returns a one-sentence answer or None if extraction fails.
    """
    if not Path(keyframe_path).exists() or not vlm:
        return None

    try:
        from PIL import Image
        img = Image.open(keyframe_path).convert("RGB")

        prompt = (
            f"Video segment [{chunk.start_time:.0f}s–{chunk.end_time:.0f}s]\n"
            f"Context: {chunk.memory_text}\n\n"
            f"Answer this specific question about the frame:\n"
            f"Question: {criterion_desc}\n\n"
            f"Answer in 1-2 sentences. Be specific. If the answer is not visible, say 'Not visible in this frame.'"
        )

        # Use VLM to answer the criterion
        answer = vlm.chat_with_frames([img], prompt, max_new_tokens=60).strip()
        return answer if answer else None
    except Exception as e:
        log.debug("  criterion answer error: %s", e)
        return None


def online_refresh(
    bank: MemoryBank,
    failed_criteria: List[dict],
    vlm,
    embedder,
    video_path: Optional[str] = None,
    top_k_chunks: int = 5,
) -> List[str]:
    """Generate temporary evidence for failed criteria without modifying the bank.

    Args:
        bank: MemoryBank to draw chunks from
        failed_criteria: List of criterion dicts from verifier output
        vlm: VLMClient for on-demand frame analysis
        embedder: BGE embedder (unused in current implementation but available)
        video_path: Path to video (optional; keyframes used if available)
        top_k_chunks: Number of chunks to query

    Returns:
        List of temporary evidence strings formatted as "[{time}s] {answer}" per criterion
    """
    if not failed_criteria or not vlm:
        return []

    evidence = []
    relevant_chunk_indices = _select_relevant_chunks(bank, failed_criteria, top_k_chunks)

    for criterion in failed_criteria:
        desc = criterion.get("description", "")
        if not desc:
            continue

        log.debug(f"Online補充 for criterion: {desc}")

        for idx in relevant_chunk_indices:
            chunk = bank.chunks[idx]

            # Try to answer using keyframe
            if chunk.keyframe_path and Path(chunk.keyframe_path).exists():
                answer = _generate_criterion_answer(desc, chunk, chunk.keyframe_path, vlm)
                if answer and "not visible" not in answer.lower():
                    # Format as evidence: [time] {answer}
                    ev_text = f"[{chunk.start_time:.0f}s] {answer}"
                    evidence.append(ev_text)
                    break  # One answer per criterion is enough for online補充

            # Fallback: try other frames if available
            if chunk.static_frames and not answer:
                for static_frame in chunk.static_frames[:1]:  # Just first static frame
                    answer = _generate_criterion_answer(
                        desc, chunk, static_frame["image_path"], vlm
                    )
                    if answer and "not visible" not in answer.lower():
                        ev_text = f"[{chunk.start_time:.0f}s] {answer}"
                        evidence.append(ev_text)
                        break

    log.debug(f"Generated {len(evidence)} online evidence items")
    return evidence
