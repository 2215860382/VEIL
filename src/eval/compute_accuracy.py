"""Compute MLVU-style accuracy: per-task and overall, plus a couple of diagnostics."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def compute_accuracy(records: List[dict]) -> Dict[str, dict]:
    """Each record: {question_type, gold, pred_letter, pred_text}.

    pred_letter ∈ A/B/C/D... or None.
    Correct iff pred_text == gold (string match — MLVU records gold as candidate text).
    """
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_task[r["question_type"]].append(r)

    out: Dict[str, dict] = {}
    total_n, total_correct = 0, 0
    for task, rs in by_task.items():
        n = len(rs)
        correct = sum(1 for r in rs if r.get("pred_text") and r["pred_text"] == r["gold"])
        no_letter = sum(1 for r in rs if r.get("pred_letter") is None)
        out[task] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "no_letter": no_letter,
        }
        total_n += n
        total_correct += correct
    out["__overall__"] = {
        "n": total_n,
        "correct": total_correct,
        "accuracy": (total_correct / total_n) if total_n else 0.0,
    }
    return out
