#!/usr/bin/env python3
"""Compare results across all ablation experiments."""
import json
from pathlib import Path
from collections import defaultdict

def load_results(jsonl_path):
    """Load JSONL results and compute accuracy metrics."""
    correct = 0
    total = 0
    by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})

    for line in Path(jsonl_path).open():
        try:
            rec = json.loads(line)
            if rec.get("error"):
                continue
            total += 1
            by_qtype[rec.get("question_type", "unknown")]["total"] += 1
            if rec.get("correct"):
                correct += 1
                by_qtype[rec["question_type"]]["correct"] += 1
        except:
            pass

    return {
        "total": total,
        "correct": correct,
        "accuracy": 100 * correct / total if total > 0 else 0,
        "by_qtype": dict(by_qtype)
    }

def main():
    results_dir = Path("outputs/results/videommeL")

    experiments = [
        ("veil_27b", "📌 Baseline (main) — 71.56%"),
        ("veil_27b_singlequery", "📍 维度1: Single query (no decomposition)"),
        ("veil_27b_no_rubric_judge", "🔴 维度2: No rubric (text-only judgment)"),
        ("veil_27b_strict_dedup", "📊 维度3a: Strict evidence dedup (0.85→0.90)"),
        ("veil_27b_high_query_threshold", "📝 维度3b: High query threshold (0.9→0.99)"),
        ("veil_27b_ignore_verifier", "🔗 维度4: Ignore verifier (no feedback)"),
        ("veil_27b_oracle", "⭐ Oracle upper bound"),
    ]

    results = {}
    for fname, label in experiments:
        path = results_dir / f"{fname}.jsonl"
        if path.exists():
            results[fname] = load_results(path)
            print(f"✓ {label:<40} {path.name}")
        else:
            print(f"✗ {label:<40} (not found)")

    if not results:
        print("\nNo results found in", results_dir)
        return

    baseline_acc = results.get("veil_27b", {}).get("accuracy", 0)

    print("\n" + "="*100)
    print(f"{'Pipeline':<30} {'Accuracy':>12} {'vs Baseline':>15} {'N':>8}")
    print("="*100)

    for fname, label in experiments:
        if fname not in results:
            continue
        res = results[fname]
        acc = res["accuracy"]
        total = res["total"]
        delta = acc - baseline_acc
        delta_str = f"{delta:+.2f}%" if fname != "veil_27b" else "baseline"

        symbol = "📌" if fname == "veil_27b" else ("✓" if delta > 0 else "✗" if delta < 0 else "=")
        print(f"{symbol} {fname:<28} {acc:>11.2f}% {delta_str:>15} {total:>8}")

    print("\n" + "="*100)
    print("Per-Question-Type Breakdown (vs Baseline):")
    print("="*100)

    baseline_by_qtype = results.get("veil_27b", {}).get("by_qtype", {})

    for qtype in sorted(baseline_by_qtype.keys()):
        baseline_res = baseline_by_qtype[qtype]
        baseline_acc_qtype = 100 * baseline_res["correct"] / baseline_res["total"] if baseline_res["total"] > 0 else 0

        print(f"\n📊 Question Type: {qtype} (baseline: {baseline_acc_qtype:.2f}%)")
        print(f"  {'Pipeline':<28} {'Accuracy':>12} {'Delta':>10}")
        print(f"  {'-'*52}")

        for fname, label in experiments:
            if fname not in results:
                continue
            res = results[fname]
            by_qtype = res.get("by_qtype", {})
            if qtype not in by_qtype:
                continue

            qtype_res = by_qtype[qtype]
            acc = 100 * qtype_res["correct"] / qtype_res["total"] if qtype_res["total"] > 0 else 0
            delta = acc - baseline_acc_qtype

            symbol = "📌" if fname == "veil_27b" else ("✓" if delta > 0 else "✗" if delta < 0 else "=")
            print(f"  {symbol} {fname:<26} {acc:>11.2f}% {delta:>+9.2f}%")

    print("\n" + "="*100)
    print("Summary:")
    print("="*100)
    print(f"Baseline: veil_27b = {baseline_acc:.2f}%")
    print()
    print("Interpretation guide:")
    print("  ✓ Delta > 0  — Ablation IMPROVES accuracy (good finding!)")
    print("  ✗ Delta < 0  — Ablation HURTS accuracy (baseline component is important)")
    print("  = Delta ≈ 0  — No significant effect")
    print()
    print("Next steps:")
    print("  1. Identify which ablations improve (keep them on)")
    print("  2. Identify which ablations hurt (confirm baseline is better)")
    print("  3. For P2 ablations, apply on top of best P1 combination")

if __name__ == "__main__":
    main()
