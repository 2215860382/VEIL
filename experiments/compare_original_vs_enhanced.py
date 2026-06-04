"""对比实验：原库 vs 增强库 的检索精度。

用法：
    cd /home2/ycj/Project/VEIL_twoLayer
    PYTHONPATH=. python experiments/compare_original_vs_enhanced.py \
        --config configs/videomme_memory_bank.yaml \
        --vlm-api-url http://localhost:8000 \
        --llm-api-url http://localhost:8001 \
        --bge-gpu cuda:1 \
        --test-samples 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logging import get_logger
from experiments.core.veil import run_veil

log = get_logger("compare_original_vs_enhanced")


def load_samples(cfg: dict, limit: Optional[int] = None):
    """Load VideoMME test samples."""
    from src.dataloader.videomme import load_videomme

    b = cfg["benchmark"]
    samples = load_videomme(
        parquet_path=b["parquet_path"],
        video_dir=b["video_dir"],
    )

    if limit:
        samples = samples[:limit]
    return samples


def run_experiment(
    config_path: str,
    bank_dir: str,
    vlm_api_url: str,
    llm_api_url: str,
    bge_gpu: str,
    test_samples: int = 50,
    exp_name: str = "test",
) -> dict:
    """Run VEIL experiment with specified bank directory.

    Returns: dict with accuracy and other metrics
    """
    log.info(f"Running {exp_name}...")
    log.info(f"  Bank dir: {bank_dir}")

    cfg = load_config(config_path)

    # Override memory bank directory
    cfg["memory"]["bank_dir"] = bank_dir

    samples = load_samples(cfg, limit=test_samples)
    log.info(f"Loaded {len(samples)} test samples")

    results = []
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            log.info(f"  [{i+1}/{len(samples)}]")

        try:
            result = run_veil(
                sample=sample,
                cfg=cfg,
                vlm_api_url=vlm_api_url,
                llm_api_url=llm_api_url,
                bge_gpu=bge_gpu,
                max_iter=5,
            )
            results.append(result)
        except Exception as e:
            log.error(f"Sample {i} failed: {e}")
            results.append({"accuracy": 0, "error": str(e)})

    # Compute metrics
    correct = sum(1 for r in results if r.get("accuracy", 0) > 0)
    accuracy = correct / len(results) if results else 0

    metrics = {
        "exp_name": exp_name,
        "bank_dir": bank_dir,
        "total_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
    }

    log.info(f"Results for {exp_name}:")
    log.info(f"  Accuracy: {accuracy:.1%} ({correct}/{len(results)})")

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="对比原库 vs 增强库的检索精度"
    )
    ap.add_argument("--config", required=True, help="Config file path")
    ap.add_argument("--vlm-api-url", required=True, help="VLM API URL")
    ap.add_argument("--llm-api-url", required=True, help="LLM API URL")
    ap.add_argument("--bge-gpu", default="cuda:1", help="GPU for BGE embedding")
    ap.add_argument("--test-samples", type=int, default=50, help="Number of test samples")
    ap.add_argument("--original-bank",
                    default="/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b",
                    help="Original bank directory")
    ap.add_argument("--enhanced-bank",
                    default="/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27b_27b_two_layer",
                    help="Enhanced bank directory")

    args = ap.parse_args()

    log.info("=" * 60)
    log.info("库对比实验：原库 vs 增强库")
    log.info("=" * 60)

    # Run both experiments
    original_metrics = run_experiment(
        config_path=args.config,
        bank_dir=args.original_bank,
        vlm_api_url=args.vlm_api_url,
        llm_api_url=args.llm_api_url,
        bge_gpu=args.bge_gpu,
        test_samples=args.test_samples,
        exp_name="原库 (Original)",
    )

    time.sleep(5)  # Cooldown

    enhanced_metrics = run_experiment(
        config_path=args.config,
        bank_dir=args.enhanced_bank,
        vlm_api_url=args.vlm_api_url,
        llm_api_url=args.llm_api_url,
        bge_gpu=args.bge_gpu,
        test_samples=args.test_samples,
        exp_name="增强库 (Enhanced)",
    )

    # Compare results
    log.info("=" * 60)
    log.info("对比结果")
    log.info("=" * 60)
    log.info(f"原库:    {original_metrics['accuracy']:.1%} ({original_metrics['correct']}/{original_metrics['total_samples']})")
    log.info(f"增强库:  {enhanced_metrics['accuracy']:.1%} ({enhanced_metrics['correct']}/{enhanced_metrics['total_samples']})")

    improvement = enhanced_metrics['accuracy'] - original_metrics['accuracy']
    log.info(f"改进:   {improvement:+.1%}")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "test_samples": args.test_samples,
        "original": original_metrics,
        "enhanced": enhanced_metrics,
        "improvement": improvement,
    }

    output_file = Path("/tmp/library_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
