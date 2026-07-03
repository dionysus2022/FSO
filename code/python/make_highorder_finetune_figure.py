# -*- coding: utf-8 -*-
"""
make_highorder_finetune_figure.py — Generate Fig.7: High-order fine-tuning comparison.

Bar chart comparing 4 configs at 5% real data across 5 metrics:
  s3_ft_5pct (baseline), ho_weighted_5pct, ho_balanced_5pct, ho_replay_5pct

Output: results/stage2/fast_publish/paper_aonly_figures/fig7_highorder_finetune_comparison.png/.pdf
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FP_ROOT = THIS_DIR.parent.parent / "results" / "stage2" / "fast_publish"
OUT_DIR = FP_ROOT / "paper_aonly_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [2026, 2027, 2028, 2029, 2030]
CONFIGS = [
    ("s3_ft_5pct",       "5% SimPretrain\n+RealFT (baseline)"),
    ("ho_weighted_5pct", "5% + HO class\n-weighted CE"),
    ("ho_balanced_5pct", "5% + HO-balanced\nbatch"),
    ("ho_replay_5pct",   "5% + real:sim\nreplay (2:1)"),
]
METRICS = [
    ("overall_acc",    "Overall"),
    ("macro_f1",       "Macro-F1"),
    ("high_order_acc", "High-order"),
    ("acc_128qam",     "128QAM"),
    ("acc_256qam",     "256QAM"),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def load_config_metrics(config: str) -> dict:
    """Load per-seed metrics for a config, return mean/std for each metric."""
    per_seed = []
    for seed in SEEDS:
        p = FP_ROOT / f"seed_{seed}" / config / "metrics.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                per_seed.append(json.load(f))
    if not per_seed:
        return {}

    result = {}
    a_tests = [r.get("A_test", {}) for r in per_seed]
    for key, _ in METRICS:
        vals = [float(t.get(key, 0.0)) * 100 for t in a_tests]
        result[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return result


def plot_fig7():
    n_configs = len(CONFIGS)
    n_metrics = len(METRICS)
    x = np.arange(n_metrics)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    all_data = {}
    for config, _ in CONFIGS:
        all_data[config] = load_config_metrics(config)

    for i, (config, label) in enumerate(CONFIGS):
        m = all_data.get(config, {})
        means = [m.get(key, {}).get("mean", 0.0) for key, _ in METRICS]
        stds = [m.get(key, {}).get("std", 0.0) for key, _ in METRICS]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=label,
                      color=COLORS[i], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("A-test Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the replay winner's Overall gain
    baseline_overall = all_data["s3_ft_5pct"].get("overall_acc", {}).get("mean", 0)
    replay_overall = all_data["ho_replay_5pct"].get("overall_acc", {}).get("mean", 0)
    gain = replay_overall - baseline_overall
    if gain > 0:
        ax.annotate(f"+{gain:.1f} pp",
                    xy=(0 + (3 - 1.5) * width, replay_overall),
                    xytext=(0.8, replay_overall + 8),
                    fontsize=10, fontweight="bold", color="#8172B2",
                    arrowprops=dict(arrowstyle="->", color="#8172B2", lw=1.2))

    plt.tight_layout()

    png_path = OUT_DIR / "fig7_highorder_finetune_comparison.png"
    pdf_path = OUT_DIR / "fig7_highorder_finetune_comparison.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    # Save data JSON
    data = {}
    for config, _ in CONFIGS:
        data[config] = all_data.get(config, {})
    json_path = OUT_DIR / "fig7_highorder_finetune_comparison_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Print summary
    print("\n  === Summary ===")
    print(f"  {'Config':<25} {'Overall':>10} {'Macro-F1':>10} {'High-ord':>10} {'128QAM':>10} {'256QAM':>10}")
    for config, label in CONFIGS:
        m = all_data.get(config, {})
        row = f"  {config:<25}"
        for key, _ in METRICS:
            row += f"  {m.get(key, {}).get('mean', 0):.1f}±{m.get(key, {}).get('std', 0):.1f}"
        print(row)


if __name__ == "__main__":
    plot_fig7()
