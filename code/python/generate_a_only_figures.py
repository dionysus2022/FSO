# -*- coding: utf-8 -*-
"""
generate_a_only_figures.py — Regenerate paper figures with Dataset-A only.

Per 13_53.txt restructure:
  fig1_a_method.png         — Method flowchart: Calibrated-Sim pretraining + real-data FT
  fig2_a_ratio_curves.png   — A-test accuracy vs real-data ratio (single panel)
  fig3_a_awgn_ablation.png  — A-test Calibrated-Sim vs AWGN-only (5 metrics: Overall, Macro-F1, High-order, 128QAM, 256QAM)
  fig4_a_confusion_5pct.png — A-test 2-panel confusion matrix: (a) 5% scratch (b) 5% SimPretrain+RealFT
  fig5_a_highorder_ratio.png — A-test high-order QAM accuracy vs real-data ratio (128QAM & 256QAM)

Outputs are written to:
  results/stage2/fast_publish/report/   (canonical report dir)
  results/docs/paper_acp/figures/       (paper source dir)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THIS_DIR = Path(__file__).resolve().parent
from stage2_common import STAGE2_ROOT

FP_ROOT = STAGE2_ROOT / "fast_publish"
AGG_JSON = FP_ROOT / "aggregated_results.json"
REPORT_DIR = FP_ROOT / "report"
PAPER_FIG_DIR = THIS_DIR.parent.parent / "results" / "docs" / "paper_acp" / "figures"
SEED_DIR = FP_ROOT / "seed_2026"

RATIO_KEYS = ["5pct", "10pct", "20pct", "100pct"]
RATIO_LABELS = ["5%", "10%", "20%", "100%"]
MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]


def load_agg() -> dict:
    with open(AGG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(agg: dict, config: str, metric: str) -> tuple[float, float]:
    d = agg.get(config)
    if d is None:
        return float("nan"), 0.0
    mean = d.get(f"A_test_{metric}_mean", float("nan"))
    std = d.get(f"A_test_{metric}_std", 0.0)
    if mean is None:
        mean = float("nan")
    if std is None:
        std = 0.0
    return float(mean), float(std)


def save_fig(fig, name: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (REPORT_DIR, PAPER_FIG_DIR):
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


# ─────────────────────────────────────────────────────────
# Fig1: Method flowchart
# ─────────────────────────────────────────────────────────
def plot_fig1_method_flowchart():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    BLUE = "#1976D2"
    GREEN = "#388E3C"
    ORANGE = "#F57C00"
    GREY = "#455A64"
    LIGHT_BLUE = "#BBDEFB"
    LIGHT_GREEN = "#C8E6C9"
    LIGHT_ORANGE = "#FFE0B2"
    LIGHT_GREY = "#CFD8DC"

    def box(cx, cy, w, h, text, fc, ec, fontsize=10, fontweight="bold", textcolor="black"):
        x = cx - w / 2
        y = cy - h / 2
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=textcolor, wrap=True)

    def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                      mutation_scale=18, color=color, linewidth=lw))

    # Title
    ax.text(6.5, 8.15, "Calibrated Simulation Pretraining + Real-Data Fine-Tuning",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ── Row 1: two data sources ──
    box(3.0, 7.0, 4.2, 0.9, "Real OFDM-FSO Dataset-A\n(900 frames, 6 QAM orders)",
        LIGHT_BLUE, BLUE, fontsize=10)
    box(10.0, 7.0, 4.2, 0.9, "Calibrated-Sim-v2.2\n(12,000 simulated frames)",
        LIGHT_GREEN, GREEN, fontsize=10)

    # Impairment list under Calibrated-Sim
    box(10.0, 5.95, 5.0, 0.85,
        "AWGN · lognormal scintillation · phase error\nbad subcarriers · impulsive noise · burst outliers",
        LIGHT_GREEN, GREEN, fontsize=8.5, fontweight="normal")
    arrow(10.0, 6.55, 10.0, 6.4, color=GREEN)

    # ── Row 2: Feature extraction (shared) ──
    box(6.5, 5.0, 6.5, 0.85,
        "Feature Extraction:  64×64 CDM  +  Radial/Statistics (48-D)",
        LIGHT_GREY, GREY, fontsize=10)
    arrow(3.0, 6.55, 5.0, 5.45, color=BLUE)
    arrow(10.0, 5.5, 8.0, 5.45, color=GREEN)

    # ── Row 3: Backbone ──
    box(6.5, 3.85, 6.5, 0.8, "CDM-Stats-Radial Network (shared backbone)",
        "#FFF9C4", "#F9A825", fontsize=10)
    arrow(6.5, 4.55, 6.5, 4.27, color=GREY)

    # ── Row 4: Two training paths ──
    # Left: Real-scratch
    box(3.0, 2.6, 4.2, 0.9, "Real-scratch\n(random init → train on A-train)",
        LIGHT_ORANGE, ORANGE, fontsize=9.5)
    arrow(4.5, 3.45, 3.5, 3.07, color=ORANGE)

    # Right: SimPretrain → RealFT
    box(10.0, 2.6, 4.2, 0.9,
        "SimPretrain → RealFT\n(SimOnly init → fine-tune on A-train)",
        LIGHT_GREEN, GREEN, fontsize=9.5)
    arrow(8.5, 3.45, 9.5, 3.07, color=GREEN)

    # Fine-tune ratio annotation
    ax.text(10.0, 1.7, "fine-tune ratio: 5% / 10% / 20% / 100%", ha="center",
            va="center", fontsize=8.5, color=GREEN, style="italic")
    ax.text(3.0, 1.7, "real-data ratio: 5% / 10% / 20% / 100%", ha="center",
            va="center", fontsize=8.5, color=ORANGE, style="italic")

    # ── Row 5: Evaluation ──
    box(6.5, 0.75, 5.5, 0.8, "A-test Evaluation (180 frames, 5 seeds)",
        LIGHT_BLUE, BLUE, fontsize=10)
    arrow(3.0, 2.13, 5.5, 1.17, color=ORANGE)
    arrow(10.0, 2.13, 7.5, 1.17, color=GREEN)

    fig.tight_layout()
    save_fig(fig, "fig1_a_method.png")


# ─────────────────────────────────────────────────────────
# Fig2: SimPretrain vs Real-scratch ratio curves (A-test only)
# ─────────────────────────────────────────────────────────
def plot_fig2_a_ratio_curves(agg: dict):
    fig, ax = plt.subplots(figsize=(8, 6))

    ft_means, ft_stds = [], []
    sc_means, sc_stds = [], []
    for rkey in RATIO_KEYS:
        ft_m, ft_s = get_metric(agg, f"s3_ft_{rkey}", "overall_acc")
        sc_m, sc_s = get_metric(agg, f"s3_scratch_{rkey}", "overall_acc")
        ft_means.append(ft_m)
        ft_stds.append(ft_s)
        sc_means.append(sc_m)
        sc_stds.append(sc_s)

    x = np.arange(len(RATIO_LABELS))
    ft_means = np.array(ft_means)
    ft_stds = np.array(ft_stds)
    sc_means = np.array(sc_means)
    sc_stds = np.array(sc_stds)

    ax.errorbar(x, ft_means * 100, yerr=ft_stds * 100,
                marker="o", linewidth=2.2, capsize=5, label="SimPretrain+RealFT",
                color="#1976D2", markersize=8)
    ax.errorbar(x, sc_means * 100, yerr=sc_stds * 100,
                marker="s", linewidth=2.2, capsize=5, label="Real-scratch",
                color="#F57C00", markersize=8)

    s2_m, _ = get_metric(agg, "s2_simonly", "overall_acc")
    if not np.isnan(s2_m):
        ax.axhline(y=s2_m * 100, color="green", linestyle="--", alpha=0.7,
                   label=f"Calibrated SimOnly ({s2_m*100:.1f}%)", linewidth=1.8)

    # Annotate the 5% gain
    gain5 = (ft_means[0] - sc_means[0]) * 100
    ax.annotate(f"+{gain5:.1f} pp", xy=(0, (ft_means[0] + sc_means[0]) * 50),
                xytext=(0.4, 78), fontsize=11, fontweight="bold", color="#C62828",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS, fontsize=12)
    ax.set_xlabel("Real fine-tune ratio", fontsize=12)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.legend(fontsize=10.5, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    save_fig(fig, "fig2_a_ratio_curves.png")


# ─────────────────────────────────────────────────────────
# Fig3: AWGN-only vs Calibrated-Sim-v2.2 ablation (A-test, 5 metrics)
# ─────────────────────────────────────────────────────────
def plot_fig3_a_awgn_ablation(agg: dict):
    metrics_to_plot = [
        ("overall_acc", "Overall"),
        ("macro_f1", "Macro-F1"),
        ("high_order_acc", "High-order"),
        ("acc_128qam", "128QAM"),
        ("acc_256qam", "256QAM"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    v22_means, v22_stds = [], []
    awgn_means, awgn_stds = [], []
    for metric, _ in metrics_to_plot:
        m1, s1 = get_metric(agg, "s2_simonly", metric)
        m2, s2 = get_metric(agg, "s5_awgn_only", metric)
        v22_means.append(m1)
        v22_stds.append(s1)
        awgn_means.append(m2)
        awgn_stds.append(s2)

    bars1 = ax.bar(x - width / 2, np.array(v22_means) * 100, width,
                   yerr=np.array(v22_stds) * 100, capsize=5,
                   label="Calibrated-Sim-v2.2", color="#388E3C", alpha=0.88,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, np.array(awgn_means) * 100, width,
                   yerr=np.array(awgn_stds) * 100, capsize=5,
                   label="AWGN-only", color="#D32F2F", alpha=0.88,
                   edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, v22_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 2,
                f"{val*100:.0f}", ha="center", va="bottom", fontsize=9, color="#1B5E20")
    for bar, val in zip(bars2, awgn_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 2,
                f"{val*100:.0f}", ha="center", va="bottom", fontsize=9, color="#B71C1C")

    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics_to_plot], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 112)

    fig.tight_layout()
    save_fig(fig, "fig3_a_awgn_ablation.png")


# ─────────────────────────────────────────────────────────
# Fig4: A-test 2-panel confusion matrix: 5% scratch vs 5% FT
# ─────────────────────────────────────────────────────────
def plot_fig4_a_confusion_5pct():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    configs = [
        ("s3_scratch_5pct", "(a) Real-scratch (5% real data)"),
        ("s3_ft_5pct", "(b) SimPretrain+RealFT (5% real data)"),
    ]

    for ax, (cfg, title) in zip(axes, configs):
        metrics_path = SEED_DIR / cfg / "metrics.json"
        if not metrics_path.exists():
            ax.text(0.5, 0.5, f"Missing: {metrics_path}", ha="center", va="center")
            ax.set_title(title)
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        cm = np.array(m["A_test"]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(MOD_NAMES)))
        ax.set_yticks(range(len(MOD_NAMES)))
        ax.set_xticklabels(MOD_NAMES, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(MOD_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        acc = m["A_test"]["overall_acc"]
        ax.set_title(f"{title}\nOverall Acc: {acc*100:.1f}%", fontsize=11)

        for i in range(len(MOD_NAMES)):
            for j in range(len(MOD_NAMES)):
                val = cm[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=9)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, "fig4_a_confusion_5pct.png")


# ─────────────────────────────────────────────────────────
# Fig5: High-order QAM accuracy vs real-data ratio (A-test)
# ─────────────────────────────────────────────────────────
def plot_fig5_a_highorder_ratio(agg: dict):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    series = [
        ("acc_128qam", "s3_scratch", "128QAM Real-scratch", "#F57C00", "s", "--"),
        ("acc_128qam", "s3_ft", "128QAM SimPretrain+RealFT", "#D32F2F", "o", "-"),
        ("acc_256qam", "s3_scratch", "256QAM Real-scratch", "#1976D2", "^", "--"),
        ("acc_256qam", "s3_ft", "256QAM SimPretrain+RealFT", "#388E3C", "D", "-"),
    ]

    x = np.arange(len(RATIO_LABELS))
    for metric, prefix, label, color, marker, ls in series:
        means, stds = [], []
        for rkey in RATIO_KEYS:
            m, s = get_metric(agg, f"{prefix}_{rkey}", metric)
            means.append(m)
            stds.append(s)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(x, means * 100, yerr=stds * 100,
                    marker=marker, linewidth=2, capsize=4, label=label,
                    color=color, linestyle=ls, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS, fontsize=12)
    ax.set_xlabel("Real fine-tune ratio", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=9.5, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    fig.tight_layout()
    save_fig(fig, "fig5_a_highorder_ratio.png")


def main():
    print("[1/4] Loading data...")
    agg = load_agg()

    print("[2/4] Generating restructured A-only figures...")
    plot_fig1_method_flowchart()
    plot_fig2_a_ratio_curves(agg)
    plot_fig3_a_awgn_ablation(agg)
    plot_fig4_a_confusion_5pct()
    plot_fig5_a_highorder_ratio(agg)

    print("[3/4] Figures written to:")
    print(f"  - {REPORT_DIR}")
    print(f"  - {PAPER_FIG_DIR}")
    print("[4/4] Done.")


if __name__ == "__main__":
    main()
