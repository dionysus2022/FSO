# -*- coding: utf-8 -*-
"""
make_paper_aonly_figures.py — Generate all A-only conference-paper figures.

Per 14_22.txt. Outputs to results/stage2/fast_publish/paper_aonly_figures/.
Each figure: PNG (dpi=300), PDF (vector), and a data JSON/CSV for verification.

Figures:
  Fig.1  fig1_experimental_setup_pipeline   — OFDM-FSO acquisition + recognition pipeline
  Fig.2  fig2_calibrated_pretraining_framework — Calibrated-Sim pretraining + real-data FT
  Fig.3  fig3_atest_accuracy_vs_real_ratio  — A-test accuracy vs real-data ratio
  Fig.4  fig4_atest_sim_realism_ablation    — Calibrated-Sim vs AWGN-only (5 metrics)
  Fig.5  fig5_atest_confusion_5seed_avg_5pct — 5-seed averaged row-normalized confusion (5%)
  Fig.6  fig6_atest_highorder_vs_ratio      — A-test high-order QAM vs ratio
  table_experimental_parameters.tex/.md     — Experimental parameters table
"""

from __future__ import annotations
import json
import csv
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
SEEDS = [2026, 2027, 2028, 2029, 2030]
OUT_DIR = FP_ROOT / "paper_aonly_figures"

RATIO_KEYS = ["5pct", "10pct", "20pct", "100pct"]
RATIO_LABELS = ["5%", "10%", "20%", "100%"]
MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]

# Unified palette
C_BLUE = "#1976D2"
C_GREEN = "#388E3C"
C_ORANGE = "#F57C00"
C_RED = "#D32F2F"
C_GREY = "#455A64"
C_LIGHT_BLUE = "#BBDEFB"
C_LIGHT_GREEN = "#C8E6C9"
C_LIGHT_ORANGE = "#FFE0B2"
C_LIGHT_GREY = "#CFD8DC"


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def load_agg() -> dict:
    with open(AGG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(agg, config, metric):
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


def save_fig_and_data(fig, name, data=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = name.removesuffix(".png")
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        json_path = OUT_DIR / f"{stem}_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        csv_path = OUT_DIR / f"{stem}_data.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                w.writerow(data[0].keys())
                for row in data:
                    w.writerow(row.values())
    print(f"  [fig] {stem}.png / .pdf" + (" + data" if data else ""))


def _box(ax, cx, cy, w, h, text, fc, ec, fontsize=9, fontweight="bold"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.10", facecolor=fc,
                                edgecolor=ec, linewidth=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color="black")


def _arrow(ax, x1, y1, x2, y2, color=C_GREY, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, color=color, linewidth=lw))


# ─────────────────────────────────────────────────────────
# Fig.1: Experimental setup + recognition pipeline
# ─────────────────────────────────────────────────────────
def plot_fig1_setup_pipeline():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # ── (a) Acquisition ──
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 17); ax1.axis("off")
    ax1.set_title("(a) OFDM-FSO experimental acquisition", fontsize=12, fontweight="bold", loc="left")

    steps_a = [
        ("TX bits", C_LIGHT_BLUE, C_BLUE),
        ("QAM mapping\nQPSK/16/32/64/128/256QAM", C_LIGHT_BLUE, C_BLUE),
        ("S/P  →  pilot insertion\n→ Hermitian symmetry", C_LIGHT_GREY, C_GREY),
        ("256-point IFFT", C_LIGHT_GREY, C_GREY),
        ("CP insertion  (CP = 16)  →  P/S", C_LIGHT_GREY, C_GREY),
        ("AWG  (Keysight M8195A)", C_LIGHT_ORANGE, C_ORANGE),
        ("MZM (Oclaro, 12.5 GHz)\n+ 1550-nm laser", C_LIGHT_ORANGE, C_ORANGE),
        ("PC / collimator / polarizer\n(Thorlabs TC18APC-1550)", C_LIGHT_ORANGE, C_ORANGE),
        ("SLM phase screens  (Holoeye Pluto)\nweak: 200 screens  |  strong: 200 screens", C_LIGHT_GREEN, C_GREEN),
        ("Receiver collimator", C_LIGHT_ORANGE, C_ORANGE),
        ("APD  (50 GHz bandwidth)", C_LIGHT_ORANGE, C_ORANGE),
        ("DSO  (Keysight DSAX92504A, 80 GSa/s)", C_LIGHT_ORANGE, C_ORANGE),
        ("RX data", C_LIGHT_BLUE, C_BLUE),
    ]
    y = 16.2
    dy = 1.22
    centers = []
    for text, fc, ec in steps_a:
        _box(ax1, 5, y, 6.8, 0.92, text, fc, ec, fontsize=8.5)
        centers.append(y)
        y -= dy
    for i in range(len(centers) - 1):
        _arrow(ax1, 5, centers[i] - 0.46, 5, centers[i + 1] + 0.46)

    # ── (b) Recognition pipeline ──
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 17); ax2.axis("off")
    ax2.set_title("(b) Recognition-oriented receive processing", fontsize=12, fontweight="bold", loc="left")

    steps_b = [
        ("RX data", C_LIGHT_BLUE, C_BLUE),
        ("Synchronization", C_LIGHT_GREY, C_GREY),
        ("S/P  →  CP removal  →  256-point FFT", C_LIGHT_GREY, C_GREY),
        ("Channel estimation / equalization", C_LIGHT_GREY, C_GREY),
        ("Active-subcarrier extraction  (123)", C_LIGHT_GREY, C_GREY),
        ("Frame construction\n128 OFDM symbols × 123 subcarriers", C_LIGHT_GREY, C_GREY),
        ("Preprocessing\ncentering + RMS normalization", C_LIGHT_GREY, C_GREY),
        ("Feature extraction\n64×64 CDM + 16 blind statistics\n+ 32-bin radial histogram", C_LIGHT_GREEN, C_GREEN),
        ("CDM-Stats-Radial classifier", "#FFF9C4", "#F9A825"),
        ("Modulation label", C_LIGHT_BLUE, C_BLUE),
    ]
    y = 16.2
    dy = 1.62
    centers = []
    for text, fc, ec in steps_b:
        _box(ax2, 5, y, 6.8, 1.18 if "\n" in text else 0.8, text, fc, ec, fontsize=8.5)
        centers.append(y)
        y -= dy
    for i in range(len(centers) - 1):
        h1 = 0.59 if "\n" in steps_b[i][0] else 0.4
        h2 = 0.59 if "\n" in steps_b[i + 1][0] else 0.4
        _arrow(ax2, 5, centers[i] - h1, 5, centers[i + 1] + h2)

    fig.suptitle("Experimental OFDM-FSO acquisition and recognition pipeline  "
                 "(FFT=256, CP=16, 123 subcarriers, 128 symbols/frame, Dataset-A=900 frames)",
                 fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig_and_data(fig, "fig1_experimental_setup_pipeline.png")


# ─────────────────────────────────────────────────────────
# Fig.2: Calibrated pretraining framework
# ─────────────────────────────────────────────────────────
def plot_fig2_pretraining_framework():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis("off")

    ax.text(6.5, 8.15, "Calibrated Simulation Pretraining + Real-Data Fine-Tuning",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # Row 1: data sources
    _box(ax, 3.0, 7.0, 4.4, 0.9, "Real OFDM-FSO Dataset-A\n900 frames, 6 QAM orders",
         C_LIGHT_BLUE, C_BLUE, fontsize=10)
    _box(ax, 10.0, 7.0, 4.4, 0.9, "Calibrated-Sim-v2.2\n12,000 simulated frames",
         C_LIGHT_GREEN, C_GREEN, fontsize=10)

    # Impairment list
    _box(ax, 10.0, 5.95, 5.2, 0.85,
         "AWGN · lognormal scintillation · phase error\nbad subcarriers · impulsive noise · burst outliers",
         C_LIGHT_GREEN, C_GREEN, fontsize=8.5, fontweight="normal")
    _arrow(ax, 10.0, 6.55, 10.0, 6.40, color=C_GREEN)

    # Feature extraction
    _box(ax, 6.5, 5.0, 6.8, 0.85,
         "Feature Extraction:  64×64 CDM  +  48-D radial/statistics",
         C_LIGHT_GREY, C_GREY, fontsize=10)
    _arrow(ax, 3.0, 6.55, 5.0, 5.45, color=C_BLUE)
    _arrow(ax, 10.0, 5.5, 8.0, 5.45, color=C_GREEN)

    # Backbone
    _box(ax, 6.5, 3.85, 6.8, 0.8, "Shared CDM-Stats-Radial Network (backbone)",
         "#FFF9C4", "#F9A825", fontsize=10)
    _arrow(ax, 6.5, 4.55, 6.5, 4.27, color=C_GREY)

    # Two branches
    _box(ax, 3.0, 2.6, 4.4, 0.9, "Branch 1: Real-scratch\nrandom init → train on A-train\n5 / 10 / 20 / 100%",
         C_LIGHT_ORANGE, C_ORANGE, fontsize=9.5)
    _arrow(ax, 4.5, 3.45, 3.5, 3.07, color=C_ORANGE)

    _box(ax, 10.0, 2.6, 4.4, 0.9,
         "Branch 2: SimPretrain+RealFT\npretrain on Calibrated-Sim-v2.2\n→ fine-tune on A-train 5/10/20/100%",
         C_LIGHT_GREEN, C_GREEN, fontsize=9.5)
    _arrow(ax, 8.5, 3.45, 9.5, 3.07, color=C_GREEN)

    # Evaluation
    _box(ax, 6.5, 0.8, 5.8, 0.8, "Evaluation:  A-test, 180 frames, 5 seeds (2026–2030)",
         C_LIGHT_BLUE, C_BLUE, fontsize=10)
    _arrow(ax, 3.0, 2.13, 5.5, 1.22, color=C_ORANGE)
    _arrow(ax, 10.0, 2.13, 7.5, 1.22, color=C_GREEN)

    fig.tight_layout()
    save_fig_and_data(fig, "fig2_calibrated_pretraining_framework.png")


# ─────────────────────────────────────────────────────────
# Fig.3: A-test accuracy vs real-data ratio
# ─────────────────────────────────────────────────────────
def plot_fig3_ratio_curves(agg):
    fig, ax = plt.subplots(figsize=(8, 6))
    ft_m, ft_s, sc_m, sc_s = [], [], [], []
    for rkey in RATIO_KEYS:
        a, b = get_metric(agg, f"s3_ft_{rkey}", "overall_acc"); ft_m.append(a); ft_s.append(b)
        c, d = get_metric(agg, f"s3_scratch_{rkey}", "overall_acc"); sc_m.append(c); sc_s.append(d)
    x = np.arange(len(RATIO_LABELS))
    ft_m, ft_s, sc_m, sc_s = map(np.array, (ft_m, ft_s, sc_m, sc_s))

    ax.errorbar(x, ft_m * 100, yerr=ft_s * 100, marker="o", linewidth=2.2, capsize=5,
                label="SimPretrain+RealFT", color=C_BLUE, markersize=8)
    ax.errorbar(x, sc_m * 100, yerr=sc_s * 100, marker="s", linewidth=2.2, capsize=5,
                label="Real-scratch", color=C_ORANGE, markersize=8)
    s2_m, _ = get_metric(agg, "s2_simonly", "overall_acc")
    ax.axhline(y=s2_m * 100, color=C_GREEN, linestyle="--", alpha=0.7,
               label=f"Calibrated SimOnly ({s2_m*100:.1f}%)", linewidth=1.8)

    gain5 = (ft_m[0] - sc_m[0]) * 100
    ax.annotate(f"+{gain5:.1f} pp", xy=(0, (ft_m[0] + sc_m[0]) * 50),
                xytext=(0.45, 78), fontsize=11, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))

    ax.set_xticks(x); ax.set_xticklabels(RATIO_LABELS, fontsize=12)
    ax.set_xlabel("Real fine-tune ratio", fontsize=12)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.legend(fontsize=10.5, loc="lower right")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    fig.tight_layout()

    data = [{"ratio": RATIO_LABELS[i], "ft_mean": ft_m[i], "ft_std": ft_s[i],
             "sc_mean": sc_m[i], "sc_std": sc_s[i]} for i in range(len(RATIO_LABELS))]
    data.append({"ratio": "SimOnly", "ft_mean": s2_m, "ft_std": 0, "sc_mean": None, "sc_std": None})
    save_fig_and_data(fig, "fig3_atest_accuracy_vs_real_ratio.png", data)


# ─────────────────────────────────────────────────────────
# Fig.4: Sim realism ablation (5 metrics)
# ─────────────────────────────────────────────────────────
def plot_fig4_ablation(agg):
    metrics = [("overall_acc", "Overall"), ("macro_f1", "Macro-F1"),
               ("high_order_acc", "High-order"), ("acc_128qam", "128QAM"),
               ("acc_256qam", "256QAM")]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics)); width = 0.35
    v22_m, v22_s, aw_m, aw_s = [], [], [], []
    for m, _ in metrics:
        a, b = get_metric(agg, "s2_simonly", m); v22_m.append(a); v22_s.append(b)
        c, d = get_metric(agg, "s5_awgn_only", m); aw_m.append(c); aw_s.append(d)

    b1 = ax.bar(x - width / 2, np.array(v22_m) * 100, width, yerr=np.array(v22_s) * 100,
                capsize=5, label="Calibrated-Sim-v2.2", color=C_GREEN, alpha=0.88,
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, np.array(aw_m) * 100, width, yerr=np.array(aw_s) * 100,
                capsize=5, label="AWGN-only", color=C_RED, alpha=0.88,
                edgecolor="black", linewidth=0.5)
    for bar, val in zip(b1, v22_m):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 2, f"{val*100:.0f}",
                ha="center", va="bottom", fontsize=9, color="#1B5E20")
    for bar, val in zip(b2, aw_m):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 2, f"{val*100:.0f}",
                ha="center", va="bottom", fontsize=9, color="#B71C1C")

    ax.annotate("AWGN-only = 0%", xy=(3 + width / 2, 2), xytext=(3.2, 18),
                fontsize=9, color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.annotate("+26.8 pp", xy=(0, (v22_m[0] + aw_m[0]) * 50), xytext=(0.5, 60),
                fontsize=10, fontweight="bold", color="#1B5E20",
                arrowprops=dict(arrowstyle="->", color="#1B5E20"))

    ax.set_xticks(x); ax.set_xticklabels([t for _, t in metrics], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 112)
    fig.tight_layout()

    data = [{"metric": t, "v22_mean": v22_m[i], "v22_std": v22_s[i],
             "awgn_mean": aw_m[i], "awgn_std": aw_s[i]} for i, (_, t) in enumerate(metrics)]
    save_fig_and_data(fig, "fig4_atest_sim_realism_ablation.png", data)


# ─────────────────────────────────────────────────────────
# Fig.5: 5-seed averaged row-normalized confusion matrices (5%)
# ─────────────────────────────────────────────────────────
def plot_fig5_confusion_5seed():
    configs = [
        ("s3_scratch_5pct", "(a) 5% Real-scratch", 69.9, 13.9),
        ("s3_ft_5pct", "(b) 5% SimPretrain+RealFT", 84.4, 1.8),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    all_data = {}

    for ax, (cfg, title, acc_m, acc_s) in zip(axes, configs):
        seed_mats = []
        for seed in SEEDS:
            p = FP_ROOT / f"seed_{seed}" / cfg / "metrics.json"
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            cm = np.array(m["A_test"]["confusion_matrix"], dtype=float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_norm = cm / row_sums * 100.0
            seed_mats.append(cm_norm)

        mean_mat = np.mean(seed_mats, axis=0)
        std_mat = np.std(seed_mats, axis=0)
        all_data[cfg] = {"mean": mean_mat.tolist(), "std": std_mat.tolist(),
                         "class_names": MOD_NAMES, "seeds": SEEDS}

        im = ax.imshow(mean_mat, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(MOD_NAMES, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(MOD_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{title}\nA-test overall = {acc_m} ± {acc_s}%", fontsize=11)

        for i in range(6):
            for j in range(6):
                color = "white" if mean_mat[i, j] > 50 else "black"
                ax.text(j, i, f"{mean_mat[i, j]:.1f}", ha="center", va="center",
                        color=color, fontsize=9)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Row-normalized (%)")
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    # Row-sum verification
    for cfg, d in all_data.items():
        row_sums = np.array(d["mean"]).sum(axis=1)
        print(f"    [check] {cfg} row sums: {np.round(row_sums, 2).tolist()}")

    save_fig_and_data(fig, "fig5_atest_confusion_5seed_avg_5pct.png", all_data)


# ─────────────────────────────────────────────────────────
# Fig.6: High-order QAM vs ratio
# ─────────────────────────────────────────────────────────
def plot_fig6_highorder_ratio(agg):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    series = [
        ("acc_128qam", "s3_scratch", "128QAM Real-scratch", C_ORANGE, "s", "--"),
        ("acc_128qam", "s3_ft", "128QAM SimPretrain+RealFT", C_RED, "o", "-"),
        ("acc_256qam", "s3_scratch", "256QAM Real-scratch", C_BLUE, "^", "--"),
        ("acc_256qam", "s3_ft", "256QAM SimPretrain+RealFT", C_GREEN, "D", "-"),
    ]
    x = np.arange(len(RATIO_LABELS))
    data = []
    for metric, prefix, label, color, marker, ls in series:
        means, stds = [], []
        for rkey in RATIO_KEYS:
            m, s = get_metric(agg, f"{prefix}_{rkey}", metric)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.errorbar(x, means * 100, yerr=stds * 100, marker=marker, linewidth=2,
                    capsize=4, label=label, color=color, linestyle=ls, markersize=7)
        for i, rkey in enumerate(RATIO_KEYS):
            data.append({"series": label, "ratio": RATIO_LABELS[i],
                         "mean": means[i], "std": stds[i]})

    ax.set_xticks(x); ax.set_xticklabels(RATIO_LABELS, fontsize=12)
    ax.set_xlabel("Real fine-tune ratio", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(True, alpha=0.3); ax.set_ylim(-5, 105)
    fig.tight_layout()
    save_fig_and_data(fig, "fig6_atest_highorder_vs_ratio.png", data)


# ─────────────────────────────────────────────────────────
# Parameter table
# ─────────────────────────────────────────────────────────
def write_parameter_table():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Wavelength", "1550 nm"),
        ("Modulator", "MZM (Oclaro, 12.5 GHz bandwidth)"),
        ("AWG", "Keysight M8195A"),
        ("SLM", "Holoeye Pluto"),
        ("SLM phase screens", "200 per turbulence condition (weak, strong)"),
        ("APD bandwidth", "50 GHz"),
        ("DSO", "Keysight DSAX92504A, 80 GSa/s"),
        ("Collimator", "Thorlabs TC18APC-1550"),
        ("VOA", "EXFO FVA-600"),
        ("FFT size", "256"),
        ("CP length", "16"),
        ("Active subcarriers", "123"),
        ("OFDM symbols per frame", "128"),
        ("Modulation formats", "QPSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM"),
        ("Turbulence conditions", "Weak, strong"),
        ("Dataset-A frames", "900"),
        ("Train / validation / test", "648 / 72 / 180"),
        ("Feature representation", "64×64 CDM + 16 blind statistics + 32-bin radial histogram"),
        ("Simulated frames (Calibrated-Sim-v2.2)", "12,000"),
        ("Training seeds", "2026–2030"),
    ]
    # Markdown
    md = OUT_DIR / "table_experimental_parameters.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("**Table II.** Experimental OFDM-FSO and recognition parameters.\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        for k, v in rows:
            f.write(f"| {k} | {v} |\n")
    # LaTeX
    tex = OUT_DIR / "table_experimental_parameters.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Experimental OFDM-FSO and recognition parameters.}\n")
        f.write("\\label{tab:params}\n\\footnotesize\n\\begin{tabular}{ll}\n\\toprule\n")
        f.write("Parameter & Value \\\\\n\\midrule\n")
        for k, v in rows:
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("  [table] table_experimental_parameters.md / .tex")


def main():
    print("[1/4] Loading aggregated results...")
    agg = load_agg()
    print("[2/4] Generating figures...")
    plot_fig1_setup_pipeline()
    plot_fig2_pretraining_framework()
    plot_fig3_ratio_curves(agg)
    plot_fig4_ablation(agg)
    plot_fig5_confusion_5seed()
    plot_fig6_highorder_ratio(agg)
    write_parameter_table()
    print(f"[3/4] All outputs in: {OUT_DIR}")
    print("[4/4] Done.")


if __name__ == "__main__":
    main()
