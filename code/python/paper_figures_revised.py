# -*- coding: utf-8 -*-
"""
paper_figures_revised.py — IEEE/ACP 会议论文投稿用正式图片生成（统一脚本）。

依据 12_51.txt 规范：
  - 全局：Times New Roman，轴标签 9pt / 刻度 8pt / 图例 8pt / 子图标题 9pt，
          线宽 1.8，标记 5，误差棒 capsize 3，浅灰网格 alpha 0.25，
          PNG 600 dpi + 矢量 PDF，输出到 paper_figures_revised/
  - 术语统一：IC-Sim / IC-Sim only / IC-Sim + RealFT / Measured-data budget / Frozen-test
  - 横轴 5%/10%/20%/100% 按离散等距显示；不生成 B-test 图
  - 数值取自已有结果文件或脚本内显式常量，不重新计算/不修改结果

主图：fig1_pipeline / fig2_cdm_six_classes / fig3_accuracy_vs_budget /
      fig4_awgn_ablation / fig5_confusion_5pct
可选：fig6_high_order_accuracy / fig7_high_order_finetuning
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─────────────────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────────────────
FP_ROOT = Path(r"D:\Project_code\FSO_research\results\stage2\fast_publish")
CACHE = FP_ROOT / "real_cache_ab.npz"
PAPER_FIG_DIR = FP_ROOT / "paper_aonly_figures"
FIG5_JSON = PAPER_FIG_DIR / "fig5_atest_confusion_5seed_avg_5pct_data.json"
FIG6_JSON = PAPER_FIG_DIR / "fig6_atest_highorder_vs_ratio_data.json"
FIG7_JSON = PAPER_FIG_DIR / "fig7_highorder_finetune_comparison_data.json"
OUT_DIR = Path(r"D:\Project_code\FSO_research\results\docs\paper_acp\paper_figures_revised")

MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
BUDGET_LABELS = ["5%", "10%", "20%", "100%"]

# ─────────────────────────────────────────────────────────
# 全局样式
# ─────────────────────────────────────────────────────────
FS_AXIS = 9      # axis label
FS_TICK = 8      # tick label
FS_LEGEND = 8    # legend
FS_TITLE = 9     # subfigure title
FS_CBAR = 8      # colorbar
LW = 1.8
MS = 5
CAP = 3
DPI = 600

C_BLUE = "#1976D2"
C_GREEN = "#388E3C"
C_ORANGE = "#F57C00"
C_RED = "#D32F2F"
C_GREY = "#455A64"
C_PURPLE = "#7B1FA2"
C_LBLUE = "#BBDEFB"
C_LGREEN = "#C8E6C9"
C_LORANGE = "#FFE0B2"
C_LGREY = "#ECEFF1"
C_LRED = "#FFCDD2"
C_LPURPLE = "#E1BEE7"


def apply_global_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.labelsize": FS_AXIS,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.titlesize": FS_TITLE,
        "axes.linewidth": 0.9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#BBBBBB",
        "legend.borderpad": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")
    print(f"  -> {pdf}")
    return png, pdf


def light_grid(ax, axis="both"):
    ax.grid(True, axis=axis, color="lightgray", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)


def thin_legend(ax, **kw):
    leg = ax.legend(**kw)
    if leg is not None:
        leg.get_frame().set_linewidth(0.6)
        leg.get_frame().set_edgecolor("#BBBBBB")
        leg.get_frame().set_alpha(0.85)
    return leg


# ─────────────────────────────────────────────────────────
# Fig.1: 横向紧凑管道框图
# ─────────────────────────────────────────────────────────
def _box(ax, cx, cy, w, h, text, fc, ec, fontsize=FS_TICK, fontweight="normal",
         textcolor="black"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06", facecolor=fc,
                                edgecolor=ec, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor)


def _harrow(ax, x1, x2, y, color=C_GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=12, color=color, linewidth=lw))


def plot_fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    modules = [
        ("TX DSP", ["QAM Map / S-P", "Pilot + Hermitian", "256-pt IFFT + CP"],
         C_LBLUE, C_BLUE),
        ("Optical FSO Link",
         ["AWG", "MZM + 1550 nm laser", "PC + Collimator",
          "SLM turbulence", "APD + DSO"],
         C_LORANGE, C_ORANGE),
        ("RX Offline DSP",
         ["Sync / CP Removal / FFT", "Ch. Est. / Equalization",
          "Active Subcarrier Extraction"],
         C_LGREY, C_GREY),
        ("Feature Extraction", ["CDM", "Stats / Radial features"],
         C_LGREEN, C_GREEN),
    ]
    centers = [10, 30, 50, 70, 90]
    w = 16
    y_top, y_bot = 92, 8
    hdr_cy = 86
    item_top, item_bot = 78, 14

    for (title, items, fc, ec), cx in zip(modules, centers):
        # 容器
        ax.add_patch(FancyBboxPatch((cx - w / 2, y_bot), w, y_top - y_bot,
                                    boxstyle="round,pad=0.10", facecolor=fc,
                                    edgecolor="black", linewidth=1.2, alpha=0.55))
        # 标题栏
        _box(ax, cx, hdr_cy, w - 1.2, 7, title, ec, ec, fontsize=FS_TITLE,
             fontweight="bold", textcolor="white")
        # 子模块
        n = len(items)
        ys = np.linspace(item_top, item_bot, n)
        ih = min(7.0, (item_top - item_bot) / max(n, 1) * 0.72)
        for y, it in zip(ys, items):
            _box(ax, cx, y, w - 2.0, ih, it, "white", ec, fontsize=FS_TICK)

    # 终端模块：Modulation Prediction（与容器等高）
    cx = centers[-1]
    ax.add_patch(FancyBboxPatch((cx - w / 2, y_bot), w, y_top - y_bot,
                                boxstyle="round,pad=0.10", facecolor=C_LBLUE,
                                edgecolor="black", linewidth=1.2, alpha=0.55))
    _box(ax, cx, (y_top + y_bot) / 2, w - 1.2, 10, "Modulation\nPrediction",
         C_BLUE, C_BLUE, fontsize=FS_TITLE, fontweight="bold", textcolor="white")

    # 模块间横向箭头
    mid_y = (y_top + y_bot) / 2
    for i in range(len(centers) - 1):
        _harrow(ax, centers[i] + w / 2 + 0.2, centers[i + 1] - w / 2 - 0.2, mid_y)

    fig.tight_layout()
    save_fig(fig, "fig1_pipeline_revised")


# ─────────────────────────────────────────────────────────
# Fig.2: 六类 CDM（2×3，共享 colorbar）
# ─────────────────────────────────────────────────────────
def plot_fig2_cdm_six_classes():
    data = np.load(CACHE, allow_pickle=True)
    X_cdm = data["X_cdm"]
    y = data["y"]
    turbs = data["turbs"]
    sources = data["sources"]
    split = data["split"]
    panels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    cdms = []
    for label, _name in enumerate(MOD_NAMES):
        mask = (y == label) & (sources == "A") & (split == "test") & (turbs == "weak")
        idx = np.where(mask)[0]
        if len(idx) == 0:
            mask = (y == label) & (sources == "A") & (split == "test")
            idx = np.where(mask)[0]
        if len(idx) == 0:
            mask = (y == label) & (sources == "A")
            idx = np.where(mask)[0]
        cdms.append(X_cdm[int(idx[0]), 0])

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6))
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.93,
                        wspace=0.30, hspace=0.40)
    im_last = None
    for k, ax in enumerate(axes.flat):
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        im_last = ax.imshow(cdms[k], cmap="viridis", origin="lower",
                            extent=[-3.0, 3.0, -3.0, 3.0],
                            aspect="equal", vmin=0.0, vmax=1.0)
        ax.set_title(f"{panels[k]} {MOD_NAMES[k]}", fontsize=FS_TITLE, pad=3)
        ax.tick_params(labelsize=FS_TICK)
        if k // 3 == 1:
            ax.set_xlabel("In-phase", fontsize=FS_AXIS)
        if k % 3 == 0:
            ax.set_ylabel("Quadrature", fontsize=FS_AXIS)

    cax = fig.add_axes([0.88, 0.13, 0.022, 0.80])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Log-normalized density", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR)
    save_fig(fig, "fig2_cdm_six_classes_revised")


# ─────────────────────────────────────────────────────────
# Fig.3: Overall accuracy vs measured-data budget
# ─────────────────────────────────────────────────────────
def plot_fig3_accuracy_vs_budget():
    # 数值为规范 (12_51.txt) 中明确给出的常量，与已有结果文件一致
    real_scratch_m = np.array([69.9, 83.2, 88.0, 90.3])
    real_scratch_s = np.array([13.9, 2.2, 2.6, 1.4])
    ic_ft_m = np.array([84.4, 87.0, 92.1, 94.9])
    ic_ft_s = np.array([1.8, 2.0, 0.5, 1.3])
    ic_only = 71.3

    x = np.arange(len(BUDGET_LABELS))
    fig, ax = plt.subplots(figsize=(4.9, 3.7))

    ax.axhline(y=ic_only, color=C_GREEN, linestyle="--", linewidth=LW,
               label=f"IC-Sim only ({ic_only:.1f}%)")
    ax.errorbar(x, ic_ft_m, yerr=ic_ft_s, marker="o", color=C_BLUE,
                linewidth=LW, markersize=MS, capsize=CAP,
                label="IC-Sim + RealFT")
    ax.errorbar(x, real_scratch_m, yerr=real_scratch_s, marker="s",
                color=C_ORANGE, linewidth=LW, markersize=MS, capsize=CAP,
                label="Real-scratch")

    # +14.6 pp：箭头从 5% Real-scratch 指向 5% IC-Sim + RealFT
    ax.annotate("", xy=(0, ic_ft_m[0]), xytext=(0, real_scratch_m[0]),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4))
    ax.text(0.18, (ic_ft_m[0] + real_scratch_m[0]) / 2, "+14.6 pp",
            color=C_RED, fontsize=FS_LEGEND, fontweight="bold", va="center")

    ax.set_xticks(x); ax.set_xticklabels(BUDGET_LABELS)
    ax.set_xlabel("Measured-data budget", fontsize=FS_AXIS)
    ax.set_ylabel("Overall accuracy (%)", fontsize=FS_AXIS)
    ax.set_ylim(50, 100)
    thin_legend(ax, fontsize=FS_LEGEND, loc="lower right")
    light_grid(ax)
    fig.tight_layout()
    save_fig(fig, "fig3_accuracy_vs_budget_revised")


# ─────────────────────────────────────────────────────────
# Fig.4: IC-Sim only vs AWGN-only 弱湍流消融
# ─────────────────────────────────────────────────────────
def plot_fig4_awgn_ablation():
    # 数值为规范 (12_51.txt) 中明确给出的常量
    metrics = ["Overall", "Macro-F1", "High-order", "128QAM", "256QAM"]
    ic_only_m = np.array([71.3, 69.5, 46.4, 52.7, 40.0])
    ic_only_s = np.array([1.4, 1.4, 5.6, 14.4, 2.1])
    awgn_m = np.array([44.6, 37.5, 30.0, 0.0, 90.0])
    awgn_s = np.array([1.8, 2.3, 4.1, 0.0, 12.3])

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.7))

    ax.bar(x - width / 2, ic_only_m, width, yerr=ic_only_s, capsize=CAP,
           color=C_GREEN, edgecolor="black", linewidth=0.5, alpha=0.9,
           label="IC-Sim only")
    ax.bar(x + width / 2, awgn_m, width, yerr=awgn_s, capsize=CAP,
           color=C_RED, edgecolor="black", linewidth=0.5, alpha=0.9,
           label="AWGN-only")

    # 128QAM 的 AWGN-only 柱为 0，标注 "0"（不使用夸张箭头）
    ax.text(x[3] + width / 2, 1.5, "0", ha="center", va="bottom",
            fontsize=FS_TICK, color=C_RED)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=FS_TICK)
    ax.set_ylabel("Frozen-test score (%)", fontsize=FS_AXIS)
    ax.set_ylim(0, 112)
    thin_legend(ax, fontsize=FS_LEGEND, loc="upper right")
    light_grid(ax, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig4_awgn_ablation_revised")


# ─────────────────────────────────────────────────────────
# Fig.5: 5% measured-data budget 下的混淆矩阵
# ─────────────────────────────────────────────────────────
def plot_fig5_confusion_5pct():
    with open(FIG5_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = [
        ("s3_scratch_5pct", "(a) 5% Real-scratch", "fig5a_confusion_5pct_revised"),
        ("s3_ft_5pct", "(b) 5% IC-Sim + RealFT", "fig5b_confusion_5pct_revised"),
    ]
    for key, title, fname in configs:
        mat = np.array(data[key]["mean"], dtype=float)
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        fig.subplots_adjust(left=0.17, right=0.80, bottom=0.16, top=0.90)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(MOD_NAMES, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(MOD_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        for i in range(6):
            for j in range(6):
                color = "white" if mat[i, j] > 50 else "black"
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        color=color, fontsize=12)
        cax = fig.add_axes([0.82, 0.16, 0.04, 0.74])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Row-normalized (%)", fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        save_fig(fig, fname)


# ─────────────────────────────────────────────────────────
# Fig.6（可选）: High-order QAM accuracy vs measured-data budget
# ─────────────────────────────────────────────────────────
def plot_fig6_high_order_accuracy():
    with open(FIG6_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    series_map = {
        "128QAM Real-scratch": ("128QAM Scratch", C_ORANGE, "s", "--"),
        "128QAM SimPretrain+RealFT": ("128QAM IC-Sim+FT", C_RED, "o", "-"),
        "256QAM Real-scratch": ("256QAM Scratch", C_BLUE, "^", "--"),
        "256QAM SimPretrain+RealFT": ("256QAM IC-Sim+FT", C_GREEN, "D", "-"),
    }
    by_series = {k: {"5%": None, "10%": None, "20%": None, "100%": None}
                 for k in series_map}
    for row in raw:
        if row["series"] in by_series:
            by_series[row["series"]][row["ratio"]] = (row["mean"] * 100.0,
                                                      row["std"] * 100.0)

    x = np.arange(len(BUDGET_LABELS))
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for src, (lab, color, marker, ls) in series_map.items():
        means = np.array([by_series[src][r][0] for r in BUDGET_LABELS])
        stds = np.array([by_series[src][r][1] for r in BUDGET_LABELS])
        ax.errorbar(x, means, yerr=stds, marker=marker, color=color,
                    linestyle=ls, linewidth=LW, markersize=MS, capsize=CAP,
                    label=lab)
    ax.set_xticks(x); ax.set_xticklabels(BUDGET_LABELS)
    ax.set_xlabel("Measured-data budget", fontsize=FS_AXIS)
    ax.set_ylabel("Class-wise accuracy (%)", fontsize=FS_AXIS)
    ax.set_ylim(-5, 105)
    thin_legend(ax, fontsize=7, loc="lower right", ncol=2)
    light_grid(ax)
    fig.tight_layout()
    save_fig(fig, "fig6_high_order_accuracy_revised")


# ─────────────────────────────────────────────────────────
# Fig.7（可选）: High-order-aware fine-tuning comparison
# ─────────────────────────────────────────────────────────
def plot_fig7_high_order_finetuning():
    with open(FIG7_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    configs = [
        ("s3_ft_5pct", "Baseline", C_GREEN),
        ("ho_weighted_5pct", "HO-weighted CE", C_BLUE),
        ("ho_balanced_5pct", "HO-balanced batch", C_ORANGE),
        ("ho_replay_5pct", "Real:sim replay", C_RED),
    ]
    metrics = ["overall_acc", "macro_f1", "high_order_acc",
               "acc_128qam", "acc_256qam"]
    metric_labels = ["Overall", "Macro-F1", "High-order", "128QAM", "256QAM"]

    x = np.arange(len(metrics))
    n = len(configs)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for i, (key, lab, color) in enumerate(configs):
        means = np.array([data[key][m]["mean"] for m in metrics])
        stds = np.array([data[key][m]["std"] for m in metrics])
        ax.bar(x + (i - (n - 1) / 2) * width, means, width, yerr=stds,
               capsize=CAP, color=color, edgecolor="black", linewidth=0.4,
               alpha=0.9, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=FS_TICK)
    ax.set_ylabel("Frozen-test accuracy (%)", fontsize=FS_AXIS)
    ax.set_ylim(0, 105)
    thin_legend(ax, fontsize=FS_LEGEND, loc="upper right", ncol=2)
    light_grid(ax, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig7_high_order_finetuning_revised")


def main():
    apply_global_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Fig.1] pipeline")
    plot_fig1_pipeline()
    print("[Fig.2] CDM six classes")
    plot_fig2_cdm_six_classes()
    print("[Fig.3] accuracy vs budget")
    plot_fig3_accuracy_vs_budget()
    print("[Fig.4] AWGN ablation")
    plot_fig4_awgn_ablation()
    print("[Fig.5] confusion 5%")
    plot_fig5_confusion_5pct()
    print("[Fig.6] high-order accuracy (optional)")
    plot_fig6_high_order_accuracy()
    print("[Fig.7] high-order finetuning (optional)")
    plot_fig7_high_order_finetuning()
    print("\nAll revised figures written to:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
