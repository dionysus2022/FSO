# -*- coding: utf-8 -*-
"""
sim_compare_distributions.py — Step 5: 真实 vs 仿真分布对比（S1 门）

生成 5 张对比图 + KS 统计量，判断仿真数据是否接近真实数据。

输出目录: results/stage2/distribution_check_v1/
  fig1_constellation_overlay.png   — 星座图叠加
  fig2_cdm_comparison.png          — CDM 对比
  fig3_radial_histogram.png        — 径向直方图
  fig4_snr_evm_distribution.png    — SNR/EVM 分布
  fig5_highorder_cdm_zoom.png      — 高阶 QAM CDM 中心区域
  ks_metrics.json                  — KS 检验结果
  summary.txt                      — 人类可读判断
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 确保能 import stage2_common 和 train_mrf_lowmem
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    FROZEN_SPLIT_CSV, SIM_MANIFEST_CSV, DIST_CHECK_ROOT,
    MOD_NAMES, N_SC, N_SYM,
    ensure_dir, now_str,
)
from train_mrf_lowmem import make_cdm_from_rx_sc, make_radial_hist, make_blind_stats

# 仿真 turb → 真实 turb 映射
SIM_TO_REAL_TURB = {"very_weak": "weak", "strong": "strong"}
REAL_TURBS = ["weak", "strong"]
HIGH_ORDER_MODS = ["64QAM", "128QAM", "256QAM"]

N_SAMPLES_PER_GROUP = 80  # 每 mod×turb 采样帧数


# ─────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────
def load_real_samples(real_turb: str, mod_name: str, n: int) -> List[np.ndarray]:
    """从 frozen split 的 test 集采样真实 rx_sc。"""
    df = pd.read_csv(FROZEN_SPLIT_CSV)
    sub = df[(df["turb_name"] == real_turb) &
             (df["mod_name"] == mod_name) &
             (df["split"] == "test")]
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    samples = []
    for _, row in sub.iterrows():
        try:
            m = sio.loadmat(row["out_mat"])
            z = np.asarray(m["rx_sc"]).astype(np.complex64)
            samples.append(z)
        except Exception as e:
            print(f"  [跳过] {row['out_mat']}: {e}")
    return samples


def load_sim_samples(sim_turb: str, mod_name: str, n: int) -> List[np.ndarray]:
    """从 sim manifest 采样 rx_sc。"""
    df = pd.read_csv(SIM_MANIFEST_CSV)
    sub = df[(df["turb_name"] == sim_turb) & (df["mod_name"] == mod_name)]
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    samples = []
    for _, row in sub.iterrows():
        try:
            m = sio.loadmat(row["out_mat"])
            z = np.asarray(m["rx_sc"]).astype(np.complex64)
            samples.append(z)
        except Exception as e:
            print(f"  [跳过] {row['out_mat']}: {e}")
    return samples


def normalize_z(z: np.ndarray) -> np.ndarray:
    """与 lowmem 一致的归一化。"""
    z = z.reshape(-1).astype(np.complex64)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    return z


# ─────────────────────────────────────────────────────────
# 图 1: 星座图叠加
# ─────────────────────────────────────────────────────────
def plot_constellation_overlay(all_real: dict, all_sim: dict):
    """6 mods × 2 turbs = 12 子图，每子图叠加 real（蓝）+ sim（红）。"""
    fig, axes = plt.subplots(6, 2, figsize=(14, 36))
    fig.suptitle("Fig 1: Constellation Overlay (Real=blue, Sim=red)", fontsize=16, y=1.01)

    for i, mod in enumerate(MOD_NAMES):
        for j, real_turb in enumerate(REAL_TURBS):
            sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
            ax = axes[i, j]

            real_samples = all_real.get((real_turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                z_real = np.concatenate([normalize_z(s) for s in real_samples[:5]])
                ax.scatter(z_real.real, z_real.imag, s=1, c="blue", alpha=0.3, label="real")
            if sim_samples:
                z_sim = np.concatenate([normalize_z(s) for s in sim_samples[:5]])
                ax.scatter(z_sim.real, z_sim.imag, s=1, c="red", alpha=0.3, label="sim")

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            ax.set_title(f"{mod} / {real_turb}", fontsize=10)
            ax.legend(fontsize=8, markerscale=5)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = DIST_CHECK_ROOT / "fig1_constellation_overlay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 2: CDM 对比
# ─────────────────────────────────────────────────────────
def plot_cdm_comparison(all_real: dict, all_sim: dict):
    """6 mods × 2 turbs × 2 (real/sim) = 24 子图。"""
    fig, axes = plt.subplots(12, 4, figsize=(20, 48))
    fig.suptitle("Fig 2: CDM Comparison (left=Real, right=Sim)", fontsize=16, y=1.01)

    for i, mod in enumerate(MOD_NAMES):
        for j, real_turb in enumerate(REAL_TURBS):
            sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
            ax_real = axes[i * 2, j * 2]
            ax_sim = axes[i * 2, j * 2 + 1]
            ax_real2 = axes[i * 2 + 1, j * 2]
            ax_sim2 = axes[i * 2 + 1, j * 2 + 1]

            # Average CDM over samples
            real_samples = all_real.get((real_turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in real_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_real.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_real.set_title(f"Real {mod}/{real_turb}", fontsize=8)
                # Also show a single example
                ax_real2.imshow(make_cdm_from_rx_sc(real_samples[0]), cmap="hot",
                                origin="lower", extent=[-3, 3, -3, 3])
                ax_real2.set_title(f"Real ex {mod}/{real_turb}", fontsize=8)

            if sim_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in sim_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_sim.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_sim.set_title(f"Sim {mod}/{sim_turb}", fontsize=8)
                ax_sim2.imshow(make_cdm_from_rx_sc(sim_samples[0]), cmap="hot",
                               origin="lower", extent=[-3, 3, -3, 3])
                ax_sim2.set_title(f"Sim ex {mod}/{sim_turb}", fontsize=8)

            for ax in [ax_real, ax_sim, ax_real2, ax_sim2]:
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)

    plt.tight_layout()
    path = DIST_CHECK_ROOT / "fig2_cdm_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 3: 径向直方图
# ─────────────────────────────────────────────────────────
def plot_radial_histogram(all_real: dict, all_sim: dict):
    """2 turbs × 1，每子图叠加 6 mod 的 real（实线）+ sim（虚线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Fig 3: Radial Histogram (Real=solid, Sim=dashed)", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for j, real_turb in enumerate(REAL_TURBS):
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
        ax = axes[j]

        for i, mod in enumerate(MOD_NAMES):
            real_samples = all_real.get((real_turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                hists = [make_radial_hist(s) for s in real_samples[:20]]
                avg_h = np.mean(hists, axis=0)
                ax.plot(avg_h, color=colors[i], linestyle="-", alpha=0.8, label=f"R-{mod}")
            if sim_samples:
                hists = [make_radial_hist(s) for s in sim_samples[:20]]
                avg_h = np.mean(hists, axis=0)
                ax.plot(avg_h, color=colors[i], linestyle="--", alpha=0.8, label=f"S-{mod}")

        ax.set_xlabel("Amplitude bin")
        ax.set_ylabel("L1-normalized density")
        ax.set_title(f"Turbulence: {real_turb}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = DIST_CHECK_ROOT / "fig3_radial_histogram.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 4: SNR / PAPR 分布
# ─────────────────────────────────────────────────────────
def plot_snr_papr_distribution(all_real: dict, all_sim: dict):
    """2 turbs × 2 (PAPR histogram + amp_mean histogram)。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fig 4: PAPR & amp_mean Distribution (Real=blue, Sim=red)", fontsize=14)

    for j, real_turb in enumerate(REAL_TURBS):
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]

        real_paprs, real_amps = [], []
        sim_paprs, sim_amps = [], []

        for mod in MOD_NAMES:
            for z in all_real.get((real_turb, mod), []):
                zn = normalize_z(z)
                real_paprs.append(10 * np.log10(np.max(np.abs(zn) ** 2) / (np.mean(np.abs(zn) ** 2) + 1e-12)))
                real_amps.append(float(np.mean(np.abs(zn))))
            for z in all_sim.get((sim_turb, mod), []):
                zn = normalize_z(z)
                sim_paprs.append(10 * np.log10(np.max(np.abs(zn) ** 2) / (np.mean(np.abs(zn) ** 2) + 1e-12)))
                sim_amps.append(float(np.mean(np.abs(zn))))

        ax_papr = axes[0, j]
        ax_amp = axes[1, j]

        ax_papr.hist(real_paprs, bins=30, alpha=0.5, color="blue", label="real", density=True)
        ax_papr.hist(sim_paprs, bins=30, alpha=0.5, color="red", label="sim", density=True)
        ax_papr.set_title(f"PAPR (dB) — {real_turb}")
        ax_papr.set_xlabel("PAPR (dB)")
        ax_papr.legend()
        ax_papr.grid(True, alpha=0.3)

        ax_amp.hist(real_amps, bins=30, alpha=0.5, color="blue", label="real", density=True)
        ax_amp.hist(sim_amps, bins=30, alpha=0.5, color="red", label="sim", density=True)
        ax_amp.set_title(f"amp_mean — {real_turb}")
        ax_amp.set_xlabel("amp_mean")
        ax_amp.legend()
        ax_amp.grid(True, alpha=0.3)

        # KS test
        if len(real_paprs) > 5 and len(sim_paprs) > 5:
            ks_papr, p_papr = scipy_stats.ks_2samp(real_paprs, sim_paprs)
            ks_amp, p_amp = scipy_stats.ks_2samp(real_amps, sim_amps)
            ax_papr.text(0.05, 0.95, f"KS={ks_papr:.3f}\np={p_papr:.4f}",
                         transform=ax_papr.transAxes, va="top", fontsize=9,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax_amp.text(0.05, 0.95, f"KS={ks_amp:.3f}\np={p_amp:.4f}",
                        transform=ax_amp.transAxes, va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = DIST_CHECK_ROOT / "fig4_snr_evm_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 5: 高阶 QAM CDM 中心区域 zoom
# ─────────────────────────────────────────────────────────
def plot_highorder_cdm_zoom(all_real: dict, all_sim: dict):
    """3 mods × 2 turbs × 2 (real/sim) = 12 子图，CDM 中心 [-1,1] 区域。"""
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    fig.suptitle("Fig 5: High-order QAM CDM Center Zoom [-1,1] (left=Real, right=Sim)",
                 fontsize=14)

    for i, mod in enumerate(HIGH_ORDER_MODS):
        for j, real_turb in enumerate(REAL_TURBS):
            sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
            ax_real = axes[i, j * 2]
            ax_sim = axes[i, j * 2 + 1]

            real_samples = all_real.get((real_turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in real_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_real.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_real.set_xlim(-1.5, 1.5)
                ax_real.set_ylim(-1.5, 1.5)
                ax_real.set_title(f"Real {mod}/{real_turb}", fontsize=9)

            if sim_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in sim_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_sim.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_sim.set_xlim(-1.5, 1.5)
                ax_sim.set_ylim(-1.5, 1.5)
                ax_sim.set_title(f"Sim {mod}/{sim_turb}", fontsize=9)

    plt.tight_layout()
    path = DIST_CHECK_ROOT / "fig5_highorder_cdm_zoom.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# KS 检验
# ─────────────────────────────────────────────────────────
def compute_ks_metrics(all_real: dict, all_sim: dict) -> dict:
    """对每个 mod×turb 组合，对 per-frame 特征做 2-sample KS 检验。"""
    metrics = {}

    for real_turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
        turb_key = real_turb
        metrics[turb_key] = {}

        for mod in MOD_NAMES:
            real_samples = all_real.get((real_turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if len(real_samples) < 5 or len(sim_samples) < 5:
                metrics[turb_key][mod] = {"error": "insufficient samples"}
                continue

            # Per-frame features
            real_paprs, sim_paprs = [], []
            real_amps, sim_amps = [], []
            real_radial_mean, sim_radial_mean = [], []

            for z in real_samples:
                zn = normalize_z(z)
                real_paprs.append(10 * np.log10(np.max(np.abs(zn) ** 2) / (np.mean(np.abs(zn) ** 2) + 1e-12)))
                real_amps.append(float(np.mean(np.abs(zn))))
                h = make_radial_hist(z)
                real_radial_mean.append(float(np.mean(h)))

            for z in sim_samples:
                zn = normalize_z(z)
                sim_paprs.append(10 * np.log10(np.max(np.abs(zn) ** 2) / (np.mean(np.abs(zn) ** 2) + 1e-12)))
                sim_amps.append(float(np.mean(np.abs(zn))))
                h = make_radial_hist(z)
                sim_radial_mean.append(float(np.mean(h)))

            # KS tests
            ks_papr, p_papr = scipy_stats.ks_2samp(real_paprs, sim_paprs)
            ks_amp, p_amp = scipy_stats.ks_2samp(real_amps, sim_amps)
            ks_radial, p_radial = scipy_stats.ks_2samp(real_radial_mean, sim_radial_mean)

            metrics[turb_key][mod] = {
                "papr": {"ks_stat": float(ks_papr), "p_value": float(p_papr),
                         "real_mean": float(np.mean(real_paprs)), "sim_mean": float(np.mean(sim_paprs))},
                "amp_mean": {"ks_stat": float(ks_amp), "p_value": float(p_amp),
                             "real_mean": float(np.mean(real_amps)), "sim_mean": float(np.mean(sim_amps))},
                "radial_mean": {"ks_stat": float(ks_radial), "p_value": float(p_radial),
                                "real_mean": float(np.mean(real_radial_mean)),
                                "sim_mean": float(np.mean(sim_radial_mean))},
            }

    return metrics


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== sim_compare_distributions.py  [{now_str()}] ===")
    ensure_dir(DIST_CHECK_ROOT)

    # 1. 加载数据
    print(f"\n[1/3] 加载数据（每 mod×turb 采样 {N_SAMPLES_PER_GROUP} 帧）")
    all_real = {}
    all_sim = {}

    for real_turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]
        for mod in MOD_NAMES:
            print(f"  real {real_turb}/{mod}...", end=" ")
            real_s = load_real_samples(real_turb, mod, N_SAMPLES_PER_GROUP)
            print(f"({len(real_s)} frames)")
            all_real[(real_turb, mod)] = real_s

            print(f"  sim  {sim_turb}/{mod}...", end=" ")
            sim_s = load_sim_samples(sim_turb, mod, N_SAMPLES_PER_GROUP)
            print(f"({len(sim_s)} frames)")
            all_sim[(sim_turb, mod)] = sim_s

    # 2. 生成 5 张图
    print(f"\n[2/3] 生成分布对比图")
    plot_constellation_overlay(all_real, all_sim)
    plot_cdm_comparison(all_real, all_sim)
    plot_radial_histogram(all_real, all_sim)
    plot_snr_papr_distribution(all_real, all_sim)
    plot_highorder_cdm_zoom(all_real, all_sim)

    # 3. KS 检验
    print(f"\n[3/3] KS 检验")
    ks_metrics = compute_ks_metrics(all_real, all_sim)

    ks_path = DIST_CHECK_ROOT / "ks_metrics.json"
    with open(ks_path, "w", encoding="utf-8") as f:
        json.dump(ks_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {ks_path.name}")

    # 打印汇总
    print(f"\n=== KS 检验汇总 ===")
    print(f"{'turb':8s} {'mod':8s} | {'PAPR':>20s} {'amp_mean':>20s} {'radial':>20s}")
    print(f"{'':8s} {'':8s} | {'KS/p/real/sim':>20s} {'KS/p/real/sim':>20s} {'KS/p/real/sim':>20s}")
    print("-" * 90)
    for turb in REAL_TURBS:
        for mod in MOD_NAMES:
            m = ks_metrics.get(turb, {}).get(mod, {})
            if "error" in m:
                print(f"{turb:8s} {mod:8s} | {m['error']}")
                continue
            papr = m["papr"]
            amp = m["amp_mean"]
            rad = m["radial_mean"]
            print(f"{turb:8s} {mod:8s} | "
                  f"KS={papr['ks_stat']:.3f} p={papr['p_value']:.3f} "
                  f"r={papr['real_mean']:.1f} s={papr['sim_mean']:.1f} | "
                  f"KS={amp['ks_stat']:.3f} p={amp['p_value']:.3f} "
                  f"r={amp['real_mean']:.3f} s={amp['sim_mean']:.3f} | "
                  f"KS={rad['ks_stat']:.3f} p={rad['p_value']:.3f} "
                  f"r={rad['real_mean']:.3f} s={rad['sim_mean']:.3f}")

    # 写 summary.txt
    summary_path = DIST_CHECK_ROOT / "summary.txt"
    lines = [
        f"Stage II S1 分布检查 — v1",
        f"生成时间: {now_str()}",
        f"",
        f"=== S1 门判断标准 ===",
        f"  PAPR KS p-value > 0.05  → PAPR 分布匹配",
        f"  amp_mean KS p-value > 0.05 → amp_mean 分布匹配",
        f"  radial KS p-value > 0.01 → 径向直方图分布匹配 (至少 4/6 mod 通过)",
        f"",
        f"=== KS 检验结果 ===",
    ]

    pass_count = {"papr": 0, "amp": 0, "radial": 0}
    total = 0
    for turb in REAL_TURBS:
        for mod in MOD_NAMES:
            m = ks_metrics.get(turb, {}).get(mod, {})
            if "error" in m:
                continue
            total += 1
            if m["papr"]["p_value"] > 0.05:
                pass_count["papr"] += 1
            if m["amp_mean"]["p_value"] > 0.05:
                pass_count["amp"] += 1
            if m["radial_mean"]["p_value"] > 0.01:
                pass_count["radial"] += 1

            lines.append(
                f"  {turb:8s}/{mod:8s}: "
                f"PAPR p={m['papr']['p_value']:.4f}({'PASS' if m['papr']['p_value']>0.05 else 'FAIL'}), "
                f"amp p={m['amp_mean']['p_value']:.4f}({'PASS' if m['amp_mean']['p_value']>0.05 else 'FAIL'}), "
                f"radial p={m['radial_mean']['p_value']:.4f}({'PASS' if m['radial_mean']['p_value']>0.01 else 'FAIL'})"
            )

    lines.append("")
    lines.append(f"=== 通过率 ===")
    lines.append(f"  PAPR:     {pass_count['papr']}/{total}")
    lines.append(f"  amp_mean: {pass_count['amp']}/{total}")
    lines.append(f"  radial:   {pass_count['radial']}/{total}")
    lines.append("")
    lines.append(f"=== 关键差异 ===")
    lines.append(f"  详见上方 PAPR/amp_mean 的 real_mean vs sim_mean")
    lines.append(f"  若 sim PAPR 远低于 real PAPR → 仿真噪声/扰动不足")
    lines.append(f"  若 sim amp_mean 远高于 real amp_mean → 仿真噪声不足或有 outlier 效应")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] {summary_path.name}")

    print(f"\n[完成] 5 张图 + KS 检验已输出到 {DIST_CHECK_ROOT}")
    print(f"  请审阅图片和 summary.txt，决定是否迭代扰动参数或进入 Step 6-9。")


if __name__ == "__main__":
    main()
