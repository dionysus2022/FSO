# -*- coding: utf-8 -*-
"""
sim_compare_distributions_v2.py — Step 5-6: S1-v2 分布检查

使用 01_29.txt 的 robust PAPR 指标和更新后的判断标准。

S1-v2 判断标准（来自 01_29.txt）:
  1. radial: 至少 10/12 PASS
  2. amp_mean: 至少 6/12 PASS 或趋势一致
  3. PAPR_max: 不要求完全 PASS，但均值趋势必须随 QAM 阶数上升
  4. PAPR_p999: 至少 6/12 接近真实
  5. outlier_rate: 真实与仿真同量级
  6. 128QAM/256QAM 的 PAPR 差距不能反向

输出: results/stage2/distribution_check_v2/
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    FROZEN_SPLIT_CSV, STAGE2_ROOT,
    MOD_NAMES, N_SC, N_SYM,
    ensure_dir, now_str,
)
from train_mrf_lowmem import make_cdm_from_rx_sc, make_radial_hist, make_blind_stats

# ─────────────────────────────────────────────────────────
# 路径与常量
# ─────────────────────────────────────────────────────────
SIM_MANIFEST_V2_CSV = STAGE2_ROOT / "sim_data_v2" / "manifest_sim_v2.csv"
DIST_CHECK_V2_ROOT = STAGE2_ROOT / "distribution_check_v2"

SIM_TO_REAL_TURB = {"very_weak": "weak", "strong": "strong"}
REAL_TURBS = ["weak", "strong"]
HIGH_ORDER_MODS = ["64QAM", "128QAM", "256QAM"]
N_SAMPLES = 80


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────
def rms_normalize(z, eps=1e-12):
    z = np.asarray(z).astype(np.complex64).reshape(-1)
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + eps)
    return z / rms


def robust_papr_metrics(z, eps=1e-12):
    z = np.asarray(z).reshape(-1)
    p = np.abs(z) ** 2
    mean_p = np.mean(p) + eps
    max_papr = 10 * np.log10(np.max(p) / mean_p + eps)
    p999 = np.percentile(p, 99.9)
    p99 = np.percentile(p, 99.0)
    papr_999 = 10 * np.log10(p999 / mean_p + eps)
    papr_99 = 10 * np.log10(p99 / mean_p + eps)
    rms = np.sqrt(mean_p)
    amp = np.abs(z)
    return {
        "papr_max_db": float(max_papr),
        "papr_p999_db": float(papr_999),
        "papr_p99_db": float(papr_99),
        "outlier_rate_5rms": float(np.mean(amp > 5 * rms)),
        "outlier_rate_8rms": float(np.mean(amp > 8 * rms)),
        "outlier_rate_10rms": float(np.mean(amp > 10 * rms)),
        "top1_energy_ratio": float(np.max(p) / (np.sum(p) + eps)),
        "amp_mean": float(np.mean(amp)),
        "amp_std": float(np.std(amp)),
        "amp_max": float(np.max(amp)),
    }


def load_rx_sc(path: str) -> np.ndarray:
    m = sio.loadmat(path)
    return np.asarray(m["rx_sc"]).astype(np.complex64)


def load_samples(manifest_path: Path, turb_filter: str, mod_filter: str,
                 n: int, turb_col: str = "turb_name") -> List[np.ndarray]:
    df = pd.read_csv(manifest_path)
    sub = df[(df[turb_col] == turb_filter) & (df["mod_name"] == mod_filter)]
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    samples = []
    for _, row in sub.iterrows():
        try:
            samples.append(load_rx_sc(row["out_mat"]))
        except Exception:
            pass
    return samples


# ─────────────────────────────────────────────────────────
# 图 1: PAPR_max / PAPR_p999 by modulation (bar chart)
# ─────────────────────────────────────────────────────────
def plot_papr_by_mod(real_stats: dict, sim_stats: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Fig 1: PAPR_max & PAPR_p999 by Modulation (Real vs Sim-v2)", fontsize=14)

    metrics = [
        ("papr_max_db", "PAPR_max (dB)", axes[0, :]),
        ("papr_p999_db", "PAPR_p999 (dB)", axes[1, :]),
    ]

    x = np.arange(len(MOD_NAMES))
    width = 0.35

    for metric_key, metric_label, (ax_weak, ax_strong) in metrics:
        for j, (turb, ax) in enumerate(zip(REAL_TURBS, [ax_weak, ax_strong])):
            real_vals = [real_stats[(turb, mod)][metric_key] for mod in MOD_NAMES]
            sim_vals = [sim_stats[(turb, mod)][metric_key] for mod in MOD_NAMES]

            ax.bar(x - width/2, real_vals, width, label="real", color="steelblue", alpha=0.8)
            ax.bar(x + width/2, sim_vals, width, label="sim_v2", color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(MOD_NAMES, rotation=30, ha="right")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — {turb}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # 标注数值
            for i, (rv, sv) in enumerate(zip(real_vals, sim_vals)):
                ax.text(i - width/2, rv + 0.3, f"{rv:.1f}", ha="center", fontsize=7)
                ax.text(i + width/2, sv + 0.3, f"{sv:.1f}", ha="center", fontsize=7)

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig1_papr_by_mod.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 2: outlier_rate by modulation
# ─────────────────────────────────────────────────────────
def plot_outlier_rate(real_stats: dict, sim_stats: dict):
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle("Fig 2: Outlier Rate by Modulation (Real vs Sim-v2)", fontsize=14)

    metrics = [
        ("outlier_rate_5rms", "Outlier 5×RMS"),
        ("outlier_rate_8rms", "Outlier 8×RMS"),
        ("outlier_rate_10rms", "Outlier 10×RMS"),
    ]

    x = np.arange(len(MOD_NAMES))
    width = 0.35

    for k, (metric_key, metric_label) in enumerate(metrics):
        for j, turb in enumerate(REAL_TURBS):
            ax = axes[j, k]
            real_vals = [real_stats[(turb, mod)][metric_key] for mod in MOD_NAMES]
            sim_vals = [sim_stats[(turb, mod)][metric_key] for mod in MOD_NAMES]

            ax.bar(x - width/2, real_vals, width, label="real", color="steelblue", alpha=0.8)
            ax.bar(x + width/2, sim_vals, width, label="sim_v2", color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(MOD_NAMES, rotation=30, ha="right")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — {turb}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_yscale("log")

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig2_outlier_rate.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 3: amp_mean / amp_std 分布
# ─────────────────────────────────────────────────────────
def plot_amp_distribution(all_real: dict, all_sim: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fig 3: amp_mean & amp_std Distribution (Real=blue, Sim-v2=red)", fontsize=14)

    for j, turb in enumerate(REAL_TURBS):
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        real_amps, sim_amps, real_stds, sim_stds = [], [], [], []

        for mod in MOD_NAMES:
            for z in all_real.get((turb, mod), []):
                zn = rms_normalize(z)
                amp = np.abs(zn)
                real_amps.append(float(np.mean(amp)))
                real_stds.append(float(np.std(amp)))
            for z in all_sim.get((sim_turb, mod), []):
                zn = rms_normalize(z)
                amp = np.abs(zn)
                sim_amps.append(float(np.mean(amp)))
                sim_stds.append(float(np.std(amp)))

        ax_mean = axes[0, j]
        ax_std = axes[1, j]

        ax_mean.hist(real_amps, bins=30, alpha=0.5, color="blue", label="real", density=True)
        ax_mean.hist(sim_amps, bins=30, alpha=0.5, color="red", label="sim_v2", density=True)
        ax_mean.set_title(f"amp_mean — {turb}")
        ax_mean.set_xlabel("amp_mean")
        ax_mean.legend()
        ax_mean.grid(True, alpha=0.3)

        ax_std.hist(real_stds, bins=30, alpha=0.5, color="blue", label="real", density=True)
        ax_std.hist(sim_stds, bins=30, alpha=0.5, color="red", label="sim_v2", density=True)
        ax_std.set_title(f"amp_std — {turb}")
        ax_std.set_xlabel("amp_std")
        ax_std.legend()
        ax_std.grid(True, alpha=0.3)

        if len(real_amps) > 5 and len(sim_amps) > 5:
            ks_m, p_m = scipy_stats.ks_2samp(real_amps, sim_amps)
            ks_s, p_s = scipy_stats.ks_2samp(real_stds, sim_stds)
            ax_mean.text(0.05, 0.95, f"KS={ks_m:.3f}\np={p_m:.4f}",
                         transform=ax_mean.transAxes, va="top", fontsize=9,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax_std.text(0.05, 0.95, f"KS={ks_s:.3f}\np={p_s:.4f}",
                        transform=ax_std.transAxes, va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig3_amp_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 4: radial histogram
# ─────────────────────────────────────────────────────────
def plot_radial_histogram(all_real: dict, all_sim: dict):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Fig 4: Radial Histogram (Real=solid, Sim-v2=dashed)", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for j, turb in enumerate(REAL_TURBS):
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        ax = axes[j]

        for i, mod in enumerate(MOD_NAMES):
            real_samples = all_real.get((turb, mod), [])
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
        ax.set_title(f"Turbulence: {turb}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig4_radial_histogram.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 5: 星座图叠加
# ─────────────────────────────────────────────────────────
def plot_constellation_overlay(all_real: dict, all_sim: dict):
    fig, axes = plt.subplots(6, 2, figsize=(14, 36))
    fig.suptitle("Fig 5: Constellation Overlay (Real=blue, Sim-v2=red)", fontsize=16, y=1.01)

    for i, mod in enumerate(MOD_NAMES):
        for j, turb in enumerate(REAL_TURBS):
            sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
            ax = axes[i, j]

            real_samples = all_real.get((turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                z_real = np.concatenate([rms_normalize(s) for s in real_samples[:5]])
                ax.scatter(z_real.real, z_real.imag, s=1, c="blue", alpha=0.3, label="real")
            if sim_samples:
                z_sim = np.concatenate([rms_normalize(s) for s in sim_samples[:5]])
                ax.scatter(z_sim.real, z_sim.imag, s=1, c="red", alpha=0.3, label="sim_v2")

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.set_title(f"{mod} / {turb}", fontsize=10)
            ax.legend(fontsize=8, markerscale=5)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig5_constellation_overlay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# 图 6: CDM 对比
# ─────────────────────────────────────────────────────────
def plot_cdm_comparison(all_real: dict, all_sim: dict):
    fig, axes = plt.subplots(6, 4, figsize=(22, 36))
    fig.suptitle("Fig 6: CDM Comparison (left=Real avg, right=Sim-v2 avg)", fontsize=14)

    for i, mod in enumerate(MOD_NAMES):
        for j, turb in enumerate(REAL_TURBS):
            sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
            ax_real = axes[i, j * 2]
            ax_sim = axes[i, j * 2 + 1]

            real_samples = all_real.get((turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if real_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in real_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_real.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_real.set_title(f"Real {mod}/{turb}", fontsize=8)

            if sim_samples:
                cdms = [make_cdm_from_rx_sc(s) for s in sim_samples[:20]]
                avg_cdm = np.mean(cdms, axis=0)
                ax_sim.imshow(avg_cdm, cmap="hot", origin="lower", extent=[-3, 3, -3, 3])
                ax_sim.set_title(f"Sim-v2 {mod}/{turb}", fontsize=8)

    plt.tight_layout()
    path = DIST_CHECK_V2_ROOT / "fig6_cdm_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────
# KS 检验
# ─────────────────────────────────────────────────────────
def compute_ks_metrics(all_real: dict, all_sim: dict) -> dict:
    metrics = {}
    for turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        metrics[turb] = {}

        for mod in MOD_NAMES:
            real_samples = all_real.get((turb, mod), [])
            sim_samples = all_sim.get((sim_turb, mod), [])

            if len(real_samples) < 5 or len(sim_samples) < 5:
                metrics[turb][mod] = {"error": "insufficient"}
                continue

            real_feats = {k: [] for k in ["papr_max", "papr_p999", "amp_mean", "amp_std",
                                           "outlier_5rms", "radial_mean"]}
            sim_feats = {k: [] for k in real_feats}

            for z in real_samples:
                zn = rms_normalize(z)
                m = robust_papr_metrics(zn)
                real_feats["papr_max"].append(m["papr_max_db"])
                real_feats["papr_p999"].append(m["papr_p999_db"])
                real_feats["amp_mean"].append(m["amp_mean"])
                real_feats["amp_std"].append(m["amp_std"])
                real_feats["outlier_5rms"].append(m["outlier_rate_5rms"])
                real_feats["radial_mean"].append(float(np.mean(make_radial_hist(z))))

            for z in sim_samples:
                zn = rms_normalize(z)
                m = robust_papr_metrics(zn)
                sim_feats["papr_max"].append(m["papr_max_db"])
                sim_feats["papr_p999"].append(m["papr_p999_db"])
                sim_feats["amp_mean"].append(m["amp_mean"])
                sim_feats["amp_std"].append(m["amp_std"])
                sim_feats["outlier_5rms"].append(m["outlier_rate_5rms"])
                sim_feats["radial_mean"].append(float(np.mean(make_radial_hist(z))))

            entry = {}
            for key in real_feats:
                ks, p = scipy_stats.ks_2samp(real_feats[key], sim_feats[key])
                entry[key] = {
                    "ks_stat": float(ks),
                    "p_value": float(p),
                    "real_mean": float(np.mean(real_feats[key])),
                    "sim_mean": float(np.mean(sim_feats[key])),
                }
            metrics[turb][mod] = entry
    return metrics


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== sim_compare_distributions_v2.py  [{now_str()}] ===")
    ensure_dir(DIST_CHECK_V2_ROOT)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据（每 mod×turb 采样 {N_SAMPLES} 帧）")
    all_real = {}
    all_sim = {}

    for turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        for mod in MOD_NAMES:
            real_s = load_samples(FROZEN_SPLIT_CSV, turb, mod, N_SAMPLES)
            all_real[(turb, mod)] = real_s
            sim_s = load_samples(SIM_MANIFEST_V2_CSV, sim_turb, mod, N_SAMPLES)
            all_sim[(sim_turb, mod)] = sim_s
            print(f"  {turb:6s}/{mod:8s}: real={len(real_s)} sim={len(sim_s)}")

    # 2. 计算 per-group robust PAPR 统计
    print(f"\n[2/4] 计算 robust PAPR 统计")
    real_stats = {}
    sim_stats = {}
    for turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        for mod in MOD_NAMES:
            real_m = [robust_papr_metrics(rms_normalize(z))
                      for z in all_real.get((turb, mod), [])]
            sim_m = [robust_papr_metrics(rms_normalize(z))
                     for z in all_sim.get((sim_turb, mod), [])]
            real_stats[(turb, mod)] = {k: float(np.mean([m[k] for m in real_m]))
                                        for k in real_m[0]} if real_m else {}
            sim_stats[(turb, mod)] = {k: float(np.mean([m[k] for m in sim_m]))
                                       for k in sim_m[0]} if sim_m else {}

    # 3. 生成图
    print(f"\n[3/4] 生成对比图")
    plot_papr_by_mod(real_stats, sim_stats)
    plot_outlier_rate(real_stats, sim_stats)
    plot_amp_distribution(all_real, all_sim)
    plot_radial_histogram(all_real, all_sim)
    plot_constellation_overlay(all_real, all_sim)
    plot_cdm_comparison(all_real, all_sim)

    # 4. KS 检验 + 判断
    print(f"\n[4/4] KS 检验")
    ks_metrics = compute_ks_metrics(all_real, all_sim)

    ks_path = DIST_CHECK_V2_ROOT / "ks_metrics.json"
    with open(ks_path, "w", encoding="utf-8") as f:
        json.dump(ks_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {ks_path.name}")

    # 打印汇总
    print(f"\n{'='*120}")
    print(f"{'turb':8s} {'mod':8s} | {'PAPR_max':>24s} | {'PAPR_p999':>24s} | {'amp_mean':>24s} | {'radial':>20s}")
    print(f"{'':8s} {'':8s} | {'real/sim/p':>24s} | {'real/sim/p':>24s} | {'real/sim/p':>24s} | {'real/sim/p':>20s}")
    print("-" * 120)

    pass_count = {"papr_max": 0, "papr_p999": 0, "amp_mean": 0, "radial": 0,
                  "outlier": 0, "amp_std": 0}
    total = 0

    for turb in REAL_TURBS:
        for mod in MOD_NAMES:
            m = ks_metrics.get(turb, {}).get(mod, {})
            if "error" in m:
                print(f"{turb:8s} {mod:8s} | {m['error']}")
                continue
            total += 1
            pm = m["papr_max"]
            pp = m["papr_p999"]
            am = m["amp_mean"]
            rd = m["radial_mean"]
            ol = m["outlier_5rms"]

            if pm["p_value"] > 0.05: pass_count["papr_max"] += 1
            if pp["p_value"] > 0.05: pass_count["papr_p999"] += 1
            if am["p_value"] > 0.05: pass_count["amp_mean"] += 1
            if rd["p_value"] > 0.01: pass_count["radial"] += 1
            if ol["p_value"] > 0.01: pass_count["outlier"] += 1

            print(f"{turb:8s} {mod:8s} | "
                  f"{pm['real_mean']:6.2f}/{pm['sim_mean']:6.2f}/p={pm['p_value']:.3f} | "
                  f"{pp['real_mean']:6.2f}/{pp['sim_mean']:6.2f}/p={pp['p_value']:.3f} | "
                  f"{am['real_mean']:6.4f}/{am['sim_mean']:6.4f}/p={am['p_value']:.3f} | "
                  f"{rd['real_mean']:6.3f}/{rd['sim_mean']:6.3f}/p={rd['p_value']:.3f}")

    print(f"{'='*120}")

    # S1-v2 判断（来自 01_29.txt）
    print(f"\n=== S1-v2 判断标准（01_29.txt）===")
    print(f"  1. radial:        {pass_count['radial']}/12  (目标 ≥10)")
    print(f"  2. amp_mean:      {pass_count['amp_mean']}/12 (目标 ≥6 或趋势一致)")
    print(f"  3. PAPR_max:      {pass_count['papr_max']}/12 (不要求全过，趋势须随阶数上升)")
    print(f"  4. PAPR_p999:     {pass_count['papr_p999']}/12 (目标 ≥6)")
    print(f"  5. outlier_rate:  {pass_count['outlier']}/12 (同量级)")

    # PAPR 趋势检查
    print(f"\n  PAPR_max 趋势检查（随 QAM 阶数上升？）:")
    for turb in REAL_TURBS:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == turb][0]
        real_vals = [real_stats[(turb, mod)]["papr_max_db"] for mod in MOD_NAMES]
        sim_vals = [sim_stats[(turb, mod)]["papr_max_db"] for mod in MOD_NAMES]
        real_trend = "↑" if real_vals[-1] > real_vals[0] else "↓"
        sim_trend = "↑" if sim_vals[-1] > sim_vals[0] else "↓"
        print(f"    {turb}: real {real_vals[0]:.1f}→{real_vals[-1]:.1f} ({real_trend})  "
              f"sim {sim_vals[0]:.1f}→{sim_vals[-1]:.1f} ({sim_trend})")

    # 128/256 反向检查
    reverse_check = True
    for turb in REAL_TURBS:
        r128 = real_stats[(turb, "128QAM")]["papr_max_db"]
        r256 = real_stats[(turb, "256QAM")]["papr_max_db"]
        s128 = sim_stats[(turb, "128QAM")]["papr_max_db"]
        s256 = sim_stats[(turb, "256QAM")]["papr_max_db"]
        real_gap = r256 - r128
        sim_gap = s256 - s128
        if (real_gap > 0 and sim_gap < -1) or (real_gap < 0 and sim_gap > 1):
            reverse_check = False
    print(f"  128→256 PAPR 反向检查: {'PASS' if reverse_check else 'FAIL'}")

    # 写 summary
    lines = [
        f"Stage II S1-v2 分布检查 — {now_str()}",
        f"",
        f"=== S1-v2 判断标准（01_29.txt）===",
        f"  1. radial:        {pass_count['radial']}/12  (目标 ≥10)  {'PASS' if pass_count['radial']>=10 else 'FAIL'}",
        f"  2. amp_mean:      {pass_count['amp_mean']}/12 (目标 ≥6)  {'PASS' if pass_count['amp_mean']>=6 else 'FAIL'}",
        f"  3. PAPR_max:      {pass_count['papr_max']}/12 (趋势须随阶数上升)",
        f"  4. PAPR_p999:     {pass_count['papr_p999']}/12 (目标 ≥6)  {'PASS' if pass_count['papr_p999']>=6 else 'FAIL'}",
        f"  5. outlier_rate:  {pass_count['outlier']}/12 (同量级)",
        f"  6. 128→256 反向:  {'PASS' if reverse_check else 'FAIL'}",
        f"",
        f"=== PAPR_max by mod ===",
    ]
    for turb in REAL_TURBS:
        for mod in MOD_NAMES:
            r = real_stats[(turb, mod)]
            s = sim_stats[(turb, mod)]
            lines.append(f"  {turb:8s}/{mod:8s}: "
                         f"PAPR_max real={r['papr_max_db']:.2f} sim={s['papr_max_db']:.2f} | "
                         f"PAPR_p999 real={r['papr_p999_db']:.2f} sim={s['papr_p999_db']:.2f} | "
                         f"amp_mean real={r['amp_mean']:.4f} sim={s['amp_mean']:.4f} | "
                         f"out5rms real={r['outlier_rate_5rms']:.5f} sim={s['outlier_rate_5rms']:.5f}")

    summary_path = DIST_CHECK_V2_ROOT / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] {summary_path.name}")

    print(f"\n[完成] S1-v2 分布检查已输出到 {DIST_CHECK_V2_ROOT}")
    print(f"  请审阅图片和 summary.txt，决定是否进入 S2-S5 训练或继续迭代。")


if __name__ == "__main__":
    main()
