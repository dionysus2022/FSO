# -*- coding: utf-8 -*-
"""
sim_v2_diagnostics.py — Step 1-2: 真实 vs Sim-v1 robust PAPR 诊断

计算 robust PAPR 指标，判断真实数据高 PAPR 是整体噪声还是少数 outlier 导致。

指标（per frame）:
  papr_max_db, papr_p999_db, papr_p99_db,
  outlier_rate_5rms, outlier_rate_8rms, outlier_rate_10rms,
  top1_energy_ratio, amp_mean, amp_std, amp_max

输出:
  results/stage2/diagnostics_v2/
    robust_papr_comparison.csv   — 逐 mod×turb 的 real vs sim-v1 均值对比
    robust_papr_raw_real.csv     — 真实数据 per-frame 明细
    robust_papr_raw_sim_v1.csv   — Sim-v1 per-frame 明细
    diagnosis.txt                — 诊断结论
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

# 确保能 import stage2_common
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    FROZEN_SPLIT_CSV, SIM_MANIFEST_CSV, STAGE2_ROOT,
    MOD_NAMES, N_SC, N_SYM,
    ensure_dir, now_str,
)


# ─────────────────────────────────────────────────────────
# robust PAPR 指标（来自 01_29.txt）
# ─────────────────────────────────────────────────────────
def robust_papr_metrics(z, eps=1e-12):
    """同时输出传统 PAPR 和鲁棒 PAPR，用来判断是否由少数 outlier 导致。"""
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

    out_5rms = np.mean(amp > 5 * rms)
    out_8rms = np.mean(amp > 8 * rms)
    out_10rms = np.mean(amp > 10 * rms)

    top1_ratio = np.max(p) / (np.sum(p) + eps)

    return {
        "papr_max_db": float(max_papr),
        "papr_p999_db": float(papr_999),
        "papr_p99_db": float(papr_99),
        "outlier_rate_5rms": float(out_5rms),
        "outlier_rate_8rms": float(out_8rms),
        "outlier_rate_10rms": float(out_10rms),
        "top1_energy_ratio": float(top1_ratio),
        "amp_mean": float(np.mean(amp)),
        "amp_std": float(np.std(amp)),
        "amp_max": float(np.max(amp)),
    }


def rms_normalize(z, eps=1e-12):
    z = np.asarray(z).astype(np.complex64).reshape(-1)
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + eps)
    return z / rms


# ─────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────
SIM_TO_REAL_TURB = {"very_weak": "weak", "strong": "strong"}


def load_rx_sc(path: str) -> np.ndarray:
    """加载 .mat 中的 rx_sc，返回 1D 归一化复数数组。"""
    m = sio.loadmat(path)
    z = np.asarray(m["rx_sc"]).astype(np.complex64).reshape(-1)
    return rms_normalize(z)


def collect_metrics(
    manifest_df: pd.DataFrame,
    turb_col: str,
    mod_col: str,
    turb_filter: str,
    mod_filter: str,
    n_max: int = 80,
) -> List[dict]:
    """从 manifest 采样并计算 robust PAPR 指标。"""
    sub = manifest_df[
        (manifest_df[turb_col] == turb_filter) &
        (manifest_df[mod_col] == mod_filter)
    ]
    if len(sub) > n_max:
        sub = sub.sample(n_max, random_state=42)

    results = []
    for _, row in sub.iterrows():
        try:
            z = load_rx_sc(row["out_mat"])
            m = robust_papr_metrics(z)
            m["source"] = "real" if "dataset_source" in row else "sim"
            results.append(m)
        except Exception as e:
            print(f"  [跳过] {row['out_mat']}: {e}")
    return results


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== sim_v2_diagnostics.py  [{now_str()}] ===")
    out_dir = STAGE2_ROOT / "diagnostics_v2"
    ensure_dir(out_dir)

    # 加载 manifest
    real_df = pd.read_csv(FROZEN_SPLIT_CSV)
    real_df = real_df[real_df["split"] == "test"]
    sim_df = pd.read_csv(SIM_MANIFEST_CSV)

    print(f"  真实 test: {len(real_df)} 行")
    print(f"  Sim-v1: {len(sim_df)} 行")

    # 1. 逐 mod×turb 计算
    print(f"\n[1/3] 计算 robust PAPR 指标")
    all_real_rows = []
    all_sim_rows = []
    summary_rows = []

    for real_turb in ["weak", "strong"]:
        sim_turb = [k for k, v in SIM_TO_REAL_TURB.items() if v == real_turb][0]

        for mod in MOD_NAMES:
            print(f"  {real_turb}/{mod}...", end=" ")

            real_metrics = collect_metrics(
                real_df, "turb_name", "mod_name", real_turb, mod, n_max=80
            )
            sim_metrics = collect_metrics(
                sim_df, "turb_name", "mod_name", sim_turb, mod, n_max=80
            )

            for m in real_metrics:
                m["turb_name"] = real_turb
                m["mod_name"] = mod
                all_real_rows.append(m)

            for m in sim_metrics:
                m["turb_name"] = real_turb  # 用真实 turb 名便于对比
                m["mod_name"] = mod
                all_sim_rows.append(m)

            # 汇总均值
            def mean_key(metrics, key):
                vals = [m[key] for m in metrics]
                return float(np.mean(vals)) if vals else float("nan")

            row = {"turb_name": real_turb, "mod_name": mod,
                   "n_real": len(real_metrics), "n_sim": len(sim_metrics)}
            for key in ["papr_max_db", "papr_p999_db", "papr_p99_db",
                        "outlier_rate_5rms", "outlier_rate_8rms", "outlier_rate_10rms",
                        "top1_energy_ratio", "amp_mean", "amp_std", "amp_max"]:
                row[f"real_{key}"] = mean_key(real_metrics, key)
                row[f"sim_{key}"] = mean_key(sim_metrics, key)
                row[f"diff_{key}"] = row[f"real_{key}"] - row[f"sim_{key}"]
            summary_rows.append(row)

            print(f"real={len(real_metrics)} sim={len(sim_metrics)}")

    # 2. 保存结果
    print(f"\n[2/3] 保存结果")
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "robust_papr_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  {summary_path.name}")

    raw_real_df = pd.DataFrame(all_real_rows)
    raw_real_path = out_dir / "robust_papr_raw_real.csv"
    raw_real_df.to_csv(raw_real_path, index=False)
    print(f"  {raw_real_path.name}")

    raw_sim_df = pd.DataFrame(all_sim_rows)
    raw_sim_path = out_dir / "robust_papr_raw_sim_v1.csv"
    raw_sim_df.to_csv(raw_sim_path, index=False)
    print(f"  {raw_sim_path.name}")

    # 3. 诊断
    print(f"\n[3/3] 诊断")

    # 打印对比表
    print(f"\n{'='*120}")
    print(f"{'turb':8s} {'mod':8s} | {'PAPR_max':>22s} | {'PAPR_p999':>22s} | {'out_5rms':>18s} | {'amp_mean':>18s}")
    print(f"{'':8s} {'':8s} | {'real':>7s} {'sim':>7s} {'diff':>6s} | {'real':>7s} {'sim':>7s} {'diff':>6s} | {'real':>8s} {'sim':>8s} | {'real':>8s} {'sim':>8s}")
    print("-" * 120)
    for _, r in summary_df.iterrows():
        print(f"{r['turb_name']:8s} {r['mod_name']:8s} | "
              f"{r['real_papr_max_db']:7.2f} {r['sim_papr_max_db']:7.2f} {r['diff_papr_max_db']:+6.2f} | "
              f"{r['real_papr_p999_db']:7.2f} {r['sim_papr_p999_db']:7.2f} {r['diff_papr_p999_db']:+6.2f} | "
              f"{r['real_outlier_rate_5rms']:8.5f} {r['sim_outlier_rate_5rms']:8.5f} | "
              f"{r['real_amp_mean']:8.4f} {r['sim_amp_mean']:8.4f}")
    print(f"{'='*120}")

    # 诊断结论
    lines = []
    lines.append(f"Sim-v2 诊断报告 — {now_str()}")
    lines.append(f"")
    lines.append(f"=== 诊断问题 ===")
    lines.append(f"  真实 PAPR_max 高是由整体噪声导致，还是少数 outlier 导致？")
    lines.append(f"")

    outlier_driven_count = 0
    noise_driven_count = 0

    for _, r in summary_df.iterrows():
        papr_max_gap = r["diff_papr_max_db"]  # real - sim
        papr_p999_gap = r["diff_papr_p999_db"]

        # 判断标准：
        # - papr_max 高但 papr_p999 接近 sim → outlier 驱动
        # - papr_max 和 papr_p999 都高 → 整体噪声/闪烁不足
        is_outlier_driven = (papr_max_gap > 3.0) and (abs(papr_p999_gap) < 2.0)
        is_noise_driven = (papr_max_gap > 3.0) and (papr_p999_gap > 3.0)

        diagnosis = "unknown"
        if is_outlier_driven:
            diagnosis = "OUTLIER_DRIVEN"
            outlier_driven_count += 1
        elif is_noise_driven:
            diagnosis = "NOISE_DRIVEN"
            noise_driven_count += 1
        elif papr_max_gap <= 3.0:
            diagnosis = "MATCH"

        lines.append(
            f"  {r['turb_name']:8s}/{r['mod_name']:8s}: "
            f"PAPR_max diff={papr_max_gap:+.2f} dB, "
            f"PAPR_p999 diff={papr_p999_gap:+.2f} dB, "
            f"outlier_5rms real={r['real_outlier_rate_5rms']:.5f} sim={r['sim_outlier_rate_5rms']:.5f}, "
            f"→ {diagnosis}"
        )

    lines.append(f"")
    lines.append(f"=== 诊断汇总 ===")
    lines.append(f"  OUTLIER_DRIVEN: {outlier_driven_count}/12")
    lines.append(f"  NOISE_DRIVEN:   {noise_driven_count}/12")
    lines.append(f"  MATCH:          {12 - outlier_driven_count - noise_driven_count}/12")
    lines.append(f"")
    lines.append(f"=== v2 改进方向 ===")
    lines.append(f"  1. 调制阶数相关 SNR（高阶 QAM SNR 更低）")
    lines.append(f"  2. Bernoulli-Gaussian 脉冲噪声（匹配 outlier 尾部）")
    lines.append(f"  3. Burst outlier（连续坏点）")
    lines.append(f"  4. Lognormal 幅度闪烁")
    lines.append(f"  5. 残余 CPE / 相位抖动")

    diag_path = out_dir / "diagnosis.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  诊断报告: {diag_path.name}")

    print(f"\n[完成] 诊断完毕。OUTLIER_DRIVEN={outlier_driven_count}, NOISE_DRIVEN={noise_driven_count}")


if __name__ == "__main__":
    main()
