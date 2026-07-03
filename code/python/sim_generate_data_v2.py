# -*- coding: utf-8 -*-
"""
sim_generate_data_v2.py — Step 3-4: Sim-v2 仿真数据生成

使用 01_29.txt 提供的 v2 impairment model：
  - mod-order-dependent SNR
  - Bernoulli-Gaussian impulsive noise
  - burst outliers
  - lognormal amplitude scintillation
  - residual CPE / phase jitter
  - bad subcarrier spike

输出 6000 .mat 文件到 results/stage2/sim_data_v2/
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

# 确保能 import stage2_common 和 train_mrf_lowmem
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    STAGE2_ROOT, MOD_NAMES, MOD_TO_LABEL, SIM_TURB_NAMES,
    TURB_TO_LABEL, N_SC, N_SYM,
    ensure_dir, now_str, set_seed,
)
from train_mrf_lowmem import make_cdm_from_rx_sc, make_blind_stats, make_radial_hist

# ─────────────────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────────────────
SIM_DATA_V2_ROOT = STAGE2_ROOT / "sim_data_v2"
SIM_MANIFEST_V2_CSV = SIM_DATA_V2_ROOT / "manifest_sim_v2.csv"
DISTURBANCE_CONFIG_V2 = SIM_DATA_V2_ROOT / "disturbance_config_v2.json"


# ─────────────────────────────────────────────────────────
# QAM 星座生成（与 v1 相同）
# ─────────────────────────────────────────────────────────
MOD_ORDER = {
    "QPSK": 4, "16QAM": 16, "32QAM": 32,
    "64QAM": 64, "128QAM": 128, "256QAM": 256,
}


def gen_qam_constellation(M: int) -> np.ndarray:
    if M == 4:
        pts = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=complex)
    elif M == 16:
        levels = np.array([-3, -1, 1, 3])
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
    elif M == 32:
        levels = np.array([-5, -3, -1, 1, 3, 5])
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
        mask = ~((np.abs(I.ravel()) == 5) & (np.abs(Q.ravel()) == 5))
        pts = pts[mask]
    elif M == 64:
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
    elif M == 128:
        levels = np.arange(-11, 12, 2)
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
        mask = ~((np.abs(I.ravel()) >= 9) & (np.abs(Q.ravel()) >= 9))
        pts = pts[mask]
    elif M == 256:
        levels = np.arange(-15, 16, 2)
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
    else:
        raise ValueError(f"不支持的 M={M}")
    power = np.mean(np.abs(pts) ** 2)
    pts = pts / np.sqrt(power)
    return pts.astype(np.complex64)


def gen_tx_sc(mod_name: str, n_sc: int, n_sym: int, rng: np.random.Generator) -> np.ndarray:
    M = MOD_ORDER[mod_name]
    constellation = gen_qam_constellation(M)
    indices = rng.integers(0, M, size=n_sc * n_sym)
    symbols = constellation[indices]
    return symbols.reshape(n_sc, n_sym).astype(np.complex64)


# ─────────────────────────────────────────────────────────
# v2 impairment model（来自 01_29.txt + 扩展）
# ─────────────────────────────────────────────────────────
def rms_normalize(z, eps=1e-12):
    z = np.asarray(z).astype(np.complex64).reshape(-1)
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + eps)
    return z / rms


def add_awgn_by_snr(z, snr_db, rng):
    sig_power = np.mean(np.abs(z) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)
    )
    return z + noise.astype(np.complex64)


def add_impulsive_noise(z, mod, turbulence, rng):
    """Bernoulli-Gaussian impulse noise. 用来匹配真实数据中的高 PAPR 尾部。"""
    order = MOD_ORDER[mod]
    base_p = {4: 0.001, 16: 0.002, 32: 0.004, 64: 0.007, 128: 0.011, 256: 0.016}[order]
    if turbulence == "strong":
        base_p *= 1.5
    scale = {4: 4.0, 16: 5.5, 32: 7.0, 64: 9.0, 128: 12.0, 256: 16.0}[order]
    if turbulence == "strong":
        scale *= 1.2

    z = z.copy()
    n = len(z)
    mask = rng.random(n) < base_p
    if np.any(mask):
        rms = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
        impulse = scale * rms * (
            rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())
        ) / np.sqrt(2)
        z[mask] += impulse.astype(np.complex64)
    return z


def add_burst_outliers(z, mod, turbulence, rng):
    """模拟连续坏点/坏符号，而不是单点 impulse。"""
    order = MOD_ORDER[mod]
    z = z.copy()
    n = len(z)
    lam = {4: 0.2, 16: 0.4, 32: 0.6, 64: 0.9, 128: 1.2, 256: 1.6}[order]
    if turbulence == "strong":
        lam *= 1.5

    n_bursts = rng.poisson(lam)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    for _ in range(n_bursts):
        length = int(rng.integers(2, 8))
        start = int(rng.integers(0, max(1, n - length)))
        amp = rng.uniform(4.0, 12.0) * rms
        if order >= 128:
            amp *= 1.5
        if turbulence == "strong":
            amp *= 1.2
        burst = amp * (
            rng.standard_normal(length) + 1j * rng.standard_normal(length)
        ) / np.sqrt(2)
        z[start:start + length] += burst.astype(np.complex64)
    return z


def add_amplitude_scintillation(z, mod, turbulence, rng):
    """湍流导致的慢变幅度闪烁（lognormal multiplicative fading）。"""
    order = MOD_ORDER[mod]
    sigma = 0.08 if turbulence == "weak" else 0.18
    if order >= 128:
        sigma *= 1.25
    fade = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=z.shape)
    return z * fade.astype(np.float32)


def add_phase_errors(z, mod, turbulence, rng):
    """残余 CPE + 小幅相位抖动。"""
    order = MOD_ORDER[mod]
    cpe_std = 0.03 if turbulence == "weak" else 0.08
    jitter_std = 0.01 if turbulence == "weak" else 0.025
    if order >= 128:
        cpe_std *= 1.25
        jitter_std *= 1.25
    cpe = rng.normal(0, cpe_std)
    jitter = rng.normal(0, jitter_std, size=z.shape)
    phase = cpe + jitter
    return z * np.exp(1j * phase).astype(np.complex64)


def add_bad_subcarrier_spike(z_2d, mod, turbulence, rng):
    """
    Bad subcarrier / bad symbol spike。
    z_2d: (n_sc, n_sym) — 在 2D 上操作，模拟个别子载波或符号异常。

    返回 (z_2d, log_dict)
    """
    order = MOD_ORDER[mod]
    z_2d = z_2d.copy()
    n_sc, n_sym = z_2d.shape
    rms = np.sqrt(np.mean(np.abs(z_2d) ** 2) + 1e-12)

    log = {"bad_sc_count": 0, "bad_sym_count": 0}

    # Bad subcarrier: 整个子载波幅度异常
    p_bad_sc = {4: 0.0, 16: 0.005, 32: 0.008, 64: 0.012, 128: 0.018, 256: 0.025}[order]
    if turbulence == "strong":
        p_bad_sc *= 1.3
    n_bad_sc = int(rng.binomial(n_sc, p_bad_sc))
    if n_bad_sc > 0:
        bad_indices = rng.choice(n_sc, n_bad_sc, replace=False)
        for idx in bad_indices:
            amp = rng.uniform(2.0, 6.0) * rms
            z_2d[idx, :] += amp * (
                rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
            ).astype(np.complex64) / np.sqrt(2)
    log["bad_sc_count"] = n_bad_sc

    # Bad symbol: 整个 OFDM 符号幅度异常
    p_bad_sym = {4: 0.0, 16: 0.003, 32: 0.005, 64: 0.008, 128: 0.012, 256: 0.016}[order]
    if turbulence == "strong":
        p_bad_sym *= 1.3
    n_bad_sym = int(rng.binomial(n_sym, p_bad_sym))
    if n_bad_sym > 0:
        bad_indices = rng.choice(n_sym, n_bad_sym, replace=False)
        for idx in bad_indices:
            amp = rng.uniform(2.0, 5.0) * rms
            z_2d[:, idx] += amp * (
                rng.standard_normal(n_sc) + 1j * rng.standard_normal(n_sc)
            ).astype(np.complex64) / np.sqrt(2)
    log["bad_sym_count"] = n_bad_sym

    return z_2d, log


def apply_v2_impairments(
    tx_sc: np.ndarray,
    mod: str,
    turbulence: str,
    rng: np.random.Generator,
    snr_db: float = None,
) -> tuple:
    """
    v2 真实链路扰动模型。

    流程:
      1. RMS 归一化
      2. AWGN (mod-order-dependent SNR)
      3. 幅度闪烁 (lognormal)
      4. 残余相位误差 (CPE + jitter)
      5. Bad subcarrier / bad symbol spike (2D)
      6. 单点 impulse (Bernoulli-Gaussian)
      7. Burst outlier

    注意: 不做 normalize_after，保留 outlier 对 PAPR 的影响。
    """
    z_2d = tx_sc.copy().astype(np.complex64)
    z_flat = z_2d.reshape(-1)

    # 1. 归一化
    z_flat = rms_normalize(z_flat)

    # 2. AWGN
    order = MOD_ORDER[mod]
    if snr_db is None:
        if turbulence == "weak":
            snr_range = {
                4: (15, 20), 16: (12, 17), 32: (9, 14),
                64: (6, 12), 128: (3, 9), 256: (0, 6),
            }[order]
        else:
            snr_range = {
                4: (10, 15), 16: (8, 13), 32: (6, 11),
                64: (3, 9), 128: (0, 6), 256: (-2, 4),
            }[order]
        snr_db = rng.uniform(*snr_range)

    z_flat = add_awgn_by_snr(z_flat, snr_db, rng)

    # 3. 幅度闪烁
    z_flat = add_amplitude_scintillation(z_flat, mod, turbulence, rng)

    # 4. 相位误差
    z_flat = add_phase_errors(z_flat, mod, turbulence, rng)

    # reshape to 2D for bad subcarrier/symbol
    z_2d = z_flat.reshape(N_SC, N_SYM)

    # 5. Bad subcarrier / bad symbol spike
    z_2d, bad_log = add_bad_subcarrier_spike(z_2d, mod, turbulence, rng)

    # back to 1D for impulse and burst
    z_flat = z_2d.reshape(-1)

    # 6. 单点 impulse
    z_flat = add_impulsive_noise(z_flat, mod, turbulence, rng)

    # 7. Burst outlier
    z_flat = add_burst_outliers(z_flat, mod, turbulence, rng)

    # reshape back to 2D
    rx_sc = z_flat.reshape(N_SC, N_SYM).astype(np.complex64)

    log = {
        "snr_db": float(snr_db),
        "bad_sc_count": bad_log["bad_sc_count"],
        "bad_sym_count": bad_log["bad_sym_count"],
    }
    return rx_sc, log


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== sim_generate_data_v2.py  [{now_str()}] ===")
    ensure_dir(SIM_DATA_V2_ROOT)

    cfg = {
        "version": "v2",
        "description": "Phase-screen sim v2 with realistic impairments",
        "impairments": [
            "mod-order-dependent SNR",
            "Bernoulli-Gaussian impulsive noise",
            "burst outliers",
            "lognormal amplitude scintillation",
            "residual CPE / phase jitter",
            "bad subcarrier / bad symbol spike",
        ],
        "snr_ranges_weak": {
            "QPSK": [15, 20], "16QAM": [12, 17], "32QAM": [9, 14],
            "64QAM": [6, 12], "128QAM": [3, 9], "256QAM": [0, 6],
        },
        "snr_ranges_strong": {
            "QPSK": [10, 15], "16QAM": [8, 13], "32QAM": [6, 11],
            "64QAM": [3, 9], "128QAM": [0, 6], "256QAM": [-2, 4],
        },
        "normalize_after": False,
    }

    with open(DISTURBANCE_CONFIG_V2, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  配置: {DISTURBANCE_CONFIG_V2}")

    # 生成数据
    print(f"\n生成 Sim-v2 数据（6 mods × 2 turbs × 500 frames = 6000）")
    set_seed(2026)
    manifest_rows = []
    t0 = time.time()

    for turb_name in SIM_TURB_NAMES:
        # v2 用 weak/strong 而非 very_weak/strong（与 01_29.txt 一致）
        turb_label = TURB_TO_LABEL[turb_name]
        # 映射到 01_29.txt 的 turbulence 参数
        turb_param = "weak" if turb_name == "very_weak" else "strong"

        for mod_name in MOD_NAMES:
            mod_label = MOD_TO_LABEL[mod_name]
            out_dir = SIM_DATA_V2_ROOT / mod_name / turb_name
            ensure_dir(out_dir)

            for frame_idx in range(500):
                rng = np.random.default_rng(
                    2026000 + hash((turb_name, mod_name, frame_idx)) % 2**31
                )

                # 生成 tx_sc
                tx_sc = gen_tx_sc(mod_name, N_SC, N_SYM, rng)

                # v2 扰动
                rx_sc, imp_log = apply_v2_impairments(
                    tx_sc, mod_name, turb_param, rng
                )

                # 特征提取（与 lowmem bit-identical）
                cdm = make_cdm_from_rx_sc(rx_sc, nbin=64, clip_val=3.0)
                blind_stats = make_blind_stats(rx_sc)
                radial = make_radial_hist(rx_sc, n_bins=32, clip_val=3.0)

                # 保存
                save_name = f"frame_{frame_idx+1:04d}.mat"
                save_path = out_dir / save_name
                sio.savemat(str(save_path), {
                    "rx_sc": rx_sc.astype(np.complex64),
                    "cdm64": cdm.astype(np.float32),
                    "blind_stats": blind_stats.reshape(1, 16).astype(np.float32),
                    "radial_hist": radial.astype(np.float32),
                    "mod_name": mod_name,
                    "mod_label": np.int32(mod_label),
                    "turb_name": turb_name,
                    "turb_label": np.int32(turb_label),
                    "snr_frame_db": np.float32(imp_log["snr_db"]),
                    "source": "sim_v2",
                    "sim_frame_idx": np.int32(frame_idx),
                    "bad_sc_count": np.int32(imp_log["bad_sc_count"]),
                    "bad_sym_count": np.int32(imp_log["bad_sym_count"]),
                })

                manifest_rows.append({
                    "out_mat": str(save_path),
                    "mod_name": mod_name,
                    "mod_label": mod_label,
                    "turb_name": turb_name,
                    "turb_label": turb_label,
                    "snr_frame_db": imp_log["snr_db"],
                    "source": "sim_v2",
                    "sim_frame_idx": frame_idx,
                    "bad_sc_count": imp_log["bad_sc_count"],
                    "bad_sym_count": imp_log["bad_sym_count"],
                })

            elapsed = time.time() - t0
            print(f"  [{turb_name}/{mod_name}] 500 帧 | "
                  f"总计 {len(manifest_rows)}/6000 | {elapsed:.1f}s")

    # 保存 manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(SIM_MANIFEST_V2_CSV, index=False)
    print(f"\n  manifest: {SIM_MANIFEST_V2_CSV} ({len(manifest_df)} rows)")

    # SNR 统计
    print(f"\n  SNR 统计:")
    for turb in SIM_TURB_NAMES:
        sub = manifest_df[manifest_df["turb_name"] == turb]
        print(f"    {turb}: mean={sub['snr_frame_db'].mean():.2f} dB, "
              f"std={sub['snr_frame_db'].std():.2f}, "
              f"min={sub['snr_frame_db'].min():.2f}, max={sub['snr_frame_db'].max():.2f}")

    # 按 mod 的 SNR 统计
    print(f"\n  SNR by mod:")
    for mod in MOD_NAMES:
        sub = manifest_df[manifest_df["mod_name"] == mod]
        print(f"    {mod:8s}: mean={sub['snr_frame_db'].mean():.2f} dB, "
              f"std={sub['snr_frame_db'].std():.2f}")

    # 样本验证
    sample = manifest_df.iloc[0]
    m = sio.loadmat(sample["out_mat"])
    print(f"\n  样本验证: {Path(sample['out_mat']).name}")
    print(f"    rx_sc: {m['rx_sc'].shape} {m['rx_sc'].dtype}")
    print(f"    cdm64: {m['cdm64'].shape}, max={m['cdm64'].max():.4f}")

    elapsed = time.time() - t0
    print(f"\n[完成] Sim-v2 6000 帧已生成，用时 {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
