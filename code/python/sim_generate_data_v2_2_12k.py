# -*- coding: utf-8 -*-
"""
sim_generate_data_v2_2_12k.py — Sim-v2.2 (Calibrated) 12000 帧

基于 sim_generate_data_v2_2.py，仅做 2 处修改：
  1. range(500) → range(1000)（每 mod×turb 1000 帧，总计 12000）
  2. 输出目录 → sim_data_v2_2_12k/，manifest → manifest_sim_v2_2_12k.csv

 impairment 参数、SNR ranges、星座生成、特征预计算、保存格式完全不变。
Round-1 命名：Calibrated-Sim-v2.2（真实分布校准的多扰动仿真）。
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    STAGE2_ROOT, MOD_NAMES, MOD_TO_LABEL, SIM_TURB_NAMES,
    TURB_TO_LABEL, N_SC, N_SYM,
    ensure_dir, now_str, set_seed,
)
from train_mrf_lowmem import make_cdm_from_rx_sc, make_blind_stats, make_radial_hist

SIM_DATA_ROOT = STAGE2_ROOT / "sim_data_v2_2_12k"
SIM_MANIFEST_CSV = SIM_DATA_ROOT / "manifest_sim_v2_2_12k.csv"
DISTURBANCE_CONFIG = SIM_DATA_ROOT / "disturbance_config_v2_2_12k.json"

FRAMES_PER_MOD_TURB = 1000

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


def add_impulsive_noise_v22(z, mod, turbulence, rng):
    order = MOD_ORDER[mod]
    base_p = {
        4: 0.0, 16: 0.0005, 32: 0.0012,
        64: 0.0035, 128: 0.0055, 256: 0.0065,
    }[order]
    if turbulence == "strong":
        base_p *= 1.2

    scale = {
        4: 0.0, 16: 2.0, 32: 2.8,
        64: 3.8, 128: 5.0, 256: 6.5,
    }[order]
    if turbulence == "strong":
        scale *= 1.2

    if base_p == 0.0 or scale == 0.0:
        return z

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


def add_burst_outliers_v22(z, mod, turbulence, rng):
    order = MOD_ORDER[mod]
    lam = {4: 0.0, 16: 0.4, 32: 0.6, 64: 0.9, 128: 1.2, 256: 1.6}[order]
    if turbulence == "strong":
        lam *= 1.2

    if lam == 0.0:
        return z

    z = z.copy()
    n = len(z)
    n_bursts = rng.poisson(lam)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    for _ in range(n_bursts):
        length = int(rng.integers(2, 8))
        start = int(rng.integers(0, max(1, n - length)))
        amp = rng.uniform(2.5, 6.0) * rms
        if order >= 128:
            amp *= 1.10
        if turbulence == "strong":
            amp *= 1.05
        burst = amp * (
            rng.standard_normal(length) + 1j * rng.standard_normal(length)
        ) / np.sqrt(2)
        z[start:start + length] += burst.astype(np.complex64)
    return z


def add_amplitude_scintillation(z, mod, turbulence, rng):
    order = MOD_ORDER[mod]
    sigma = 0.08 if turbulence == "weak" else 0.18
    if order >= 128:
        sigma *= 1.25
    fade = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=z.shape)
    return z * fade.astype(np.float32)


def add_phase_errors(z, mod, turbulence, rng):
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
    order = MOD_ORDER[mod]
    z_2d = z_2d.copy()
    n_sc, n_sym = z_2d.shape
    rms = np.sqrt(np.mean(np.abs(z_2d) ** 2) + 1e-12)

    log = {"bad_sc_count": 0, "bad_sym_count": 0}

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


SNR_RANGES_V22 = {
    "weak": {
        4: (15, 20), 16: (12, 17), 32: (9, 14),
        64: (8, 14), 128: (6, 12), 256: (3, 9),
    },
    "strong": {
        4: (10, 15), 16: (8, 13), 32: (6, 11),
        64: (5, 10), 128: (3, 8), 256: (0, 6),
    },
}


def apply_v2_2_impairments(
    tx_sc: np.ndarray,
    mod: str,
    turbulence: str,
    rng: np.random.Generator,
    snr_db: float = None,
) -> tuple:
    z_2d = tx_sc.copy().astype(np.complex64)
    z_flat = z_2d.reshape(-1)

    z_flat = rms_normalize(z_flat)

    order = MOD_ORDER[mod]
    if snr_db is None:
        snr_range = SNR_RANGES_V22[turbulence][order]
        snr_db = rng.uniform(*snr_range)
    z_flat = add_awgn_by_snr(z_flat, snr_db, rng)

    z_flat = add_amplitude_scintillation(z_flat, mod, turbulence, rng)
    z_flat = add_phase_errors(z_flat, mod, turbulence, rng)

    z_2d = z_flat.reshape(N_SC, N_SYM)
    z_2d, bad_log = add_bad_subcarrier_spike(z_2d, mod, turbulence, rng)
    z_flat = z_2d.reshape(-1)

    z_flat = add_impulsive_noise_v22(z_flat, mod, turbulence, rng)
    z_flat = add_burst_outliers_v22(z_flat, mod, turbulence, rng)

    rx_sc = z_flat.reshape(N_SC, N_SYM).astype(np.complex64)

    log = {
        "snr_db": float(snr_db),
        "bad_sc_count": bad_log["bad_sc_count"],
        "bad_sym_count": bad_log["bad_sym_count"],
    }
    return rx_sc, log


def main():
    print(f"=== sim_generate_data_v2_2_12k.py  [{now_str()}] ===")
    print(f"  Calibrated-Sim-v2.2: 6 mods x 2 turbs x {FRAMES_PER_MOD_TURB} = {6*2*FRAMES_PER_MOD_TURB}")
    ensure_dir(SIM_DATA_ROOT)

    cfg = {
        "version": "v2.2_12k",
        "calibration_name": "Calibrated-Sim-v2.2",
        "description": "Real-distribution-calibrated multi-impairment simulation (12k frames)",
        "frames_per_mod_turb": FRAMES_PER_MOD_TURB,
        "total_frames": 6 * 2 * FRAMES_PER_MOD_TURB,
        "impairments": ["awgn", "amplitude_scintillation", "phase_errors",
                        "bad_subcarrier_spike", "impulsive_noise", "burst_outliers"],
        "snr_ranges_weak": {
            "QPSK": [15, 20], "16QAM": [12, 17], "32QAM": [9, 14],
            "64QAM": [8, 14], "128QAM": [6, 12], "256QAM": [3, 9],
        },
        "snr_ranges_strong": {
            "QPSK": [10, 15], "16QAM": [8, 13], "32QAM": [6, 11],
            "64QAM": [5, 10], "128QAM": [3, 8], "256QAM": [0, 6],
        },
        "impulsive_scale": {"QPSK": 0.0, "16QAM": 2.0, "32QAM": 2.8, "64QAM": 3.8, "128QAM": 5.0, "256QAM": 6.5},
        "impulsive_base_p": {"QPSK": 0.0, "16QAM": 0.0005, "32QAM": 0.0012, "64QAM": 0.0035, "128QAM": 0.0055, "256QAM": 0.0065},
        "burst_lam": {"QPSK": 0.0, "16QAM": 0.4, "32QAM": 0.6, "64QAM": 0.9, "128QAM": 1.2, "256QAM": 1.6},
        "burst_amp_range": [2.5, 6.0],
        "high_order_burst_mult": 1.10,
        "strong_burst_mult": 1.05,
        "normalize_after": False,
    }

    with open(DISTURBANCE_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  配置: {DISTURBANCE_CONFIG}")

    total = 6 * 2 * FRAMES_PER_MOD_TURB
    print(f"\n生成 Calibrated-Sim-v2.2 数据（{total} 帧）")
    set_seed(2026)
    manifest_rows = []
    t0 = time.time()

    for turb_name in SIM_TURB_NAMES:
        turb_label = TURB_TO_LABEL[turb_name]
        turb_param = "weak" if turb_name == "very_weak" else "strong"

        for mod_name in MOD_NAMES:
            mod_label = MOD_TO_LABEL[mod_name]
            out_dir = SIM_DATA_ROOT / mod_name / turb_name
            ensure_dir(out_dir)

            for frame_idx in range(FRAMES_PER_MOD_TURB):
                rng = np.random.default_rng(
                    2026000 + hash((turb_name, mod_name, frame_idx)) % 2**31
                )

                tx_sc = gen_tx_sc(mod_name, N_SC, N_SYM, rng)
                rx_sc, imp_log = apply_v2_2_impairments(
                    tx_sc, mod_name, turb_param, rng
                )

                cdm = make_cdm_from_rx_sc(rx_sc, nbin=64, clip_val=3.0)
                blind_stats = make_blind_stats(rx_sc)
                radial = make_radial_hist(rx_sc, n_bins=32, clip_val=3.0)

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
                    "source": "sim_v2_2_12k",
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
                    "source": "sim_v2_2_12k",
                    "sim_frame_idx": frame_idx,
                    "bad_sc_count": imp_log["bad_sc_count"],
                    "bad_sym_count": imp_log["bad_sym_count"],
                })

            elapsed = time.time() - t0
            print(f"  [{turb_name}/{mod_name}] {FRAMES_PER_MOD_TURB} 帧 | "
                  f"总计 {len(manifest_rows)}/{total} | {elapsed:.1f}s")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(SIM_MANIFEST_CSV, index=False)
    print(f"\n  manifest: {SIM_MANIFEST_CSV} ({len(manifest_df)} rows)")

    print(f"\n  SNR by mod x turb:")
    for turb in SIM_TURB_NAMES:
        for mod in MOD_NAMES:
            sub = manifest_df[(manifest_df["turb_name"] == turb) & (manifest_df["mod_name"] == mod)]
            print(f"    {turb:10s}/{mod:8s}: mean={sub['snr_frame_db'].mean():.2f} dB, "
                  f"std={sub['snr_frame_db'].std():.2f}")

    elapsed = time.time() - t0
    print(f"\n[完成] Calibrated-Sim-v2.2 {total} 帧已生成，用时 {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
