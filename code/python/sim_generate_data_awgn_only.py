# -*- coding: utf-8 -*-
"""
sim_generate_data_awgn_only.py — AWGN-only 仿真数据（S5 消融对照）

基于 sim_generate_data_v2_2.py，仅保留 AWGN 扰动，跳过：
  - amplitude scintillation
  - phase errors (CPE + jitter)
  - bad subcarrier / bad symbol spike
  - impulsive noise
  - burst outliers

SNR ranges、星座生成、特征预计算、保存格式与 v2.2 完全一致，确保公平对比。
6 mods x 2 turbs x 1000 = 12000 帧。

S5 消融：AWGN-only vs Calibrated-Sim-v2.2，证明真实分布校准的多扰动模型有效性。
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

SIM_DATA_ROOT = STAGE2_ROOT / "sim_data_awgn_only"
SIM_MANIFEST_CSV = SIM_DATA_ROOT / "manifest_sim_awgn_only.csv"
DISTURBANCE_CONFIG = SIM_DATA_ROOT / "disturbance_config_awgn_only.json"

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


SNR_RANGES = {
    "weak": {
        4: (15, 20), 16: (12, 17), 32: (9, 14),
        64: (8, 14), 128: (6, 12), 256: (3, 9),
    },
    "strong": {
        4: (10, 15), 16: (8, 13), 32: (6, 11),
        64: (5, 10), 128: (3, 8), 256: (0, 6),
    },
}


def apply_awgn_only_impairments(
    tx_sc: np.ndarray,
    mod: str,
    turbulence: str,
    rng: np.random.Generator,
    snr_db: float = None,
) -> tuple:
    """仅 AWGN 扰动，跳过所有其他扰动。"""
    z_flat = tx_sc.copy().astype(np.complex64).reshape(-1)

    z_flat = rms_normalize(z_flat)

    order = MOD_ORDER[mod]
    if snr_db is None:
        snr_range = SNR_RANGES[turbulence][order]
        snr_db = rng.uniform(*snr_range)
    z_flat = add_awgn_by_snr(z_flat, snr_db, rng)

    rx_sc = z_flat.reshape(N_SC, N_SYM).astype(np.complex64)

    log = {
        "snr_db": float(snr_db),
        "bad_sc_count": 0,
        "bad_sym_count": 0,
    }
    return rx_sc, log


def main():
    print(f"=== sim_generate_data_awgn_only.py  [{now_str()}] ===")
    print(f"  AWGN-only: 6 mods x 2 turbs x {FRAMES_PER_MOD_TURB} = {6*2*FRAMES_PER_MOD_TURB}")
    ensure_dir(SIM_DATA_ROOT)

    cfg = {
        "version": "awgn_only",
        "description": "AWGN-only simulation for S5 ablation (no scintillation/phase/impulse/burst)",
        "frames_per_mod_turb": FRAMES_PER_MOD_TURB,
        "total_frames": 6 * 2 * FRAMES_PER_MOD_TURB,
        "impairments": ["awgn"],
        "snr_ranges_weak": {
            "QPSK": [15, 20], "16QAM": [12, 17], "32QAM": [9, 14],
            "64QAM": [8, 14], "128QAM": [6, 12], "256QAM": [3, 9],
        },
        "snr_ranges_strong": {
            "QPSK": [10, 15], "16QAM": [8, 13], "32QAM": [6, 11],
            "64QAM": [5, 10], "128QAM": [3, 8], "256QAM": [0, 6],
        },
    }

    with open(DISTURBANCE_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  配置: {DISTURBANCE_CONFIG}")

    total = 6 * 2 * FRAMES_PER_MOD_TURB
    print(f"\n生成 AWGN-only 数据（{total} 帧）")
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
                rx_sc, imp_log = apply_awgn_only_impairments(
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
                    "source": "sim_awgn_only",
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
                    "source": "sim_awgn_only",
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
    print(f"\n[完成] AWGN-only {total} 帧已生成，用时 {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
