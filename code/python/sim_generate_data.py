# -*- coding: utf-8 -*-
"""
sim_generate_data.py — Step 4: 生成 phase-screen 仿真数据

频域直生 rx_sc[123,128] + 扰动模型 + 特征提取（与 lowmem bit-identical）。
输出 6000 .mat 文件（6 调制 × 2 湍流 × 500 帧）+ manifest_sim.csv + disturbance_config.json

设计要点：
  - 不做完整时域 OFDM+DSP；直接在频域生成后均衡的 rx_sc
  - 扰动模型注入能存活 LTS 均衡的效应：CPE、残余 CFO、per-sc 相位噪声、
    闪烁调制 SNR、AWGN、RX 幅度缩放
  - 特征提取调用 train_mrf_lowmem 的 make_cdm_from_rx_sc / make_blind_stats / make_radial_hist
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
    SIM_DATA_ROOT, STAGE2_ROOT, HTURB_NPZ, DISTURBANCE_CONFIG,
    SIM_MANIFEST_CSV, MOD_NAMES, MOD_TO_LABEL, SIM_TURB_NAMES,
    TURB_TO_LABEL, N_SC, N_SYM, REAL_DATA_ROOT,
    ensure_dir, now_str, get_hturb_pool, set_seed,
)
from train_mrf_lowmem import make_cdm_from_rx_sc, make_blind_stats, make_radial_hist


# ─────────────────────────────────────────────────────────
# QAM 星座生成
# ─────────────────────────────────────────────────────────
def gen_qam_constellation(M: int) -> np.ndarray:
    """生成 M-QAM 星座点，归一化到单位平均功率。"""
    if M == 4:
        pts = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=complex)
    elif M == 16:
        levels = np.array([-3, -1, 1, 3])
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
    elif M == 32:
        # Cross QAM: 6×6 grid, remove 4 corners
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
        # Cross QAM: 12×12 grid, remove 4 corner 2×2 blocks
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


MOD_TO_M = {"QPSK": 4, "16QAM": 16, "32QAM": 32, "64QAM": 64, "128QAM": 128, "256QAM": 256}


def gen_tx_sc(mod_name: str, n_sc: int, n_sym: int, rng: np.random.Generator) -> np.ndarray:
    """随机采样 QAM 符号，返回 (n_sc, n_sym) complex64, 单位功率。"""
    M = MOD_TO_M[mod_name]
    constellation = gen_qam_constellation(M)
    indices = rng.integers(0, M, size=n_sc * n_sym)
    symbols = constellation[indices]
    return symbols.reshape(n_sc, n_sym).astype(np.complex64)


# ─────────────────────────────────────────────────────────
# 扰动模型
# ─────────────────────────────────────────────────────────
def get_default_config() -> dict:
    """v1 初始扰动参数（基于真实数据 PAPR/amp_mean 统计推断）。"""
    return {
        "version": "v1",
        "description": "Phase-screen sim v1, frequency-domain direct generation",
        # CPE: 角度(h_norm) 提供基准 CPE，附加随机漂移
        "cpe_drift_std_rad": 0.08,
        # 残余 CFO: per-subcarrier 线性相位斜率
        "cfo_std_rad_per_sc": 0.0015,
        # per-subcarrier 相位噪声（LTS 均衡后残留）
        "phase_noise_std_rad": 0.03,
        # 接收端幅度缩放
        "amp_scale_std": 0.06,
        # 闪烁对 SNR 的调制范围（|h_norm| 偏离 1 时 SNR 变化）
        "snr_scint_range_db": 4.0,
        # SNR 随机波动
        "snr_std_db": 2.0,
        # 基础 SNR（per turbulence）
        "snr_base_db": {"very_weak": 25.0, "strong": 19.0},
        "snr_min_db": 10.0,
        "snr_max_db": 32.0,
        # 是否使用 h_norm 的角度作为 CPE 基准
        "use_h_angle_as_cpe": True,
        # 是否使用 |h_norm| 调制 SNR
        "use_scint_for_snr": True,
    }


def apply_disturbances(
    tx_sc: np.ndarray,
    h_norm: complex,
    cfg: dict,
    turb_name: str,
    rng: np.random.Generator,
) -> tuple:
    """
    对 tx_sc 注入扰动，返回 (rx_sc, snr_db, disturbance_log)。

    扰动清单（对应 23_57.txt 的 8 项要求）：
      1. 相位屏扰动 → h_norm 的 angle 作为 CPE 基准
      2. 幅度闪烁 → |h_norm| 调制 SNR
      3. 相位畸变 → per-subcarrier 残余相位
      4. AWGN
      5. CPE 漂移
      6. 残余 CFO
      7. RX 幅度缩放
      8. 归一化（与真实数据一致：零均值、单位 RMS）
    """
    n_sc, n_sym = tx_sc.shape
    rx_sc = tx_sc.copy().astype(np.complex128)
    scint = np.abs(h_norm)

    # --- 1+5. CPE (Common Phase Error) ---
    if cfg["use_h_angle_as_cpe"]:
        cpe_base = np.angle(h_norm)
    else:
        cpe_base = 0.0
    cpe_drift = cfg["cpe_drift_std_rad"] * rng.standard_normal()
    cpe_total = cpe_base + cpe_drift

    # --- 6. 残余 CFO (linear phase across subcarriers) ---
    cfo = cfg["cfo_std_rad_per_sc"] * rng.standard_normal()
    k_idx = np.arange(n_sc) - n_sc // 2  # 以中心子载波为参考

    # --- 3. per-subcarrier 相位噪声 ---
    phase_noise = cfg["phase_noise_std_rad"] * rng.standard_normal(n_sc)

    # 组合相位: phi[k] = CPE + CFO*k + noise[k]
    phase_per_sc = cpe_total + cfo * k_idx + phase_noise
    rx_sc = rx_sc * np.exp(1j * phase_per_sc)[:, None]

    # --- 7. RX 幅度缩放 ---
    amp_scale = 1.0 + cfg["amp_scale_std"] * rng.standard_normal()
    rx_sc = rx_sc * amp_scale

    # --- 2+4. 闪烁调制 SNR + AWGN ---
    snr_base = cfg["snr_base_db"][turb_name]
    if cfg["use_scint_for_snr"]:
        # |h_norm| < 1 → 信号衰减 → SNR 降低
        # |h_norm| > 1 → 信号增强 → SNR 升高
        snr_scint = snr_base + cfg["snr_scint_range_db"] * (scint - 1.0) / 0.3
    else:
        snr_scint = snr_base
    snr_db = snr_scint + cfg["snr_std_db"] * rng.standard_normal()
    snr_db = float(np.clip(snr_db, cfg["snr_min_db"], cfg["snr_max_db"]))

    snr_lin = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_lin  # tx_sc 是单位功率
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal((n_sc, n_sym)) + 1j * rng.standard_normal((n_sc, n_sym))
    )
    rx_sc = rx_sc + noise

    # --- 8. 归一化：零均值、单位 RMS（与 lowmem 特征提取一致）---
    rx_sc = rx_sc - np.mean(rx_sc)
    power = np.mean(np.abs(rx_sc) ** 2)
    rx_sc = rx_sc / np.sqrt(power + 1e-12)

    log = {
        "snr_db": snr_db,
        "scint": float(scint),
        "cpe_rad": float(cpe_total),
        "cfo_rad_per_sc": float(cfo),
        "amp_scale": float(amp_scale),
        "phase_noise_rms_rad": float(np.sqrt(np.mean(phase_noise ** 2))),
    }
    return rx_sc.astype(np.complex64), snr_db, log


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== sim_generate_data.py  [{now_str()}] ===")
    ensure_dir(SIM_DATA_ROOT)

    # 1. 加载 h_turb 池
    print(f"\n[1/4] 加载预计算 h_turb")
    hturb_pool = get_hturb_pool(HTURB_NPZ)
    for turb, arr in hturb_pool.items():
        print(f"  {turb}: {len(arr)} 屏, |h| mean={np.mean(np.abs(arr)):.4f} std={np.std(np.abs(arr)):.4f}")

    # 2. 扰动配置
    cfg = get_default_config()
    print(f"\n[2/4] 扰动配置 v1:")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # 保存配置
    with open(DISTURBANCE_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  配置已保存: {DISTURBANCE_CONFIG}")

    # 3. 生成数据
    print(f"\n[3/4] 生成仿真数据（6 mods × 2 turbs × 500 frames = 6000）")
    set_seed(2026)
    manifest_rows = []
    t0 = time.time()

    for turb_name in SIM_TURB_NAMES:
        h_pool = hturb_pool[turb_name]
        turb_label = TURB_TO_LABEL[turb_name]

        for mod_name in MOD_NAMES:
            mod_label = MOD_TO_LABEL[mod_name]
            out_dir = SIM_DATA_ROOT / mod_name / turb_name
            ensure_dir(out_dir)

            for frame_idx in range(500):
                rng = np.random.default_rng(
                    2026000 + hash((turb_name, mod_name, frame_idx)) % 2**31
                )

                # 随机选一个 h_norm
                h_idx = rng.integers(0, len(h_pool))
                h_norm = complex(h_pool[h_idx])

                # 生成 tx_sc
                tx_sc = gen_tx_sc(mod_name, N_SC, N_SYM, rng)

                # 注入扰动
                rx_sc, snr_db, log = apply_disturbances(tx_sc, h_norm, cfg, turb_name, rng)

                # 特征提取（与 lowmem bit-identical）
                cdm = make_cdm_from_rx_sc(rx_sc, nbin=64, clip_val=3.0)
                blind_stats = make_blind_stats(rx_sc)
                radial = make_radial_hist(rx_sc, n_bins=32, clip_val=3.0)

                # 保存 .mat
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
                    "snr_frame_db": np.float32(snr_db),
                    "h_norm_abs": np.float32(np.abs(h_norm)),
                    "h_norm_angle": np.float32(np.angle(h_norm)),
                    "source": "sim",
                    "sim_frame_idx": np.int32(frame_idx),
                })

                manifest_rows.append({
                    "out_mat": str(save_path),
                    "mod_name": mod_name,
                    "mod_label": mod_label,
                    "turb_name": turb_name,
                    "turb_label": turb_label,
                    "snr_frame_db": snr_db,
                    "h_norm_abs": float(np.abs(h_norm)),
                    "h_norm_angle": float(np.angle(h_norm)),
                    "source": "sim",
                    "sim_frame_idx": frame_idx,
                })

            # 进度
            elapsed = time.time() - t0
            total_done = len(manifest_rows)
            print(f"  [{turb_name}/{mod_name}] 完成 500 帧 | "
                  f"总计 {total_done}/6000 | elapsed {elapsed:.1f}s")

    # 4. 保存 manifest
    print(f"\n[4/4] 保存 manifest")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(SIM_MANIFEST_CSV, index=False)
    print(f"  {SIM_MANIFEST_CSV} ({len(manifest_df)} rows)")

    # 验证
    print(f"\n=== 生成验证 ===")
    print(f"  总帧数: {len(manifest_df)}")
    print(f"  mod 分布:\n{manifest_df['mod_name'].value_counts().sort_index()}")
    print(f"  turb 分布:\n{manifest_df['turb_name'].value_counts()}")

    # SNR 统计
    print(f"\n  SNR 统计 (per turb):")
    for turb in SIM_TURB_NAMES:
        sub = manifest_df[manifest_df["turb_name"] == turb]
        print(f"    {turb}: mean={sub['snr_frame_db'].mean():.2f} dB, "
              f"std={sub['snr_frame_db'].std():.2f}, "
              f"min={sub['snr_frame_db'].min():.2f}, max={sub['snr_frame_db'].max():.2f}")

    # 验证一个 .mat 文件
    sample_path = manifest_df.iloc[0]["out_mat"]
    m = sio.loadmat(sample_path)
    print(f"\n  样本 .mat 验证: {Path(sample_path).name}")
    print(f"    rx_sc: {m['rx_sc'].shape} {m['rx_sc'].dtype}")
    print(f"    cdm64: {m['cdm64'].shape} {m['cdm64'].dtype}")
    print(f"    blind_stats: {m['blind_stats'].shape}")
    print(f"    cdm max={m['cdm64'].max():.4f}, has_nan={np.any(np.isnan(m['blind_stats']))}")

    elapsed_total = time.time() - t0
    print(f"\n[完成] 6000 仿真帧已生成，用时 {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    main()
