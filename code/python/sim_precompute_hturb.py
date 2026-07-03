# -*- coding: utf-8 -*-
"""
sim_precompute_hturb.py — Step 3: 相位屏角谱传播预计算

numpy 复刻 fso_full_pipeline.m 的角谱传播，对 400 个相位屏各算一个 h_norm。
h_norm = h_turb / h_ref，其中
    h_turb = sum(E_rxf * conj(E_rx_mode)) * pixel_pitch^2  （有湍流）
    h_ref  = 同上但 phi=0                                   （无湍流基准）

输出：
    results/stage2/hturb_precompute.npz
        h_norm_very_weak: (200,) complex128
        h_norm_strong:    (200,) complex128
        screen_files_very_weak: (200,) object
        screen_files_strong:    (200,) object
        h_ref: complex128
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

# 确保能 import stage2_common
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    SCREEN_VERYWEAK_DIR,
    SCREEN_STRONG_DIR,
    STAGE2_ROOT,
    HTURB_NPZ,
    LAMBDA, K_WAVE, NX, NY, PIXEL_PITCH, LX, LY,
    Z_PROP, W0, D_RX, NA_RX, F_MAX,
    ensure_dir, now_str,
)


def build_grids():
    """构建空间网格和频率网格（对应 .m 文件的 meshgrid）。"""
    x = np.linspace(-LX / 2, LX / 2, NX)
    y = np.linspace(-LY / 2, LY / 2, NY)
    X, Y = np.meshgrid(x, y)  # shape (NY, NX) = (1080, 1920)

    fx = np.linspace(-1 / (2 * PIXEL_PITCH), 1 / (2 * PIXEL_PITCH), NX)
    fy = np.linspace(-1 / (2 * PIXEL_PITCH), 1 / (2 * PIXEL_PITCH), NY)
    FX, FY = np.meshgrid(fx, fy)
    return X, Y, FX, FY


def build_propagation_kernel(FX, FY):
    """传播核 H_prop 和 NA 滤波 H_NA。"""
    # 传播核
    arg = 1 - (LAMBDA * FX) ** 2 - (LAMBDA * FY) ** 2
    arg = np.maximum(arg, 0.0)  # 消失波 → 0
    H_prop = np.exp(1j * K_WAVE * Z_PROP * np.sqrt(arg))
    # 消失波区域置零
    evanescent = FX ** 2 + FY ** 2 > (1.0 / LAMBDA) ** 2
    H_prop[evanescent] = 0.0

    # NA 滤波（二值掩膜）
    H_NA = ((FX ** 2 + FY ** 2) <= F_MAX ** 2).astype(np.float64)
    return H_prop, H_NA


def build_fields(X, Y):
    """入射高斯光束 E_in 和接收模式 E_rx_mode。"""
    E_in = np.exp(-(X ** 2 + Y ** 2) / W0 ** 2)
    aperture = (X ** 2 + Y ** 2) <= (D_RX / 2) ** 2
    w_rx = D_RX / 2.5
    E_rx_mode = np.exp(-(X ** 2 + Y ** 2) / w_rx ** 2) * aperture
    return E_in, E_rx_mode


def compute_h(E_in, E_rx_mode, H_prop, H_NA, phi=None):
    """
    计算耦合系数 h = sum(E_rxf * conj(E_rx_mode)) * pixel_pitch^2
    phi=None → 无湍流基准
    phi=相位屏 → 有湍流
    """
    if phi is None:
        E_slm = E_in
    else:
        E_slm = E_in * np.exp(1j * phi)

    # 角谱传播: fft2 → fftshift(居中) → ×H_prop → ifftshift(回角) → ifft2
    E_rxp = np.fft.ifft2(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft2(E_slm)) * H_prop
    ))
    # NA 滤波: 同样 fftshift 居中后再乘掩膜，再 ifftshift 回去
    E_rxf = np.fft.ifft2(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft2(E_rxp)) * H_NA
    ))

    h = np.sum(E_rxf * np.conj(E_rx_mode)) * (PIXEL_PITCH ** 2)
    return h


def load_screens(screen_dir: Path, pattern: str = "screen_*_sim.mat") -> list:
    """加载目录下所有相位屏 .mat，返回 [(filename, phz_crop), ...]"""
    files = sorted(screen_dir.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {screen_dir} 找不到 {pattern}")

    screens = []
    for f in files:
        mat = sio.loadmat(f, squeeze_me=True)
        phi = np.asarray(mat["phz_crop"], dtype=np.float64)
        if phi.shape != (NY, NX):
            print(f"  [警告] {f.name} shape={phi.shape}, 期望 ({NY},{NX})")
        screens.append((f.name, phi))
    return screens


def process_turbulence(screens, E_in, E_rx_mode, H_prop, H_NA, h_ref, label):
    """对一组相位屏计算 h_norm，返回 (h_norm_array, filenames_array)。"""
    n = len(screens)
    h_norms = np.zeros(n, dtype=np.complex128)
    filenames = np.empty(n, dtype=object)

    t0 = time.time()
    for i, (fname, phi) in enumerate(screens):
        h_turb = compute_h(E_in, E_rx_mode, H_prop, H_NA, phi)
        h_norms[i] = h_turb / h_ref
        filenames[i] = fname

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{n} | "
                  f"|h_norm| mean={np.mean(np.abs(h_norms[:i+1])):.4f} "
                  f"std={np.std(np.abs(h_norms[:i+1])):.4f} | "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  [{label}] 完成 {n} 屏，用时 {elapsed:.1f}s")
    return h_norms, filenames


def main():
    print(f"=== sim_precompute_hturb.py  [{now_str()}] ===")
    ensure_dir(STAGE2_ROOT)

    # 1. 构建网格和场
    print(f"\n[1/4] 构建空间网格和传播核")
    X, Y, FX, FY = build_grids()
    H_prop, H_NA = build_propagation_kernel(FX, FY)
    E_in, E_rx_mode = build_fields(X, Y)
    print(f"  网格: X{X.shape}, H_prop{H_prop.shape}")
    print(f"  E_in sum={np.sum(np.abs(E_in)):.4e}")

    # 2. 无湍流基准
    print(f"\n[2/4] 计算无湍流基准 h_ref")
    h_ref = compute_h(E_in, E_rx_mode, H_prop, H_NA, phi=None)
    print(f"  h_ref = {h_ref:.6e}")
    print(f"  |h_ref| = {np.abs(h_ref):.6e}")
    assert np.abs(h_ref) > 0, "h_ref 为零，传播计算有误"

    # 3. 加载并处理相位屏
    print(f"\n[3/4] 加载相位屏")
    screens_vw = load_screens(SCREEN_VERYWEAK_DIR)
    screens_st = load_screens(SCREEN_STRONG_DIR)
    print(f"  极弱: {len(screens_vw)} 屏")
    print(f"  强:   {len(screens_st)} 屏")

    print(f"\n  处理极弱湍流...")
    h_vw, f_vw = process_turbulence(
        screens_vw, E_in, E_rx_mode, H_prop, H_NA, h_ref, "very_weak"
    )

    print(f"\n  处理强湍流...")
    h_st, f_st = process_turbulence(
        screens_st, E_in, E_rx_mode, H_prop, H_NA, h_ref, "strong"
    )

    # 4. 诊断和保存
    print(f"\n[4/4] 诊断与保存")
    print(f"\n  === |h_norm| 统计 ===")
    for label, h in [("very_weak", h_vw), ("strong", h_st)]:
        amp = np.abs(h)
        ang = np.angle(h)
        print(f"  {label:12s}: "
              f"|h| mean={amp.mean():.4f} std={amp.std():.4f} "
              f"min={amp.min():.4f} max={amp.max():.4f}")
        print(f"  {'':12s}  "
              f"angle mean={ang.mean():.4f} std={ang.std():.4f} rad "
              f"({np.degrees(ang.std()):.2f} deg)")

    print(f"\n  保存 → {HTURB_NPZ}")
    np.savez_compressed(
        HTURB_NPZ,
        h_norm_very_weak=h_vw,
        h_norm_strong=h_st,
        screen_files_very_weak=f_vw,
        screen_files_strong=f_st,
        h_ref=np.complex128(h_ref),
    )
    print(f"  [OK] 保存完成")

    # 物理性验证
    print(f"\n  === 物理性验证 ===")
    vw_std = np.std(np.abs(h_vw))
    st_std = np.std(np.abs(h_st))
    print(f"  弱湍流 |h| std = {vw_std:.4f}  (期望 < 0.1)")
    print(f"  强湍流 |h| std = {st_std:.4f}  (期望 > 0.3)")
    if vw_std < 0.1:
        print(f"  [OK] 弱湍流闪烁合理")
    else:
        print(f"  [警告] 弱湍流闪烁偏大")
    if st_std > 0.3:
        print(f"  [OK] 强湍流闪烁合理")
    else:
        print(f"  [警告] 强湍流闪烁偏小")

    print(f"\n[完成] h_turb 预计算完毕。")


if __name__ == "__main__":
    main()
