"""
time_series_utils.py — GAF (Gramian Angular Field) 时序编码工具
===============================================================
基于 pyts 将 I/Q 复数符号序列转换为 64×64 GAF 图像，
用于 CDM(空间域) + GAF(时序域) 多域特征融合。

用法:
    from time_series_utils import complex_to_gaf
    gaf_img = complex_to_gaf(rx_IQ, image_size=64)
"""

import numpy as np
from pyts.image import GramianAngularField


def complex_to_gaf(
    complex_symbols: np.ndarray,
    image_size: int = 64,
    method: str = "summation",
    downsample_to: int = 4096,
) -> np.ndarray:
    """
    将复数符号序列转换为 GAF 时序编码图像。

    处理流程:
      1. 计算复数序列的幅度 (magnitude)
      2. 重采样/截断至 downsample_to 长度 (确保 >= image_size)
      3. 归一化至 [-1, 1] 区间 (GAF 要求)
      4. GramianAngularField 编码 → (image_size, image_size)

    参数:
      complex_symbols: 形状 (N,) 或 (N,2) 的输入
                       - (N,): 直接作为复数向量
                       - (N,2): 第0列为实部, 第1列为虚部, 合并为复数
      image_size:      GAF 输出图像的边长 (默认 64)
      method:          GAF 求和方式 ("summation" 或 "difference")
      downsample_to:   重采样目标长度 (默认 4096)

    返回:
      gaf_img: 形状 (image_size, image_size) 的 GAF 图像, 值域 [0, 1]
    """
    # ---- 1. 解析输入 ----
    c = np.asarray(complex_symbols, dtype=np.float64)

    if c.ndim == 2 and c.shape[1] == 2:
        # (N, 2) → 复数
        complex_vec = c[:, 0] + 1j * c[:, 1]
    elif c.ndim == 1:
        complex_vec = c
    else:
        raise ValueError(f"输入形状必须为 (N,) 或 (N,2), 实际: {c.shape}")

    # ---- 2. 计算幅度 ----
    mag = np.abs(complex_vec).astype(np.float64)

    # ---- 3. 重采样至固定长度 ----
    n = len(mag)
    if n > downsample_to:
        indices = np.linspace(0, n - 1, downsample_to, dtype=int)
        mag = mag[indices]
    elif n < image_size:
        raise ValueError(
            f"符号数量 ({n}) 必须 >= image_size ({image_size})"
        )

    # ---- 4. 归一化至 [-1, 1] ----
    mag_min, mag_max = mag.min(), mag.max()
    if mag_max > mag_min:
        mag = 2.0 * (mag - mag_min) / (mag_max - mag_min) - 1.0
    else:
        mag = np.zeros_like(mag)

    # ---- 5. GAF 编码 ----
    gaf = GramianAngularField(image_size=image_size, method=method)
    gaf_img = gaf.fit_transform(mag.reshape(1, -1))  # (1, H, W)

    # ---- 6. 归一化至 [0, 1] 并返回单通道 ----
    img = np.squeeze(gaf_img).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def complex_to_gaf_pair(
    complex_symbols: np.ndarray,
    image_size: int = 64,
    downsample_to: int = 4096,
) -> np.ndarray:
    """
    同时生成 summation + difference GAF, 拼为 (2, H, W) 双通道输出。

    参数同上。
    """
    img_sum = complex_to_gaf(complex_symbols, image_size, "summation", downsample_to)
    img_dif = complex_to_gaf(complex_symbols, image_size, "difference", downsample_to)
    return np.stack([img_sum, img_dif], axis=0)  # (2, H, W)


def phase_to_gaf(
    complex_symbols: np.ndarray,
    image_size: int = 64,
    downsample_to: int = 4096,
) -> np.ndarray:
    """
    使用复数符号的相位 (而非幅度) 生成 GAF。

    参数同上。
    """
    c = np.asarray(complex_symbols, dtype=np.float64)
    if c.ndim == 2 and c.shape[1] == 2:
        complex_vec = c[:, 0] + 1j * c[:, 1]
    elif c.ndim == 1:
        complex_vec = c
    else:
        raise ValueError(f"输入形状必须为 (N,) 或 (N,2), 实际: {c.shape}")

    phase = np.angle(complex_vec)

    n = len(phase)
    if n > downsample_to:
        indices = np.linspace(0, n - 1, downsample_to, dtype=int)
        phase = phase[indices]

    # 相位已在 [-π, π], 归一化至 [-1, 1]
    phase = phase / np.pi

    gaf = GramianAngularField(image_size=image_size, method="summation")
    gaf_img = gaf.fit_transform(phase.reshape(1, -1))

    img = np.squeeze(gaf_img).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 模拟 15,000 个复数符号 (OFDM 解调后典型长度)
    N = 15000
    np.random.seed(42)
    signal = np.random.randn(N) + 1j * np.random.randn(N)
    signal_IQ = np.column_stack([signal.real, signal.imag])  # (N, 2)

    print(f"输入: {signal_IQ.shape}")

    gaf_mag = complex_to_gaf(signal_IQ)
    print(f"幅度 GAF: {gaf_mag.shape}, range=[{gaf_mag.min():.3f}, {gaf_mag.max():.3f}]")

    gaf_phase = phase_to_gaf(signal_IQ)
    print(f"相位 GAF: {gaf_phase.shape}, range=[{gaf_phase.min():.3f}, {gaf_phase.max():.3f}]")

    gaf_pair = complex_to_gaf_pair(signal_IQ)
    print(f"双通道 GAF: {gaf_pair.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(gaf_mag, cmap="inferno"); axes[0].set_title("GAF (Magnitude)")
    axes[1].imshow(gaf_phase, cmap="inferno"); axes[1].set_title("GAF (Phase)")
    axes[2].imshow(gaf_pair[0], cmap="inferno"); axes[2].set_title("GAF pair (Sum)")
    for ax in axes: ax.axis("off")
    fig.savefig("results/figures/gaf_verification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 自测完成 → results/figures/gaf_verification.png")
