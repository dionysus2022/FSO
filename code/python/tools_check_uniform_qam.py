#!/usr/bin/env python3
"""
tools_check_uniform_qam.py

检测 QAM 符号分布是否为均匀分布（判断是否经过概率整形）。

用法:
    python tools_check_uniform_qam.py --data_dir <npz_dir> --mod 16QAM
    python tools_check_uniform_qam.py --data_dir <npz_dir> --all      # 测试所有调制
    python tools_check_uniform_qam.py --iq_file <npy_or_npz>          # 直接加载 I/Q 文件

原理:
    1. 对接收符号做硬判决→最近星座点
    2. 统计每个星座点出现的频次
    3. 卡方拟合优度检验 (H0: 均匀分布)
    
    均匀分布 → 没有概率整形 (CCDM)
    非均匀 (低幅度点偏多) → 有概率整形
"""

import numpy as np
import argparse
import glob
import os
import sys
from scipy import stats


def make_constellation(mod: str, norm: str = "unit_avg_power") -> np.ndarray:
    """生成标准 QAM 星座点。"""
    mod = mod.upper()
    
    if mod == "QPSK":
        c = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    elif mod == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        c = (I + 1j*Q).ravel()
    elif mod == "32QAM":
        levels = np.arange(-5, 6, 2, dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        keep = ~((np.abs(I) == 5) & (np.abs(Q) == 5))
        c = (I[keep] + 1j*Q[keep]).ravel()
    elif mod == "64QAM":
        levels = np.arange(-7, 8, 2, dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        c = (I + 1j*Q).ravel()
    elif mod == "128QAM":
        levels = np.arange(-11, 12, 2, dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        keep = ~((np.abs(I) >= 9) & (np.abs(Q) >= 9))
        c = (I[keep] + 1j*Q[keep]).ravel()
    elif mod == "256QAM":
        levels = np.arange(-15, 16, 2, dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        c = (I + 1j*Q).ravel()
    else:
        raise ValueError(f"不支持的调制: {mod}")
    
    if norm == "unit_avg_power":
        avg_power = np.mean(np.abs(c) ** 2)
        c = c / np.sqrt(avg_power)
    
    return c.astype(np.complex64)


def test_uniform(samples: np.ndarray, mod: str, norm: str = "unit_avg_power"):
    """
    检测 QAM 符号分布是否为均匀分布。
    
    Parameters
    ----------
    samples : np.ndarray, complex
        QAM 符号数组（已解调、已归一化）
    mod : str
        调制格式
    norm : str
        星座归一化方式
    
    Returns
    -------
    dict : {'is_uniform', 'p_value', 'chi2', 'counts', 'expected_per_point'}
    """
    const = make_constellation(mod, norm)
    n_points = len(const)
    
    # 硬判决：每个符号归类到最近的星座点
    # 用广播加速，避免逐点循环
    dist = np.abs(samples[:, None] - const[None, :])  # [N, M]
    nearest = np.argmin(dist, axis=1)
    
    # 计数 + 卡方检验
    counts = np.bincount(nearest, minlength=n_points)
    expected = len(samples) / n_points
    
    chi2, p = stats.chisquare(counts)
    
    # 计算幅度分布的偏度（概率整形的特征：低幅度偏多）
    amp = np.abs(samples)
    amp_skew = stats.skew(amp)
    
    # 各环幅度统计
    unique_amps = np.sort(np.unique(np.round(np.abs(const), 4)))
    
    return {
        "is_uniform": p > 0.05,
        "p_value": p,
        "chi2": chi2,
        "counts": counts,
        "expected_per_point": expected,
        "n_symbols": len(samples),
        "n_points": n_points,
        "counts_min": int(counts.min()),
        "counts_max": int(counts.max()),
        "counts_var_ratio": float(np.var(counts) / max(expected, 1)),
        "amp_skew": float(amp_skew),
    }


def print_result(mod: str, r: dict):
    """打印检测结果。"""
    status = "✅ 均匀分布" if r["is_uniform"] else "❌ 非均匀 (概率整形)"
    print(f"  {mod:8s} | "
          f"N={r['n_symbols']:5d} | "
          f"星座点={r['n_points']:3d} | "
          f"期望每点={r['expected_per_point']:5.1f} | "
          f"实际范围=[{r['counts_min']:3d},{r['counts_max']:3d}] | "
          f"var_ratio={r['counts_var_ratio']:.2f} | "
          f"p={r['p_value']:.4f} | "
          f"{status}")


def load_from_npz(path: str, mod_label: int = None, max_samples: int = 1):
    """
    从 NPZ shard 加载 I/Q 样本。
    
    Returns: list of complex arrays (每个样本一个数组)
    """
    z = np.load(path, allow_pickle=True)
    if mod_label is not None:
        mask = z["mod_label"] == mod_label
    else:
        mask = np.ones(len(z["mod_label"]), dtype=bool)
    
    if mask.sum() == 0:
        return []
    
    # 限制样本数
    indices = np.where(mask)[0][:max_samples]
    
    samples_list = []
    for idx in indices:
        y = z["Y_iq"][idx]  # [2, K, T]
        c = (y[0] + 1j * y[1]).ravel()
        samples_list.append(c)
    
    return samples_list


def main():
    p = argparse.ArgumentParser(description="QAM 分布均匀性检测工具")
    p.add_argument("--data_dir", help="NPZ shard 目录")
    p.add_argument("--mod", default="16QAM", help="调制格式")
    p.add_argument("--all", action="store_true", help="测试所有 6 种调制")
    p.add_argument("--iq_file", help="直接加载 .npy/.npz I/Q 文件")
    p.add_argument("--norm", default="unit_avg_power", choices=["unit_avg_power", "none"])
    p.add_argument("--max_samples", type=int, default=3, help="每种调制最多测试几个样本")
    args = p.parse_args()
    
    mods = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
    mod_labels = [0, 1, 2, 3, 4, 5]
    
    if args.iq_file:
        # 从单文件加载
        data = np.load(args.iq_file)
        if "Y_iq" in data:
            y = data["Y_iq"][0]
            s = (y[0] + 1j*y[1]).ravel()
        else:
            s = data.ravel()
        r = test_uniform(s, args.mod, args.norm)
        print(f"\n文件: {args.iq_file}")
        print_result(args.mod, r)
        return
    
    if not args.data_dir:
        p.print_help()
        return
    
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not paths:
        print(f"错误: {args.data_dir} 中未找到 .npz 文件")
        sys.exit(1)
    
    print(f"数据目录: {args.data_dir}")
    print(f"Shard 数: {len(paths)}")
    print(f"{'调制':8s} | {'N':>5s} | {'星座点':>5s} | {'期望/点':>7s} | {'实际范围':>10s} | {'var_ratio':>9s} | {'p':>6s} | 结论")
    print("-" * 95)
    
    test_mods = zip(mods, mod_labels) if args.all else [(args.mod, mod_labels[mods.index(args.mod.upper())])]
    
    for mod_name, mod_label in test_mods:
        all_samples = []
        for path in paths:
            samples = load_from_npz(path, mod_label, args.max_samples)
            all_samples.extend(samples)
            if len(all_samples) >= args.max_samples:
                break
        
        if not all_samples:
            print(f"  {mod_name:8s} | ⚠️  未找到样本")
            continue
        
        # 将多个样本的符号拼接在一起以获得更可靠的统计
        combined = np.concatenate(all_samples)
        r = test_uniform(combined, mod_name, args.norm)
        print_result(mod_name, r)
    
    print()
    print("说明:")
    print("  - p>0.05 → 不能拒绝均匀分布假设 → 无概率整形")
    print("  - p<0.05 → 非均匀分布 → 可能存在概率整形")
    print("  - 注意: 高阶 QAM(64+) 在符号数不足时噪声会导致假阳性")
    print("  - var_ratio ≈ 1: 均匀; >2: 可能有整形")


if __name__ == "__main__":
    main()
