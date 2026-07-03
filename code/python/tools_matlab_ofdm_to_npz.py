#!/usr/bin/env python3
"""
tools_matlab_ofdm_to_npz.py

将 MATLAB 生成的 OFDM 信号转换为 QAM 符号 NPZ，用于均匀性检测。

用法 1: 直接加载 TX 频域信号（最快，无需解调）
    python tools_matlab_ofdm_to_npz.py ^
      --mat_dir "D:/.../tx_3frame_6mod/16QAM/sub1" ^
      --mod 16QAM ^
      --out "tx_16qam_samples.npz"

用法 2: 加载 RX 时域信号（需 OFDM 解调）
    python tools_matlab_ofdm_to_npz.py ^
      --rx_time_dir "D:/.../rx_data" ^
      --ofdm_params ofdm_params.json ^
      --mod 16QAM ^
      --out "rx_16qam_samples.npz"

用法 3: 加载后直接用 uniform check 检测
    python tools_check_uniform_qam.py --iq_file "tx_16qam_samples.npz" --mod 16QAM
"""

import numpy as np
import argparse
import glob
import json
import os
import re
import sys
from scipy import signal as scipy_signal


# ========== OFDM 参数（默认与 tx_3frame_6mod.m 一致） ==========
DEFAULT_OFDM = {
    "n_fft": 256,           # IFFT/FFT 点数
    "cp_len": 16,           # 循环前缀长度
    "data_carriers": [4, 126],  # 数据子载波范围 [start, end]（MATLAB 1-indexed）
    "fs_tx": 16e9,          # OFDM 基带采样率 (Hz)
    "fs_rx": 80e9,          # 重采样后采样率 (Hz)
    "n_syms_per_frame": 128,  # 每帧 OFDM 符号数（SIG.nSyms）
}


def parse_ofdm_params(params: dict) -> dict:
    """解析 OFDM 参数，统一格式。"""
    p = DEFAULT_OFDM.copy()
    if params:
        p.update(params)
    p["data_carrier_indices"] = list(range(
        p["data_carriers"][0] - 1,  # MATLAB 1-indexed → Python 0-indexed
        p["data_carriers"][1]
    ))
    p["n_data_carriers"] = len(p["data_carrier_indices"])
    return p


# ========== OFDM 解调 ==========

def ofdm_demodulate(rx_time: np.ndarray, params: dict) -> np.ndarray:
    """
    将时域 OFDM 信号解调为频域 QAM 符号。
    
    Parameters
    ----------
    rx_time : np.ndarray, complex
        时域接收信号（1D）
    params : dict
        OFDM 参数
    
    Returns
    -------
    qam_symbols : np.ndarray, complex, shape [n_symbols, n_data_carriers]
        频域 QAM 符号
    """
    p = parse_ofdm_params(params)
    
    # 1. 重采样到 OFDM 基带采样率
    if abs(p["fs_rx"] - p["fs_tx"]) > 1:
        # 从 fs_rx 重采样到 fs_tx
        num = int(round(p["fs_tx"]))
        den = int(round(p["fs_rx"]))
        # 约分
        from math import gcd
        g = gcd(num, den)
        num //= g
        den //= g
        rx_baseband = scipy_signal.resample_poly(rx_time, num, den)
    else:
        rx_baseband = rx_time
    
    # 2. 找到 OFDM 符号边界（假设帧对齐）
    # 每个 OFDM 符号长度 = n_fft + cp_len
    sym_len = p["n_fft"] + p["cp_len"]
    
    # 丢弃末尾不完整的符号
    n_full_syms = len(rx_baseband) // sym_len
    if n_full_syms == 0:
        raise ValueError(f"信号太短: {len(rx_baseband)} 样本, 需要至少 {sym_len}")
    
    rx_baseband = rx_baseband[:n_full_syms * sym_len]
    rx_2d = rx_baseband.reshape(-1, sym_len)  # [n_syms, sym_len]
    
    # 3. 去 CP + FFT
    qam_fd = np.fft.fft(rx_2d[:, p["cp_len"]:], axis=1)  # [n_syms, n_fft]
    
    # 4. 提取数据子载波
    qam_symbols = qam_fd[:, p["data_carrier_indices"]]  # [n_syms, n_data_carriers]
    
    return qam_symbols


# ========== 加载 MATLAB TX 文件 ==========

def load_matlab_tx_mat(mat_path: str) -> np.ndarray:
    """
    加载 MATLAB tx_3frame_6mod 输出的 .mat 文件。
    
    文件格式:
        sig_XXXX.mat → data_tx_all (频域 QAM 符号, [nSyms*3, nCarriers])
        sig_XXXX_frame1.mat → data_tx (频域 QAM 符号, [nSyms, nCarriers])
    
    Returns: [N, K] complex 数组, N=符号数, K=子载波数
    """
    import scipy.io as sio
    
    try:
        mat = sio.loadmat(mat_path)
    except Exception as e:
        print(f"  无法加载 {mat_path}: {e}")
        return None
    
    if "data_tx_all" in mat:
        data = mat["data_tx_all"]
        print(f"  加载 data_tx_all: shape={data.shape}")
    elif "data_tx" in mat:
        data = mat["data_tx"]
        print(f"  加载 data_tx: shape={data.shape}")
    else:
        print(f"  文件 {mat_path} 中没有 data_tx_all 或 data_tx 变量")
        print(f"  可用变量: {list(mat.keys())[:10]}")
        return None
    
    # 确保是复数
    if np.iscomplexobj(data):
        return data.astype(np.complex64)
    else:
        # 可能是两个实数通道（I/Q）拼接的
        print(f"  警告: 数据不是复数，尝试解释为 I/Q")
        print(f"  dtype={data.dtype}, shape={data.shape}")
        return None


def load_matlab_tx_time(txt_path: str, params: dict) -> np.ndarray:
    """
    加载 MATLAB TX 时域信号（.txt 文件）并解调。
    
    sig_XXXX.txt → InputFSO_all（时域, 列向量）
    """
    try:
        rx_time = np.loadtxt(txt_path, dtype=np.float64)
    except Exception as e:
        print(f"  无法加载 {txt_path}: {e}")
        return None
    
    if rx_time.ndim == 2 and rx_time.shape[1] >= 2:
        # 可能是 I/Q 双列
        rx_complex = rx_time[:, 0] + 1j * rx_time[:, 1]
    else:
        rx_complex = rx_time.ravel()
    
    print(f"  加载时域信号: {len(rx_complex)} 样本")
    
    # OFDM 解调
    qam_syms = ofdm_demodulate(rx_complex, params)
    print(f"  解调后: {qam_syms.shape[0]} 符号 × {qam_syms.shape[1]} 子载波")
    
    return qam_syms.astype(np.complex64)


# ========== 主函数 ==========

def main():
    ap = argparse.ArgumentParser(description="MATLAB OFDM 信号 → QAM 符号 NPZ")
    
    # 输入来源
    ap.add_argument("--mat_dir", help="包含 sig_XXXX.mat 的目录")
    ap.add_argument("--rx_time_dir", help="包含时域信号文件的目录")
    ap.add_argument("--single_file", help="单个 .mat 或 .txt 文件")
    
    # 调制信息
    ap.add_argument("--mod", default="16QAM", 
                    choices=["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"],
                    help="调制格式")
    
    # OFDM 参数
    ap.add_argument("--ofdm_params", help="OFDM 参数 JSON 文件（可选，默认使用 TX 参数）")
    ap.add_argument("--n_fft", type=int, default=256)
    ap.add_argument("--cp_len", type=int, default=16)
    ap.add_argument("--data_carrier_start", type=int, default=4)
    ap.add_argument("--data_carrier_end", type=int, default=126)
    ap.add_argument("--fs_tx", type=float, default=16e9)
    ap.add_argument("--fs_rx", type=float, default=80e9)
    
    # 输出
    ap.add_argument("--out", default="qam_symbols.npz", help="输出 NPZ 文件路径")
    ap.add_argument("--max_files", type=int, default=5, help="最多处理几个文件")
    
    args = ap.parse_args()
    
    # 解析 OFDM 参数
    params = {
        "n_fft": args.n_fft,
        "cp_len": args.cp_len,
        "data_carriers": [args.data_carrier_start, args.data_carrier_end],
        "fs_tx": args.fs_tx,
        "fs_rx": args.fs_rx,
    }
    if args.ofdm_params:
        with open(args.ofdm_params) as f:
            params.update(json.load(f))
    
    p = parse_ofdm_params(params)
    print(f"OFDM 参数: n_fft={p['n_fft']}, CP={p['cp_len']}, "
          f"数据子载波={p['data_carrier_indices'][0]}-{p['data_carrier_indices'][-1]} "
          f"({p['n_data_carriers']} 个)")
    
    all_qam = []
    
    # === 模式 1: 单个文件 ===
    if args.single_file:
        ext = os.path.splitext(args.single_file)[1].lower()
        if ext == ".mat":
            qam = load_matlab_tx_mat(args.single_file)
        elif ext == ".txt":
            qam = load_matlab_tx_time(args.single_file, params)
        else:
            print(f"不支持的文件格式: {ext}")
            sys.exit(1)
        
        if qam is not None:
            all_qam.append(qam.ravel())  # 展平所有符号
    
    # === 模式 2: TX 频域目录 ===
    if args.mat_dir:
        paths = sorted(glob.glob(os.path.join(args.mat_dir, "sig_*.mat")))
        # 排除 frame 文件（只取主文件）
        paths = [p for p in paths if "_frame" not in os.path.basename(p)]
        print(f"\n找到 {len(paths)} 个 TX .mat 文件")
        
        count = 0
        for path in paths[:args.max_files]:
            qam = load_matlab_tx_mat(path)
            if qam is not None:
                all_qam.append(qam.ravel())
                count += 1
        print(f"已加载 {count} 个文件")
    
    # === 模式 3: 时域信号目录 ===
    if args.rx_time_dir:
        paths = sorted(glob.glob(os.path.join(args.rx_time_dir, "sig_*.txt")))
        print(f"\n找到 {len(paths)} 个时域 .txt 文件")
        
        count = 0
        for path in paths[:args.max_files]:
            qam = load_matlab_tx_time(path, params)
            if qam is not None:
                all_qam.append(qam.ravel())
                count += 1
        print(f"已解调 {count} 个文件")
    
    # === 输出 ===
    if not all_qam:
        print("\n错误: 没有加载到任何 QAM 符号")
        sys.exit(1)
    
    combined = np.concatenate(all_qam)
    print(f"\n总计 {len(combined)} 个 QAM 符号")
    print(f"调制: {args.mod}")
    
    # 保存 NPZ
    np.savez_compressed(
        args.out,
        Y_iq=np.stack([combined.real, combined.imag], axis=0).astype(np.float32).reshape(2, 1, -1),
        mod_label=np.array([["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"].index(args.mod)]),
        mod_name=np.array([args.mod]),
    )
    print(f"已保存: {args.out}")
    print(f"\n接下来运行检测:")
    print(f"  python tools_check_uniform_qam.py --iq_file \"{args.out}\" --mod {args.mod}")


if __name__ == "__main__":
    main()
