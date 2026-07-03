#!/usr/bin/env python3
"""
preprocess_dsp_ablation.py
Python 版 DSP 消融预处理：NoEQ / Blind-like
替代 MATLAB 脚本 preprocess_6mod_dsp_ablation_lowmem.m

输出：.mat 文件，包含 rx_sc, cdm64, blind_stats 等
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.interpolate import interp1d
from pathlib import Path
import argparse
import os
import sys
import time
import traceback

# ========== OFDM 参数 ==========
N_FFT = 256
N_GUARD = 16
N_SYMS = 128
SYM_LEN = N_FFT + N_GUARD
CARRIER_LOC = slice(3, 126)  # MATLAB: 4:126 (1-indexed) → Python 0-indexed: 3:125
N_SC = 123
FS_RX = 80e9
FS_BASE = 16e9
ZEROS_HEAD = 80
FRAME_PRE_LTS = ZEROS_HEAD + N_GUARD - 5
FRAME_LEN_16 = ZEROS_HEAD + N_GUARD + 2 * N_FFT + SYM_LEN * N_SYMS
NEXT_SEARCH_BACKOFF = 800
N_FRAMES_PER_FILE = 3
CDM_BINS = 64
CDM_CLIP = 3.0

MOD_LIST = ['QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
MOD_LABEL = [0, 1, 2, 3, 4, 5]
# 默认 Dataset-A; Dataset-B 用 sub02/sub04
TURB_SUBDIRS = ['sub01', 'sub03']
TURB_NAMES = ['weak', 'strong']
TURB_LABELS = [0, 1]

# ========== LTS 构建 ==========
def build_lts_time(lts_mat_path):
    """从 LongTrainSym_ini.mat 构建时域 LTS 模板"""
    mat = sio.loadmat(lts_mat_path)
    lts_raw = mat['LongTrainSym_ini'].ravel().astype(np.complex128)
    LTS_f = np.zeros(N_FFT, dtype=np.complex128)
    LTS_f[:N_FFT] = lts_raw[:N_FFT]
    LTS_f[0] = 0
    LTS_f[N_FFT // 2] = 0
    # Mirror conjugate: MATLAB ltrs(1, n_fft/2+2:n_fft) = conj(ltrs(1, n_fft/2:-1:2))
    LTS_f[N_FFT // 2 + 1:N_FFT] = np.conj(LTS_f[N_FFT // 2 - 1:0:-1])
    lts_time = np.fft.ifft(LTS_f, n=N_FFT) * np.sqrt(N_FFT)
    return lts_time, LTS_f


# ========== 文件读取 ==========
def read_keysight_bin(bin_path):
    """读取 Keysight 示波器 .bin 文件，返回 float32 实数信号"""
    with open(bin_path, 'rb') as f:
        f.read(2)  # file_id[0]
        f.read(2)  # file_id[1]
        f.read(5 * 4)  # 5 x int32
        num_points = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.read(4)  # int32
        f.read(4)  # float32
        f.read(3 * 8)  # 3 x float64
        f.read(2 * 4)  # 2 x int32
        f.read(16)  # char
        f.read(16)  # char
        f.read(24)  # char
        f.read(16)  # char
        f.read(8)  # float64
        f.read(4)  # uint32
        f.read(4)  # int32
        f.read(2)  # int16
        bpp = np.frombuffer(f.read(2), dtype=np.int16)[0]
        f.read(4)  # int32 (buffer size)
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data


# ========== 信号处理 ==========
def resample_signal(rx, fs_from, fs_to):
    """重采样信号"""
    from math import gcd
    num = int(fs_to)
    den = int(fs_from)
    g = gcd(num, den)
    num //= g
    den //= g
    return sig.resample_poly(rx.astype(np.float64), num, den).astype(np.float32)


def find_lts_xcorr(rx, lts_time):
    """使用互相关找到 LTS 起始位置"""
    lts_t = np.concatenate([lts_time, lts_time])  # 两个 LTS 符号
    lts_len = len(lts_t)
    
    if len(rx) < lts_len * 2:
        return None, None
    
    # 互相关
    xcorr = np.abs(sig.correlate(rx, lts_t, mode='valid'))
    
    # 找峰值
    search_len = min(len(rx), 2 * FRAME_LEN_16)
    if search_len < lts_len:
        return None, None
    
    xcorr_short = xcorr[:search_len]
    peak_idx = np.argmax(xcorr_short)
    
    # LTS 起始位置 = 峰值位置 + N_FFT (第一个 LTS 符号的起始)
    lts_start = peak_idx  # 互相关峰值对应第一个 LTS 符号的起始
    
    # 帧起始位置
    frame_start = max(0, lts_start - FRAME_PRE_LTS)
    
    return lts_start, frame_start


def est_cfo_lts(rx, lts_start):
    """基于 LTS 估计 CFO"""
    if lts_start + 2 * N_FFT > len(rx):
        return 0.0
    lts1 = rx[lts_start:lts_start + N_FFT]
    lts2 = rx[lts_start + N_FFT:lts_start + 2 * N_FFT]
    cfo = np.angle(np.sum(lts1 * np.conj(lts2))) / (2 * np.pi * N_FFT)
    return float(cfo)


def est_cfo_cp(rx, lts_start, n_use):
    """基于 CP 估计 CFO"""
    sym_len = N_FFT + N_GUARD
    data_start = lts_start + 2 * N_FFT
    if data_start >= len(rx):
        return 0.0
    
    dp = rx[data_start:]
    nd = len(dp) // sym_len
    n_use = min(n_use, nd, 32)
    if n_use <= 0:
        return 0.0
    
    acc = 0.0 + 0.0j
    for k in range(n_use):
        st = k * sym_len
        if st + sym_len > len(dp):
            break
        cp = dp[st:st + N_GUARD]
        tail = dp[st + N_FFT:st + N_FFT + N_GUARD]
        acc += np.sum(np.conj(cp) * tail)
    
    if abs(acc) < 1e-12:
        return 0.0
    return float(np.angle(acc) / (2 * np.pi))


def fft_payload(dp, n_use):
    """FFT 解调 payload"""
    sym_len = N_FFT + N_GUARD
    dp = dp[:n_use * sym_len]
    dm = dp.reshape(n_use, sym_len)
    dn = dm[:, N_GUARD:]  # 去 CP
    fd = np.fft.fft(dn, n=N_FFT, axis=1) / np.sqrt(N_FFT)
    return fd


def demod_modes(rx, lts_start, lts_freq):
    """OFDM 解调：返回 eq, noeq, blindlike 三种模式"""
    remaining = len(rx) - lts_start
    if remaining < 2 * N_FFT + SYM_LEN:
        return None
    
    # CFO 估计 (LTS)
    cfo_lts = est_cfo_lts(rx, lts_start)
    
    # LTS-CFO 补偿
    n = np.arange(remaining)
    shift = np.exp(-1j * 2 * np.pi * cfo_lts * n / N_FFT)
    rx_lts = rx[lts_start:lts_start + remaining] * shift
    
    # FFT 解调 (noeq)
    dp_lts = rx_lts[2 * N_FFT:]
    nd = len(dp_lts) // SYM_LEN
    n_use = min(nd, N_SYMS)
    if n_use <= 0:
        return None
    
    fd_noeq = fft_payload(dp_lts, n_use)
    rx_sc_noeq = fd_noeq[:, CARRIER_LOC].T  # [n_sc, n_use] — MATLAB format
    
    # 信道估计 + 均衡 (eq)
    lts1c = rx_lts[:N_FFT]
    lts2c = rx_lts[N_FFT:2 * N_FFT]
    lts_avg = (lts1c + lts2c) / 2
    lts_fd = np.fft.fft(lts_avg, n=N_FFT) / np.sqrt(N_FFT)
    H = lts_fd / (lts_freq + 1e-12)
    H[np.abs(lts_freq) < 0.5] = 1.0
    fd_eq = fd_noeq / H
    rx_sc_eq = fd_eq[:, CARRIER_LOC].T  # [n_sc, n_use]
    
    # CP-CFO 估计 (blindlike)
    cfo_cp = est_cfo_cp(rx, lts_start, n_use)
    shift_cp = np.exp(-1j * 2 * np.pi * cfo_cp * n / N_FFT)
    rx_cp = rx[lts_start:lts_start + remaining] * shift_cp
    
    dp_cp = rx_cp[2 * N_FFT:]
    fd_blind = fft_payload(dp_cp, n_use)
    rx_sc_blindlike = fd_blind[:, CARRIER_LOC].T  # [n_sc, n_use]
    
    return {
        'rx_sc_eq': rx_sc_eq,
        'rx_sc_noeq': rx_sc_noeq,
        'rx_sc_blindlike': rx_sc_blindlike,
        'cfo_lts': cfo_lts,
        'cfo_cp': cfo_cp,
        'n_use': n_use,
    }


# ========== 特征提取 ==========
def make_cdm(rx_sc, nbin=64, clip_val=3.0):
    """生成星座密度图"""
    z = rx_sc.ravel()
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if len(z) == 0:
        return np.zeros((nbin, nbin), dtype=np.float32)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    zr = np.clip(z.real, -clip_val, clip_val)
    zi = np.clip(z.imag, -clip_val, clip_val)
    edges = np.linspace(-clip_val, clip_val, nbin + 1)
    H, _, _ = np.histogram2d(zi, zr, bins=[edges, edges])
    H = np.log1p(H)
    H = H / (np.max(H) + 1e-12)
    return H.astype(np.float32)


def skew_np(x):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return np.mean(((x - np.mean(x)) / (np.std(x) + 1e-12)) ** 3)


def kurt_np(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    return np.mean(((x - np.mean(x)) / (np.std(x) + 1e-12)) ** 4)


def make_blind_stats(rx_sc):
    """生成盲统计特征"""
    z = rx_sc.ravel()
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if len(z) == 0:
        return np.full(16, np.nan, dtype=np.float32)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    a = np.abs(z)
    i = z.real
    q = z.imag
    dph = np.diff(np.unwrap(np.angle(z)))
    
    if np.std(i) < 1e-12 or np.std(q) < 1e-12:
        iq_corr = 0.0
    else:
        iq_corr = float(np.corrcoef(i, q)[0, 1])
    
    stats = np.array([
        np.mean(a), np.std(a), skew_np(a), kurt_np(a),
        10 * np.log10(np.max(np.abs(z) ** 2) / (np.mean(np.abs(z) ** 2) + 1e-12)),
        np.mean(i), np.std(i), skew_np(i), kurt_np(i),
        np.mean(q), np.std(q), skew_np(q), kurt_np(q),
        np.std(dph) if len(dph) > 0 else np.nan,
        np.abs(np.mean(np.exp(1j * np.angle(z)))),
        iq_corr,
    ], dtype=np.float32)
    return stats


# ========== 主流程 ==========
def discover_jobs(rx_data_root, turb_subdirs=None, turb_names=None):
    """发现所有 .bin 文件"""
    if turb_subdirs is None:
        turb_subdirs = TURB_SUBDIRS
    if turb_names is None:
        turb_names = TURB_NAMES
    jobs = []
    for mi, mod_name in enumerate(MOD_LIST):
        for ti, turb_subdir in enumerate(turb_subdirs):
            rx_dir = Path(rx_data_root) / mod_name / turb_subdir
            if not rx_dir.exists():
                print(f'  [skip] missing {rx_dir}')
                continue
            bins = sorted(rx_dir.glob('*.bin'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
            n_total = len(bins)
            n_empty = 0
            for k, bin_file in enumerate(bins):
                if bin_file.stat().st_size == 0:
                    n_empty += 1
                    continue
                jobs.append({
                    'mod_name': mod_name,
                    'mod_label': MOD_LABEL[mi],
                    'turb_subdir': turb_subdir,
                    'turb_name': turb_names[ti],
                    'turb_label': TURB_LABELS[ti],
                    'rx_file': str(bin_file),
                    'rx_name': bin_file.name,
                    'sig_idx': k + 1,
                })
            print(f'  [discover] {mod_name} {turb_names[ti]} {n_total} files (skip {n_empty} empty), {n_total - n_empty} valid')
    return jobs


def process_file(rx, lts_time, lts_freq, cfg):
    """处理一个文件，返回每帧的解调结果"""
    # 预处理
    rx = rx.astype(np.float64)
    rx = rx - np.mean(rx)
    rx = rx / (np.std(rx) + 1e-12)
    
    # 重采样 80G -> 16G
    rx16 = resample_signal(rx, FS_RX, FS_BASE)
    rx16 = rx16 - np.mean(rx16)
    rx16 = rx16 / (np.mean(np.abs(rx16)) + 1e-12)
    
    # 扩展信号（循环拼接）
    wrap_len = min(len(rx16), 3 * FRAME_LEN_16)
    rx_full = np.concatenate([rx16, rx16[:wrap_len]])
    
    results = []
    cursor = 0
    
    for rk in range(N_FRAMES_PER_FILE):
        if cursor >= len(rx_full) - FRAME_LEN_16:
            break
        
        search_sig = rx_full[cursor:]
        lts_start, frame_start = find_lts_xcorr(search_sig, lts_time)
        
        if lts_start is None or frame_start is None:
            cursor += FRAME_LEN_16 // 2
            continue
        
        lts_abs = cursor + lts_start
        frame_abs = cursor + frame_start
        
        dem = demod_modes(rx_full, lts_abs, lts_freq)
        if dem is None:
            cursor += FRAME_LEN_16 // 2
            continue
        
        results.append({
            'frame_idx': rk,
            'lts_abs': lts_abs,
            'frame_abs': frame_abs,
            'dem': dem,
        })
        
        cursor = max(cursor + 1, frame_abs + FRAME_LEN_16 - NEXT_SEARCH_BACKOFF)
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rx-data-root', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--lts-mat', required=True)
    ap.add_argument('--modes', nargs='+', default=['noeq', 'blindlike'])
    ap.add_argument('--date-tag', default='2026.06.28')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--turb-subdirs', nargs='+', default=None,
                    help='湍流子目录名，如 sub01 sub03 (默认) 或 sub02 sub04')
    ap.add_argument('--turb-names', nargs='+', default=None,
                    help='湍流名称，如 weak strong')
    args = ap.parse_args()
    
    out_root = Path(args.out_root) / args.date_tag
    rx_data_root = Path(args.rx_data_root) / args.date_tag
    lts_time, lts_freq = build_lts_time(args.lts_mat)
    
    print(f'LTS loaded, time-domain shape: {lts_time.shape}')
    print(f'RX data root: {rx_data_root}')
    print(f'Output root: {out_root}')
    print(f'Modes: {args.modes}')
    
    jobs = discover_jobs(rx_data_root, args.turb_subdirs, args.turb_names)
    if args.max_files > 0:
        jobs = jobs[:args.max_files]
    print(f'Total jobs: {len(jobs)}')
    
    # 初始化输出目录
    for mode in args.modes:
        (out_root / mode).mkdir(parents=True, exist_ok=True)
    
    # 初始化 manifest
    manifests = {}
    for mode in args.modes:
        f = open(out_root / mode / 'manifest_all.csv', 'w', encoding='utf-8')
        f.write('global_frame_id,status,message,mod_name,mod_label,turb_name,turb_label,'
                'turb_subdir,rx_file,rx_name,sig_idx,rx_frame_idx,lts_start_abs,'
                'frame_start_abs,cfo_lts,cfo_cp,n_use,dsp_mode,'
                'amp_mean,amp_std,amp_skew,amp_kurt,papr_db,'
                'i_mean,i_std,i_skew,i_kurt,'
                'q_mean,q_std,q_skew,q_kurt,'
                'phase_diff_std,phase_concentration,iq_corr,out_mat\n')
        manifests[mode] = f
    
    frame_ok = {m: 0 for m in args.modes}
    frame_fail = {m: 0 for m in args.modes}
    gid = {m: 0 for m in args.modes}
    
    t0 = time.time()
    
    for j, job in enumerate(jobs):
        print(f'\n[{j+1}/{len(jobs)}] {job["mod_name"]} | {job["turb_name"]} | {job["rx_name"]}')
        try:
            rx = read_keysight_bin(job['rx_file'])
            results = process_file(rx, lts_time, lts_freq, {})
            
            for r in results:
                dem = r['dem']
                for mode in args.modes:
                    if mode == 'eq':
                        rx_sc = dem['rx_sc_eq']
                    elif mode == 'noeq':
                        rx_sc = dem['rx_sc_noeq']
                    elif mode == 'blindlike':
                        rx_sc = dem['rx_sc_blindlike']
                    else:
                        continue
                    
                    # 构建输出路径
                    file_base = Path(job['rx_name']).stem
                    safe_name = f"{job['mod_name']}_sig{job['sig_idx']:04d}_{file_base}"
                    out_dir = out_root / mode / job['mod_name'] / job['turb_name'] / safe_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    mat_path = out_dir / f'frame_{r["frame_idx"]+1:02d}.mat'
                    
                    # 计算特征
                    cdm64 = make_cdm(rx_sc, CDM_BINS, CDM_CLIP)
                    blind_stats = make_blind_stats(rx_sc)
                    
                    # 保存
                    sample = {
                        'rx_sc': rx_sc.astype(np.complex64),
                        'cdm64': cdm64,
                        'blind_stats': blind_stats,
                        'mod_name': job['mod_name'],
                        'mod_label': job['mod_label'],
                        'turb_name': job['turb_name'],
                        'turb_label': job['turb_label'],
                        'turb_subdir': job['turb_subdir'],
                        'rx_file': job['rx_file'],
                        'rx_name': job['rx_name'],
                        'sig_idx': job['sig_idx'],
                        'rx_frame_idx': r['frame_idx'] + 1,
                        'lts_start_abs': r['lts_abs'],
                        'frame_start_abs': r['frame_abs'],
                        'dsp_mode': mode,
                        'cfo_lts': dem['cfo_lts'],
                        'cfo_cp': dem['cfo_cp'],
                        'n_use': dem['n_use'],
                        'snr_sc_db': np.full((N_SC,), np.nan, dtype=np.float32),
                        'snr_frame_db': np.nan,
                    }
                    sio.savemat(str(mat_path), sample, do_compression=False)
                    
                    gid[mode] += 1
                    frame_ok[mode] += 1
                    
                    # 写入 manifest
                    bs = blind_stats
                    if len(bs) < 16:
                        bs = np.pad(bs, (0, 16 - len(bs)), constant_values=np.nan)
                    manifests[mode].write(
                        f'{gid[mode]},ok,,{job["mod_name"]},{job["mod_label"]},'
                        f'{job["turb_name"]},{job["turb_label"]},'
                        f'{job["turb_subdir"]},{job["rx_file"]},{job["rx_name"]},'
                        f'{job["sig_idx"]},{r["frame_idx"]+1},'
                        f'{r["lts_abs"]},{r["frame_abs"]},'
                        f'{dem["cfo_lts"]:.12g},{dem["cfo_cp"]:.12g},{dem["n_use"]},'
                        f'{mode},'
                        f'{bs[0]:.12g},{bs[1]:.12g},{bs[2]:.12g},{bs[3]:.12g},{bs[4]:.12g},'
                        f'{bs[5]:.12g},{bs[6]:.12g},{bs[7]:.12g},{bs[8]:.12g},'
                        f'{bs[9]:.12g},{bs[10]:.12g},{bs[11]:.12g},{bs[12]:.12g},'
                        f'{bs[13]:.12g},{bs[14]:.12g},{bs[15]:.12g},'
                        f'{mat_path}\n'
                    )
        except Exception as e:
            print(f'  ERROR: {e}')
            traceback.print_exc()
            for mode in args.modes:
                frame_fail[mode] += N_FRAMES_PER_FILE
        
        if (j + 1) % 5 == 0 or j == len(jobs) - 1:
            elapsed = time.time() - t0
            remaining = elapsed / (j + 1) * (len(jobs) - j - 1) if j < len(jobs) - 1 else 0
            print(f'  [progress] {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min remaining')
    
    # 关闭 manifest
    for f in manifests.values():
        f.close()
    
    # 写摘要
    for mode in args.modes:
        with open(out_root / mode / 'run_summary_dsp_ablation.txt', 'w') as f:
            f.write(f'mode={mode}\n')
            f.write(f'processed_frames={frame_ok[mode]}\n')
            f.write(f'failed_frames={frame_fail[mode]}\n')
            f.write(f'time_min={(time.time()-t0)/60:.3f}\n')
    
    print(f'\nDone. Total {(time.time()-t0)/60:.1f} min')
    for mode in args.modes:
        print(f'  {mode}: ok={frame_ok[mode]}, fail={frame_fail[mode]}')


if __name__ == '__main__':
    main()