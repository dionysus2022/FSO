# -*- coding: utf-8 -*-
"""
train_mrf_lowmem.py

真实实验湍流数据 + 多表征融合 + 小样本鲁棒识别训练脚本

适用输入：
    MATLAB 低内存预处理脚本 batch_preprocess_6mod_rx_lowmem.m 的输出目录：

    preprocessed_uniform_qam_rx_lowmem/2026.06.28/
        QPSK/weak/frame_xx.mat
        QPSK/strong/frame_xx.mat
        ...
        manifest_all.csv

核心设计：
    1) 按湍流条件独立训练：
        weak   单独训练、验证、测试
        strong 单独训练、验证、测试

    2) 按 rx_file 文件级划分数据：
        同一个 .bin 里面的 3 帧不会同时进入训练集和测试集，避免数据泄漏。

    3) 多表征输入：
        cdm64        星座密度图，1 × 64 × 64
        rx_sc-derived IQ 序列，默认 5 × L，包含 I/Q/幅度/sin相位/cos相位
        blind_stats  盲统计特征

    4) 运行时间控制：
        默认 model=cdm_stats，使用星座密度图 + 盲统计特征，速度快，适合先跑基线；
        最终论文模型可用 model=fusion，使用 CDM + IQ + statistics 三分支融合。

推荐运行：
    快速基线：
        python train_mrf_lowmem.py --data-root "D:\\...\\preprocessed_uniform_qam_rx_lowmem\\2026.06.28" --model cdm_stats

    最终融合：
        python train_mrf_lowmem.py --data-root "D:\\...\\preprocessed_uniform_qam_rx_lowmem\\2026.06.28" --model fusion --epochs 60

输出：
    recognition_mrf_lowmem/
        weak_cdm_stats/
            best_model.pt
            metrics.json
            confusion_matrix.png
            training_curve.png
            predictions.csv
            split.csv
        strong_cdm_stats/
            ...
        summary_all_turbulence.csv
"""

import os
import json
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import scipy.io as sio
except Exception as e:
    raise RuntimeError("需要 scipy 读取 .mat 文件，请先安装：pip install scipy") from e

try:
    import h5py
    H5PY_OK = True
except Exception:
    H5PY_OK = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
MOD_TO_LABEL = {m: i for i, m in enumerate(MOD_NAMES)}
LABEL_TO_MOD = {i: m for i, m in enumerate(MOD_NAMES)}


# =========================
# 通用工具
# =========================

def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def read_mat_scipy(path: Path) -> dict:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def read_mat_h5py(path: Path) -> dict:
    """
    兜底读取 v7.3 MAT。你的 frame_*.mat 默认是 -v7，一般不会走到这里。
    """
    if not H5PY_OK:
        raise RuntimeError("h5py 未安装，无法读取 v7.3 MAT。")

    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            obj = f[k]
            arr = np.array(obj)
            arr = arr.T
            out[k] = arr
    return out


def load_mat(path: Path) -> dict:
    try:
        return read_mat_scipy(path)
    except NotImplementedError:
        return read_mat_h5py(path)


def matlab_str_to_py(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "S"):
            return "".join(x.astype(str).ravel().tolist())
        if x.size == 1:
            return str(x.item())
    return str(x)


def load_sample_from_mat(path: Path, use_iq: bool = True, use_radial: bool = False,
                         use_angle: bool = False, use_polar: bool = False) -> dict:
    mat = load_mat(path)

    if "rx_sc" not in mat:
        raise KeyError(f"{path} 里面没有 rx_sc。请确认 MATLAB 低内存预处理是否成功。")

    rx_sc = np.asarray(mat["rx_sc"])
    rx_sc = np.squeeze(rx_sc)

    # scipy 读取 MATLAB complex single 通常会直接得到 complex64/complex128
    if not np.iscomplexobj(rx_sc):
        # 兜底：如果是 [real, imag] 形式
        if rx_sc.ndim >= 3 and rx_sc.shape[0] == 2:
            rx_sc = rx_sc[0] + 1j * rx_sc[1]
        elif rx_sc.ndim >= 3 and rx_sc.shape[-1] == 2:
            rx_sc = rx_sc[..., 0] + 1j * rx_sc[..., 1]
        else:
            rx_sc = rx_sc.astype(np.float32).astype(np.complex64)

    rx_sc = rx_sc.astype(np.complex64)

    if "cdm64" in mat:
        cdm = np.asarray(mat["cdm64"], dtype=np.float32)
        cdm = np.squeeze(cdm)
    else:
        cdm = make_cdm_from_rx_sc(rx_sc, nbin=64, clip_val=3.0)

    if cdm.ndim != 2:
        cdm = np.reshape(cdm, (64, 64)).astype(np.float32)
    cdm = cdm.astype(np.float32)
    cdm = np.nan_to_num(cdm, nan=0.0, posinf=0.0, neginf=0.0)
    cdm = cdm[None, :, :]  # [1,64,64]

    if "blind_stats" in mat:
        stats = np.asarray(mat["blind_stats"], dtype=np.float32).reshape(-1)
    else:
        stats = make_blind_stats(rx_sc)

    stats = np.asarray(stats, dtype=np.float32).reshape(-1)
    if stats.size < 16:
        stats = np.pad(stats, (0, 16 - stats.size), constant_values=np.nan)
    stats = stats[:16]

    # 径向幅度直方图（可选）：拼接在 stats 之后，由 StatsBranch MLP 统一处理
    if use_radial:
        radial = make_radial_hist(rx_sc, n_bins=32, clip_val=3.0)
        stats = np.concatenate([stats, radial], axis=0)

    # 相位角直方图（可选）
    if use_angle:
        angle = make_angle_hist(rx_sc, n_bins=32, clip_val=3.0)
        stats = np.concatenate([stats, angle], axis=0)

    # 极坐标 2D 直方图（可选）
    if use_polar:
        polar = make_polar_hist(rx_sc, n_r=8, n_theta=8, clip_val=3.0)
        stats = np.concatenate([stats, polar], axis=0)

    if "mod_label" in mat:
        y = int(np.asarray(mat["mod_label"]).reshape(-1)[0])
    elif "mod_name" in mat:
        y = MOD_TO_LABEL.get(matlab_str_to_py(mat["mod_name"]), -1)
    else:
        y = -1

    if "turb_label" in mat:
        turb_label = int(np.asarray(mat["turb_label"]).reshape(-1)[0])
    else:
        turb_label = -1

    if "turb_name" in mat:
        turb_name = matlab_str_to_py(mat["turb_name"])
    else:
        turb_name = ""

    if use_iq:
        iq = rx_sc_to_iq_channels(rx_sc)
    else:
        iq = np.zeros((1, 1), dtype=np.float32)

    return {
        "cdm": cdm.astype(np.float32),
        "iq": iq.astype(np.float32),
        "stats": stats.astype(np.float32),
        "label": y,
        "turb_label": turb_label,
        "turb_name": turb_name,
    }


def rx_sc_to_iq_channels(rx_sc: np.ndarray) -> np.ndarray:
    """
    rx_sc: complex [n_sc, n_sym]
    输出: [5, L]，通道为 I, Q, Amp, sin(Phase), cos(Phase)
    """
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
        return np.zeros((5, 1), dtype=np.float32)

    z = z - np.mean(z)
    power = np.mean(np.abs(z) ** 2)
    z = z / np.sqrt(power + 1e-12)

    i = z.real.astype(np.float32)
    q = z.imag.astype(np.float32)
    amp = np.abs(z).astype(np.float32)
    ph = np.angle(z).astype(np.float32)
    sinp = np.sin(ph).astype(np.float32)
    cosp = np.cos(ph).astype(np.float32)

    x = np.stack([i, q, amp, sinp, cosp], axis=0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_cdm_from_rx_sc(rx_sc: np.ndarray, nbin: int = 64, clip_val: float = 3.0) -> np.ndarray:
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
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


def make_radial_hist(rx_sc: np.ndarray, n_bins: int = 32, clip_val: float = 3.0) -> np.ndarray:
    """
    径向幅度直方图：histogram(|z|)，刻画星座点半径层级分布。
    对高阶 QAM（64/128/256）的半径环结构敏感。
    归一化方式与 make_cdm / make_blind_stats 一致：零均值、单位 RMS。
    输出：n_bins 维 L1 归一化直方图。
    """
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
        return np.zeros((n_bins,), dtype=np.float32)

    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    amp = np.clip(np.abs(z), 0.0, clip_val)
    edges = np.linspace(0.0, clip_val, n_bins + 1)
    h, _ = np.histogram(amp, bins=edges)
    h = h.astype(np.float32)
    h = h / (np.sum(h) + 1e-12)  # L1 归一化
    return h


def make_angle_hist(rx_sc: np.ndarray, n_bins: int = 32, clip_val: float = 3.0) -> np.ndarray:
    """
    相位角直方图：histogram(angle(z_aligned))，刻画星座点角度分布。
    使用 4 次方盲相位对齐（对方形 QAM 旋转不变）：
        phi = angle(mean(z^4)) / 4 ; z_aligned = z * exp(-j*phi)
    输出：n_bins 维 L1 归一化直方图，范围 [-pi, pi]。
    """
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
        return np.zeros((n_bins,), dtype=np.float32)

    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    # 4 次方盲相位对齐（对方形 QAM 旋转不变）
    z4 = z ** 4
    if np.abs(np.mean(z4)) > 1e-8:
        phi = float(np.angle(np.mean(z4))) / 4.0
        z = z * np.exp(-1j * phi)

    ang = np.angle(z)  # [-pi, pi]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    h, _ = np.histogram(ang, bins=edges)
    h = h.astype(np.float32)
    h = h / (np.sum(h) + 1e-12)
    return h


def make_polar_hist(rx_sc: np.ndarray, n_r: int = 8, n_theta: int = 8, clip_val: float = 3.0) -> np.ndarray:
    """
    极坐标 2D 直方图：histogram2d(|z|, angle(z_aligned))，联合刻画半径×角度结构。
    与 CDM（笛卡尔 2D）互补，对环形/辐条结构更敏感。
    使用 4 次方盲相位对齐。输出：n_r × n_theta 维 L1 归一化直方图（展平）。
    """
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
        return np.zeros((n_r * n_theta,), dtype=np.float32)

    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    # 4 次方盲相位对齐
    z4 = z ** 4
    if np.abs(np.mean(z4)) > 1e-8:
        phi = float(np.angle(np.mean(z4))) / 4.0
        z = z * np.exp(-1j * phi)

    amp = np.clip(np.abs(z), 0.0, clip_val)
    ang = np.angle(z)  # [-pi, pi]

    r_edges = np.linspace(0.0, clip_val, n_r + 1)
    t_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    H, _, _ = np.histogram2d(amp, ang, bins=[r_edges, t_edges])
    H = H.astype(np.float32).flatten()
    H = H / (np.sum(H) + 1e-12)
    return H


def skew_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return np.mean(((x - mu) / sd) ** 3)


def kurt_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return np.mean(((x - mu) / sd) ** 4)


def make_blind_stats(rx_sc: np.ndarray) -> np.ndarray:
    z = rx_sc.reshape(-1).astype(np.complex64)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[finite]
    if z.size == 0:
        return np.full((16,), np.nan, dtype=np.float32)

    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    amp = np.abs(z)
    i = z.real
    q = z.imag
    ph = np.unwrap(np.angle(z))
    dph = np.diff(ph)

    if np.std(i) < 1e-12 or np.std(q) < 1e-12:
        iq_corr = 0.0
    else:
        iq_corr = float(np.corrcoef(i, q)[0, 1])

    stats = np.array([
        np.mean(amp),
        np.std(amp),
        skew_np(amp),
        kurt_np(amp),
        10 * np.log10(np.max(np.abs(z) ** 2) / (np.mean(np.abs(z) ** 2) + 1e-12)),
        np.mean(i),
        np.std(i),
        skew_np(i),
        kurt_np(i),
        np.mean(q),
        np.std(q),
        skew_np(q),
        kurt_np(q),
        np.std(dph) if dph.size > 0 else np.nan,
        np.abs(np.mean(np.exp(1j * np.angle(z)))),
        iq_corr,
    ], dtype=np.float32)

    return stats


# =========================
# 数据划分与缓存
# =========================

def load_manifest(data_root: Path) -> pd.DataFrame:
    manifest = data_root / "manifest_all.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"找不到 manifest_all.csv：{manifest}")

    df = pd.read_csv(manifest)
    df = df[df["status"].astype(str) == "ok"].copy()

    if "out_mat" not in df.columns:
        raise KeyError("manifest_all.csv 中没有 out_mat 列。")

    df["out_mat"] = df["out_mat"].astype(str)
    df = df[df["out_mat"].str.len() > 0].copy()

    # 路径存在性检查；Windows 下运行时应该直接存在
    exists_mask = df["out_mat"].apply(lambda p: Path(p).exists())
    if exists_mask.sum() == 0:
        raise FileNotFoundError(
            "manifest 中的 out_mat 路径都不存在。请确认在 Windows/PyCharm 下运行，且路径未移动。"
        )

    if exists_mask.sum() < len(df):
        print(f"[警告] 有 {len(df)-exists_mask.sum()} 个 out_mat 不存在，将跳过。")
        df = df[exists_mask].copy()

    df["mod_label"] = df["mod_label"].astype(int)
    df["turb_label"] = df["turb_label"].astype(int)
    df["turb_name"] = df["turb_name"].astype(str)
    df["rx_file"] = df["rx_file"].astype(str)
    df["rx_frame_idx"] = df["rx_frame_idx"].astype(int)
    return df


def group_kfold_split_by_rx_file(df_turb: pd.DataFrame, fold: int = 0, n_folds: int = 5, seed: int = 2026) -> pd.DataFrame:
    """
    文件级 K-Fold 交叉验证划分。
    每个调制格式内部按 rx_file 做 K-Fold，每折测试 1/K 的文件，其余再分 train/val。
    典型每类 25 个 bin 文件，5-fold：每折测试 5 files = 15 frames。
    """
    rng = np.random.default_rng(seed)
    split_rows = []

    for y in sorted(df_turb["mod_label"].unique()):
        d = df_turb[df_turb["mod_label"] == y].copy()
        groups = np.array(sorted(d["rx_file"].unique()))
        rng.shuffle(groups)

        n = len(groups)
        if n < n_folds:
            raise RuntimeError(f"类别 {y} 的 rx_file 数量 {n} 少于 n_folds {n_folds}")

        # Split into n_folds chunks
        fold_chunks = np.array_split(groups, n_folds)
        test_g = set(fold_chunks[fold])
        remaining = [g for i, chunk in enumerate(fold_chunks) if i != fold for g in chunk]
        remaining = np.array(remaining)

        # From remaining, take ~15% as val
        n_val = max(1, int(round(0.15 * len(remaining))))
        rng.shuffle(remaining)
        val_g = set(remaining[:n_val])
        train_g = set(remaining[n_val:])

        for idx, row in d.iterrows():
            g = row["rx_file"]
            if g in train_g:
                split = "train"
            elif g in val_g:
                split = "val"
            elif g in test_g:
                split = "test"
            else:
                split = "train"  # fallback
            rr = row.to_dict()
            rr["split"] = split
            split_rows.append(rr)

    split_df = pd.DataFrame(split_rows)
    return split_df


def group_split_by_rx_file(df_turb: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """
    每个调制格式内部按 rx_file 文件级划分。

    - 若 df_turb 不含 dataset_source 列：按原逻辑对全部 rx_file 划分。
      典型每类 25 个 bin 文件：train 17 / val 3 / test 5。
    - 若 df_turb 含 dataset_source 列（如合并 Dataset-A + Dataset-B）：
      在每个 dataset_source 内部各自按比例划分，再合并。
      保证每个 split 都包含来自不同采集批次的样本，避免域偏移。
      典型 A 25 + B 20 = 45 bin：train A17+B14=31 / val A3+B2=5 / test A5+B4=9。
    """
    rng = np.random.default_rng(seed)
    split_rows = []

    has_source = "dataset_source" in df_turb.columns

    # 决定 n_test/n_val 的目标比例
    pct_test = 0.20
    pct_val = 0.12  # 取整后约 10%

    for y in sorted(df_turb["mod_label"].unique()):
        d = df_turb[df_turb["mod_label"] == y].copy()

        # 按 dataset_source 分组（若无该列则视为单一组）
        if has_source:
            sources = sorted(d["dataset_source"].astype(str).unique())
        else:
            sources = ["__all__"]

        train_g, val_g, test_g = set(), set(), set()

        for src in sources:
            if has_source:
                d_src = d[d["dataset_source"].astype(str) == src]
            else:
                d_src = d
            groups = np.array(sorted(d_src["rx_file"].unique()))
            rng.shuffle(groups)

            n = len(groups)
            if n < 3:
                # 数据太少的来源至少留 1 个给 val/test
                if n == 2:
                    _tr, _va, _te = 1, 0, 1
                elif n == 1:
                    _tr, _va, _te = 1, 0, 0
                else:
                    _tr, _va, _te = 0, 0, 0
            else:
                _te = max(1, int(round(pct_test * n)))
                _va = max(1, int(round(pct_val * n)))
                _tr = n - _va - _te
                if _tr < 1:
                    _tr = max(1, n - 2)
                    _va = 1 if n >= 2 else 0
                    _te = n - _tr - _va

            train_g.update(groups[:_tr])
            val_g.update(groups[_tr:_tr + _va])
            test_g.update(groups[_tr + _va:_tr + _va + _te])

        for idx, row in d.iterrows():
            g = row["rx_file"]
            if g in train_g:
                split = "train"
            elif g in val_g:
                split = "val"
            elif g in test_g:
                split = "test"
            else:
                split = "test"
            rr = row.to_dict()
            rr["split"] = split
            split_rows.append(rr)

    split_df = pd.DataFrame(split_rows)
    return split_df


def build_cache_for_turbulence(
    split_df: pd.DataFrame,
    cache_path: Path,
    model_mode: str,
    rebuild_cache: bool = False,
) -> Dict[str, np.ndarray]:
    """
    每个湍流独立缓存，避免每次训练反复读取 .mat。
    """
    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    use_iq = model_mode in ("iq", "fusion")
    use_radial = model_mode in ("cdm_stats_radial", "cdm_stats_radial_angle", "cdm_stats_rap")
    use_angle = model_mode in ("cdm_stats_radial_angle", "cdm_stats_rap")
    use_polar = model_mode in ("cdm_stats_rap",)

    cdm_list, iq_list, stats_list, y_list, split_list, path_list, rx_file_list = [], [], [], [], [], [], []

    t0 = time.time()
    n = len(split_df)
    for i, (_, row) in enumerate(split_df.iterrows(), start=1):
        p = Path(row["out_mat"])
        try:
            s = load_sample_from_mat(p, use_iq=use_iq, use_radial=use_radial,
                                     use_angle=use_angle, use_polar=use_polar)
        except Exception as e:
            print(f"[跳过] 读取失败：{p} | {e}")
            continue

        y = int(row["mod_label"])
        if y < 0 or y >= len(MOD_NAMES):
            continue

        cdm_list.append(s["cdm"])
        if use_iq:
            iq_list.append(s["iq"])
        stats_list.append(s["stats"])
        y_list.append(y)
        split_list.append(str(row["split"]))
        path_list.append(str(p))
        rx_file_list.append(str(row["rx_file"]))

        if i % 100 == 0 or i == n:
            print(f"  cache reading {i}/{n} | elapsed {time.time()-t0:.1f}s")

    if len(y_list) == 0:
        raise RuntimeError("没有成功读取任何样本。")

    X_cdm = np.stack(cdm_list, axis=0).astype(np.float32)
    X_stats = np.stack(stats_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    split = np.asarray(split_list, dtype=object)
    paths = np.asarray(path_list, dtype=object)
    rx_files = np.asarray(rx_file_list, dtype=object)

    if use_iq:
        # 所有帧的 OFDM 维度应该一致；如果不一致，截断到最短长度
        min_len = min(x.shape[-1] for x in iq_list)
        X_iq = np.stack([x[:, :min_len] for x in iq_list], axis=0).astype(np.float32)
    else:
        X_iq = np.zeros((len(y), 1, 1), dtype=np.float32)

    np.savez_compressed(
        cache_path,
        X_cdm=X_cdm,
        X_iq=X_iq,
        X_stats=X_stats,
        y=y,
        split=split,
        paths=paths,
        rx_files=rx_files,
    )

    return {
        "X_cdm": X_cdm,
        "X_iq": X_iq,
        "X_stats": X_stats,
        "y": y,
        "split": split,
        "paths": paths,
        "rx_files": rx_files,
    }


def normalize_stats_by_train(X_stats: np.ndarray, split: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X_stats.copy().astype(np.float32)
    train_mask = split == "train"

    train = X[train_mask]
    col_mean = np.nanmean(train, axis=0)
    col_std = np.nanstd(train, axis=0)

    col_mean = np.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
    col_std = np.nan_to_num(col_std, nan=1.0, posinf=1.0, neginf=1.0)
    col_std[col_std < 1e-6] = 1.0

    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])

    X = (X - col_mean) / col_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), col_mean.astype(np.float32), col_std.astype(np.float32)


class MultiRepDataset(Dataset):
    def __init__(
        self,
        X_cdm: np.ndarray,
        X_iq: np.ndarray,
        X_stats: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ):
        self.X_cdm = X_cdm
        self.X_iq = X_iq
        self.X_stats = X_stats
        self.y = y
        self.indices = indices.astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.from_numpy(self.X_cdm[i]),
            torch.from_numpy(self.X_iq[i]),
            torch.from_numpy(self.X_stats[i]),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# =========================
# 模型
# =========================

class CDMBranch(nn.Module):
    def __init__(self, out_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 8

            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class IQBranch(nn.Module):
    def __init__(self, in_ch: int = 5, out_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),

            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(inplace=True),

            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class StatsBranch(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiRepNet(nn.Module):
    def __init__(
        self,
        model_mode: str,
        num_classes: int = 6,
        iq_in_ch: int = 5,
        stats_dim: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.model_mode = model_mode

        self.use_cdm = model_mode in ("cdm", "cdm_stats", "cdm_stats_radial",
                                      "cdm_stats_radial_angle", "cdm_stats_rap", "fusion")
        self.use_iq = model_mode in ("iq", "fusion")
        self.use_stats = model_mode in ("stats", "cdm_stats", "cdm_stats_radial",
                                        "cdm_stats_radial_angle", "cdm_stats_rap", "fusion")

        feat_dim = 0

        if self.use_cdm:
            self.cdm_branch = CDMBranch(out_dim=64, dropout=dropout * 0.5)
            feat_dim += 64
        else:
            self.cdm_branch = None

        if self.use_iq:
            self.iq_branch = IQBranch(in_ch=iq_in_ch, out_dim=96, dropout=dropout * 0.5)
            feat_dim += 96
        else:
            self.iq_branch = None

        if self.use_stats:
            self.stats_branch = StatsBranch(in_dim=stats_dim, out_dim=32, dropout=dropout * 0.5)
            feat_dim += 32
        else:
            self.stats_branch = None

        if feat_dim == 0:
            raise ValueError(f"未知 model_mode: {model_mode}")

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, cdm, iq, stats):
        feats = []
        if self.use_cdm:
            feats.append(self.cdm_branch(cdm))
        if self.use_iq:
            feats.append(self.iq_branch(iq))
        if self.use_stats:
            feats.append(self.stats_branch(stats))
        h = torch.cat(feats, dim=1)
        return self.classifier(h)


# =========================
# 训练与评估
# =========================

def class_weights_from_y(y: np.ndarray, indices: np.ndarray, num_classes: int = 6) -> torch.Tensor:
    yy = y[indices]
    counts = np.bincount(yy, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for cdm, iq, stats, y in loader:
        cdm = cdm.to(device, non_blocking=True)
        iq = iq.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        logits = model(cdm, iq, stats)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())

    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def simple_confusion_matrix(y_true, y_pred, num_classes=6):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        if 0 <= int(a) < num_classes and 0 <= int(b) < num_classes:
            cm[int(a), int(b)] += 1
    return cm


def save_confusion_matrix(cm, labels, path: Path, title: str):
    if not MPL_OK:
        return
    plt.figure(figsize=(7.0, 6.0))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_training_curve(history: List[dict], path: Path):
    if not MPL_OK:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_acc = [h["train_acc"] for h in history]

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, train_acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate_split(model, loader, device, out_dir: Path, split_name: str, prefix: str) -> dict:
    y_true, y_pred, prob = predict_loader(model, loader, device)

    acc = float((y_true == y_pred).mean())
    if SKLEARN_OK:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
        report = classification_report(
            y_true, y_pred,
            labels=list(range(6)),
            target_names=MOD_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
    else:
        bal_acc = acc
        f1_macro = acc
        cm = simple_confusion_matrix(y_true, y_pred, 6)
        report = {}

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [LABEL_TO_MOD[int(x)] for x in y_true],
        "pred_name": [LABEL_TO_MOD[int(x)] for x in y_pred],
    })
    for k, name in enumerate(MOD_NAMES):
        pred_df[f"prob_{name}"] = prob[:, k]
    pred_df.to_csv(out_dir / f"{prefix}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    if split_name == "test":
        save_confusion_matrix(
            cm,
            MOD_NAMES,
            out_dir / "confusion_matrix.png",
            title=f"{prefix} test confusion matrix",
        )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "num_samples": int(len(y_true)),
        "classification_report": report,
    }


def train_one_turbulence(df: pd.DataFrame, turb_name: str, args, device) -> dict:
    out_dir = Path(args.out) / f"{turb_name}_{args.model}"
    ensure_dir(out_dir)

    print("\n" + "=" * 90)
    print(f"[{now()}] Turbulence = {turb_name} | model = {args.model}")
    print("=" * 90)

    df_turb = df[df["turb_name"].astype(str) == turb_name].copy()
    if len(df_turb) == 0:
        raise RuntimeError(f"没有找到 turb_name={turb_name} 的样本。")

    print("样本统计：")
    print(df_turb.groupby(["mod_name", "mod_label"]).size())

    if getattr(args, 'fold', -1) >= 0:
        split_df = group_kfold_split_by_rx_file(df_turb, fold=args.fold, n_folds=args.n_folds, seed=args.seed)
        print(f"\n[K-Fold] fold {args.fold}/{args.n_folds}")
    else:
        split_df = group_split_by_rx_file(df_turb, seed=args.seed)
    split_df.to_csv(out_dir / "split.csv", index=False, encoding="utf-8-sig")

    print("\n划分统计：")
    print(split_df.groupby(["split", "mod_name"]).size().unstack(fill_value=0))

    if getattr(args, 'fold', -1) >= 0:
        cache_path = out_dir / f"cache_{turb_name}_{args.model}_fold{args.fold}.npz"
    else:
        cache_path = out_dir / f"cache_{turb_name}_{args.model}.npz"
    data = build_cache_for_turbulence(
        split_df,
        cache_path=cache_path,
        model_mode=args.model,
        rebuild_cache=args.rebuild_cache,
    )

    X_cdm = data["X_cdm"].astype(np.float32)
    X_iq = data["X_iq"].astype(np.float32)
    X_stats = data["X_stats"].astype(np.float32)
    y = data["y"].astype(np.int64)
    split = data["split"].astype(str)

    X_stats, stats_mean, stats_std = normalize_stats_by_train(X_stats, split)

    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]

    print(f"\n缓存数据形状：")
    print(f"  X_cdm   : {X_cdm.shape}")
    print(f"  X_iq    : {X_iq.shape}")
    print(f"  X_stats : {X_stats.shape}")
    print(f"  train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    train_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, idx_train)
    val_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, idx_val)
    test_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, idx_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    iq_in_ch = int(X_iq.shape[1])
    stats_dim = int(X_stats.shape[1])

    model = MultiRepNet(
        model_mode=args.model,
        num_classes=6,
        iq_in_ch=iq_in_ch,
        stats_dim=stats_dim,
        dropout=args.dropout,
    ).to(device)

    weights = class_weights_from_y(y, idx_train, 6).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []
    ckpt_path = out_dir / "best_model.pt"

    t_train = time.time()
    print(f"\n开始训练：device={device}, amp={args.amp}, epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for cdm, iq, stats, yy in train_loader:
            cdm = cdm.to(device, non_blocking=True)
            iq = iq.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(cdm, iq, stats)
                loss = criterion(logits, yy)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = yy.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n_sum += bs

        scheduler.step()

        train_loss = loss_sum / max(1, n_sum)
        ytr, ptr, _ = predict_loader(model, train_loader, device)
        yv, pv, _ = predict_loader(model, val_loader, device)

        train_acc = float((ytr == ptr).mean())
        val_acc = float((yv == pv).mean())
        lr_now = optimizer.param_groups[0]["lr"]

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": lr_now,
        }
        history.append(rec)

        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | lr={lr_now:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_mode": args.model,
                    "mod_names": MOD_NAMES,
                    "turb_name": turb_name,
                    "stats_mean": stats_mean,
                    "stats_std": stats_std,
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "iq_in_ch": iq_in_ch,
                    "stats_dim": stats_dim,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"  Early stopping: best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
            break

    train_seconds = time.time() - t_train

    # 读取最佳模型评估
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    metrics = {
        "train": evaluate_split(model, train_loader, device, out_dir, "train", f"{turb_name}_{args.model}"),
        "val": evaluate_split(model, val_loader, device, out_dir, "val", f"{turb_name}_{args.model}"),
        "test": evaluate_split(model, test_loader, device, out_dir, "test", f"{turb_name}_{args.model}"),
    }

    save_training_curve(history, out_dir / "training_curve.png")

    result = {
        "turb_name": turb_name,
        "model": args.model,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "train_seconds": float(train_seconds),
        "metrics": metrics,
        "history": history,
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt_path),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"\n[结果] {turb_name} | model={args.model} | "
        f"test_acc={metrics['test']['accuracy']:.4f} | "
        f"test_f1={metrics['test']['f1_macro']:.4f} | "
        f"time={train_seconds/60:.2f} min"
    )
    print(f"[保存] {out_dir}")

    del model
    torch.cuda.empty_cache()

    return result


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="真实实验湍流数据多表征融合调制识别训练"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\2026.06.28",
        help="MATLAB 低内存预处理输出目录，里面应有 manifest_all.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem",
        help="训练结果输出目录",
    )

    parser.add_argument(
        "--turbs",
        type=str,
        nargs="*",
        default=["weak", "strong"],
        help="要训练的湍流条件，默认 weak strong",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cdm_stats",
        choices=["cdm", "iq", "stats", "cdm_stats", "cdm_stats_radial",
                 "cdm_stats_radial_angle", "cdm_stats_rap", "fusion"],
        help="模型模式：cdm_stats最快；cdm_stats_radial增加径向直方图；cdm_stats_radial_angle加角度直方图；cdm_stats_rap加极坐标2D直方图",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=3.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows建议0")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="关闭CUDA AMP混合精度")

    parser.add_argument("--fold", type=int, default=-1, help="K-Fold 折号 (0-indexed)，设为 -1 禁用 K-Fold")
    parser.add_argument("--n-folds", type=int, default=5, help="K-Fold 总折数")

    return parser.parse_args()


def main():
    args = parse_args()
    args.amp = not args.no_amp

    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    ensure_dir(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("Experimental turbulence modulation recognition - MRF lowmem training")
    print("=" * 90)
    print(f"Time      : {now()}")
    print(f"Data root : {data_root}")
    print(f"Out root  : {out_root}")
    print(f"Model     : {args.model}")
    print(f"Device    : {device}")
    if device.type == "cuda":
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        print(f"CUDA      : {torch.version.cuda}")
    print("=" * 90)

    df = load_manifest(data_root)

    print("\n总样本统计：")
    print(df.groupby(["turb_name", "mod_name"]).size().unstack(fill_value=0))

    all_results = []
    t0 = time.time()

    for turb in args.turbs:
        r = train_one_turbulence(df, turb, args, device)
        all_results.append(r)

    total_seconds = time.time() - t0

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "turb_name": r["turb_name"],
            "model": r["model"],
            "best_epoch": r["best_epoch"],
            "best_val_acc": r["best_val_acc"],
            "train_seconds": r["train_seconds"],
            "test_accuracy": r["metrics"]["test"]["accuracy"],
            "test_balanced_accuracy": r["metrics"]["test"]["balanced_accuracy"],
            "test_f1_macro": r["metrics"]["test"]["f1_macro"],
            "out_dir": r["out_dir"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / f"summary_all_turbulence_{args.model}.csv", index=False, encoding="utf-8-sig")

    with open(out_root / f"summary_all_turbulence_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_seconds": total_seconds,
            "results": summary_rows,
            "args": vars(args),
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("训练全部完成")
    print("=" * 90)
    print(summary_df)
    print(f"Total time: {total_seconds/60:.2f} min")
    print(f"Summary: {out_root / f'summary_all_turbulence_{args.model}.csv'}")


if __name__ == "__main__":
    main()
