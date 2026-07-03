# -*- coding: utf-8 -*-
"""
stage2_common.py — Stage II 共享工具与路径常量

所有 Stage II 脚本（make_frozen_split / sim_precompute_hturb /
sim_generate_data / sim_compare_distributions / 后续 train_sim2real）
都从此模块导入路径常量和共享函数，保证一致性。
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path("D:/Project_code/FSO_research")

REAL_DATA_ROOT = Path(
    "D:/Project_code/OptDSP_FSO_2/practical waterfiling-exp/"
    "OptDSP_lite-master/2_Data_Results/"
    "preprocessed_uniform_qam_rx_lowmem/merged_AB"
)

SCREEN_VERYWEAK_DIR = Path(
    "D:/Project_code/FSO_research/code/matlab/screens/screens_极弱_mat"
)
SCREEN_STRONG_DIR = Path(
    "D:/Project_code/FSO_research/code/matlab/screens/screens_强_mat"
)

STAGE2_ROOT = Path("D:/Project_code/FSO_research/results/stage2")
SIM_DATA_ROOT = STAGE2_ROOT / "sim_data_v1"
DIST_CHECK_ROOT = STAGE2_ROOT / "distribution_check_v1"

FROZEN_SPLIT_CSV = STAGE2_ROOT / "frozen_real_split.csv"
FROZEN_SPLIT_SUMMARY = STAGE2_ROOT / "frozen_real_split_summary.json"
HTURB_NPZ = STAGE2_ROOT / "hturb_precompute.npz"
SIM_MANIFEST_CSV = SIM_DATA_ROOT / "manifest_sim.csv"
DISTURBANCE_CONFIG = SIM_DATA_ROOT / "disturbance_config.json"

# train_mrf_lowmem.py 所在目录（用于 import）
LOWMEM_DIR = Path("D:/Project_code/FSO_research/code/python")

# ─────────────────────────────────────────────────────────
# OFDM 参数（与真实数据一致）
# ─────────────────────────────────────────────────────────
N_SC = 123          # 活跃子载波（索引 4:126，1-based）
N_SYM = 128         # OFDM 符号 / 帧
N_FFT = 256
CP = 16

MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
MOD_TO_LABEL = {m: i for i, m in enumerate(MOD_NAMES)}
LABEL_TO_MOD = {i: m for i, m in enumerate(MOD_NAMES)}

# 仿真湍流名称 ↔ 真实湍流名称映射
# 真实数据用 weak / strong；相位屏目录用 very_weak / strong
SIM_TURB_NAMES = ["very_weak", "strong"]          # 仿真 .mat 中的 turb_name
REAL_TURB_MAP = {"very_weak": "weak", "strong": "strong"}  # 仿真→真实映射
TURB_TO_LABEL = {"very_weak": 0, "strong": 1}

# ─────────────────────────────────────────────────────────
# 物理参数（复刻 fso_full_pipeline.m）
# ─────────────────────────────────────────────────────────
LAMBDA = 1550e-9
K_WAVE = 2 * np.pi / LAMBDA
NX, NY = 1920, 1080
PIXEL_PITCH = 8e-6
LX = NX * PIXEL_PITCH
LY = NY * PIXEL_PITCH
Z_PROP = 0.36
W0 = 0.8e-3
D_RX = 10e-3
NA_RX = 0.545
F_MAX = NA_RX / LAMBDA


# ─────────────────────────────────────────────────────────
# train_mrf_lowmem 安全导入
# ─────────────────────────────────────────────────────────
def _ensure_lowmem_importable():
    """把 LOWMEM_DIR 加到 sys.path，使 train_mrf_lowmem 可被 import。"""
    p = str(LOWMEM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_lowmem_importable()


# ─────────────────────────────────────────────────────────
# 冻结划分工具
# ─────────────────────────────────────────────────────────
def load_frozen_split(path: Path = FROZEN_SPLIT_CSV) -> pd.DataFrame:
    """加载冻结的真实数据划分 CSV。"""
    if not path.exists():
        raise FileNotFoundError(
            f"冻结划分文件不存在：{path}\n请先运行 make_frozen_split.py"
        )
    df = pd.read_csv(path)
    return df


def subset_real_train(
    split_df: pd.DataFrame,
    ratio: float,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    从 frozen split 的 train 子集中按 ratio 分层采样。

    分层维度：mod_label × turb_name × dataset_source
    ratio=1.0 → 返回全部 train
    ratio<1.0 → 每组按 ratio 采样（至少 1 个）
    """
    train_df = split_df[split_df["split"].astype(str) == "train"].copy()
    if ratio >= 1.0:
        return train_df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    parts = []
    for (mod, turb, src), grp in train_df.groupby(
        ["mod_label", "turb_name", "dataset_source"]
    ):
        n = max(1, int(round(len(grp) * ratio)))
        sampled = grp.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────────────────
# 统一缓存构建（真实或仿真均可用）
# ─────────────────────────────────────────────────────────
def build_unified_cache(
    rows_df: pd.DataFrame,
    model_mode: str,
    cache_path: Path,
    rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    """
    对任意 rows_df（真实或仿真 manifest 子集）构建特征缓存。

    rows_df 必须含列：out_mat, mod_label, split（若无 split 则全部视为 train）
    model_mode: 见 train_mrf_lowmem 的 choices
    cache_path: .npz 输出路径
    """
    from train_mrf_lowmem import (
        load_sample_from_mat,
        MOD_NAMES,
    )

    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    use_iq = model_mode in ("iq", "fusion")
    use_radial = model_mode in ("cdm_stats_radial", "cdm_stats_radial_angle", "cdm_stats_rap")
    use_angle = model_mode in ("cdm_stats_radial_angle", "cdm_stats_rap")
    use_polar = model_mode in ("cdm_stats_rap",)

    cdm_list, iq_list, stats_list = [], [], []
    y_list, split_list, path_list, rx_file_list = [], [], [], []

    t0 = time.time()
    n = len(rows_df)
    for i, (_, row) in enumerate(rows_df.iterrows(), start=1):
        p = Path(row["out_mat"])
        try:
            s = load_sample_from_mat(
                p, use_iq=use_iq, use_radial=use_radial,
                use_angle=use_angle, use_polar=use_polar,
            )
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
        split_list.append(str(row.get("split", "train")))
        path_list.append(str(p))
        rx_file_list.append(str(row.get("rx_file", "")))

        if i % 200 == 0 or i == n:
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
        min_len = min(x.shape[-1] for x in iq_list)
        X_iq = np.stack([x[:, :min_len] for x in iq_list], axis=0).astype(np.float32)
    else:
        X_iq = np.zeros((len(y), 1, 1), dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_cdm=X_cdm, X_iq=X_iq, X_stats=X_stats,
        y=y, split=split, paths=paths, rx_files=rx_files,
    )
    print(f"[cache] 保存 {len(y)} 样本 → {cache_path}")
    return {"X_cdm": X_cdm, "X_iq": X_iq, "X_stats": X_stats,
            "y": y, "split": split, "paths": paths, "rx_files": rx_files}


# ─────────────────────────────────────────────────────────
# h_turb 池加载
# ─────────────────────────────────────────────────────────
def get_hturb_pool(hturb_npz_path: Path = HTURB_NPZ) -> Dict[str, np.ndarray]:
    """
    加载预计算 h_norm，返回 {turb_name: complex array}。
    """
    if not hturb_npz_path.exists():
        raise FileNotFoundError(
            f"h_turb 预计算文件不存在：{hturb_npz_path}\n请先运行 sim_precompute_hturb.py"
        )
    data = np.load(hturb_npz_path, allow_pickle=True)
    return {
        "very_weak": data["h_norm_very_weak"],
        "strong": data["h_norm_strong"],
    }


# ─────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int = 2026):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    print("=== stage2_common.py 路径检查 ===")
    print(f"REAL_DATA_ROOT      : {REAL_DATA_ROOT}  exists={REAL_DATA_ROOT.exists()}")
    print(f"SCREEN_VERYWEAK_DIR : {SCREEN_VERYWEAK_DIR}  exists={SCREEN_VERYWEAK_DIR.exists()}")
    print(f"SCREEN_STRONG_DIR   : {SCREEN_STRONG_DIR}  exists={SCREEN_STRONG_DIR.exists()}")
    print(f"STAGE2_ROOT         : {STAGE2_ROOT}  exists={STAGE2_ROOT.exists()}")
    print(f"LOWMEM_DIR          : {LOWMEM_DIR}  exists={LOWMEM_DIR.exists()}")
    # 验证 train_mrf_lowmem 可导入
    try:
        from train_mrf_lowmem import load_manifest, group_split_by_rx_file
        print("[OK] train_mrf_lowmem 导入成功")
    except Exception as e:
        print(f"[FAIL] train_mrf_lowmem 导入失败: {e}")
