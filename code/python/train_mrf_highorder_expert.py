# -*- coding: utf-8 -*-
"""
train_mrf_highorder_expert.py

M7 / M8:
    M7 = CDM-Stats-Radial + Augmentation + High-order QAM Expert
    M8 = M7 + High-order-only MMD domain alignment

目标：
    1. 不再盲目堆叠 Angle / Polar 特征
    2. 保留 Radial 的高性价比
    3. 沿用 M4 的真实信号增强
    4. 增加高阶 QAM 专家分支，专门解决 64/128/256QAM 混淆
    5. 可选 high-order-only MMD，对齐 A/B 中高阶 QAM 特征

作者建议：
    先跑 M7：--domain_loss none
    再跑 M8：--domain_loss high_mmd --mmd_lambda 0.03 或 0.05
"""

import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import h5py
except Exception:
    h5py = None

try:
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
except Exception:
    confusion_matrix = None
    f1_score = None
    accuracy_score = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. 基本配置
# ============================================================

MODS = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
MOD_TO_ID = {m: i for i, m in enumerate(MODS)}
ID_TO_MOD = {i: m for m, i in MOD_TO_ID.items()}

HIGH_IDS = [3, 4, 5]  # 64QAM / 128QAM / 256QAM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 1. Manifest 自动识别
# ============================================================

def find_col(df: pd.DataFrame, candidates: List[str], required=True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # 模糊匹配
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c

    if required:
        raise ValueError(f"Cannot find column from candidates: {candidates}. Existing columns: {list(df.columns)}")
    return None


def normalize_mod_name(x) -> str:
    s = str(x).strip().upper()
    s = s.replace("-", "").replace("_", "")
    if s in ["QPSK", "4QAM"]:
        return "QPSK"
    if s in ["16QAM", "QAM16"]:
        return "16QAM"
    if s in ["32QAM", "QAM32"]:
        return "32QAM"
    if s in ["64QAM", "QAM64"]:
        return "64QAM"
    if s in ["128QAM", "QAM128"]:
        return "128QAM"
    if s in ["256QAM", "QAM256"]:
        return "256QAM"

    # 如果 label 是数字
    try:
        v = int(float(s))
        if 0 <= v < len(MODS):
            return ID_TO_MOD[v]
    except Exception:
        pass

    raise ValueError(f"Unknown modulation label: {x}")


def normalize_turb(x) -> str:
    s = str(x).strip().lower()
    if "weak" in s or "sub01" in s or "sub02" in s or s in ["w", "0", "low"]:
        return "weak"
    if "strong" in s or "sub03" in s or "sub04" in s or s in ["s", "1", "high"]:
        return "strong"
    return s


def normalize_source(x) -> str:
    s = str(x).strip().upper()
    if s in ["A", "DATASET-A", "DATASETA", "2026.06.28", "20260628"]:
        return "A"
    if s in ["B", "DATASET-B", "DATASETB", "2026.06.29", "20260629"]:
        return "B"
    if "A" == s[-1:]:
        return "A"
    if "B" == s[-1:]:
        return "B"
    return s


def resolve_path(raw_path: str, root: str) -> str:
    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return str(p)

    p1 = Path(root) / str(raw_path)
    if p1.exists():
        return str(p1)

    p2 = Path.cwd() / str(raw_path)
    if p2.exists():
        return str(p2)

    # Windows 路径在 CSV 里可能以反斜杠保存
    raw2 = str(raw_path).replace("\\", os.sep).replace("/", os.sep)
    p3 = Path(root) / raw2
    if p3.exists():
        return str(p3)

    return str(p1)


def load_manifest(
    manifest_path: str,
    data_root: Optional[str],
    turbulence: str,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    # 过滤 status != ok 的样本
    status_col = find_col(df, ["status"], required=False)
    if status_col is not None:
        df = df[df[status_col].astype(str).str.lower() == "ok"].copy()

    path_col = find_col(df, ["mat_path", "file_path", "path", "sample_path",
                             "preprocessed_path", "mat_file", "filename", "out_mat"])
    mod_col = find_col(df, ["modulation", "mod_name", "mod", "label", "class", "class_name"])
    split_col = find_col(df, ["split", "train_split", "data_split"], required=False)
    turb_col = find_col(df, ["turbulence", "turb_name", "turb", "condition", "turbulence_level"], required=False)
    src_col = find_col(df, ["dataset_source", "source", "dataset", "domain"], required=False)
    rxfile_col = find_col(df, ["rx_file", "bin_file", "file", "rx_name"], required=False)

    root = data_root if data_root is not None else str(Path(manifest_path).parent)

    out = pd.DataFrame()
    out["path"] = df[path_col].apply(lambda x: resolve_path(x, root))
    out["mod"] = df[mod_col].apply(normalize_mod_name)
    out["label"] = out["mod"].map(MOD_TO_ID).astype(int)

    if split_col is not None:
        out["split"] = df[split_col].astype(str).str.lower().str.strip()
        out["split"] = out["split"].replace({
            "training": "train",
            "tr": "train",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "testing": "test",
            "te": "test",
        })
    else:
        out["split"] = ""

    if turb_col is not None:
        out["turbulence"] = df[turb_col].apply(normalize_turb)
    else:
        out["turbulence"] = out["path"].apply(normalize_turb)

    if src_col is not None:
        out["source"] = df[src_col].apply(normalize_source)
    else:
        def infer_source_from_path(p):
            ps = str(p).lower()
            if "2026.06.28" in ps or "20260628" in ps or "dataset-a" in ps:
                return "A"
            if "2026.06.29" in ps or "20260629" in ps or "dataset-b" in ps:
                return "B"
            return "UNK"
        out["source"] = out["path"].apply(infer_source_from_path)

    out["domain"] = out["source"].map({"A": 0, "B": 1}).fillna(-1).astype(int)

    if rxfile_col is not None:
        out["rx_file"] = df[rxfile_col].astype(str)
    else:
        out["rx_file"] = out["path"].apply(lambda p: Path(str(p)).stem)

    turbulence = turbulence.lower()
    if turbulence in ["weak", "strong"]:
        out = out[out["turbulence"] == turbulence].reset_index(drop=True)

    return out


def group_split_by_rx_file(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    按 rx_file 文件级划分 train/val/test。
    若含 dataset_source 列，则分层划分（每个 source 内部各自按比例分配）。
    典型 A 25 + B 20 = 45 bin：train A17+B14=31 / val A3+B2=5 / test A5+B4=9。
    """
    rng = np.random.default_rng(seed)
    has_source = "source" in df.columns
    pct_test = 0.20
    pct_val = 0.12

    split_rows = []
    for y in sorted(df["label"].unique()):
        d = df[df["label"] == y].copy()

        if has_source:
            sources = sorted(d["source"].astype(str).unique())
        else:
            sources = ["__all__"]

        train_g, val_g, test_g = set(), set(), set()

        for src in sources:
            if has_source:
                d_src = d[d["source"].astype(str) == src]
            else:
                d_src = d
            groups = np.array(sorted(d_src["rx_file"].unique()))
            rng.shuffle(groups)

            n = len(groups)
            if n < 3:
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
                s = "train"
            elif g in val_g:
                s = "val"
            elif g in test_g:
                s = "test"
            else:
                s = "test"
            rr = row.to_dict()
            rr["split"] = s
            split_rows.append(rr)

    return pd.DataFrame(split_rows).reset_index(drop=True)


# ============================================================
# 2. MAT 文件读取与特征计算
# ============================================================

def read_mat_file(path: str) -> Dict[str, np.ndarray]:
    path = str(path)

    if sio is not None:
        try:
            mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            mat = {k: v for k, v in mat.items() if not k.startswith("__")}
            return mat
        except Exception:
            pass

    if h5py is not None:
        try:
            out = {}
            with h5py.File(path, "r") as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            arr = np.array(obj)
                            out[name.split("/")[-1]] = arr
                        except Exception:
                            pass
                f.visititems(visit)
            return out
        except Exception as e:
            raise RuntimeError(f"Failed to read mat file: {path}, error={e}")

    raise RuntimeError("Neither scipy.io nor h5py can read .mat files.")


def to_numpy_complex(x) -> np.ndarray:
    arr = np.array(x)

    # MATLAB v7.3 complex 有时会存成 compound
    if arr.dtype.fields is not None:
        names = list(arr.dtype.fields.keys())
        if "real" in names and "imag" in names:
            arr = arr["real"] + 1j * arr["imag"]

    arr = np.squeeze(arr)

    if np.iscomplexobj(arr):
        z = arr.astype(np.complex64)
    else:
        arr = arr.astype(np.float32)
        # 可能是 [2, N] 或 [N, 2]
        if arr.ndim >= 2:
            if arr.shape[0] == 2:
                z = arr[0].reshape(-1) + 1j * arr[1].reshape(-1)
            elif arr.shape[-1] == 2:
                z = arr[..., 0].reshape(-1) + 1j * arr[..., 1].reshape(-1)
            else:
                z = arr.reshape(-1).astype(np.complex64)
        else:
            z = arr.reshape(-1).astype(np.complex64)

    z = z.reshape(-1)
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    return z.astype(np.complex64)


def find_first_key(mat: Dict[str, np.ndarray], keys: List[str]) -> Optional[str]:
    lower = {k.lower(): k for k in mat.keys()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    for k in mat.keys():
        kl = k.lower()
        for key in keys:
            if key.lower() in kl:
                return k
    return None


def normalize_complex(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z).reshape(-1).astype(np.complex64)
    if len(z) == 0:
        return z

    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if len(z) == 0:
        return z

    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    z = z / rms
    return z.astype(np.complex64)


def compute_cdm(z: np.ndarray, bins: int = 64, clip_val: float = 3.0) -> np.ndarray:
    z = normalize_complex(z)
    if len(z) == 0:
        return np.zeros((bins, bins), dtype=np.float32)

    x = np.clip(z.real, -clip_val, clip_val)
    y = np.clip(z.imag, -clip_val, clip_val)

    hist, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[[-clip_val, clip_val], [-clip_val, clip_val]]
    )
    hist = hist.astype(np.float32)

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def compute_blind_stats(z: np.ndarray) -> np.ndarray:
    z = normalize_complex(z)
    if len(z) == 0:
        return np.zeros(16, dtype=np.float32)

    amp = np.abs(z)
    phase = np.angle(z)
    i = z.real
    q = z.imag

    def stats4(v):
        v = np.asarray(v, dtype=np.float32)
        mu = np.mean(v)
        std = np.std(v) + 1e-8
        skew = np.mean(((v - mu) / std) ** 3)
        kurt = np.mean(((v - mu) / std) ** 4)
        return [mu, std, skew, kurt]

    papr = np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-12)

    feat = []
    feat += stats4(amp)
    feat += stats4(phase)
    feat += stats4(i)
    feat += stats4(q)

    feat = np.array(feat, dtype=np.float32)
    # 把 PAPR 融入最后一维，避免维度改变影响已有结构
    feat[-1] = papr
    feat[~np.isfinite(feat)] = 0
    return feat.astype(np.float32)


def compute_radial(z: np.ndarray, bins: int = 32, max_r: float = 3.0) -> np.ndarray:
    z = normalize_complex(z)
    if len(z) == 0:
        return np.zeros(bins, dtype=np.float32)

    r = np.clip(np.abs(z), 0, max_r)
    hist, _ = np.histogram(r, bins=bins, range=(0, max_r))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def safe_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def extract_sample_features(
    path: str,
    use_radial: bool = True,
    cdm_bins: int = 64,
    clip_val: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mat = read_mat_file(path)

    cdm_keys = [
        "cdm", "cdm_img", "cdm_map", "constellation_density",
        "density_map", "hist2d", "H"
    ]
    stats_keys = [
        "stats", "blind_stats", "stat_features", "features",
        "blind_feature", "feat"
    ]
    z_keys = [
        "rx_sc", "rx_symbols", "rx_symbol", "constellation",
        "z", "iq", "iq_data", "rx_iq", "data", "x"
    ]

    cdm_key = find_first_key(mat, cdm_keys)
    stats_key = find_first_key(mat, stats_keys)
    z_key = find_first_key(mat, z_keys)

    z = None
    if z_key is not None:
        try:
            z = to_numpy_complex(mat[z_key])
        except Exception:
            z = None

    if cdm_key is not None:
        cdm = np.array(mat[cdm_key]).squeeze().astype(np.float32)
        if cdm.ndim == 3:
            cdm = cdm.squeeze()
        if cdm.shape != (cdm_bins, cdm_bins):
            # 如果尺寸不对，优先从 z 重新计算
            if z is not None:
                cdm = compute_cdm(z, bins=cdm_bins, clip_val=clip_val)
            else:
                cdm = np.resize(cdm, (cdm_bins, cdm_bins)).astype(np.float32)
        cdm[~np.isfinite(cdm)] = 0
        if cdm.sum() > 0:
            cdm = cdm / cdm.sum()
    else:
        if z is None:
            cdm = np.zeros((cdm_bins, cdm_bins), dtype=np.float32)
        else:
            cdm = compute_cdm(z, bins=cdm_bins, clip_val=clip_val)

    if stats_key is not None:
        stats = np.array(mat[stats_key]).squeeze().astype(np.float32).reshape(-1)
        stats[~np.isfinite(stats)] = 0
    else:
        if z is None:
            stats = np.zeros(16, dtype=np.float32)
        else:
            stats = compute_blind_stats(z)

    if use_radial:
        if z is None:
            radial = np.zeros(32, dtype=np.float32)
        else:
            radial = compute_radial(z, bins=32, max_r=clip_val)
        stats = np.concatenate([stats, radial], axis=0).astype(np.float32)

    cdm = cdm.astype(np.float32)[None, :, :]  # [1, H, W]
    return cdm, stats.astype(np.float32)


# ============================================================
# 3. Dataset
# ============================================================

class MRFDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        use_radial: bool = True,
        augment: bool = False,
        stats_mean: Optional[np.ndarray] = None,
        stats_std: Optional[np.ndarray] = None,
        cache_in_memory: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.use_radial = use_radial
        self.augment = augment
        self.stats_mean = stats_mean
        self.stats_std = stats_std
        self.cache_in_memory = cache_in_memory
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _load(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]

        if self.cache_in_memory and idx in self.cache:
            cdm, stats = self.cache[idx]
        else:
            cdm, stats = extract_sample_features(
                path=path,
                use_radial=self.use_radial,
                cdm_bins=64,
                clip_val=3.0,
            )
            if self.cache_in_memory:
                self.cache[idx] = (cdm, stats)

        return cdm.copy(), stats.copy()

    def _augment(self, cdm: np.ndarray, stats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # CDM 90°旋转：QAM 星座具有 90°旋转对称性
        if random.random() < 0.75:
            k = random.randint(0, 3)
            cdm[0] = np.rot90(cdm[0], k).copy()

        # CDM 幅度缩放
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            cdm = cdm * scale

        # CDM 高斯噪声
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, size=cdm.shape).astype(np.float32)
            cdm = cdm + noise

        cdm = np.clip(cdm, 0, None)
        if cdm.sum() > 0:
            cdm = cdm / cdm.sum()

        # stats 轻噪声，不要过强，避免破坏 128QAM
        if random.random() < 0.5:
            stats = stats + np.random.normal(0, 0.005, size=stats.shape).astype(np.float32)

        return cdm.astype(np.float32), stats.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cdm, stats = self._load(idx)

        if self.augment:
            cdm, stats = self._augment(cdm, stats)

        if self.stats_mean is not None and self.stats_std is not None:
            stats = safe_standardize(stats, self.stats_mean, self.stats_std)

        y = int(row["label"])
        domain = int(row["domain"])
        source = row["source"]

        return {
            "cdm": torch.from_numpy(cdm).float(),
            "stats": torch.from_numpy(stats).float(),
            "label": torch.tensor(y).long(),
            "domain": torch.tensor(domain).long(),
            "source": source,
            "path": row["path"],
        }


def compute_stats_norm(df_train: pd.DataFrame, use_radial: bool) -> Tuple[np.ndarray, np.ndarray, int]:
    feats = []
    for i, row in df_train.iterrows():
        _, stats = extract_sample_features(row["path"], use_radial=use_radial)
        feats.append(stats)
    feats = np.stack(feats, axis=0).astype(np.float32)
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, feats.shape[1]


# ============================================================
# 4. 模型：主分类器 + 高阶专家分支
# ============================================================

class CDMBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class StatsBranch(nn.Module):
    def __init__(self, in_dim: int, out_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.20),

            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(64, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class HighOrderExpertNet(nn.Module):
    """
    输出：
        logits6: 6 类主分类
        logits_coarse: 4 类粗分类，QPSK / 16QAM / 32QAM / HighQAM
        logits_expert: 3 类高阶专家，64QAM / 128QAM / 256QAM
        feat: 融合特征，用于 MMD
    """
    def __init__(self, stats_dim: int, num_classes=6):
        super().__init__()
        self.cdm_branch = CDMBranch(out_dim=64)
        self.stats_branch = StatsBranch(stats_dim, out_dim=48)

        fusion_dim = 64 + 48

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
        )

        self.main_head = nn.Linear(96, num_classes)
        self.coarse_head = nn.Linear(96, 4)
        self.expert_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 3)
        )

    def forward(self, cdm, stats):
        f1 = self.cdm_branch(cdm)
        f2 = self.stats_branch(stats)
        feat = self.fusion(torch.cat([f1, f2], dim=1))

        logits6 = self.main_head(feat)
        logits_coarse = self.coarse_head(feat)
        logits_expert = self.expert_head(feat)

        return {
            "logits6": logits6,
            "logits_coarse": logits_coarse,
            "logits_expert": logits_expert,
            "feat": feat,
        }


# ============================================================
# 5. Loss：Focal + Expert + Coarse + High-MMD
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def labels_to_coarse(y: torch.Tensor) -> torch.Tensor:
    # 0 QPSK, 1 16QAM, 2 32QAM, 3 high-order
    coarse = y.clone()
    coarse[y >= 3] = 3
    return coarse


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, sigmas=(1, 2, 4, 8, 16)) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)

    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2

    loss = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma * sigma)
        kxx = torch.exp(-gamma * xx).mean()
        kyy = torch.exp(-gamma * yy).mean()
        kxy = torch.exp(-gamma * xy).mean()
        loss = loss + kxx + kyy - 2 * kxy

    return loss / len(sigmas)


def high_order_mmd(feat, y, domain):
    # 只对 64/128/256QAM 样本做 A/B MMD
    mask = (y >= 3) & (domain >= 0)
    if mask.sum() < 4:
        return torch.tensor(0.0, device=feat.device)

    f = feat[mask]
    d = domain[mask]

    fa = f[d == 0]
    fb = f[d == 1]

    if fa.size(0) < 2 or fb.size(0) < 2:
        return torch.tensor(0.0, device=feat.device)

    return gaussian_mmd(fa, fb)


def compute_loss(
    outputs,
    y,
    domain,
    criterion_main,
    device,
    expert_lambda=0.8,
    coarse_lambda=0.3,
    domain_loss="none",
    mmd_lambda=0.0,
):
    logits6 = outputs["logits6"]
    logits_coarse = outputs["logits_coarse"]
    logits_expert = outputs["logits_expert"]
    feat = outputs["feat"]

    # 主 6 类 focal loss
    loss_main = criterion_main(logits6, y)

    # 粗分类 loss
    coarse_y = labels_to_coarse(y)
    coarse_weight = torch.tensor([1.0, 1.0, 1.0, 1.3], device=device)
    loss_coarse = F.cross_entropy(logits_coarse, coarse_y, weight=coarse_weight, label_smoothing=0.03)

    # 高阶专家 loss：只在 y >= 3 的样本上计算
    high_mask = y >= 3
    if high_mask.sum() > 0:
        expert_y = y[high_mask] - 3
        expert_weight = torch.tensor([1.2, 2.5, 1.2], device=device)  # 强调 128QAM
        loss_expert = F.cross_entropy(
            logits_expert[high_mask],
            expert_y,
            weight=expert_weight,
            label_smoothing=0.03,
        )
    else:
        loss_expert = torch.tensor(0.0, device=device)

    loss = loss_main + coarse_lambda * loss_coarse + expert_lambda * loss_expert

    loss_da = torch.tensor(0.0, device=device)
    if domain_loss == "high_mmd":
        loss_da = high_order_mmd(feat, y, domain)
        loss = loss + mmd_lambda * loss_da

    return loss, {
        "loss_main": float(loss_main.detach().cpu()),
        "loss_coarse": float(loss_coarse.detach().cpu()),
        "loss_expert": float(loss_expert.detach().cpu()),
        "loss_da": float(loss_da.detach().cpu()),
    }


# ============================================================
# 6. 训练与评估
# ============================================================

def combine_logits(outputs, expert_mix=0.65):
    """
    将专家分支融入最终 6 类预测。
    低阶类仍主要由主分类器决定。
    高阶类 logits = (1-mix)*主分类器高阶 logits + mix*专家 logits
    """
    logits6 = outputs["logits6"]
    logits_expert = outputs["logits_expert"]

    combined = logits6.clone()
    combined[:, 3:6] = (1.0 - expert_mix) * logits6[:, 3:6] + expert_mix * logits_expert
    return combined


@torch.no_grad()
def evaluate(model, loader, device, expert_mix=0.65):
    model.eval()

    all_y = []
    all_pred = []
    all_source = []
    all_domain = []
    total_loss = 0.0

    for batch in loader:
        cdm = batch["cdm"].to(device)
        stats = batch["stats"].to(device)
        y = batch["label"].to(device)
        domain = batch["domain"].to(device)

        outputs = model(cdm, stats)
        logits = combine_logits(outputs, expert_mix=expert_mix)
        pred = torch.argmax(logits, dim=1)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_source.extend(list(batch["source"]))
        all_domain.extend(domain.cpu().numpy().tolist())

    y_true = np.array(all_y, dtype=int)
    y_pred = np.array(all_pred, dtype=int)
    sources = np.array(all_source)

    if len(y_true) == 0:
        return {}

    acc = float((y_true == y_pred).mean())

    if f1_score is not None:
        macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(6)), average="macro", zero_division=0))
    else:
        macro_f1 = 0.0

    per_class = {}
    for i, m in enumerate(MODS):
        mask = y_true == i
        if mask.sum() == 0:
            per_class[m] = np.nan
        else:
            per_class[m] = float((y_pred[mask] == y_true[mask]).mean())

    source_acc = {}
    for src in ["A", "B"]:
        mask = sources == src
        if mask.sum() == 0:
            source_acc[src] = np.nan
        else:
            source_acc[src] = float((y_pred[mask] == y_true[mask]).mean())

    high_acc = float(np.mean([per_class["64QAM"], per_class["128QAM"], per_class["256QAM"]]))

    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    else:
        cm = np.zeros((6, 6), dtype=int)
        for a, b in zip(y_true, y_pred):
            cm[a, b] += 1

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "source_acc": source_acc,
        "ab_gap": None if (np.isnan(source_acc.get("A", np.nan)) or np.isnan(source_acc.get("B", np.nan))) else float(source_acc["A"] - source_acc["B"]),
        "high_order_acc": high_acc,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "sources": sources,
    }


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    cm = np.asarray(cm)
    fig = plt.figure(figsize=(6.4, 5.6), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(MODS, rotation=45, ha="right")
    ax.set_yticklabels(MODS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def train_one_seed(
    df: pd.DataFrame,
    seed: int,
    turbulence: str,
    args,
):
    set_seed(seed)

    out_dir = Path(args.out_dir) / f"seed_{seed}" / f"{turbulence}_{args.exp_name}"
    ensure_dir(str(out_dir))

    # 若 manifest 无预定义 split，则按 seed 做文件级分层划分
    need_split = len(df) == 0 or str(df["split"].iloc[0]) == ""
    if need_split:
        df = group_split_by_rx_file(df, seed=seed)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError(f"Empty split. train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    print(f"\n[{now()}] Seed={seed}, turbulence={turbulence}")
    print(f"Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print("Train label count:")
    print(df_train["mod"].value_counts().sort_index())
    print("Train source count:")
    print(df_train["source"].value_counts().sort_index())

    stats_mean, stats_std, stats_dim = compute_stats_norm(df_train, use_radial=args.use_radial)
    np.save(out_dir / "stats_mean.npy", stats_mean)
    np.save(out_dir / "stats_std.npy", stats_std)

    train_ds = MRFDataset(
        df_train,
        use_radial=args.use_radial,
        augment=args.augment,
        stats_mean=stats_mean,
        stats_std=stats_std,
        cache_in_memory=args.cache,
    )
    val_ds = MRFDataset(
        df_val,
        use_radial=args.use_radial,
        augment=False,
        stats_mean=stats_mean,
        stats_std=stats_std,
        cache_in_memory=args.cache,
    )
    test_ds = MRFDataset(
        df_test,
        use_radial=args.use_radial,
        augment=False,
        stats_mean=stats_mean,
        stats_std=stats_std,
        cache_in_memory=args.cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}, stats_dim={stats_dim}")

    model = HighOrderExpertNet(stats_dim=stats_dim).to(device)

    # 主分类权重：128QAM 权重最高
    main_weight = torch.tensor([1.0, 1.0, 1.0, 1.3, 2.2, 1.3], device=device)
    criterion_main = FocalLoss(
        weight=main_weight,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    best_val = -1.0
    best_epoch = -1
    best_path = out_dir / "best_model.pt"
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0
        log_parts = {"loss_main": 0.0, "loss_coarse": 0.0, "loss_expert": 0.0, "loss_da": 0.0}

        for batch in train_loader:
            cdm = batch["cdm"].to(device, non_blocking=True)
            stats = batch["stats"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            domain = batch["domain"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(cdm, stats)

            loss, parts = compute_loss(
                outputs=outputs,
                y=y,
                domain=domain,
                criterion_main=criterion_main,
                device=device,
                expert_lambda=args.expert_lambda,
                coarse_lambda=args.coarse_lambda,
                domain_loss=args.domain_loss,
                mmd_lambda=args.mmd_lambda,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = y.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n_sum += bs
            for k in log_parts:
                log_parts[k] += parts[k] * bs

        scheduler.step()

        train_loss = loss_sum / max(1, n_sum)
        for k in log_parts:
            log_parts[k] = log_parts[k] / max(1, n_sum)

        val_metrics = evaluate(model, val_loader, device, expert_mix=args.expert_mix)
        val_acc = val_metrics["acc"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_high_order_acc": val_metrics["high_order_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update(log_parts)
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"main={log_parts['loss_main']:.4f} | "
            f"expert={log_parts['loss_expert']:.4f} | "
            f"da={log_parts['loss_da']:.4f} | "
            f"val_acc={val_acc*100:.2f}% | "
            f"val_HO={val_metrics['high_order_acc']*100:.2f}%"
        )

        # 以 val_acc 为主，val_high_order_acc 为次级判断
        score = val_acc + 0.15 * val_metrics["high_order_acc"]

        if score > best_val:
            best_val = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "stats_dim": stats_dim,
                    "args": vars(args),
                    "seed": seed,
                    "turbulence": turbulence,
                    "best_epoch": best_epoch,
                    "stats_mean": stats_mean,
                    "stats_std": stats_std,
                },
                best_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    # 加载最佳模型测试
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_metrics = evaluate(model, val_loader, device, expert_mix=args.expert_mix)
    test_metrics = evaluate(model, test_loader, device, expert_mix=args.expert_mix)

    plot_confusion_matrix(
        test_metrics["cm"],
        out_dir / "confusion_matrix_test.png",
        title=f"{turbulence} {args.exp_name} seed={seed}",
    )

    result = {
        "seed": seed,
        "turbulence": turbulence,
        "exp_name": args.exp_name,
        "best_epoch": best_epoch,
        "val_acc": val_metrics["acc"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_high_order_acc": val_metrics["high_order_acc"],
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_high_order_acc": test_metrics["high_order_acc"],
        "test_ab_gap": test_metrics["ab_gap"],
        "test_source_A": test_metrics["source_acc"].get("A", np.nan),
        "test_source_B": test_metrics["source_acc"].get("B", np.nan),
    }

    for m in MODS:
        result[f"test_acc_{m}"] = test_metrics["per_class"][m]

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存逐样本预测
    pred_df = pd.DataFrame({
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "true_mod": [ID_TO_MOD[int(x)] for x in test_metrics["y_true"]],
        "pred_mod": [ID_TO_MOD[int(x)] for x in test_metrics["y_pred"]],
        "source": test_metrics["sources"],
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nTest result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


# ============================================================
# 7. 多 seed 汇总
# ============================================================

def summarize_results(results: List[Dict], save_dir: str):
    df = pd.DataFrame(results)
    df.to_csv(Path(save_dir) / "summary_all_seeds.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for turb in sorted(df["turbulence"].unique()):
        sub = df[df["turbulence"] == turb]
        row = {
            "turbulence": turb,
            "n_seeds": len(sub),
            "test_acc_mean": sub["test_acc"].mean(),
            "test_acc_std": sub["test_acc"].std(),
            "macro_f1_mean": sub["test_macro_f1"].mean(),
            "macro_f1_std": sub["test_macro_f1"].std(),
            "high_order_acc_mean": sub["test_high_order_acc"].mean(),
            "high_order_acc_std": sub["test_high_order_acc"].std(),
            "A_acc_mean": sub["test_source_A"].mean(),
            "B_acc_mean": sub["test_source_B"].mean(),
            "AB_gap_mean": sub["test_ab_gap"].mean(),
        }
        for m in MODS:
            row[f"{m}_mean"] = sub[f"test_acc_{m}"].mean()
            row[f"{m}_std"] = sub[f"test_acc_{m}"].std()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(Path(save_dir) / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    print("\n========== Mean ± Std Summary ==========")
    for _, r in summary.iterrows():
        print(f"\nTurbulence: {r['turbulence']}")
        print(f"Acc        : {r['test_acc_mean']*100:.2f} ± {r['test_acc_std']*100:.2f}%")
        print(f"Macro-F1   : {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}")
        print(f"High-order : {r['high_order_acc_mean']*100:.2f} ± {r['high_order_acc_std']*100:.2f}%")
        print(f"A Acc      : {r['A_acc_mean']*100:.2f}%")
        print(f"B Acc      : {r['B_acc_mean']*100:.2f}%")
        print(f"A-B Gap    : {r['AB_gap_mean']*100:.2f}%")
        for m in MODS:
            print(f"{m:7s}    : {r[f'{m}_mean']*100:.2f} ± {r[f'{m}_std']*100:.2f}%")


# ============================================================
# 8. main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="merged_AB/manifest_all.csv 路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="如果 manifest 里是相对路径，这里填数据根目录；默认使用 manifest 所在目录",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--turbulence",
        type=str,
        default="all",
        choices=["weak", "strong", "all"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
    )

    parser.add_argument("--exp_name", type=str, default="m7_radial_aug_hqexpert")
    parser.add_argument("--use_radial", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cache", action="store_true")

    parser.add_argument("--domain_loss", type=str, default="none", choices=["none", "high_mmd"])
    parser.add_argument("--mmd_lambda", type=float, default=0.03)

    parser.add_argument("--expert_lambda", type=float, default=0.8)
    parser.add_argument("--coarse_lambda", type=float, default=0.3)
    parser.add_argument("--expert_mix", type=float, default=0.65)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    with open(Path(args.out_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    turbs = ["weak", "strong"] if args.turbulence == "all" else [args.turbulence]

    all_results = []

    for turb in turbs:
        df = load_manifest(
            manifest_path=args.manifest,
            data_root=args.data_root,
            turbulence=turb,
        )

        print(f"\nLoaded manifest for turbulence={turb}: {len(df)} samples")
        print(df.groupby(["split", "mod", "source"]).size())

        for seed in args.seeds:
            result = train_one_seed(
                df=df,
                seed=seed,
                turbulence=turb,
                args=args,
            )
            all_results.append(result)

    summarize_results(all_results, args.out_dir)


if __name__ == "__main__":
    main()