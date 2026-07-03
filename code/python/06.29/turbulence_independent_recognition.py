# -*- coding: utf-8 -*-
"""
turbulence_independent_recognition.py

目的：
    对已经完成 OFDM 解调/均衡/预处理后的不同调制格式信号，
    按“每一种湍流条件独立”进行调制格式识别。

核心原则：
    1) sub01 / sub02 / sub03 等湍流条件分别训练、分别测试、分别保存模型；
    2) 不把不同湍流条件的数据混在一起训练一个总模型；
    3) 数据划分按“文件级”划分，避免同一个接收文件切出来的窗口同时进入训练集和测试集；
    4) 输入为解调后的复数符号序列，默认构造 I / Q / Amp / sin(Phase) / cos(Phase) 五通道；
    5) 使用 PyTorch + CUDA + AMP 加速；
    6) 保存 best_model.pt、metrics.json、confusion_matrix.png、features_*.npz，方便后续论文分析和识别调用。

推荐目录结构：
    ROOT_DIR/
        QPSK/
            sub01/
                *.npz / *.mat / *.npy / *.bin
            sub02/
            sub03/
        16QAM/
            sub01/
            sub02/
            sub03/
        32QAM/
        64QAM/
        128QAM/
        256QAM/

运行示例：
    python turbulence_independent_recognition.py ^
        --root "D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx_data\2026.06.28" ^
        --turbs sub01 sub03 ^
        --out "D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_results"

如果你有 sub02：
    python turbulence_independent_recognition.py --root "你的根目录" --turbs sub01 sub02 sub03

说明：
    - 如果你的 npz/mat 里面复数符号变量名不是默认候选名，可以在 PREFERRED_KEYS 里加入。
    - 如果你的 .bin 是 interleaved float32 IQ，即 [I0,Q0,I1,Q1,...]，本脚本可以直接读。
    - 如果你的 .bin 已经是 complex64，也可以用 --bin-format complex64。
"""

import os
import json
import math
import time
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import scipy.io as sio
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# =========================
# 1. 默认配置
# =========================

DEFAULT_ROOT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx_data\2026.06.28"
DEFAULT_OUT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_results"

# 根据你真实已有的调制格式自动发现；如果需要固定顺序，可在命令行 --mods 指定
DEFAULT_MODS = None

# 湍流文件夹：sub01 弱湍流，sub02 中湍流，sub03 强湍流
DEFAULT_TURBS = ["sub01", "sub02", "sub03"]
TURB_NAME_MAP = {
    "sub00": "no_turbulence",
    "sub01": "weak",
    "sub02": "moderate",
    "sub03": "strong",
}

# 解调/均衡后复数符号在 npz/mat 中可能使用的变量名
PREFERRED_KEYS = [
    "rx_eq",
    "rx_syms",
    "rx_symbols",
    "symbols",
    "data_symbols",
    "demod_symbols",
    "equalized_symbols",
    "rx_1sps",
    "ofdm_symbols",
    "X",
    "x",
    "iq",
]

SUPPORTED_EXTS = [".npz", ".npy", ".mat", ".bin", ".csv", ".txt"]


# =========================
# 2. 通用工具函数
# =========================

def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 速度优先；如果你要完全可复现，可设置 deterministic=True，但会慢
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def natural_mod_key(name: str):
    """
    让 QPSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM 排序更自然。
    """
    s = name.upper().replace("-", "").replace("_", "")
    if s == "BPSK":
        return 1
    if s == "QPSK":
        return 2
    if "QAM" in s:
        num = "".join([c for c in s if c.isdigit()])
        if num:
            return int(num)
    return 9999


def discover_mods(root: Path, turbs: Sequence[str]) -> List[str]:
    mods = []
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在：{root}")

    for child in root.iterdir():
        if not child.is_dir():
            continue
        ok = False
        for tb in turbs:
            if (child / tb).exists():
                ok = True
                break
        if ok:
            mods.append(child.name)

    mods = sorted(mods, key=natural_mod_key)
    return mods


def list_signal_files(mod_dir: Path, turbs: Sequence[str]) -> Dict[str, List[Path]]:
    out = {}
    for tb in turbs:
        folder = mod_dir / tb
        files = []
        if folder.exists():
            for ext in SUPPORTED_EXTS:
                files.extend(folder.rglob(f"*{ext}"))
        out[tb] = sorted(files)
    return out


def choose_numeric_array_from_npz(npz_obj) -> np.ndarray:
    keys = list(npz_obj.keys())

    for k in PREFERRED_KEYS:
        if k in keys:
            arr = npz_obj[k]
            if np.issubdtype(arr.dtype, np.number):
                return arr

    # 兜底：选最大数字数组
    candidates = []
    for k in keys:
        arr = npz_obj[k]
        if np.issubdtype(arr.dtype, np.number):
            candidates.append((arr.size, k, arr))
    if not candidates:
        raise ValueError("npz 中没有可用数字数组。")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][2]


def choose_numeric_array_from_mat(mat_dict: dict) -> np.ndarray:
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]

    for k in PREFERRED_KEYS:
        if k in keys:
            arr = mat_dict[k]
            if np.issubdtype(arr.dtype, np.number):
                return arr

    candidates = []
    for k in keys:
        arr = mat_dict[k]
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
            candidates.append((arr.size, k, arr))
    if not candidates:
        raise ValueError("mat 中没有可用数字数组。")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][2]


def array_to_complex_1d(arr: np.ndarray) -> np.ndarray:
    """
    支持以下形式：
        - complex ndarray: 直接 flatten
        - shape [..., 2] 的实数数组：最后一维为 I/Q
        - shape [2, N] 的实数数组：第一维为 I/Q
        - 纯实数数组：当成实信号，虚部为 0
    """
    arr = np.asarray(arr)

    # 去掉多余单维度
    arr = np.squeeze(arr)

    if np.iscomplexobj(arr):
        x = arr.astype(np.complex64).reshape(-1)
        return x

    if arr.ndim >= 2 and arr.shape[-1] == 2:
        real = arr[..., 0]
        imag = arr[..., 1]
        x = real + 1j * imag
        return x.astype(np.complex64).reshape(-1)

    if arr.ndim == 2 and arr.shape[0] == 2:
        x = arr[0, :] + 1j * arr[1, :]
        return x.astype(np.complex64).reshape(-1)

    # 如果是 [N, 2] 前面已经处理；其他纯实数兜底
    x = arr.astype(np.float32).reshape(-1).astype(np.complex64)
    return x


def load_complex_sequence(path: Path, bin_format: str = "interleaved_float32") -> np.ndarray:
    ext = path.suffix.lower()

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as obj:
            arr = choose_numeric_array_from_npz(obj)
        x = array_to_complex_1d(arr)

    elif ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        x = array_to_complex_1d(arr)

    elif ext == ".mat":
        if not SCIPY_OK:
            raise RuntimeError("读取 .mat 需要 scipy，请先 pip install scipy")
        mat = sio.loadmat(path)
        arr = choose_numeric_array_from_mat(mat)
        x = array_to_complex_1d(arr)

    elif ext == ".bin":
        if bin_format == "interleaved_float32":
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size < 2:
                raise ValueError("bin 文件长度过短。")
            if raw.size % 2 == 1:
                raw = raw[:-1]
            x = raw[0::2] + 1j * raw[1::2]
            x = x.astype(np.complex64)
        elif bin_format == "complex64":
            x = np.fromfile(path, dtype=np.complex64)
        else:
            raise ValueError(f"未知 bin_format：{bin_format}")

    elif ext in [".csv", ".txt"]:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        x = array_to_complex_1d(arr)

    else:
        raise ValueError(f"不支持的文件类型：{path}")

    x = x.reshape(-1)
    finite = np.isfinite(x.real) & np.isfinite(x.imag)
    x = x[finite]
    return x.astype(np.complex64)


def normalize_complex_window(x: np.ndarray, mode: str = "power") -> np.ndarray:
    """
    mode:
        power: 每个窗口按平均功率归一化，推荐用于真实实验数据；
        zscore: 对 I/Q 做均值方差归一化，可能削弱幅度环信息；
        none: 不归一化。
    """
    x = x.astype(np.complex64)

    if mode == "none":
        return x

    if mode == "power":
        p = np.mean(np.abs(x) ** 2)
        if p > 1e-12:
            x = x / np.sqrt(p + 1e-12)
        return x.astype(np.complex64)

    if mode == "zscore":
        xr = x.real
        xi = x.imag
        xr = (xr - xr.mean()) / (xr.std() + 1e-6)
        xi = (xi - xi.mean()) / (xi.std() + 1e-6)
        return (xr + 1j * xi).astype(np.complex64)

    raise ValueError(f"未知 normalize mode：{mode}")


def complex_to_channels(x: np.ndarray, channel_mode: str = "iq_amp_phase") -> np.ndarray:
    """
    输入:
        x: complex sequence, shape [L]
    输出:
        features: float32, shape [C, L]

    channel_mode:
        iq:
            [I, Q]
        iq_amp:
            [I, Q, |x|]
        iq_amp_phase:
            [I, Q, |x|, sin(angle), cos(angle)]
    """
    i = x.real.astype(np.float32)
    q = x.imag.astype(np.float32)

    if channel_mode == "iq":
        return np.stack([i, q], axis=0).astype(np.float32)

    amp = np.abs(x).astype(np.float32)

    if channel_mode == "iq_amp":
        return np.stack([i, q, amp], axis=0).astype(np.float32)

    ph = np.angle(x).astype(np.float32)
    sinp = np.sin(ph).astype(np.float32)
    cosp = np.cos(ph).astype(np.float32)

    if channel_mode == "iq_amp_phase":
        return np.stack([i, q, amp, sinp, cosp], axis=0).astype(np.float32)

    raise ValueError(f"未知 channel_mode：{channel_mode}")


def make_windows_from_sequence(
    x: np.ndarray,
    window_len: int,
    stride: int,
    max_windows_per_file: int,
    normalize_mode: str,
    channel_mode: str,
) -> np.ndarray:
    if x.size < window_len:
        return np.empty((0, 5, window_len), dtype=np.float32)

    starts = list(range(0, x.size - window_len + 1, stride))
    if len(starts) == 0:
        starts = [0]

    if max_windows_per_file > 0 and len(starts) > max_windows_per_file:
        # 均匀抽取，避免只取文件开头
        idx = np.linspace(0, len(starts) - 1, max_windows_per_file).round().astype(int)
        starts = [starts[i] for i in idx]

    feats = []
    for st in starts:
        w = x[st: st + window_len]
        w = normalize_complex_window(w, mode=normalize_mode)
        c = complex_to_channels(w, channel_mode=channel_mode)
        feats.append(c)

    return np.stack(feats, axis=0).astype(np.float32)


def split_files_per_class(
    files_by_label: Dict[int, List[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[int, List[Path]], Dict[int, List[Path]], Dict[int, List[Path]]]:
    rng = random.Random(seed)

    train, val, test = {}, {}, {}
    for y, files in files_by_label.items():
        files = list(files)
        rng.shuffle(files)

        n = len(files)
        if n == 0:
            train[y], val[y], test[y] = [], [], []
            continue

        if n == 1:
            train[y], val[y], test[y] = files, [], []
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0

        if n_train + n_val >= n:
            n_train = max(1, n - 1)
            n_val = 0 if n == 2 else 1

        train[y] = files[:n_train]
        val[y] = files[n_train:n_train + n_val]
        test[y] = files[n_train + n_val:]

        # 如果 test 为空，从 train 拿一个出来，保证能测试
        if len(test[y]) == 0 and len(train[y]) > 1:
            test[y] = [train[y].pop()]

    return train, val, test


def flatten_split_dict(split_dict: Dict[int, List[Path]]) -> List[Tuple[Path, int]]:
    items = []
    for y, files in split_dict.items():
        for p in files:
            items.append((p, y))
    return items


def build_split_cache(
    items: List[Tuple[Path, int]],
    cache_path: Path,
    window_len: int,
    stride: int,
    max_windows_per_file: int,
    normalize_mode: str,
    channel_mode: str,
    bin_format: str,
) -> dict:
    """
    把文件列表切片为 X/y 并保存 npz。
    """
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "X": data["X"],
            "y": data["y"],
            "file": data["file"],
        }

    X_list, y_list, file_list = [], [], []
    failed = []

    for p, y in items:
        try:
            x = load_complex_sequence(p, bin_format=bin_format)
            Xw = make_windows_from_sequence(
                x,
                window_len=window_len,
                stride=stride,
                max_windows_per_file=max_windows_per_file,
                normalize_mode=normalize_mode,
                channel_mode=channel_mode,
            )
            if Xw.shape[0] == 0:
                failed.append((str(p), "too_short"))
                continue

            X_list.append(Xw)
            y_list.append(np.full((Xw.shape[0],), y, dtype=np.int64))
            file_list.extend([str(p)] * Xw.shape[0])

        except Exception as e:
            failed.append((str(p), repr(e)))

    if len(X_list) == 0:
        raise RuntimeError(f"没有成功构建任何样本。cache={cache_path}，失败={failed[:5]}")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    file_arr = np.asarray(file_list, dtype=object)

    np.savez_compressed(cache_path, X=X, y=y, file=file_arr)

    if failed:
        fail_path = cache_path.with_suffix(".failed.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    return {"X": X, "y": y, "file": file_arr}


# =========================
# 3. Dataset
# =========================

class NpzSignalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# =========================
# 4. 模型：1D ResNet + SE 注意力
# =========================

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)
        self.drop = nn.Dropout(dropout)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out, inplace=True)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + identity
        out = F.silu(out, inplace=True)
        return out


class TurbulenceIndependentNet(nn.Module):
    """
    输入:  [B, C, L]
    输出:  logits, embedding
    """
    def __init__(self, in_ch: int, num_classes: int, emb_dim: int = 256, dropout: float = 0.25):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResBlock1D(64, 64, stride=1, dropout=0.05),
            ResBlock1D(64, 64, stride=1, dropout=0.05),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(64, 128, stride=2, dropout=0.08),
            ResBlock1D(128, 128, stride=1, dropout=0.08),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 256, stride=2, dropout=0.10),
            ResBlock1D(256, 256, stride=1, dropout=0.10),
        )
        self.layer4 = nn.Sequential(
            ResBlock1D(256, 384, stride=2, dropout=0.10),
            ResBlock1D(384, 384, stride=1, dropout=0.10),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(384 * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        a = self.avg_pool(x).squeeze(-1)
        m = self.max_pool(x).squeeze(-1)
        h = torch.cat([a, m], dim=1)

        emb = self.embedding(h)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits


# =========================
# 5. 训练与评估
# =========================

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_loader(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        all_y.append(yb.numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    y = np.concatenate(all_y, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    prob = np.concatenate(all_prob, axis=0)
    return y, pred, prob


@torch.no_grad()
def extract_embeddings(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_pred, all_emb = [], [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits, emb = model(xb, return_embedding=True)
        pred = torch.argmax(logits, dim=1)

        all_y.append(yb.numpy())
        all_pred.append(pred.cpu().numpy())
        all_emb.append(emb.cpu().numpy())

    y = np.concatenate(all_y, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    emb = np.concatenate(all_emb, axis=0)
    return y, pred, emb


def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], save_path: Path, title: str):
    if not MATPLOTLIB_OK:
        return

    fig_w = max(7, 1.0 * len(labels))
    fig_h = max(6, 0.9 * len(labels))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_and_save(
    model,
    loaders: Dict[str, DataLoader],
    device,
    mod_names: List[str],
    out_dir: Path,
    prefix: str,
) -> dict:
    results = {}

    for split, loader in loaders.items():
        y, pred, prob = predict_loader(model, loader, device)
        acc = float((y == pred).mean())

        if SKLEARN_OK:
            bal_acc = float(balanced_accuracy_score(y, pred))
            cm = confusion_matrix(y, pred, labels=list(range(len(mod_names))))
            report = classification_report(
                y, pred,
                labels=list(range(len(mod_names))),
                target_names=mod_names,
                digits=4,
                zero_division=0,
                output_dict=True,
            )
        else:
            bal_acc = acc
            cm = np.zeros((len(mod_names), len(mod_names)), dtype=int)
            for yi, pi in zip(y, pred):
                cm[int(yi), int(pi)] += 1
            report = {}

        results[split] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "num_samples": int(len(y)),
            "classification_report": report,
        }

        np.savez_compressed(
            out_dir / f"{prefix}_{split}_predictions.npz",
            y_true=y,
            y_pred=pred,
            prob=prob,
            mod_names=np.asarray(mod_names, dtype=object),
        )

        save_confusion_matrix_png(
            cm,
            labels=mod_names,
            save_path=out_dir / f"{prefix}_{split}_confusion_matrix.png",
            title=f"{prefix} - {split}",
        )

    return results


def train_one_turbulence(
    turb: str,
    mod_names: List[str],
    files_by_mod_turb: Dict[str, Dict[str, List[Path]]],
    args,
    device,
) -> Optional[dict]:

    turb_readable = TURB_NAME_MAP.get(turb, turb)
    out_dir = Path(args.out) / "independent_by_turbulence" / f"{turb}_{turb_readable}"
    cache_dir = out_dir / "dataset_cache"
    ensure_dir(out_dir)
    ensure_dir(cache_dir)

    print("\n" + "=" * 88)
    print(f"[{now_time()}] 开始湍流条件：{turb} ({turb_readable})")
    print("=" * 88)

    # 组装每个类别在当前湍流条件下的文件列表
    files_by_label: Dict[int, List[Path]] = {}
    for y, mod in enumerate(mod_names):
        files = files_by_mod_turb.get(mod, {}).get(turb, [])
        files_by_label[y] = files
        print(f"  Mod={mod:<10s}  files={len(files)}")

    # 检查类别是否完整
    non_empty_classes = [y for y, files in files_by_label.items() if len(files) > 0]
    if len(non_empty_classes) < 2:
        print(f"[跳过] {turb} 可用类别少于2类。")
        return None

    missing = [mod_names[y] for y, files in files_by_label.items() if len(files) == 0]
    if missing:
        print(f"[警告] 当前湍流 {turb} 缺少这些调制格式：{missing}")
        if args.strict_mods:
            print("[跳过] strict_mods=True，类别不完整。")
            return None

    # 如果不严格，删除空类别，并重新映射标签
    if missing and not args.strict_mods:
        kept_mod_names = [mod_names[y] for y in non_empty_classes]
        old_to_new = {old_y: new_y for new_y, old_y in enumerate(non_empty_classes)}
        new_files_by_label = {}
        for old_y in non_empty_classes:
            new_files_by_label[old_to_new[old_y]] = files_by_label[old_y]
        mod_names_use = kept_mod_names
        files_by_label_use = new_files_by_label
    else:
        mod_names_use = mod_names
        files_by_label_use = files_by_label

    num_classes = len(mod_names_use)

    split_json_path = out_dir / "file_split.json"

    if split_json_path.exists() and not args.rebuild_cache:
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        def read_items(split_name):
            return [(Path(d["path"]), int(d["label"])) for d in split_info[split_name]]

        train_items = read_items("train")
        val_items = read_items("val")
        test_items = read_items("test")
        print(f"[读取已有文件级划分] {split_json_path}")

    else:
        train_d, val_d, test_d = split_files_per_class(
            files_by_label_use,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        train_items = flatten_split_dict(train_d)
        val_items = flatten_split_dict(val_d)
        test_items = flatten_split_dict(test_d)

        split_info = {
            "turbulence": turb,
            "turbulence_name": turb_readable,
            "mod_names": mod_names_use,
            "train": [{"path": str(p), "label": int(y), "label_name": mod_names_use[int(y)]} for p, y in train_items],
            "val": [{"path": str(p), "label": int(y), "label_name": mod_names_use[int(y)]} for p, y in val_items],
            "test": [{"path": str(p), "label": int(y), "label_name": mod_names_use[int(y)]} for p, y in test_items],
        }
        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"  file split: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    cache_train = cache_dir / "train.npz"
    cache_val = cache_dir / "val.npz"
    cache_test = cache_dir / "test.npz"

    if args.rebuild_cache:
        for p in [cache_train, cache_val, cache_test]:
            if p.exists():
                p.unlink()

    print(f"[{now_time()}] 构建/读取窗口数据 cache ...")
    train_data = build_split_cache(
        train_items, cache_train,
        window_len=args.window_len,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        normalize_mode=args.normalize,
        channel_mode=args.channel_mode,
        bin_format=args.bin_format,
    )
    val_data = build_split_cache(
        val_items, cache_val,
        window_len=args.window_len,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        normalize_mode=args.normalize,
        channel_mode=args.channel_mode,
        bin_format=args.bin_format,
    ) if len(val_items) > 0 else None
    test_data = build_split_cache(
        test_items, cache_test,
        window_len=args.window_len,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        normalize_mode=args.normalize,
        channel_mode=args.channel_mode,
        bin_format=args.bin_format,
    )

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    if val_data is not None:
        X_val, y_val = val_data["X"], val_data["y"]
    else:
        # 没有验证文件时，从训练窗口中切一小部分作为 val；这只在文件数极少时使用
        n = X_train.shape[0]
        idx = np.arange(n)
        np.random.default_rng(args.seed).shuffle(idx)
        n_val = max(1, int(0.15 * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]

    in_ch = int(X_train.shape[1])

    print(f"  windows: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  input shape: C={in_ch}, L={X_train.shape[-1]}, classes={num_classes}")
    print(f"  labels: {mod_names_use}")

    # 保存数据摘要
    data_summary = {
        "turbulence": turb,
        "turbulence_name": turb_readable,
        "mod_names": mod_names_use,
        "input_channels": in_ch,
        "window_len": args.window_len,
        "stride": args.stride,
        "max_windows_per_file": args.max_windows_per_file,
        "normalize": args.normalize,
        "channel_mode": args.channel_mode,
        "num_train_windows": int(len(y_train)),
        "num_val_windows": int(len(y_val)),
        "num_test_windows": int(len(y_test)),
        "class_count_train": np.bincount(y_train, minlength=num_classes).astype(int).tolist(),
        "class_count_val": np.bincount(y_val, minlength=num_classes).astype(int).tolist(),
        "class_count_test": np.bincount(y_test, minlength=num_classes).astype(int).tolist(),
    }
    with open(out_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(data_summary, f, ensure_ascii=False, indent=2)

    train_ds = NpzSignalDataset(X_train, y_train)
    val_ds = NpzSignalDataset(X_val, y_val)
    test_ds = NpzSignalDataset(X_test, y_test)

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

    model = TurbulenceIndependentNet(
        in_ch=in_ch,
        num_classes=num_classes,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.03,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val_acc = -1.0
    best_epoch = -1
    patience_count = 0
    history = []

    ckpt_path = out_dir / "best_model.pt"

    print(f"[{now_time()}] 训练开始 device={device}, amp={args.amp}")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = yb.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n_sum += bs

        scheduler.step()

        train_loss = loss_sum / max(1, n_sum)

        yv, pv, _ = predict_loader(model, val_loader, device)
        val_acc = float((yv == pv).mean())

        yt, pt, _ = predict_loader(model, train_loader, device)
        train_acc = float((yt == pt).mean())

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
            f"val_acc={val_acc:.4f} | lr={lr_now:.3e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mod_names": mod_names_use,
                    "turbulence": turb,
                    "turbulence_name": turb_readable,
                    "args": vars(args),
                    "input_channels": in_ch,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"  Early stopping at epoch {epoch}, best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
            break

    train_seconds = time.time() - train_start
    print(f"[{now_time()}] 训练结束，用时 {train_seconds / 60:.2f} min")

    # 读取最佳模型
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    metrics = evaluate_and_save(
        model,
        loaders=loaders,
        device=device,
        mod_names=mod_names_use,
        out_dir=out_dir,
        prefix=f"{turb}_{turb_readable}",
    )

    # 提取 embedding，方便后续 t-SNE、SVM、跨湍流分析、论文可视化调用
    for split_name, loader in loaders.items():
        yy, pp, emb = extract_embeddings(model, loader, device)
        np.savez_compressed(
            out_dir / f"features_{split_name}.npz",
            embedding=emb,
            y_true=yy,
            y_pred=pp,
            mod_names=np.asarray(mod_names_use, dtype=object),
            turbulence=turb,
            turbulence_name=turb_readable,
        )

    final_record = {
        "turbulence": turb,
        "turbulence_name": turb_readable,
        "mod_names": mod_names_use,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "train_seconds": float(train_seconds),
        "metrics": metrics,
        "history": history,
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_record, f, ensure_ascii=False, indent=2)

    print(f"[结果] {turb} test_acc={metrics['test']['accuracy']:.4f}, "
          f"balanced_acc={metrics['test']['balanced_accuracy']:.4f}")
    print(f"[保存] {out_dir}")

    # 主动释放显存
    del model
    torch.cuda.empty_cache()

    return final_record


# =========================
# 6. 总控：每种湍流独立训练
# =========================

def collect_all_files(root: Path, mods: List[str], turbs: List[str]) -> Dict[str, Dict[str, List[Path]]]:
    files_by_mod_turb = {}
    for mod in mods:
        mod_dir = root / mod
        files_by_mod_turb[mod] = list_signal_files(mod_dir, turbs)
    return files_by_mod_turb


def parse_args():
    p = argparse.ArgumentParser(
        description="每种湍流条件独立进行调制格式识别：OFDM解调后复数序列 -> 1D ResNet-SE 分类"
    )

    p.add_argument("--root", type=str, default=DEFAULT_ROOT, help="数据根目录，下面应包含各调制格式文件夹")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="结果输出目录")

    p.add_argument("--mods", type=str, nargs="*", default=None,
                   help="调制格式列表；不填则自动发现，例如 QPSK 16QAM 32QAM 64QAM 128QAM 256QAM")
    p.add_argument("--turbs", type=str, nargs="*", default=DEFAULT_TURBS,
                   help="湍流子目录，例如 sub01 sub02 sub03")
    p.add_argument("--strict-mods", action="store_true",
                   help="如果某个湍流下缺少某些调制格式，则跳过该湍流")

    p.add_argument("--window-len", type=int, default=2048,
                   help="每个样本的符号窗口长度；若你之前论文实验用8192，可改为8192")
    p.add_argument("--stride", type=int, default=1024, help="滑窗步长")
    p.add_argument("--max-windows-per-file", type=int, default=80,
                   help="每个接收文件最多切多少个窗口；0 表示不限制")
    p.add_argument("--normalize", type=str, default="power",
                   choices=["power", "zscore", "none"],
                   help="窗口归一化方式")
    p.add_argument("--channel-mode", type=str, default="iq_amp_phase",
                   choices=["iq", "iq_amp", "iq_amp_phase"],
                   help="输入通道：iq / iq_amp / iq_amp_phase")
    p.add_argument("--bin-format", type=str, default="interleaved_float32",
                   choices=["interleaved_float32", "complex64"],
                   help="读取 .bin 的格式")

    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rebuild-cache", action="store_true", help="重新构建 npz cache")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--grad-clip", type=float, default=3.0)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Windows/PyCharm 下建议先用0；Linux可设为4或8")
    p.add_argument("--no-amp", action="store_true", help="关闭CUDA AMP混合精度")

    return p.parse_args()


def main():
    args = parse_args()
    args.amp = not args.no_amp

    set_seed(args.seed)

    root = Path(args.root)
    out = Path(args.out)
    ensure_dir(out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("Turbulence-independent modulation recognition")
    print("=" * 88)
    print(f"Time       : {now_time()}")
    print(f"Root       : {root}")
    print(f"Out        : {out}")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA       : {torch.version.cuda}")
    print("=" * 88)

    if args.mods is None or len(args.mods) == 0:
        mods = discover_mods(root, args.turbs)
    else:
        mods = args.mods

    if len(mods) < 2:
        raise RuntimeError(f"有效调制格式少于2类，请检查目录：{root}")

    mods = sorted(mods, key=natural_mod_key)
    print(f"Modulations: {mods}")
    print(f"Turbulence : {args.turbs}")

    files_by_mod_turb = collect_all_files(root, mods, args.turbs)

    # 保存全局配置
    global_config = {
        "time": now_time(),
        "root": str(root),
        "out": str(out),
        "mods": mods,
        "turbs": args.turbs,
        "args": vars(args),
    }
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(global_config, f, ensure_ascii=False, indent=2)

    all_results = []
    total_start = time.time()

    for turb in args.turbs:
        result = train_one_turbulence(
            turb=turb,
            mod_names=mods,
            files_by_mod_turb=files_by_mod_turb,
            args=args,
            device=device,
        )
        if result is not None:
            all_results.append(result)

    total_seconds = time.time() - total_start

    summary = {
        "time": now_time(),
        "total_seconds": float(total_seconds),
        "results": [
            {
                "turbulence": r["turbulence"],
                "turbulence_name": r["turbulence_name"],
                "best_val_acc": r["best_val_acc"],
                "test_acc": r["metrics"]["test"]["accuracy"],
                "test_balanced_acc": r["metrics"]["test"]["balanced_accuracy"],
                "output_dir": r["output_dir"],
            }
            for r in all_results
        ],
    }

    summary_path = out / "independent_turbulence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("全部湍流条件独立识别完成")
    print("=" * 88)
    for r in summary["results"]:
        print(
            f"{r['turbulence']} ({r['turbulence_name']}): "
            f"test_acc={r['test_acc']:.4f}, "
            f"balanced_acc={r['test_balanced_acc']:.4f}, "
            f"out={r['output_dir']}"
        )
    print(f"Summary: {summary_path}")
    print(f"Total time: {total_seconds / 60:.2f} min")


if __name__ == "__main__":
    main()
