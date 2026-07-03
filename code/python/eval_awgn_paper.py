# -*- coding: utf-8 -*-
"""
eval_awgn_paper.py

Paper-level AWGN robustness evaluation for experimental FSO-OFDM modulation recognition.

This script adds complex AWGN to experimental received symbols rx_sc, regenerates
constellation density maps (CDM) and blind statistical features, then evaluates a trained
CDM+Stats classifier.

Protocols
---------
1) additional:
   Adds AWGN with a specified Additional SNR to every test frame.
   X-axis: Additional SNR (dB). This is NOT the original channel SNR.

2) drop:
   Degrades every frame by a specified SNR drop relative to its measured frame SNR.
   Requires --snr-csv with columns out_mat and snr_frame_db.
   X-axis: SNR degradation (dB).

3) target:
   Uses only frames with measured SNR higher than target SNR and degrades them
   to the target effective SNR.
   Requires --snr-csv with columns out_mat and snr_frame_db.
   X-axis: Target effective SNR (dB).

Outputs
-------
metrics_all_trials.csv
metrics_mean.csv
per_class_metrics_all_trials.csv
per_class_metrics_mean.csv
figures/*.png, *.pdf, *.svg
confusion_matrices/*.png, *.pdf, *.svg
protocol_description.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
NUM_CLASSES = len(MOD_NAMES)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rx_sc(path: Path) -> np.ndarray:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "rx_sc" not in mat:
        raise KeyError(f"{path} does not contain rx_sc")
    x = np.asarray(mat["rx_sc"])
    x = np.squeeze(x)
    if not np.iscomplexobj(x):
        x = x.astype(np.float32).astype(np.complex64)
    return x.astype(np.complex64)


def get_signal_power(x: np.ndarray) -> float:
    z = x.reshape(-1).astype(np.complex64)
    m = np.isfinite(z.real) & np.isfinite(z.imag)
    if m.sum() == 0:
        return 1.0
    p = float(np.mean(np.abs(z[m]) ** 2))
    return p if np.isfinite(p) and p > 0 else 1.0


def add_complex_noise_by_variance(x: np.ndarray, noise_var: float, rng: np.random.Generator) -> np.ndarray:
    if noise_var <= 0 or not np.isfinite(noise_var):
        return x.astype(np.complex64)
    noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return (x + noise.astype(np.complex64)).astype(np.complex64)


def add_awgn_additional_snr(x: np.ndarray, additional_snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if additional_snr_db is None:
        return x.astype(np.complex64)
    p = get_signal_power(x)
    noise_var = p / (10.0 ** (float(additional_snr_db) / 10.0))
    return add_complex_noise_by_variance(x, noise_var, rng)


def add_awgn_snr_drop(x: np.ndarray, measured_snr_db: float, drop_db: float, rng: np.random.Generator) -> Tuple[np.ndarray, float, bool]:
    if not np.isfinite(measured_snr_db):
        return x.astype(np.complex64), np.nan, False
    if drop_db <= 0:
        return x.astype(np.complex64), float(measured_snr_db), True
    target_snr_db = float(measured_snr_db) - float(drop_db)
    p = get_signal_power(x)
    snr_orig = 10.0 ** (float(measured_snr_db) / 10.0)
    snr_target = 10.0 ** (float(target_snr_db) / 10.0)
    noise_var_add = max(0.0, p / snr_target - p / snr_orig)
    return add_complex_noise_by_variance(x, noise_var_add, rng), target_snr_db, True


def add_awgn_target_snr(x: np.ndarray, measured_snr_db: float, target_snr_db: float, rng: np.random.Generator, margin_db: float = 0.1) -> Tuple[np.ndarray, float, bool]:
    if not np.isfinite(measured_snr_db):
        return x.astype(np.complex64), np.nan, False
    if float(measured_snr_db) <= float(target_snr_db) + margin_db:
        return x.astype(np.complex64), np.nan, False
    p = get_signal_power(x)
    snr_orig = 10.0 ** (float(measured_snr_db) / 10.0)
    snr_target = 10.0 ** (float(target_snr_db) / 10.0)
    noise_var_add = max(0.0, p / snr_target - p / snr_orig)
    return add_complex_noise_by_variance(x, noise_var_add, rng), float(target_snr_db), True


def make_cdm(rx_sc: np.ndarray, nbin: int = 64, clip_val: float = 3.0) -> np.ndarray:
    z = rx_sc.reshape(-1).astype(np.complex64)
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if z.size == 0:
        return np.zeros((1, nbin, nbin), dtype=np.float32)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    zr = np.clip(z.real, -clip_val, clip_val)
    zi = np.clip(z.imag, -clip_val, clip_val)
    edges = np.linspace(-clip_val, clip_val, nbin + 1)
    H, _, _ = np.histogram2d(zi, zr, bins=[edges, edges])
    H = np.log1p(H)
    H = H / (np.max(H) + 1e-12)
    return H[None, :, :].astype(np.float32)


def skew_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return float(np.mean(((x - mu) / sd) ** 3))


def kurt_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return float(np.mean(((x - mu) / sd) ** 4))


def make_stats(rx_sc: np.ndarray) -> np.ndarray:
    z = rx_sc.reshape(-1).astype(np.complex64)
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if z.size == 0:
        return np.full((16,), np.nan, dtype=np.float32)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    amp = np.abs(z)
    ii = z.real
    qq = z.imag
    ph = np.unwrap(np.angle(z))
    dph = np.diff(ph)
    if np.std(ii) < 1e-12 or np.std(qq) < 1e-12:
        iq_corr = 0.0
    else:
        iq_corr = float(np.corrcoef(ii, qq)[0, 1])
    return np.array([
        np.mean(amp), np.std(amp), skew_np(amp), kurt_np(amp),
        10 * np.log10(np.max(np.abs(z) ** 2) / (np.mean(np.abs(z) ** 2) + 1e-12)),
        np.mean(ii), np.std(ii), skew_np(ii), kurt_np(ii),
        np.mean(qq), np.std(qq), skew_np(qq), kurt_np(qq),
        np.std(dph) if dph.size > 0 else np.nan,
        np.abs(np.mean(np.exp(1j * np.angle(z)))),
        iq_corr,
    ], dtype=np.float32)


class CDMBranch(nn.Module):
    def __init__(self, out_dim: int = 64, dropout: float = 0.125):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(96, out_dim), nn.SiLU(inplace=True), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.fc(self.net(x))


class StatsBranch(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 32, dropout: float = 0.125):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class CDMStatsNet(nn.Module):
    def __init__(self, stats_dim: int = 16, num_classes: int = 6, dropout: float = 0.25):
        super().__init__()
        self.cdm_branch = CDMBranch(64, dropout * 0.5)
        self.stats_branch = StatsBranch(stats_dim, 32, dropout * 0.5)
        self.classifier = nn.Sequential(
            nn.Linear(96, 128), nn.BatchNorm1d(128), nn.SiLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
    def forward(self, cdm, stats):
        h = torch.cat([self.cdm_branch(cdm), self.stats_branch(stats)], dim=1)
        return self.classifier(h)


def infer_turb_from_model_dir(model_dir: Path) -> str:
    s = (model_dir.name + "_" + model_dir.parent.name).lower()
    if "weak" in s:
        return "weak"
    if "strong" in s:
        return "strong"
    return "unknown"


def infer_model_label(model_dir: Path, user_label: Optional[str] = None) -> str:
    if user_label:
        return user_label
    s = (model_dir.name + "_" + model_dir.parent.name).lower()
    if "noiseaug" in s:
        return "NoiseAug"
    if "baseline" in s:
        return "Baseline"
    return model_dir.parent.name if "seed" in model_dir.parent.name.lower() else model_dir.name


def _remap_noiseaug_state_dict(state: dict) -> dict:
    """Remap state dict from train_cdm_stats_noiseaug.py key names to eval_awgn_paper.py names."""
    if not any(k.startswith("cdm.") for k in state):
        return state  # already in target format
    remapped = {}
    for k, v in state.items():
        if k.startswith("cdm.net."):
            idx = int(k.split(".")[2])
            if idx == 17:  # Linear(96,64) -> cdm_branch.fc.1
                nk = k.replace("cdm.net.17", "cdm_branch.fc.1")
            else:
                nk = k.replace("cdm.net.", "cdm_branch.net.")
            remapped[nk] = v
        elif k.startswith("stats.net."):
            remapped[k.replace("stats.net.", "stats_branch.net.")] = v
        elif k.startswith("cls."):
            remapped[k.replace("cls.", "classifier.")] = v
        else:
            remapped[k] = v
    return remapped


def load_model(model_dir: Path, device: torch.device):
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    stats_mean = np.asarray(ckpt["stats_mean"], dtype=np.float32).reshape(-1)
    stats_std = np.asarray(ckpt["stats_std"], dtype=np.float32).reshape(-1)
    stats_std[stats_std < 1e-6] = 1.0
    args = ckpt.get("args", {})
    dropout = float(args.get("dropout", 0.25))
    stats_dim = int(ckpt.get("stats_dim", len(stats_mean)))
    model = CDMStatsNet(stats_dim=stats_dim, dropout=dropout).to(device)
    state = _remap_noiseaug_state_dict(ckpt["model_state"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [WARN] load_state_dict: missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}")
    model.eval()
    turb_name = str(ckpt.get("turb_name", infer_turb_from_model_dir(model_dir)))
    return model, stats_mean, stats_std, turb_name


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        if 0 <= int(a) < NUM_CLASSES and 0 <= int(b) < NUM_CLASSES:
            cm[int(a), int(b)] += 1
    return cm


def per_class_metrics_from_cm(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(MOD_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        rows.append({
            "class_id": i, "mod_name": name,
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "support": int(support), "class_accuracy": float(recall)
        })
    return pd.DataFrame(rows)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix_np(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else np.nan
    per = per_class_metrics_from_cm(cm)
    return {
        "accuracy": acc,
        "macro_f1": float(per["f1"].mean()),
        "macro_precision": float(per["precision"].mean()),
        "macro_recall": float(per["recall"].mean()),
    }


def normalize_path_str(x: str) -> str:
    return str(x).replace("\\", "/").lower()


def load_snr_table(path: Optional[str], snr_col: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"snr-csv not found: {p}")
    df = pd.read_csv(p)
    if "out_mat" not in df.columns:
        raise KeyError("snr-csv must contain out_mat column")
    if snr_col not in df.columns:
        raise KeyError(f"{snr_col} not found in snr-csv. Columns: {list(df.columns)}")
    df = df.copy()
    df["out_mat_norm"] = df["out_mat"].astype(str).apply(normalize_path_str)
    df["out_mat_name"] = df["out_mat"].astype(str).apply(lambda x: Path(x).name.lower())
    df[snr_col] = pd.to_numeric(df[snr_col], errors="coerce")
    return df[["out_mat_norm", "out_mat_name", snr_col]].drop_duplicates()


def attach_snr_to_rows(rows: pd.DataFrame, snr_table: Optional[pd.DataFrame], snr_col: str) -> pd.DataFrame:
    rows = rows.copy()
    if snr_table is None:
        rows["measured_snr_db"] = np.nan
        return rows
    rows["out_mat_norm"] = rows["out_mat"].astype(str).apply(normalize_path_str)
    rows["out_mat_name"] = rows["out_mat"].astype(str).apply(lambda x: Path(x).name.lower())
    merged = rows.merge(snr_table[["out_mat_norm", snr_col]], on="out_mat_norm", how="left")
    merged = merged.rename(columns={snr_col: "measured_snr_db"})
    missing = merged["measured_snr_db"].isna()
    if missing.any():
        fb = rows.loc[missing, ["out_mat_name"]].merge(snr_table[["out_mat_name", snr_col]], on="out_mat_name", how="left")
        merged.loc[missing, "measured_snr_db"] = fb[snr_col].values
    return merged.drop(columns=[c for c in ["out_mat_norm", "out_mat_name"] if c in merged.columns])


def parse_float_or_clean(x: str) -> Optional[float]:
    s = str(x).lower().strip()
    if s in ["clean", "none", "inf", "infinity"]:
        return None
    return float(s)


def condition_values_from_args(args):
    if args.protocol == "additional":
        return [(str(v).lower(), parse_float_or_clean(str(v))) for v in args.additional_snrs]
    if args.protocol == "drop":
        return [(str(v), float(v)) for v in args.drops]
    if args.protocol == "target":
        return [(str(v), float(v)) for v in args.target_snrs]
    raise ValueError(args.protocol)


def x_axis_label(protocol: str) -> str:
    if protocol == "additional":
        return "Additional SNR (dB)"
    if protocol == "drop":
        return "SNR degradation (dB)"
    if protocol == "target":
        return "Target effective SNR (dB)"
    return "SNR (dB)"


def x_numeric(label: str, value: Optional[float], protocol: str) -> float:
    if str(label).lower() == "clean":
        return 100.0 if protocol == "additional" else 0.0
    return float(value)


@torch.no_grad()
def evaluate_condition(model, stats_mean, stats_std, rows, protocol, condition_value, trial, batch_size, device, base_seed):
    rng = np.random.default_rng(base_seed + 10007 * trial + int(1000 * (condition_value if condition_value is not None else 999)))
    records, cdms, statss, ys, metas = [], [], [], [], []

    def flush():
        nonlocal cdms, statss, ys, metas, records
        if not ys:
            return
        xb = torch.from_numpy(np.stack(cdms)).to(device)
        sb = torch.from_numpy(np.stack(statss)).to(device)
        prob = torch.softmax(model(xb, sb), dim=1).detach().cpu().numpy()
        pred = prob.argmax(axis=1)
        for i in range(len(ys)):
            rec = metas[i].copy()
            rec.update({
                "y_true": int(ys[i]), "y_pred": int(pred[i]),
                "true_name": MOD_NAMES[int(ys[i])], "pred_name": MOD_NAMES[int(pred[i])]
            })
            for k, name in enumerate(MOD_NAMES):
                rec[f"prob_{name}"] = float(prob[i, k])
            records.append(rec)
        cdms, statss, ys, metas = [], [], [], []

    for _, row in rows.iterrows():
        try:
            rx_sc = load_rx_sc(Path(row["out_mat"]))
        except Exception:
            continue
        y = int(row["mod_label"])
        measured_snr_db = float(row.get("measured_snr_db", np.nan))
        used = True
        effective_snr_db = np.nan

        if protocol == "additional":
            rx_noisy = add_awgn_additional_snr(rx_sc, condition_value, rng)
        elif protocol == "drop":
            rx_noisy, effective_snr_db, used = add_awgn_snr_drop(rx_sc, measured_snr_db, float(condition_value), rng)
        elif protocol == "target":
            rx_noisy, effective_snr_db, used = add_awgn_target_snr(rx_sc, measured_snr_db, float(condition_value), rng)
        else:
            raise ValueError(protocol)
        if not used:
            continue

        cdm = make_cdm(rx_noisy)
        stats = make_stats(rx_noisy)
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        stats = (stats - stats_mean) / stats_std
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        cdms.append(cdm)
        statss.append(stats)
        ys.append(y)
        metas.append({
            "out_mat": str(row["out_mat"]),
            "mod_label": y,
            "mod_name": row.get("mod_name", MOD_NAMES[y]),
            "rx_file": row.get("rx_file", ""),
            "measured_snr_db": measured_snr_db,
            "effective_snr_db": effective_snr_db,
        })
        if len(ys) >= batch_size:
            flush()
    flush()
    pred_df = pd.DataFrame(records)
    cm = confusion_matrix_np(pred_df["y_true"].values, pred_df["y_pred"].values) if len(pred_df) else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    return pred_df, cm


def setup_paper_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig_all(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def plot_metric_curve(metrics_mean: pd.DataFrame, protocol: str, metric: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (turb, model_label), d in metrics_mean.groupby(["turb_name", "model_label"]):
        d = d.sort_values("x_value")
        ax.plot(d["x_value"], d[metric], marker="o", label=f"{model_label}-{turb}")
        std_col = f"{metric}_std"
        if std_col in d.columns:
            y = d[metric].values
            s = d[std_col].fillna(0).values
            ax.fill_between(d["x_value"].values, y - s, y + s, alpha=0.12)
    ax.set_xlabel(x_axis_label(protocol))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.03)
    if protocol == "additional":
        ticks = sorted([v for v in metrics_mean["x_value"].unique() if v < 99]) + [100]
        labels = [str(int(v)) for v in ticks if v < 99] + ["clean"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ticks = sorted(metrics_mean["x_value"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in ticks])
    ax.set_title(f"{metric.replace('_', ' ').title()} under controlled AWGN degradation")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    savefig_all(fig, out_base)
    plt.close(fig)


def plot_per_class_heatmaps(per_class_mean: pd.DataFrame, protocol: str, out_dir: Path):
    for (model_label, turb), d0 in per_class_mean.groupby(["model_label", "turb_name"]):
        d = d0.copy()
        cond_order = sorted(d["x_value"].unique())
        mat = np.zeros((NUM_CLASSES, len(cond_order)), dtype=float)
        for i, cls in enumerate(MOD_NAMES):
            for j, xv in enumerate(cond_order):
                sub = d[(d["mod_name"] == cls) & (d["x_value"] == xv)]
                mat[i, j] = sub["class_accuracy"].mean() if len(sub) else np.nan
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(np.arange(NUM_CLASSES))
        ax.set_yticklabels(MOD_NAMES)
        ax.set_xticks(np.arange(len(cond_order)))
        labels = ["clean" if (protocol == "additional" and v >= 99) else (str(int(v)) if float(v).is_integer() else str(v)) for v in cond_order]
        ax.set_xticklabels(labels)
        ax.set_xlabel(x_axis_label(protocol))
        ax.set_ylabel("Modulation format")
        ax.set_title(f"Per-class accuracy: {model_label}-{turb}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Class accuracy")
        for i in range(NUM_CLASSES):
            for j in range(len(cond_order)):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val*100:.0f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        safe = f"per_class_accuracy_{model_label}_{turb}".replace(" ", "_").replace("/", "_")
        savefig_all(fig, out_dir / safe)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(MOD_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_yticklabels(MOD_NAMES)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    savefig_all(fig, out_base)
    plt.close(fig)


def write_protocol_description(out_dir: Path, protocol: str):
    if protocol == "additional":
        detail = "The x-axis is Additional SNR, which controls the power of artificially injected AWGN only. It is not the original channel SNR of the experimental FSO link."
    elif protocol == "drop":
        detail = "The x-axis is SNR degradation. Each frame is degraded by a specified dB relative to its measured original frame-level SNR."
    else:
        detail = "The x-axis is Target effective SNR. Only frames whose measured original SNR is higher than the target SNR are used."
    text = f"""# AWGN robustness evaluation protocol: {protocol}

This experiment adds complex AWGN to experimentally received complex symbols `rx_sc`, then regenerates CDM and blind statistical features before classification.

{detail}

The noise-degraded samples are used for robustness evaluation only. They should not be described as newly collected real low-SNR experimental data.
"""
    (out_dir / "protocol_description.md").write_text(text, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Paper-level AWGN robustness evaluation for CDM+Stats models.")
    p.add_argument("--protocol", default="additional", choices=["additional", "drop", "target"])
    p.add_argument("--model-dirs", nargs="+", required=True)
    p.add_argument("--model-labels", nargs="*", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--additional-snrs", nargs="+", default=["clean", "25", "22", "20", "18", "16", "14", "12", "10"])
    p.add_argument("--drops", nargs="+", type=float, default=[0, 2, 4, 6, 8, 10, 12])
    p.add_argument("--target-snrs", nargs="+", type=float, default=[18, 16, 14, 12, 10])
    p.add_argument("--snr-csv", default=None)
    p.add_argument("--snr-col", default="snr_frame_db")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--confusion-conditions", nargs="*", default=["clean", "20", "16", "12"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    model_dirs = [Path(x) for x in args.model_dirs]
    if args.model_labels and len(args.model_labels) != len(model_dirs):
        raise ValueError("--model-labels must match --model-dirs length")
    if args.out_root:
        out_dir = Path(args.out_root)
    else:
        out_dir = model_dirs[0].parent / f"awgn_paper_eval_{args.protocol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    setup_paper_style()

    if args.protocol in ["drop", "target"] and not args.snr_csv:
        raise ValueError("--snr-csv is required for drop/target protocols")
    snr_table = load_snr_table(args.snr_csv, args.snr_col)
    conds = condition_values_from_args(args)
    confusion_set = set(str(x).lower() for x in args.confusion_conditions)

    print("=" * 90)
    print(f"Protocol: {args.protocol}")
    print(f"Device: {device}")
    print(f"Out dir: {out_dir}")
    print("=" * 90)

    all_metrics, all_per_class = [], []

    for mi, model_dir in enumerate(model_dirs):
        label = infer_model_label(model_dir, args.model_labels[mi] if args.model_labels else None)
        print(f"\nModel: {label} | {model_dir}")
        model, stats_mean, stats_std, turb = load_model(model_dir, device)
        split_file = model_dir / "split.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split.csv in {model_dir}")
        split = pd.read_csv(split_file)
        rows = split[split["split"].astype(str) == "test"].copy()
        rows["out_mat"] = rows["out_mat"].astype(str)
        rows["mod_label"] = rows["mod_label"].astype(int)
        rows = attach_snr_to_rows(rows, snr_table, args.snr_col)
        print(f"Turbulence: {turb}, test frames: {len(rows)}")
        if args.protocol in ["drop", "target"]:
            print(f"Valid SNR frames: {int(np.isfinite(rows['measured_snr_db']).sum())}")

        for cond_label, cond_value in conds:
            pred_dfs = []
            for trial in range(args.trials):
                t0 = time.time()
                pred_df, cm = evaluate_condition(model, stats_mean, stats_std, rows, args.protocol, cond_value, trial, args.batch_size, device, args.seed + 100000 * mi)
                if len(pred_df):
                    met = metrics_from_predictions(pred_df["y_true"].values, pred_df["y_pred"].values)
                    n_eff = len(pred_df)
                else:
                    met = {"accuracy": np.nan, "macro_f1": np.nan, "macro_precision": np.nan, "macro_recall": np.nan}
                    n_eff = 0
                xv = x_numeric(cond_label, cond_value, args.protocol)
                rec = {"model_label": label, "model_dir": str(model_dir), "turb_name": turb, "protocol": args.protocol,
                       "condition_label": str(cond_label), "condition_value": np.nan if cond_value is None else float(cond_value),
                       "x_value": xv, "trial": trial, "num_test_frames": int(len(rows)), "num_effective_frames": int(n_eff), **met}
                all_metrics.append(rec)
                per = per_class_metrics_from_cm(cm)
                per["model_label"] = label
                per["model_dir"] = str(model_dir)
                per["turb_name"] = turb
                per["protocol"] = args.protocol
                per["condition_label"] = str(cond_label)
                per["condition_value"] = np.nan if cond_value is None else float(cond_value)
                per["x_value"] = xv
                per["trial"] = trial
                per["num_effective_frames"] = int(n_eff)
                all_per_class.append(per)
                pred_df["model_label"] = label
                pred_df["turb_name"] = turb
                pred_df["protocol"] = args.protocol
                pred_df["condition_label"] = str(cond_label)
                pred_df["trial"] = trial
                pred_dfs.append(pred_df)
                if trial == 0 and str(cond_label).lower() in confusion_set:
                    safe = f"cm_{label}_{turb}_{args.protocol}_{cond_label}".replace(" ", "_").replace("/", "_")
                    plot_confusion_matrix(cm, f"{label}-{turb}, {args.protocol}={cond_label}", out_dir / "confusion_matrices" / safe)
                print(f"  {cond_label:>6} trial {trial}: Acc={rec['accuracy']:.4f}, F1={rec['macro_f1']:.4f}, N={n_eff}, {time.time()-t0:.1f}s")
            if pred_dfs:
                safe = f"pred_{label}_{turb}_{args.protocol}_{cond_label}".replace(" ", "_").replace("/", "_")
                pd.concat(pred_dfs, ignore_index=True).to_csv(out_dir / "predictions" / f"{safe}.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(all_metrics)
    per_class_df = pd.concat(all_per_class, ignore_index=True) if all_per_class else pd.DataFrame()
    metrics_df.to_csv(out_dir / "metrics_all_trials.csv", index=False, encoding="utf-8-sig")
    per_class_df.to_csv(out_dir / "per_class_metrics_all_trials.csv", index=False, encoding="utf-8-sig")
    group_cols = ["model_label", "turb_name", "protocol", "condition_label", "condition_value", "x_value"]
    metrics_mean = metrics_df.groupby(group_cols, dropna=False).agg(
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        macro_precision=("macro_precision", "mean"), macro_precision_std=("macro_precision", "std"),
        macro_recall=("macro_recall", "mean"), macro_recall_std=("macro_recall", "std"),
        num_effective_frames=("num_effective_frames", "mean"),
    ).reset_index()
    metrics_mean.to_csv(out_dir / "metrics_mean.csv", index=False, encoding="utf-8-sig")
    per_class_mean = per_class_df.groupby(
        ["model_label", "turb_name", "protocol", "condition_label", "condition_value", "x_value", "class_id", "mod_name"],
        dropna=False
    ).agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), support=("support", "mean"), class_accuracy=("class_accuracy", "mean")).reset_index()
    per_class_mean.to_csv(out_dir / "per_class_metrics_mean.csv", index=False, encoding="utf-8-sig")
    plot_metric_curve(metrics_mean, args.protocol, "accuracy", out_dir / "figures" / f"{args.protocol}_accuracy_curve")
    plot_metric_curve(metrics_mean, args.protocol, "macro_f1", out_dir / "figures" / f"{args.protocol}_macro_f1_curve")
    plot_per_class_heatmaps(per_class_mean, args.protocol, out_dir / "figures")
    write_protocol_description(out_dir, args.protocol)
    (out_dir / "summary.json").write_text(json.dumps({"protocol": args.protocol, "out_dir": str(out_dir), "trials": args.trials}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
