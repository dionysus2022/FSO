# -*- coding: utf-8 -*-
"""
stage2_round1.py — Stage II Round-1 完整 S2-S5 编排脚本

模型: M1 CDM-Stats-Radial
Sim 数据: Calibrated-Sim-v2.2 (12000 帧) — 真实分布校准的多扰动仿真
AWGN-only: 12000 帧 (S5 消融对照)
Seeds: 2026, 2027, 2028, 2029, 2030
Real test: frozen split test set (324 行, 每 turb 162 行)

阶段:
  S2: SimOnly (Calibrated-Sim-v2.2 train → real test)
  S3: SimPretrain+RealFT × {5%,10%,20%,50%,100%} + Real-scratch 对照
  S4: Joint Training (direct mix / weighted 3:1 / MMD)
  S5: AWGN-only SimOnly (消融)

可恢复: 每个 run 检查 metrics.json 是否存在，存在则跳过。
"""

from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    FROZEN_SPLIT_CSV, STAGE2_ROOT, REAL_TURB_MAP,
    load_frozen_split, subset_real_train,
    ensure_dir, now_str,
)
from train_mrf_lowmem import (
    MultiRepNet, MultiRepDataset, load_sample_from_mat,
    set_seed, MOD_NAMES, LABEL_TO_MOD,
    class_weights_from_y, predict_loader,
    save_confusion_matrix, save_training_curve,
)

# ─────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────
MODEL_MODE = "cdm_stats_radial"
STATS_DIM = 48
NUM_CLASSES = 6
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
REAL_TURBS = ["weak", "strong"]
RATIOS = [0.05, 0.10, 0.20, 0.50, 1.00]
RATIO_LABELS = ["5pct", "10pct", "20pct", "50pct", "100pct"]
HIGH_ORDER_LABELS = [3, 4, 5]  # 64QAM, 128QAM, 256QAM

OUT_ROOT = STAGE2_ROOT / "round1"
SIM_MANIFEST_12K = STAGE2_ROOT / "sim_data_v2_2_12k" / "manifest_sim_v2_2_12k.csv"
AWGN_MANIFEST_12K = STAGE2_ROOT / "sim_data_awgn_only" / "manifest_sim_awgn_only.csv"

MMD_WEIGHT = 0.1


@dataclass
class HyperParams:
    epochs: int = 50
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    label_smoothing: float = 0.03
    patience: int = 8
    grad_clip: float = 3.0


HP_PRETRAIN = HyperParams(epochs=50, batch=32, lr=1e-3, patience=8)
HP_SCRATCH = HyperParams(epochs=50, batch=16, lr=1e-3, patience=8)
HP_FINETUNE = HyperParams(epochs=20, batch=16, lr=1e-4, patience=5)
HP_JOINT = HyperParams(epochs=50, batch=32, lr=1e-3, patience=8)
HP_AWGN = HyperParams(epochs=50, batch=32, lr=1e-3, patience=8)


# ─────────────────────────────────────────────────────────
# Cache 构建（扩展版，存储 dataset_source 和 turb_name）
# ─────────────────────────────────────────────────────────
def build_pilot_cache(
    rows_df: pd.DataFrame,
    cache_path: Path,
    rebuild: bool = False,
) -> dict:
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    use_iq = False
    use_radial = True

    cdm_list, stats_list = [], []
    y_list, split_list, path_list = [], [], []
    src_list, turb_list = [], []

    n = len(rows_df)
    t0 = time.time()
    for i, (_, row) in enumerate(rows_df.iterrows(), start=1):
        p = Path(row["out_mat"])
        try:
            s = load_sample_from_mat(p, use_iq=use_iq, use_radial=use_radial)
        except Exception as e:
            print(f"  [跳过] {p.name}: {e}")
            continue

        y = int(row["mod_label"])
        if y < 0 or y >= NUM_CLASSES:
            continue

        cdm_list.append(s["cdm"])
        stats_list.append(s["stats"])
        y_list.append(y)
        split_list.append(str(row.get("split", "train")))
        path_list.append(str(p))
        src_list.append(str(row.get("dataset_source", "")))
        turb_list.append(str(row.get("turb_name", "")))

        if i % 500 == 0 or i == n:
            print(f"    cache {i}/{n} | {time.time()-t0:.1f}s")

    if len(y_list) == 0:
        raise RuntimeError("没有成功读取任何样本")

    X_cdm = np.stack(cdm_list, axis=0).astype(np.float32)
    X_stats = np.stack(stats_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    split = np.asarray(split_list, dtype=object)
    paths = np.asarray(path_list, dtype=object)
    sources = np.asarray(src_list, dtype=object)
    turbs = np.asarray(turb_list, dtype=object)
    X_iq = np.zeros((len(y), 1, 1), dtype=np.float32)

    if X_stats.shape[1] < STATS_DIM:
        X_stats = np.pad(X_stats, ((0, 0), (0, STATS_DIM - X_stats.shape[1])),
                         constant_values=np.nan)
    X_stats = X_stats[:, :STATS_DIM]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_cdm=X_cdm, X_iq=X_iq, X_stats=X_stats, y=y,
        split=split, paths=paths, sources=sources, turbs=turbs,
    )
    print(f"    [cache] {len(y)} 样本 → {cache_path.name}")
    return {
        "X_cdm": X_cdm, "X_iq": X_iq, "X_stats": X_stats, "y": y,
        "split": split, "paths": paths, "sources": sources, "turbs": turbs,
    }


# ─────────────────────────────────────────────────────────
# Sim val 划分
# ─────────────────────────────────────────────────────────
def stratified_sim_split(sim_df: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = sim_df.copy()
    df["split"] = "train"
    val_idx = []
    for mod, grp in df.groupby("mod_name"):
        n_val = max(1, int(round(len(grp) * 0.1)))
        idx = rng.choice(grp.index.values, n_val, replace=False)
        val_idx.extend(idx)
    df.loc[val_idx, "split"] = "val"
    return df


# ─────────────────────────────────────────────────────────
# Stats 归一化
# ─────────────────────────────────────────────────────────
def compute_stats_mean_std(X_stats: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train = X_stats[train_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_stats_norm(X_stats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X_stats.copy().astype(np.float32)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(mean, inds[1])
    X = (X - mean) / std
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# 特征提取 + MMD（从 train_mrf_aug_da.py 复制）
# ─────────────────────────────────────────────────────────
def extract_features(model, cdm, iq, stats):
    feats = []
    if model.use_cdm:
        feats.append(model.cdm_branch(cdm))
    if model.use_iq:
        feats.append(model.iq_branch(iq))
    if model.use_stats:
        feats.append(model.stats_branch(stats))
    return torch.cat(feats, dim=1)


def gaussian_mmd(feat_a, feat_b, sigmas=(1, 2, 4, 8, 16)):
    """多核高斯 MMD^2 = E[k(a,a)] + E[k(b,b)] - 2*E[k(a,b)]"""
    if feat_a.size(0) < 2 or feat_b.size(0) < 2:
        return torch.tensor(0.0, device=feat_a.device)

    xx = torch.cdist(feat_a, feat_a, p=2) ** 2
    yy = torch.cdist(feat_b, feat_b, p=2) ** 2
    xy = torch.cdist(feat_a, feat_b, p=2) ** 2

    loss = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma * sigma)
        kxx = torch.exp(-gamma * xx).mean()
        kyy = torch.exp(-gamma * yy).mean()
        kxy = torch.exp(-gamma * xy).mean()
        loss = loss + kxx + kyy - 2 * kxy

    return loss / len(sigmas)


# ─────────────────────────────────────────────────────────
# Joint Dataset（带 domain flag）
# ─────────────────────────────────────────────────────────
class JointDataset(Dataset):
    def __init__(self, X_cdm, X_iq, X_stats, y, domain, indices):
        self.X_cdm = X_cdm
        self.X_iq = X_iq
        self.X_stats = X_stats
        self.y = y
        self.domain = domain
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (self.X_cdm[idx], self.X_iq[idx], self.X_stats[idx],
                self.y[idx], self.domain[idx])


# ─────────────────────────────────────────────────────────
# 训练：标准（S2/S3/S5 用）
# ─────────────────────────────────────────────────────────
def train_m1(
    cache: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    hp: HyperParams,
    ckpt_path: Path,
    device: torch.device,
    pretrained_ckpt: Optional[dict] = None,
    log_prefix: str = "",
    class_weights_override: Optional[torch.Tensor] = None,
    sampler_mode: Optional[str] = None,
) -> Tuple[nn.Module, List[dict]]:
    X_cdm = cache["X_cdm"]
    X_iq = cache["X_iq"]
    X_stats_raw = cache["X_stats"]
    y = cache["y"]

    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)

    train_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, train_idx)
    val_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, val_idx)

    if sampler_mode == "highorder_heavy":
        y_train = y[train_idx]
        ho_weight = np.where(y_train >= 3, 2.0, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=ho_weight.tolist(),
            num_samples=len(train_idx),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=hp.batch, shuffle=False,
                                  num_workers=0, drop_last=False, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=hp.batch, shuffle=True,
                                  num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=hp.batch, shuffle=False, num_workers=0)

    model = MultiRepNet(
        model_mode=MODEL_MODE, num_classes=NUM_CLASSES,
        iq_in_ch=1, stats_dim=STATS_DIM, dropout=hp.dropout,
    ).to(device)

    if pretrained_ckpt is not None:
        model.load_state_dict(pretrained_ckpt["model_state"])
        print(f"  {log_prefix}加载预训练权重 (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp.epochs, eta_min=hp.lr * 0.05
    )
    if class_weights_override is not None:
        class_w = class_weights_override.to(device)
    else:
        class_w = class_weights_from_y(y, train_idx, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=hp.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, hp.epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for cdm, iq, stats, yy in train_loader:
            cdm = cdm.to(device, non_blocking=True)
            iq = iq.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(cdm, iq, stats)
                loss = criterion(logits, yy)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(yy)
            total_correct += (logits.argmax(dim=1) == yy).sum().item()
            total_n += len(yy)

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for cdm, iq, stats, yy in val_loader:
                cdm = cdm.to(device, non_blocking=True)
                iq = iq.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(cdm, iq, stats)
                val_correct += (logits.argmax(dim=1) == yy).sum().item()
                val_n += len(yy)
        val_acc = val_correct / max(val_n, 1)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_acc": val_acc, "lr": lr_now,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {log_prefix}epoch {epoch:3d}/{hp.epochs} | "
                  f"loss={train_loss:.4f} tr_acc={train_acc:.4f} "
                  f"va_acc={val_acc:.4f} lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_mode": MODEL_MODE,
                "stats_dim": STATS_DIM,
                "stats_mean": stats_mean,
                "stats_std": stats_std,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "mod_names": MOD_NAMES,
            }, str(ckpt_path))
        else:
            patience_counter += 1
            if patience_counter >= hp.patience:
                print(f"  {log_prefix}early stop @ epoch {epoch} (best={best_epoch}, val_acc={best_val_acc:.4f})")
                break

    print(f"  {log_prefix}完成: best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, history


# ─────────────────────────────────────────────────────────
# 训练：Joint（S4 用 — direct / weighted / mmd）
# ─────────────────────────────────────────────────────────
def train_m1_joint(
    combined: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    hp: HyperParams,
    ckpt_path: Path,
    device: torch.device,
    mode: str = "direct",
    log_prefix: str = "",
    pretrained_ckpt: Optional[dict] = None,
) -> Tuple[nn.Module, List[dict]]:
    """
    mode: "direct" (普通 shuffle), "weighted" (3:1 sim:real), "replay" (2:1 real:sim), "mmd" (CE + MMD)
    val 只在 real_val 上做。
    """
    X_cdm = combined["X_cdm"]
    X_iq = combined["X_iq"]
    X_stats_raw = combined["X_stats"]
    y = combined["y"]
    domain = combined["domain"]

    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)

    val_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, val_idx)
    val_loader = DataLoader(val_ds, batch_size=hp.batch, shuffle=False, num_workers=0)

    if mode in ("weighted", "replay"):
        n_sim = int((domain[train_idx] == 0).sum())
        n_real = int((domain[train_idx] == 1).sum())
        if n_sim == 0 or n_real == 0:
            print(f"  {log_prefix}[WARNING] {mode} mode 缺域: sim={n_sim} real={n_real}, fallback to direct")
            sampler = None
            shuffle = True
        else:
            if mode == "weighted":
                w_sim = 3.0 / n_sim
                w_real = 1.0 / n_real
            else:  # replay: real:sim = 2:1
                w_sim = 1.0 / n_sim
                w_real = 2.0 / n_real
            sample_weights = np.where(domain[train_idx] == 0, w_sim, w_real).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=sample_weights.tolist(),
                num_samples=len(train_idx),
                replacement=True,
            )
            shuffle = False
        train_ds = JointDataset(X_cdm, X_iq, X_stats, y, domain, train_idx)
        train_loader = DataLoader(train_ds, batch_size=hp.batch, shuffle=shuffle,
                                  num_workers=0, drop_last=False, sampler=sampler)
    else:
        train_ds = JointDataset(X_cdm, X_iq, X_stats, y, domain, train_idx)
        train_loader = DataLoader(train_ds, batch_size=hp.batch, shuffle=True,
                                  num_workers=0, drop_last=False)

    model = MultiRepNet(
        model_mode=MODEL_MODE, num_classes=NUM_CLASSES,
        iq_in_ch=1, stats_dim=STATS_DIM, dropout=hp.dropout,
    ).to(device)

    if pretrained_ckpt is not None:
        model.load_state_dict(pretrained_ckpt["model_state"])
        print(f"  {log_prefix}加载预训练权重 (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp.epochs, eta_min=hp.lr * 0.05
    )
    class_w = class_weights_from_y(y, train_idx, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=hp.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, hp.epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        mmd_count = 0
        for cdm, iq, stats, yy, dd in train_loader:
            cdm = cdm.to(device, non_blocking=True)
            iq = iq.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            dd = dd.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(cdm, iq, stats)
                loss = criterion(logits, yy)

                if mode == "mmd":
                    sim_mask = (dd == 0)
                    real_mask = (dd == 1)
                    if sim_mask.sum() >= 2 and real_mask.sum() >= 2:
                        feat = extract_features(model, cdm, iq, stats)
                        mmd = gaussian_mmd(feat[sim_mask], feat[real_mask])
                        loss = loss + MMD_WEIGHT * mmd
                        mmd_count += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(yy)
            total_correct += (logits.argmax(dim=1) == yy).sum().item()
            total_n += len(yy)

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for cdm, iq, stats, yy in val_loader:
                cdm = cdm.to(device, non_blocking=True)
                iq = iq.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(cdm, iq, stats)
                val_correct += (logits.argmax(dim=1) == yy).sum().item()
                val_n += len(yy)
        val_acc = val_correct / max(val_n, 1)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_acc": val_acc, "lr": lr_now,
            "mmd_batches": mmd_count,
        })

        if epoch % 5 == 0 or epoch == 1:
            mmd_info = f" mmd_batches={mmd_count}" if mode == "mmd" else ""
            print(f"  {log_prefix}epoch {epoch:3d}/{hp.epochs} | "
                  f"loss={train_loss:.4f} tr_acc={train_acc:.4f} "
                  f"va_acc={val_acc:.4f} lr={lr_now:.2e}{mmd_info}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_mode": MODEL_MODE,
                "stats_dim": STATS_DIM,
                "stats_mean": stats_mean,
                "stats_std": stats_std,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "mod_names": MOD_NAMES,
            }, str(ckpt_path))
        else:
            patience_counter += 1
            if patience_counter >= hp.patience:
                print(f"  {log_prefix}early stop @ epoch {epoch} (best={best_epoch}, val_acc={best_val_acc:.4f})")
                break

    print(f"  {log_prefix}完成: best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, history


# ─────────────────────────────────────────────────────────
# 评估（增强版：含 128QAM, 256QAM, A-B gap）
# ─────────────────────────────────────────────────────────
def evaluate_on_real_test(
    model: nn.Module,
    cache: dict,
    test_idx: np.ndarray,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    device: torch.device,
    out_dir: Path,
    prefix: str,
) -> dict:
    X_cdm = cache["X_cdm"]
    X_iq = cache["X_iq"]
    X_stats_raw = cache["X_stats"]
    y = cache["y"]
    sources = cache["sources"]

    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)
    test_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred, _ = predict_loader(model, test_loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[MOD_NAMES[c]] = float((y_pred[mask] == c).mean())
        else:
            per_class[MOD_NAMES[c]] = float("nan")

    test_sources = sources[test_idx]
    a_mask = test_sources == "A"
    b_mask = test_sources == "B"
    a_acc = float((y_pred[a_mask] == y_true[a_mask]).mean()) if a_mask.sum() > 0 else float("nan")
    b_acc = float((y_pred[b_mask] == y_true[b_mask]).mean()) if b_mask.sum() > 0 else float("nan")
    a_b_gap = a_acc - b_acc

    ho_mask = np.isin(y_true, HIGH_ORDER_LABELS)
    ho_acc = float((y_pred[ho_mask] == y_true[ho_mask]).mean()) if ho_mask.sum() > 0 else float("nan")

    save_confusion_matrix(
        cm, MOD_NAMES,
        out_dir / f"confusion_matrix.png",
        title=f"{prefix} test confusion matrix",
    )

    result = {
        "overall_acc": acc,
        "macro_f1": f1_macro,
        "per_class_acc": per_class,
        "acc_128qam": per_class.get("128QAM", float("nan")),
        "acc_256qam": per_class.get("256QAM", float("nan")),
        "A_acc": a_acc,
        "B_acc": b_acc,
        "a_b_gap": a_b_gap,
        "high_order_acc": ho_acc,
        "num_test": int(len(y_true)),
        "num_A": int(a_mask.sum()),
        "num_B": int(b_mask.sum()),
        "confusion_matrix": cm.tolist(),
    }
    return result


# ─────────────────────────────────────────────────────────
# 通用：检查是否已完成
# ─────────────────────────────────────────────────────────
def load_existing_metrics(out_dir: Path) -> Optional[dict]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_result(result: dict, out_dir: Path):
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────
# S2: SimOnly
# ─────────────────────────────────────────────────────────
def run_s2_simonly(
    seed: int, turb: str,
    sim_cache: dict, real_cache: dict,
    device: torch.device,
) -> dict:
    config_name = f"s2_simonly_{turb}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config_name
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config_name} seed={seed} (已存在)")
        return existing

    print(f"\n  [S2] SimOnly seed={seed} turb={turb}")
    ensure_dir(out_dir)

    split = sim_cache["split"]
    sim_train_idx = np.where(split == "train")[0]
    sim_val_idx = np.where(split == "val")[0]
    sim_mean, sim_std = compute_stats_mean_std(sim_cache["X_stats"], sim_train_idx)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        sim_cache, sim_train_idx, sim_val_idx,
        sim_mean, sim_std, HP_PRETRAIN, ckpt_path, device,
        log_prefix=f"[S2/{turb}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]

    result = evaluate_on_real_test(
        model, real_cache, test_idx, sim_mean, sim_std,
        device, out_dir, f"S2_SimOnly_{turb}",
    )
    result["stage"] = "S2"
    result["config"] = config_name
    result["seed"] = seed
    result["turb"] = turb
    result["best_val_acc"] = max(h["val_acc"] for h in history)
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# S3: Real-scratch
# ─────────────────────────────────────────────────────────
def run_s3_real_scratch(
    seed: int, turb: str, ratio: float, ratio_label: str,
    real_cache: dict, frozen_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    config_name = f"s3_scratch_{ratio_label}_{turb}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config_name
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config_name} seed={seed}")
        return existing

    print(f"\n  [S3-scratch] {ratio_label} seed={seed} turb={turb}")
    ensure_dir(out_dir)

    train_subset = subset_real_train(frozen_df, ratio=ratio, seed=seed)
    train_subset_turb = train_subset[train_subset["turb_name"].astype(str) == turb].copy()

    paths = real_cache["paths"]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_idx = np.array(
        [path_to_idx[p] for p in train_subset_turb["out_mat"].astype(str)
         if p in path_to_idx],
        dtype=np.int64,
    )

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    val_idx = np.where((real_split == "val") & (real_turbs == turb))[0]

    real_mean, real_std = compute_stats_mean_std(real_cache["X_stats"], train_idx)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_idx, val_idx,
        real_mean, real_std, HP_SCRATCH, ckpt_path, device,
        log_prefix=f"[S3-scratch/{turb}/{ratio_label}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]
    result = evaluate_on_real_test(
        model, real_cache, test_idx, real_mean, real_std,
        device, out_dir, f"S3_scratch_{ratio_label}_{turb}",
    )
    result["stage"] = "S3"
    result["config"] = config_name
    result["seed"] = seed
    result["turb"] = turb
    result["ratio"] = ratio
    result["ratio_label"] = ratio_label
    result["mode"] = "scratch"
    result["best_val_acc"] = max(h["val_acc"] for h in history)
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# S3: SimPretrain + RealFT
# ─────────────────────────────────────────────────────────
def run_s3_simpretrain_realfinetune(
    seed: int, turb: str, ratio: float, ratio_label: str,
    real_cache: dict, frozen_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    config_name = f"s3_ft_{ratio_label}_{turb}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config_name
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config_name} seed={seed}")
        return existing

    print(f"\n  [S3-FT] {ratio_label} seed={seed} turb={turb}")
    ensure_dir(out_dir)

    s2_ckpt_path = OUT_ROOT / f"seed_{seed}" / f"s2_simonly_{turb}" / "best_model.pt"
    if not s2_ckpt_path.exists():
        raise FileNotFoundError(f"S2 checkpoint 不存在: {s2_ckpt_path}")
    s2_ckpt = torch.load(str(s2_ckpt_path), map_location=device, weights_only=False)
    sim_mean = s2_ckpt["stats_mean"]
    sim_std = s2_ckpt["stats_std"]

    train_subset = subset_real_train(frozen_df, ratio=ratio, seed=seed)
    train_subset_turb = train_subset[train_subset["turb_name"].astype(str) == turb].copy()

    paths = real_cache["paths"]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_idx = np.array(
        [path_to_idx[p] for p in train_subset_turb["out_mat"].astype(str)
         if p in path_to_idx],
        dtype=np.int64,
    )

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    val_idx = np.where((real_split == "val") & (real_turbs == turb))[0]

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_idx, val_idx,
        sim_mean, sim_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=s2_ckpt,
        log_prefix=f"[S3-FT/{turb}/{ratio_label}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]
    result = evaluate_on_real_test(
        model, real_cache, test_idx, sim_mean, sim_std,
        device, out_dir, f"S3_FT_{ratio_label}_{turb}",
    )
    result["stage"] = "S3"
    result["config"] = config_name
    result["seed"] = seed
    result["turb"] = turb
    result["ratio"] = ratio
    result["ratio_label"] = ratio_label
    result["mode"] = "simpretrain_realfinetune"
    result["best_val_acc"] = max(h["val_acc"] for h in history)
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# S4: Joint Training (direct / weighted / mmd)
# ─────────────────────────────────────────────────────────
def run_s4_joint(
    seed: int, turb: str, mode: str,
    sim_cache: dict, real_cache: dict, frozen_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    config_name = f"s4_joint_{mode}_{turb}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config_name
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config_name} seed={seed}")
        return existing

    print(f"\n  [S4-joint-{mode}] seed={seed} turb={turb}")
    ensure_dir(out_dir)

    sim_split = sim_cache["split"]
    sim_train_idx_local = np.where(sim_split == "train")[0]

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    real_train_idx_local = np.where((real_split == "train") & (real_turbs == turb))[0]
    real_val_idx_local = np.where((real_split == "val") & (real_turbs == turb))[0]

    n_sim = len(sim_cache["y"])
    n_real_total = len(real_cache["y"])

    combined_X_cdm = np.concatenate([sim_cache["X_cdm"], real_cache["X_cdm"]], axis=0)
    combined_X_iq = np.concatenate([sim_cache["X_iq"], real_cache["X_iq"]], axis=0)
    combined_X_stats = np.concatenate([sim_cache["X_stats"], real_cache["X_stats"]], axis=0)
    combined_y = np.concatenate([sim_cache["y"], real_cache["y"]], axis=0)
    combined_domain = np.concatenate([
        np.zeros(n_sim, dtype=np.int64),
        np.ones(n_real_total, dtype=np.int64),
    ], axis=0)

    combined = {
        "X_cdm": combined_X_cdm,
        "X_iq": combined_X_iq,
        "X_stats": combined_X_stats,
        "y": combined_y,
        "domain": combined_domain,
    }

    train_idx = np.concatenate([
        sim_train_idx_local,
        n_sim + real_train_idx_local,
    ], axis=0)
    val_idx = (n_sim + real_val_idx_local).astype(np.int64)

    combined_mean, combined_std = compute_stats_mean_std(combined_X_stats, train_idx)

    hp = HP_JOINT
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1_joint(
        combined, train_idx, val_idx,
        combined_mean, combined_std, hp, ckpt_path, device,
        mode=mode,
        log_prefix=f"[S4-{mode}/{turb}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    test_idx = (n_sim + np.where((real_split == "test") & (real_turbs == turb))[0]).astype(np.int64)

    test_ds = MultiRepDataset(combined_X_cdm, combined_X_iq,
                               apply_stats_norm(combined_X_stats, combined_mean, combined_std),
                               combined_y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    y_true, y_pred, _ = predict_loader(model, test_loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        per_class[MOD_NAMES[c]] = float((y_pred[mask] == c).mean()) if mask.sum() > 0 else float("nan")

    test_sources = real_cache["sources"][np.where((real_split == "test") & (real_turbs == turb))[0]]
    a_mask = test_sources == "A"
    b_mask = test_sources == "B"
    a_acc = float((y_pred[a_mask] == y_true[a_mask]).mean()) if a_mask.sum() > 0 else float("nan")
    b_acc = float((y_pred[b_mask] == y_true[b_mask]).mean()) if b_mask.sum() > 0 else float("nan")

    ho_mask = np.isin(y_true, HIGH_ORDER_LABELS)
    ho_acc = float((y_pred[ho_mask] == y_true[ho_mask]).mean()) if ho_mask.sum() > 0 else float("nan")

    save_confusion_matrix(
        cm, MOD_NAMES, out_dir / "confusion_matrix.png",
        title=f"S4_joint_{mode}_{turb} test confusion matrix",
    )

    result = {
        "overall_acc": acc,
        "macro_f1": f1_macro,
        "per_class_acc": per_class,
        "acc_128qam": per_class.get("128QAM", float("nan")),
        "acc_256qam": per_class.get("256QAM", float("nan")),
        "A_acc": a_acc,
        "B_acc": b_acc,
        "a_b_gap": a_acc - b_acc,
        "high_order_acc": ho_acc,
        "num_test": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "stage": "S4",
        "config": config_name,
        "seed": seed,
        "turb": turb,
        "joint_mode": mode,
        "best_val_acc": max(h["val_acc"] for h in history),
    }
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# S5: AWGN-only SimOnly
# ─────────────────────────────────────────────────────────
def run_s5_awgn_only(
    seed: int, turb: str,
    awgn_cache: dict, real_cache: dict,
    device: torch.device,
) -> dict:
    config_name = f"s5_awgn_only_{turb}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config_name
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config_name} seed={seed}")
        return existing

    print(f"\n  [S5] AWGN-only seed={seed} turb={turb}")
    ensure_dir(out_dir)

    split = awgn_cache["split"]
    awgn_train_idx = np.where(split == "train")[0]
    awgn_val_idx = np.where(split == "val")[0]
    awgn_mean, awgn_std = compute_stats_mean_std(awgn_cache["X_stats"], awgn_train_idx)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        awgn_cache, awgn_train_idx, awgn_val_idx,
        awgn_mean, awgn_std, HP_AWGN, ckpt_path, device,
        log_prefix=f"[S5/{turb}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]

    result = evaluate_on_real_test(
        model, real_cache, test_idx, awgn_mean, awgn_std,
        device, out_dir, f"S5_AWGNonly_{turb}",
    )
    result["stage"] = "S5"
    result["config"] = config_name
    result["seed"] = seed
    result["turb"] = turb
    result["best_val_acc"] = max(h["val_acc"] for h in history)
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────
def aggregate_results(all_results: List[dict]) -> dict:
    """按 config 跨 seeds 聚合 mean±std。"""
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    agg = {}
    for config, runs in by_config.items():
        metrics_keys = ["overall_acc", "macro_f1", "acc_128qam", "acc_256qam",
                        "A_acc", "B_acc", "a_b_gap", "high_order_acc", "best_val_acc"]
        per_mod_keys = MOD_NAMES
        entry = {
            "config": config,
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "turb": runs[0].get("turb", ""),
            "stage": runs[0].get("stage", ""),
        }
        for k in metrics_keys:
            vals = [r.get(k, float("nan")) for r in runs]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                entry[f"{k}_mean"] = float(np.mean(vals))
                entry[f"{k}_std"] = float(np.std(vals))
            else:
                entry[f"{k}_mean"] = float("nan")
                entry[f"{k}_std"] = float("nan")
        for mk in per_mod_keys:
            vals = [r.get("per_class_acc", {}).get(mk, float("nan")) for r in runs]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                entry[f"acc_{mk}_mean"] = float(np.mean(vals))
                entry[f"acc_{mk}_std"] = float(np.std(vals))
            else:
                entry[f"acc_{mk}_mean"] = float("nan")
                entry[f"acc_{mk}_std"] = float("nan")
        agg[config] = entry
    return agg


def write_summary(all_results: List[dict], agg: dict, path: Path):
    lines = [
        f"# Stage II Round-1 训练结果汇总",
        f"",
        f"生成时间: {now_str()}",
        f"模型: M1 (CDM-Stats-Radial)",
        f"Sim 数据: Calibrated-Sim-v2.2 (12000 帧)",
        f"AWGN-only: 12000 帧 (S5 消融)",
        f"Seeds: {ALL_SEEDS}",
        f"Real test: frozen split test set (324 行, 每 turb 162 行)",
        f"",
        f"## 总 run 数: {len(all_results)}",
        f"",
        f"## 聚合结果 (mean ± std across seeds)",
        f"",
        f"| Config | Turb | nSeeds | Overall | MacroF1 | 128QAM | 256QAM | HighOrd | A_acc | B_acc | A-B gap |",
        f"|--------|------|--------|---------|---------|--------|--------|---------|-------|-------|---------|",
    ]
    for config in sorted(agg.keys()):
        e = agg[config]
        def fmt(m, s):
            if isinstance(m, float) and np.isnan(m):
                return "—"
            return f"{m*100:.1f}±{s*100:.1f}"
        lines.append(
            f"| {config} | {e['turb']} | {e['n_seeds']} | "
            f"{fmt(e['overall_acc_mean'], e['overall_acc_std'])} | "
            f"{fmt(e['macro_f1_mean'], e['macro_f1_std'])} | "
            f"{fmt(e['acc_128qam_mean'], e['acc_128qam_std'])} | "
            f"{fmt(e['acc_256qam_mean'], e['acc_256qam_std'])} | "
            f"{fmt(e['high_order_acc_mean'], e['high_order_acc_std'])} | "
            f"{fmt(e['A_acc_mean'], e['A_acc_std'])} | "
            f"{fmt(e['B_acc_mean'], e['B_acc_std'])} | "
            f"{fmt(e['a_b_gap_mean'], e['a_b_gap_std'])} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] summary → {path}")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS,
                        help="seeds to run (default: all 5)")
    args = parser.parse_args()
    seeds = args.seeds

    print(f"=== stage2_round1.py  [{now_str()}] ===")
    print(f"  seeds: {seeds}")
    ensure_dir(OUT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # 1. 加载 frozen split
    print(f"\n[1/4] 加载 frozen split")
    frozen_df = load_frozen_split()
    print(f"  frozen split: {len(frozen_df)} 行 (train={len(frozen_df[frozen_df['split']=='train'])}, "
          f"val={len(frozen_df[frozen_df['split']=='val'])}, test={len(frozen_df[frozen_df['split']=='test'])})")

    # 2. 构建 real cache（一次性）
    print(f"\n[2/4] 构建 real cache")
    real_cache_path = OUT_ROOT / "real_cache.npz"
    real_cache = build_pilot_cache(frozen_df, real_cache_path)
    print(f"  real cache: {len(real_cache['y'])} 样本")

    # 3. 构建 sim 12k cache + awgn 12k cache（分 turb）
    sim_caches = {}
    awgn_caches = {}

    print(f"\n[3/4] 构建 Calibrated-Sim-v2.2 12k cache")
    sim_df = pd.read_csv(SIM_MANIFEST_12K)
    for turb in REAL_TURBS:
        sim_turb_name = [k for k, v in REAL_TURB_MAP.items() if v == turb][0]
        sub = sim_df[sim_df["turb_name"].astype(str) == sim_turb_name].copy()
        sub["dataset_source"] = "sim"
        sub = stratified_sim_split(sub, seed=2026)
        cache_path = OUT_ROOT / f"sim12k_cache_{turb}.npz"
        sim_caches[turb] = build_pilot_cache(sub, cache_path)
        print(f"  sim12k cache {turb}: {len(sim_caches[turb]['y'])} 样本")

    print(f"\n  构建 AWGN-only 12k cache")
    awgn_df = pd.read_csv(AWGN_MANIFEST_12K)
    for turb in REAL_TURBS:
        sim_turb_name = [k for k, v in REAL_TURB_MAP.items() if v == turb][0]
        sub = awgn_df[awgn_df["turb_name"].astype(str) == sim_turb_name].copy()
        sub["dataset_source"] = "sim_awgn"
        sub = stratified_sim_split(sub, seed=2026)
        cache_path = OUT_ROOT / f"awgn12k_cache_{turb}.npz"
        awgn_caches[turb] = build_pilot_cache(sub, cache_path)
        print(f"  awgn12k cache {turb}: {len(awgn_caches[turb]['y'])} 样本")

    # 4. 运行所有 seeds
    all_results = []
    print(f"\n[4/4] 运行 {len(seeds)} seeds × {len(REAL_TURBS)} turbs × 15 configs = {len(seeds)*len(REAL_TURBS)*15} runs")

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed = {seed}")
        print(f"{'='*70}")
        set_seed(seed)

        for turb in REAL_TURBS:
            print(f"\n  --- turb={turb} ---")

            # S2
            r = run_s2_simonly(seed, turb, sim_caches[turb], real_cache, device)
            all_results.append(r)

            # S3
            for ratio, rlabel in zip(RATIOS, RATIO_LABELS):
                r = run_s3_real_scratch(seed, turb, ratio, rlabel, real_cache, frozen_df, device)
                all_results.append(r)
                r = run_s3_simpretrain_realfinetune(seed, turb, ratio, rlabel, real_cache, frozen_df, device)
                all_results.append(r)

            # S4
            for mode in ["direct", "weighted", "mmd"]:
                r = run_s4_joint(seed, turb, mode, sim_caches[turb], real_cache, frozen_df, device)
                all_results.append(r)

            # S5
            r = run_s5_awgn_only(seed, turb, awgn_caches[turb], real_cache, device)
            all_results.append(r)

    # 5. 汇总
    print(f"\n{'='*70}")
    print(f"  汇总 {len(all_results)} runs")
    print(f"{'='*70}")

    agg = aggregate_results(all_results)
    with open(OUT_ROOT / "aggregated_results.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"  aggregated_results.json → {OUT_ROOT / 'aggregated_results.json'}")

    write_summary(all_results, agg, OUT_ROOT / "summary.md")

    print(f"\n[完成] Stage II Round-1 结果已输出到 {OUT_ROOT}")
    print(f"  请审阅 summary.md 和 aggregated_results.json")


if __name__ == "__main__":
    main()
