# -*- coding: utf-8 -*-
"""
pilot_s2_s3.py — Stage II Pilot: S2 (SimOnly) + S3 (SimPretrain+10%RealFT)

小规模 pilot 训练，验证 Sim-v2.2 → real 迁移可行性。
- seed=2026 only, no multi-seed
- weak/strong 分开训练和评估
- 全部在同一个 frozen real test set (324 行) 上评估
- 完成后暂停，不进入完整 S4/S5

3 个 phase × 2 turbs:
  Phase A (SimOnly):              Sim-v2.2 3000/turb → real test
  Phase B (10% Real from scratch): 10% real_train → real test
  Phase C (SimPretrain+10%RealFT): Phase A ckpt → 10% real_train fine-tune → real test

输出: results/stage2/pilot_s2_s3/
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# 确保能 import 同目录模块
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
STATS_DIM = 48  # 16 blind stats + 32 radial bins
NUM_CLASSES = 6
SEED = 2026
OUT_ROOT = STAGE2_ROOT / "pilot_s2_s3"
SIM_MANIFEST = STAGE2_ROOT / "sim_data_v2_2" / "manifest_sim_v2_2.csv"
REAL_TURBS = ["weak", "strong"]
HIGH_ORDER_LABELS = [3, 4, 5]  # 64QAM, 128QAM, 256QAM


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


# ─────────────────────────────────────────────────────────
# Cache 构建（扩展版，存储 dataset_source 和 turb_name）
# ─────────────────────────────────────────────────────────
def build_pilot_cache(
    rows_df: pd.DataFrame,
    cache_path: Path,
    rebuild: bool = False,
) -> dict:
    """
    构建特征缓存。扩展 build_unified_cache，额外存储 dataset_source 和 turb_name。
    rows_df 必须含列：out_mat, mod_label
    可选列：split (默认 "train"), dataset_source (默认 ""), turb_name (默认 "")
    """
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    use_iq = False  # cdm_stats_radial 不用 IQ
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

        if i % 200 == 0 or i == n:
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

    # 确保 stats_dim
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
    """按 mod_name 分层 90/10 划分 train/val。"""
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
    """从 train 子集计算 mean/std。"""
    train = X_stats[train_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_stats_norm(X_stats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """应用归一化。"""
    X = X_stats.copy().astype(np.float32)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(mean, inds[1])
    X = (X - mean) / std
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# 训练
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
) -> Tuple[nn.Module, List[dict]]:
    """
    通用 M1 训练函数。
    pretrained_ckpt 非 None 时加载预训练权重（fine-tune）。
    """
    X_cdm = cache["X_cdm"]
    X_iq = cache["X_iq"]
    X_stats_raw = cache["X_stats"]
    y = cache["y"]

    # 归一化
    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)

    # Dataset
    train_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, train_idx)
    val_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, val_idx)
    train_loader = DataLoader(train_ds, batch_size=hp.batch, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=hp.batch, shuffle=False, num_workers=0)

    # Model
    model = MultiRepNet(
        model_mode=MODEL_MODE, num_classes=NUM_CLASSES,
        iq_in_ch=1, stats_dim=STATS_DIM, dropout=hp.dropout,
    ).to(device)

    if pretrained_ckpt is not None:
        model.load_state_dict(pretrained_ckpt["model_state"])
        print(f"  {log_prefix}加载预训练权重 (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")

    # Optimizer / Scheduler / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp.epochs, eta_min=hp.lr * 0.05
    )
    class_w = class_weights_from_y(y, train_idx, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=hp.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, hp.epochs + 1):
        # Train
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

        # Val
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

        # Early stop
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best
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

    # Load best for return
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, history


# ─────────────────────────────────────────────────────────
# 评估
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
    """在 real test 上评估，返回完整指标。"""
    X_cdm = cache["X_cdm"]
    X_iq = cache["X_iq"]
    X_stats_raw = cache["X_stats"]
    y = cache["y"]
    sources = cache["sources"]
    turbs = cache["turbs"]

    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)
    test_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred, _ = predict_loader(model, test_loader, device)

    # Overall
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Per-class
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[MOD_NAMES[c]] = float((y_pred[mask] == c).mean())
        else:
            per_class[MOD_NAMES[c]] = float("nan")

    # A/B source
    test_sources = sources[test_idx]
    a_mask = test_sources == "A"
    b_mask = test_sources == "B"
    a_acc = float((y_pred[a_mask] == y_true[a_mask]).mean()) if a_mask.sum() > 0 else float("nan")
    b_acc = float((y_pred[b_mask] == y_true[b_mask]).mean()) if b_mask.sum() > 0 else float("nan")

    # High-order
    ho_mask = np.isin(y_true, HIGH_ORDER_LABELS)
    ho_acc = float((y_pred[ho_mask] == y_true[ho_mask]).mean()) if ho_mask.sum() > 0 else float("nan")

    # Save confusion matrix
    save_confusion_matrix(
        cm, MOD_NAMES,
        out_dir / f"confusion_matrix.png",
        title=f"{prefix} test confusion matrix",
    )

    result = {
        "overall_acc": acc,
        "macro_f1": f1_macro,
        "per_class_acc": per_class,
        "A_acc": a_acc,
        "B_acc": b_acc,
        "high_order_acc": ho_acc,
        "num_test": int(len(y_true)),
        "num_A": int(a_mask.sum()),
        "num_B": int(b_mask.sum()),
        "confusion_matrix": cm.tolist(),
    }
    return result


# ─────────────────────────────────────────────────────────
# 单个 phase×turb 流程
# ─────────────────────────────────────────────────────────
def run_phase_a_simonly(
    turb: str,
    real_cache: dict,
    device: torch.device,
) -> dict:
    """Phase A: SimOnly — 仿真训练，real test 评估。"""
    print(f"\n{'='*70}")
    print(f"Phase A (SimOnly) — turb={turb}")
    print(f"{'='*70}")

    out_dir = OUT_ROOT / f"simonly_{turb}"
    ensure_dir(out_dir)

    # 1. 加载 sim 数据
    sim_df = pd.read_csv(SIM_MANIFEST)
    sim_turb_name = [k for k, v in REAL_TURB_MAP.items() if v == turb][0]
    sim_df = sim_df[sim_df["turb_name"].astype(str) == sim_turb_name].copy()
    sim_df["dataset_source"] = "sim"
    sim_df = stratified_sim_split(sim_df, seed=SEED)
    print(f"  sim 数据: {len(sim_df)} 行 (train={(sim_df['split']=='train').sum()}, val={(sim_df['split']=='val').sum()})")

    # 2. 构建 sim cache
    sim_cache_path = OUT_ROOT / f"sim_cache_{turb}.npz"
    sim_cache = build_pilot_cache(sim_df, sim_cache_path)

    # 3. Stats 归一化（用 sim train）
    split = sim_cache["split"]
    sim_train_idx = np.where(split == "train")[0]
    sim_val_idx = np.where(split == "val")[0]
    sim_mean, sim_std = compute_stats_mean_std(sim_cache["X_stats"], sim_train_idx)
    print(f"  sim train_idx={len(sim_train_idx)} val_idx={len(sim_val_idx)}")

    # 4. 训练
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        sim_cache, sim_train_idx, sim_val_idx,
        sim_mean, sim_std, HP_PRETRAIN, ckpt_path, device,
        log_prefix=f"[A/{turb}] ",
    )

    # 5. 保存 training curve
    save_training_curve(history, out_dir / "training_curve.png")

    # 6. 评估 real test
    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]
    print(f"  real test: {len(test_idx)} 样本")

    result = evaluate_on_real_test(
        model, real_cache, test_idx, sim_mean, sim_std,
        device, out_dir, f"SimOnly_{turb}",
    )
    result["phase"] = "SimOnly"
    result["turb"] = turb
    result["best_val_acc"] = max(h["val_acc"] for h in history)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


def run_phase_b_real10pct(
    turb: str,
    real_cache: dict,
    frozen_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Phase B: 10% Real from scratch — 小样本基线。"""
    print(f"\n{'='*70}")
    print(f"Phase B (10% Real from scratch) — turb={turb}")
    print(f"{'='*70}")

    out_dir = OUT_ROOT / f"real10pct_{turb}"
    ensure_dir(out_dir)

    # 1. 10% real train
    train_10pct = subset_real_train(frozen_df, ratio=0.1, seed=SEED)
    train_10pct_turb = train_10pct[train_10pct["turb_name"].astype(str) == turb].copy()
    print(f"  10% real train: {len(train_10pct_turb)} 行")

    # 2. 匹配到 real_cache 位置
    paths = real_cache["paths"]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_idx = np.array(
        [path_to_idx[p] for p in train_10pct_turb["out_mat"].astype(str)
         if p in path_to_idx],
        dtype=np.int64,
    )

    # 3. Real val
    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    val_idx = np.where((real_split == "val") & (real_turbs == turb))[0]
    print(f"  train_idx={len(train_idx)} val_idx={len(val_idx)}")

    # 4. Stats 归一化（用 real 10% train）
    real_mean, real_std = compute_stats_mean_std(real_cache["X_stats"], train_idx)

    # 5. 训练
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_idx, val_idx,
        real_mean, real_std, HP_SCRATCH, ckpt_path, device,
        log_prefix=f"[B/{turb}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    # 6. 评估
    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]
    result = evaluate_on_real_test(
        model, real_cache, test_idx, real_mean, real_std,
        device, out_dir, f"10pctReal_{turb}",
    )
    result["phase"] = "10%Real"
    result["turb"] = turb
    result["best_val_acc"] = max(h["val_acc"] for h in history)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


def run_phase_c_simpretrain_realfinetune(
    turb: str,
    real_cache: dict,
    frozen_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Phase C: SimPretrain + 10% Real FineTune。"""
    print(f"\n{'='*70}")
    print(f"Phase C (SimPretrain + 10% RealFT) — turb={turb}")
    print(f"{'='*70}")

    out_dir = OUT_ROOT / f"simpretrain_realfinetune_{turb}"
    ensure_dir(out_dir)

    # 1. 加载 Phase A checkpoint
    phase_a_ckpt_path = OUT_ROOT / f"simonly_{turb}" / "best_model.pt"
    if not phase_a_ckpt_path.exists():
        raise FileNotFoundError(f"Phase A checkpoint 不存在: {phase_a_ckpt_path}")
    phase_a_ckpt = torch.load(str(phase_a_ckpt_path), map_location=device, weights_only=False)
    sim_mean = phase_a_ckpt["stats_mean"]
    sim_std = phase_a_ckpt["stats_std"]
    print(f"  加载 Phase A ckpt: best_val_acc={phase_a_ckpt['best_val_acc']:.4f}")

    # 2. 10% real train (同 Phase B)
    train_10pct = subset_real_train(frozen_df, ratio=0.1, seed=SEED)
    train_10pct_turb = train_10pct[train_10pct["turb_name"].astype(str) == turb].copy()

    paths = real_cache["paths"]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_idx = np.array(
        [path_to_idx[p] for p in train_10pct_turb["out_mat"].astype(str)
         if p in path_to_idx],
        dtype=np.int64,
    )

    real_split = real_cache["split"]
    real_turbs = real_cache["turbs"]
    val_idx = np.where((real_split == "val") & (real_turbs == turb))[0]
    print(f"  train_idx={len(train_idx)} val_idx={len(val_idx)}")

    # 3. 训练（用 sim_mean/std 归一化，加载预训练权重）
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_idx, val_idx,
        sim_mean, sim_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=phase_a_ckpt,
        log_prefix=f"[C/{turb}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    # 4. 评估（用 sim_mean/std）
    test_idx = np.where((real_split == "test") & (real_turbs == turb))[0]
    result = evaluate_on_real_test(
        model, real_cache, test_idx, sim_mean, sim_std,
        device, out_dir, f"SimPretrain_RealFT_{turb}",
    )
    result["phase"] = "SimPretrain+10%RealFT"
    result["turb"] = turb
    result["best_val_acc"] = max(h["val_acc"] for h in history)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


# ─────────────────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────────────────
def write_summary(results: List[dict], path: Path):
    lines = [
        f"# Pilot S2-S3 训练结果汇总",
        f"",
        f"生成时间: {now_str()}",
        f"模型: M1 (CDM-Stats-Radial), seed={SEED}",
        f"Sim 数据: Sim-v2.2 (6000 帧)",
        f"Real test: frozen split test set (324 行, 每 turb 162 行)",
        f"",
        f"## 结果对比表",
        f"",
        f"| Phase | Turb | Overall | MacroF1 | QPSK | 16QAM | 32QAM | 64QAM | 128QAM | 256QAM | A_acc | B_acc | HighOrder | best_val |",
        f"|-------|------|---------|---------|------|-------|-------|-------|--------|--------|-------|-------|-----------|----------|",
    ]

    for r in results:
        pc = r["per_class_acc"]
        def fmt(x):
            return f"{x*100:.1f}" if isinstance(x, (int, float)) and not np.isnan(x) else "—"
        lines.append(
            f"| {r['phase']} | {r['turb']} | "
            f"{fmt(r['overall_acc'])} | {fmt(r['macro_f1'])} | "
            f"{fmt(pc.get('QPSK', float('nan')))} | {fmt(pc.get('16QAM', float('nan')))} | "
            f"{fmt(pc.get('32QAM', float('nan')))} | {fmt(pc.get('64QAM', float('nan')))} | "
            f"{fmt(pc.get('128QAM', float('nan')))} | {fmt(pc.get('256QAM', float('nan')))} | "
            f"{fmt(r['A_acc'])} | {fmt(r['B_acc'])} | {fmt(r['high_order_acc'])} | "
            f"{fmt(r.get('best_val_acc', float('nan')))} |"
        )

    lines.append(f"")
    lines.append(f"## 关键对比")
    lines.append(f"")
    lines.append(f"### Phase C (SimPretrain+10%RealFT) vs Phase B (10%Real)")
    lines.append(f"")
    for turb in REAL_TURBS:
        b = next((r for r in results if r["phase"] == "10%Real" and r["turb"] == turb), None)
        c = next((r for r in results if r["phase"] == "SimPretrain+10%RealFT" and r["turb"] == turb), None)
        if b and c:
            delta = c["overall_acc"] - b["overall_acc"]
            sign = "+" if delta >= 0 else ""
            lines.append(f"- **{turb}**: Phase C {c['overall_acc']*100:.1f}% vs Phase B {b['overall_acc']*100:.1f}% (Δ={sign}{delta*100:.1f}%)")
            ho_delta = c["high_order_acc"] - b["high_order_acc"]
            ho_sign = "+" if ho_delta >= 0 else ""
            lines.append(f"  - High-order: C {c['high_order_acc']*100:.1f}% vs B {b['high_order_acc']*100:.1f}% (Δ={ho_sign}{ho_delta*100:.1f}%)")

    lines.append(f"")
    lines.append(f"### Phase A (SimOnly) — sim→real 迁移下限")
    lines.append(f"")
    for turb in REAL_TURBS:
        a = next((r for r in results if r["phase"] == "SimOnly" and r["turb"] == turb), None)
        if a:
            lines.append(f"- **{turb}**: overall={a['overall_acc']*100:.1f}%, high_order={a['high_order_acc']*100:.1f}%, A={a['A_acc']*100:.1f}%, B={a['B_acc']*100:.1f}%")

    lines.append(f"")
    lines.append(f"## 结论")
    lines.append(f"")
    lines.append(f"- 若 Phase C > Phase B：仿真预训练有增益，可考虑扩大到完整 S2-S5")
    lines.append(f"- 若 Phase C ≈ Phase B：仿真预训练无显著增益，需改进仿真域")
    lines.append(f"- 若 Phase C < Phase B：负迁移，仿真域与真实域差异过大")
    lines.append(f"- 若 Phase A > 16.7%（随机基线）：仿真数据有一定判别力")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] summary → {path}")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def main():
    print(f"=== pilot_s2_s3.py  [{now_str()}] ===")
    ensure_dir(OUT_ROOT)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # 1. 加载 frozen split
    print(f"\n[1/5] 加载 frozen split")
    frozen_df = load_frozen_split()
    print(f"  frozen split: {len(frozen_df)} 行 (train={len(frozen_df[frozen_df['split']=='train'])}, "
          f"val={len(frozen_df[frozen_df['split']=='val'])}, test={len(frozen_df[frozen_df['split']=='test'])})")

    # 2. 构建 real cache（一次性）
    print(f"\n[2/5] 构建 real cache")
    real_cache_path = OUT_ROOT / "real_cache.npz"
    real_cache = build_pilot_cache(frozen_df, real_cache_path)
    print(f"  real cache: {len(real_cache['y'])} 样本")

    # 3. 运行 3 phases × 2 turbs
    all_results = []

    for turb in REAL_TURBS:
        print(f"\n[3/5] Phase A — SimOnly — turb={turb}")
        r_a = run_phase_a_simonly(turb, real_cache, device)
        all_results.append(r_a)
        print(f"  → overall={r_a['overall_acc']*100:.1f}% macro_f1={r_a['macro_f1']*100:.1f}% "
              f"high_order={r_a['high_order_acc']*100:.1f}%")

        print(f"\n[4/5] Phase B — 10% Real from scratch — turb={turb}")
        r_b = run_phase_b_real10pct(turb, real_cache, frozen_df, device)
        all_results.append(r_b)
        print(f"  → overall={r_b['overall_acc']*100:.1f}% macro_f1={r_b['macro_f1']*100:.1f}% "
              f"high_order={r_b['high_order_acc']*100:.1f}%")

        print(f"\n[5/5] Phase C — SimPretrain + 10% RealFT — turb={turb}")
        r_c = run_phase_c_simpretrain_realfinetune(turb, real_cache, frozen_df, device)
        all_results.append(r_c)
        print(f"  → overall={r_c['overall_acc']*100:.1f}% macro_f1={r_c['macro_f1']*100:.1f}% "
              f"high_order={r_c['high_order_acc']*100:.1f}%")

    # 4. 汇总
    print(f"\n{'='*70}")
    print(f"汇总")
    print(f"{'='*70}")
    write_summary(all_results, OUT_ROOT / "summary.md")

    # 打印简要对比
    print(f"\n=== 结果简要对比 ===")
    print(f"{'Phase':<28s} {'Turb':<8s} {'Overall':<10s} {'MacroF1':<10s} {'HighOrd':<10s} {'A_acc':<10s} {'B_acc':<10s}")
    print("-" * 86)
    for r in all_results:
        print(f"{r['phase']:<28s} {r['turb']:<8s} "
              f"{r['overall_acc']*100:>7.1f}%  {r['macro_f1']*100:>7.1f}%  "
              f"{r['high_order_acc']*100:>7.1f}%  {r['A_acc']*100:>7.1f}%  {r['B_acc']*100:>7.1f}%")

    print(f"\n[完成] Pilot 结果已输出到 {OUT_ROOT}")
    print(f"  请审阅 summary.md 和各 phase 的 confusion_matrix.png")
    print(f"  决定是否进入完整 S4/S5 或继续迭代仿真域。")


if __name__ == "__main__":
    main()
