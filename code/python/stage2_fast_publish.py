# -*- coding: utf-8 -*-
"""
stage2_fast_publish.py — Stage II Fast-Publish A/B Domain Pipeline

Dataset-A = in-domain (train/val/test)
Dataset-B = out-of-domain (finetune/test, never in main training)

Merged turb training: one model per config on combined weak+strong.
Per-turb evaluation: A-test and B-test each broken down by weak/strong.

15 configs × 5 seeds = 75 runs:
  S2: s2_simonly
  S3: s3_scratch_{5,10,20,100}pct + s3_ft_{5,10,20,100}pct
  S4: s4_joint_{direct,weighted,mmd}
  S5: s5_awgn_only
  B-FT: b_ft_{5,10}pct (from s3_ft_100pct checkpoint)

可恢复: 每个 run 检查 metrics.json 是否存在，存在则跳过。
"""

from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    STAGE2_ROOT, ensure_dir, now_str,
)
from train_mrf_lowmem import (
    MultiRepNet, MultiRepDataset, set_seed, MOD_NAMES,
    class_weights_from_y, predict_loader,
    save_confusion_matrix, save_training_curve,
)
from stage2_round1 import (
    build_pilot_cache, stratified_sim_split,
    compute_stats_mean_std, apply_stats_norm,
    train_m1, train_m1_joint, extract_features, gaussian_mmd,
    JointDataset, load_existing_metrics, save_result,
    HyperParams, HP_PRETRAIN, HP_SCRATCH, HP_FINETUNE, HP_JOINT, HP_AWGN,
    MODEL_MODE, STATS_DIM, NUM_CLASSES, HIGH_ORDER_LABELS, MMD_WEIGHT,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
RATIOS = [0.05, 0.10, 0.20, 1.00]
RATIO_LABELS = ["5pct", "10pct", "20pct", "100pct"]

OUT_ROOT = STAGE2_ROOT / "fast_publish"
AB_SPLIT_CSV = OUT_ROOT / "ab_domain_split.csv"
SIM_MANIFEST_12K = STAGE2_ROOT / "sim_data_v2_2_12k" / "manifest_sim_v2_2_12k.csv"
AWGN_MANIFEST_12K = STAGE2_ROOT / "sim_data_awgn_only" / "manifest_sim_awgn_only.csv"

# Existing caches from Round-1
ROUND1_ROOT = STAGE2_ROOT / "round1"
SIM_CACHE_WEAK = ROUND1_ROOT / "sim12k_cache_weak.npz"
SIM_CACHE_STRONG = ROUND1_ROOT / "sim12k_cache_strong.npz"
AWGN_CACHE_WEAK = ROUND1_ROOT / "awgn12k_cache_weak.npz"
AWGN_CACHE_STRONG = ROUND1_ROOT / "awgn12k_cache_strong.npz"

REAL_CACHE_PATH = OUT_ROOT / "real_cache_ab.npz"


# ─────────────────────────────────────────────────────────
# Cache building
# ─────────────────────────────────────────────────────────
def build_real_cache() -> dict:
    """Build real data cache from ab_domain_split.csv."""
    if REAL_CACHE_PATH.exists():
        print(f"  [cache] Loading existing real cache: {REAL_CACHE_PATH.name}")
        data = np.load(REAL_CACHE_PATH, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"  [cache] Building real cache from {AB_SPLIT_CSV.name}...")
    df = pd.read_csv(AB_SPLIT_CSV)
    cache = build_pilot_cache(df, REAL_CACHE_PATH, rebuild=False)
    return cache


def merge_sim_caches() -> dict:
    """Merge per-turb sim caches into one combined cache."""
    print(f"  [cache] Merging sim caches (weak + strong)...")
    weak = np.load(SIM_CACHE_WEAK, allow_pickle=True)
    strong = np.load(SIM_CACHE_STRONG, allow_pickle=True)

    merged = {}
    for key in ["X_cdm", "X_iq", "X_stats", "y", "paths"]:
        merged[key] = np.concatenate([weak[key], strong[key]], axis=0)
    # split, sources, turbs are object arrays
    for key in ["split", "sources", "turbs"]:
        merged[key] = np.concatenate([weak[key], strong[key]], axis=0)

    print(f"  [cache] Sim merged: {len(merged['y'])} frames")
    return merged


def merge_awgn_caches() -> dict:
    """Merge per-turb AWGN caches into one combined cache."""
    print(f"  [cache] Merging AWGN caches (weak + strong)...")
    weak = np.load(AWGN_CACHE_WEAK, allow_pickle=True)
    strong = np.load(AWGN_CACHE_STRONG, allow_pickle=True)

    merged = {}
    for key in ["X_cdm", "X_iq", "X_stats", "y", "paths"]:
        merged[key] = np.concatenate([weak[key], strong[key]], axis=0)
    for key in ["split", "sources", "turbs"]:
        merged[key] = np.concatenate([weak[key], strong[key]], axis=0)

    print(f"  [cache] AWGN merged: {len(merged['y'])} frames")
    return merged


# ─────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────
def get_indices(cache: dict) -> dict:
    """Get index arrays for A/B domain splits from real cache."""
    split = cache["split"]
    sources = cache["sources"]

    a_train = np.where((split == "train") & (sources == "A"))[0]
    a_val = np.where((split == "val") & (sources == "A"))[0]
    a_test = np.where((split == "test") & (sources == "A"))[0]
    b_ft_pool = np.where((split == "finetune") & (sources == "B"))[0]
    b_test = np.where((split == "test") & (sources == "B"))[0]

    print(f"  [indices] A-train={len(a_train)}, A-val={len(a_val)}, A-test={len(a_test)}, "
          f"B-finetune={len(b_ft_pool)}, B-test={len(b_test)}")
    return {
        "a_train": a_train, "a_val": a_val, "a_test": a_test,
        "b_ft_pool": b_ft_pool, "b_test": b_test,
    }


def subset_indices(train_idx: np.ndarray, y: np.ndarray, turbs: np.ndarray,
                   ratio: float, seed: int) -> np.ndarray:
    """Stratified subset of train indices by (mod_label, turb_name)."""
    if ratio >= 1.0:
        return train_idx
    rng = np.random.default_rng(seed)
    parts = []
    for (mod_val, turb_val) in sorted(set(zip(y[train_idx], turbs[train_idx]))):
        mask = (y[train_idx] == mod_val) & (turbs[train_idx] == turb_val)
        grp = train_idx[mask]
        n = max(1, int(round(len(grp) * ratio)))
        sampled = rng.choice(grp, n, replace=False)
        parts.append(sampled)
    return np.concatenate(parts)


def subset_b_finetune(b_ft_pool: np.ndarray, y: np.ndarray, turbs: np.ndarray,
                      fraction: float, seed: int) -> np.ndarray:
    """Sample B fine-tune subset. fraction=0.05 → 36 frames, 0.10 → 72 frames."""
    total_b = 720
    n_target = max(1, int(round(fraction * total_b)))
    rng = np.random.default_rng(seed)
    parts = []
    for (mod_val, turb_val) in sorted(set(zip(y[b_ft_pool], turbs[b_ft_pool]))):
        mask = (y[b_ft_pool] == mod_val) & (turbs[b_ft_pool] == turb_val)
        grp = b_ft_pool[mask]
        n_per = max(1, n_target // 12)  # 12 strata
        n_per = min(n_per, len(grp))
        sampled = rng.choice(grp, n_per, replace=False)
        parts.append(sampled)
    result = np.concatenate(parts)
    print(f"    B-finetune subset: {len(result)} frames (target {n_target}, fraction={fraction})")
    return result


# ─────────────────────────────────────────────────────────
# Evaluation (with per-turb breakdown)
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard metrics from y_true and y_pred."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        per_class[MOD_NAMES[c]] = float((y_pred[mask] == c).mean()) if mask.sum() > 0 else float("nan")

    ho_mask = np.isin(y_true, HIGH_ORDER_LABELS)
    ho_acc = float((y_pred[ho_mask] == y_true[ho_mask]).mean()) if ho_mask.sum() > 0 else float("nan")

    return {
        "overall_acc": acc,
        "macro_f1": f1_macro,
        "per_class_acc": per_class,
        "acc_128qam": per_class.get("128QAM", float("nan")),
        "acc_256qam": per_class.get("256QAM", float("nan")),
        "high_order_acc": ho_acc,
        "confusion_matrix": cm,
        "num_test": int(len(y_true)),
    }


def eval_on_test(model: nn.Module, cache: dict, test_idx: np.ndarray,
                 stats_mean: np.ndarray, stats_std: np.ndarray,
                 device: torch.device, out_dir: Path, prefix: str) -> dict:
    """Evaluate model on test set with per-turb breakdown."""
    X_cdm = cache["X_cdm"]
    X_iq = cache["X_iq"]
    X_stats_raw = cache["X_stats"]
    y = cache["y"]
    turbs = cache["turbs"]

    X_stats = apply_stats_norm(X_stats_raw, stats_mean, stats_std)
    test_ds = MultiRepDataset(X_cdm, X_iq, X_stats, y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    y_true, y_pred, _ = predict_loader(model, test_loader, device)

    overall = compute_metrics(y_true, y_pred)

    # Per-turb breakdown
    test_turbs = turbs[test_idx]
    weak_mask = test_turbs == "weak"
    strong_mask = test_turbs == "strong"

    weak_metrics = compute_metrics(y_true[weak_mask], y_pred[weak_mask]) if weak_mask.sum() > 0 else {}
    strong_metrics = compute_metrics(y_true[strong_mask], y_pred[strong_mask]) if strong_mask.sum() > 0 else {}

    # Save confusion matrix
    cm = np.array(overall["confusion_matrix"])
    save_confusion_matrix(
        cm, MOD_NAMES, out_dir / f"confusion_matrix_{prefix}.png",
        title=f"{prefix} confusion matrix",
    )

    return {
        **overall,
        "weak": weak_metrics,
        "strong": strong_metrics,
    }


# ─────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────
def run_s2_simonly(seed: int, sim_cache: dict, real_cache: dict, idx: dict, device) -> dict:
    config = "s2_simonly"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [S2-SimOnly] seed={seed}")
    ensure_dir(out_dir)

    sim_split = sim_cache["split"]
    sim_train_idx = np.where(sim_split == "train")[0]
    sim_val_idx = np.where(sim_split == "val")[0]

    stats_mean, stats_std = compute_stats_mean_std(sim_cache["X_stats"], sim_train_idx)
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        sim_cache, sim_train_idx, sim_val_idx,
        stats_mean, stats_std, HP_PRETRAIN, ckpt_path, device,
        log_prefix=f"[S2/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, real_cache, idx["b_test"], stats_mean, stats_std, device, out_dir, "B_test")

    result = {
        "stage": "S2", "config": config, "seed": seed,
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


def run_s3_scratch(seed: int, ratio: float, ratio_label: str,
                   real_cache: dict, idx: dict, device) -> dict:
    config = f"s3_scratch_{ratio_label}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [S3-scratch-{ratio_label}] seed={seed}")
    ensure_dir(out_dir)

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    train_subset = subset_indices(idx["a_train"], y, turbs, ratio, seed)

    stats_mean, stats_std = compute_stats_mean_std(real_cache["X_stats"], train_subset)
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_subset, idx["a_val"],
        stats_mean, stats_std, HP_SCRATCH, ckpt_path, device,
        log_prefix=f"[S3-sc-{ratio_label}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, real_cache, idx["b_test"], stats_mean, stats_std, device, out_dir, "B_test")

    result = {
        "stage": "S3", "config": config, "seed": seed, "ratio": ratio, "mode": "scratch",
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


def run_s3_ft(seed: int, ratio: float, ratio_label: str,
              sim_cache: dict, real_cache: dict, idx: dict, device,
              s2_ckpt_path: Optional[Path] = None) -> dict:
    config = f"s3_ft_{ratio_label}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [S3-ft-{ratio_label}] seed={seed}")
    ensure_dir(out_dir)

    # Load S2 pretrained checkpoint
    pretrained_ckpt = None
    if s2_ckpt_path is not None and s2_ckpt_path.exists():
        pretrained_ckpt = torch.load(str(s2_ckpt_path), map_location=device, weights_only=False)
        print(f"    Loaded S2 pretrain (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")
    else:
        print(f"    [WARNING] S2 checkpoint not found: {s2_ckpt_path}")

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    train_subset = subset_indices(idx["a_train"], y, turbs, ratio, seed)

    # Use sim stats for normalization (from pretrained model)
    if pretrained_ckpt is not None:
        stats_mean = pretrained_ckpt["stats_mean"]
        stats_std = pretrained_ckpt["stats_std"]
    else:
        stats_mean, stats_std = compute_stats_mean_std(real_cache["X_stats"], train_subset)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_subset, idx["a_val"],
        stats_mean, stats_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=pretrained_ckpt,
        log_prefix=f"[S3-ft-{ratio_label}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, real_cache, idx["b_test"], stats_mean, stats_std, device, out_dir, "B_test")

    result = {
        "stage": "S3", "config": config, "seed": seed, "ratio": ratio, "mode": "simpretrain_realfinetune",
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


def run_s4_joint(seed: int, mode: str, sim_cache: dict, real_cache: dict, idx: dict, device) -> dict:
    config = f"s4_joint_{mode}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [S4-joint-{mode}] seed={seed}")
    ensure_dir(out_dir)

    sim_split = sim_cache["split"]
    sim_train_idx_local = np.where(sim_split == "train")[0]

    n_sim = len(sim_cache["y"])
    n_real = len(real_cache["y"])

    combined_X_cdm = np.concatenate([sim_cache["X_cdm"], real_cache["X_cdm"]], axis=0)
    combined_X_iq = np.concatenate([sim_cache["X_iq"], real_cache["X_iq"]], axis=0)
    combined_X_stats = np.concatenate([sim_cache["X_stats"], real_cache["X_stats"]], axis=0)
    combined_y = np.concatenate([sim_cache["y"], real_cache["y"]], axis=0)
    combined_domain = np.concatenate([
        np.zeros(n_sim, dtype=np.int64),
        np.ones(n_real, dtype=np.int64),
    ], axis=0)

    combined = {
        "X_cdm": combined_X_cdm, "X_iq": combined_X_iq,
        "X_stats": combined_X_stats, "y": combined_y,
        "domain": combined_domain,
    }

    train_idx = np.concatenate([
        sim_train_idx_local,
        n_sim + idx["a_train"],
    ], axis=0)
    val_idx = (n_sim + idx["a_val"]).astype(np.int64)

    combined_mean, combined_std = compute_stats_mean_std(combined_X_stats, train_idx)
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1_joint(
        combined, train_idx, val_idx,
        combined_mean, combined_std, HP_JOINT, ckpt_path, device,
        mode=mode, log_prefix=f"[S4-{mode}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test_idx = (n_sim + idx["a_test"]).astype(np.int64)
    b_test_idx = (n_sim + idx["b_test"]).astype(np.int64)

    # Build a cache dict for evaluation (use combined arrays)
    eval_cache = {
        "X_cdm": combined_X_cdm, "X_iq": combined_X_iq,
        "X_stats": combined_X_stats, "y": combined_y,
        "turbs": np.concatenate([sim_cache["turbs"], real_cache["turbs"]], axis=0),
    }

    a_test = eval_on_test(model, eval_cache, a_test_idx, combined_mean, combined_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, eval_cache, b_test_idx, combined_mean, combined_std, device, out_dir, "B_test")

    result = {
        "stage": "S4", "config": config, "seed": seed, "joint_mode": mode,
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


def run_s5_awgn(seed: int, awgn_cache: dict, real_cache: dict, idx: dict, device) -> dict:
    config = "s5_awgn_only"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [S5-AWGN-only] seed={seed}")
    ensure_dir(out_dir)

    awgn_split = awgn_cache["split"]
    awgn_train_idx = np.where(awgn_split == "train")[0]
    awgn_val_idx = np.where(awgn_split == "val")[0]

    stats_mean, stats_std = compute_stats_mean_std(awgn_cache["X_stats"], awgn_train_idx)
    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        awgn_cache, awgn_train_idx, awgn_val_idx,
        stats_mean, stats_std, HP_AWGN, ckpt_path, device,
        log_prefix=f"[S5/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, real_cache, idx["b_test"], stats_mean, stats_std, device, out_dir, "B_test")

    result = {
        "stage": "S5", "config": config, "seed": seed,
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


def run_b_ft(seed: int, fraction: float, fraction_label: str,
             real_cache: dict, idx: dict, device,
             s3_ft100_ckpt_path: Optional[Path] = None) -> dict:
    config = f"b_ft_{fraction_label}"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [B-FT-{fraction_label}] seed={seed}")
    ensure_dir(out_dir)

    # Load S3 ft_100% checkpoint (A-pretrained model)
    pretrained_ckpt = None
    if s3_ft100_ckpt_path is not None and s3_ft100_ckpt_path.exists():
        pretrained_ckpt = torch.load(str(s3_ft100_ckpt_path), map_location=device, weights_only=False)
        print(f"    Loaded S3-ft-100pct pretrain (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")
    else:
        print(f"    [ERROR] S3-ft-100pct checkpoint not found: {s3_ft100_ckpt_path}")
        return {}

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    b_ft_subset = subset_b_finetune(idx["b_ft_pool"], y, turbs, fraction, seed)

    # Use A-pretrained model's stats
    stats_mean = pretrained_ckpt["stats_mean"]
    stats_std = pretrained_ckpt["stats_std"]

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, b_ft_subset, idx["a_val"],
        stats_mean, stats_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=pretrained_ckpt,
        log_prefix=f"[B-FT-{fraction_label}/s{seed}] ",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")
    b_test = eval_on_test(model, real_cache, idx["b_test"], stats_mean, stats_std, device, out_dir, "B_test")

    result = {
        "stage": "B-FT", "config": config, "seed": seed, "fraction": fraction,
        "A_test": a_test, "B_test": b_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────
def aggregate_results(seeds: list[int]) -> dict:
    """Aggregate metrics across seeds."""
    configs = [
        "s2_simonly",
        "s3_scratch_5pct", "s3_scratch_10pct", "s3_scratch_20pct", "s3_scratch_100pct",
        "s3_ft_5pct", "s3_ft_10pct", "s3_ft_20pct", "s3_ft_100pct",
        "s4_joint_direct", "s4_joint_weighted", "s4_joint_mmd",
        "s5_awgn_only",
        "b_ft_5pct", "b_ft_10pct",
    ]

    agg = {}
    for cfg in configs:
        results = []
        for seed in seeds:
            metrics_path = OUT_ROOT / f"seed_{seed}" / cfg / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))

        if not results:
            continue

        agg[cfg] = {"config": cfg, "n_seeds": len(results), "seeds": [r["seed"] for r in results]}

        # Aggregate A_test and B_test separately
        for test_key in ["A_test", "B_test"]:
            for metric in ["overall_acc", "macro_f1", "acc_128qam", "acc_256qam",
                          "high_order_acc", "a_b_gap"]:
                vals = [r.get(test_key, {}).get(metric, float("nan")) for r in results]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vals:
                    agg[cfg][f"{test_key}_{metric}_mean"] = float(np.mean(vals))
                    agg[cfg][f"{test_key}_{metric}_std"] = float(np.std(vals))
                else:
                    agg[cfg][f"{test_key}_{metric}_mean"] = float("nan")
                    agg[cfg][f"{test_key}_{metric}_std"] = 0.0

            # Per-class
            for mod in MOD_NAMES:
                vals = [r.get(test_key, {}).get("per_class_acc", {}).get(mod, float("nan"))
                        for r in results]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vals:
                    agg[cfg][f"{test_key}_acc_{mod}_mean"] = float(np.mean(vals))
                    agg[cfg][f"{test_key}_acc_{mod}_std"] = float(np.std(vals))

            # Per-turb
            for turb in ["weak", "strong"]:
                for metric in ["overall_acc", "macro_f1", "high_order_acc", "acc_128qam", "acc_256qam"]:
                    vals = [r.get(test_key, {}).get(turb, {}).get(metric, float("nan"))
                            for r in results]
                    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if vals:
                        agg[cfg][f"{test_key}_{turb}_{metric}_mean"] = float(np.mean(vals))
                        agg[cfg][f"{test_key}_{turb}_{metric}_std"] = float(np.std(vals))

    return agg


def write_summary(agg: dict, seeds: list[int]):
    """Write summary.md."""
    lines = [
        "# Stage II Fast-Publish 训练结果汇总",
        f"\n生成时间: {now_str()}",
        f"模型: M1 (CDM-Stats-Radial)",
        f"Sim 数据: Calibrated-Sim-v2.2 (12000 帧, merged weak+strong)",
        f"AWGN-only: 12000 帧 (S5 消融)",
        f"Seeds: {seeds}",
        f"总 run 数: {len(agg)} configs × {len(seeds)} seeds",
        f"\n## 聚合结果 (mean±std across seeds)\n",
        "| Config | nSeeds | A-test Overall | A-test 128QAM | A-test 256QAM | B-test Overall | B-test 128QAM | B-test 256QAM |",
        "|--------|--------|----------------|---------------|---------------|----------------|---------------|---------------|",
    ]
    for cfg, d in sorted(agg.items()):
        ns = d.get("n_seeds", 0)
        a_oa = d.get("A_test_overall_acc_mean", float("nan"))
        a_128 = d.get("A_test_acc_128qam_mean", float("nan"))
        a_256 = d.get("A_test_acc_256qam_mean", float("nan"))
        b_oa = d.get("B_test_overall_acc_mean", float("nan"))
        b_128 = d.get("B_test_acc_128qam_mean", float("nan"))
        b_256 = d.get("B_test_acc_256qam_mean", float("nan"))
        def fmt(v):
            return f"{v*100:.1f}" if not np.isnan(v) else "N/A"
        lines.append(f"| {cfg} | {ns} | {fmt(a_oa)} | {fmt(a_128)} | {fmt(a_256)} | {fmt(b_oa)} | {fmt(b_128)} | {fmt(b_256)} |")

    summary_path = OUT_ROOT / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] summary → {summary_path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else ALL_SEEDS
    print(f"=== Stage II Fast-Publish ===")
    print(f"Seeds: {seeds}")
    print(f"Output: {OUT_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build caches
    print(f"\n=== Building caches ===")
    real_cache = build_real_cache()
    sim_cache = merge_sim_caches()
    awgn_cache = merge_awgn_caches()
    idx = get_indices(real_cache)

    # Run experiments
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed = {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        # S2
        s2_result = run_s2_simonly(seed, sim_cache, real_cache, idx, device)
        s2_ckpt = OUT_ROOT / f"seed_{seed}" / "s2_simonly" / "best_model.pt"

        # S3 scratch
        for ratio, rlabel in zip(RATIOS, RATIO_LABELS):
            run_s3_scratch(seed, ratio, rlabel, real_cache, idx, device)

        # S3 ft (needs S2 checkpoint)
        for ratio, rlabel in zip(RATIOS, RATIO_LABELS):
            run_s3_ft(seed, ratio, rlabel, sim_cache, real_cache, idx, device, s2_ckpt_path=s2_ckpt)

        # S4 joint
        for mode in ["direct", "weighted", "mmd"]:
            run_s4_joint(seed, mode, sim_cache, real_cache, idx, device)

        # S5 AWGN
        run_s5_awgn(seed, awgn_cache, real_cache, idx, device)

        # B-FT (needs S3 ft_100% checkpoint)
        s3_ft100_ckpt = OUT_ROOT / f"seed_{seed}" / "s3_ft_100pct" / "best_model.pt"
        for fraction, flabel in [(0.05, "5pct"), (0.10, "10pct")]:
            run_b_ft(seed, fraction, flabel, real_cache, idx, device, s3_ft100_ckpt_path=s3_ft100_ckpt)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"  汇总 {len(seeds)} seeds × 15 configs")
    print(f"{'='*60}")
    agg = aggregate_results(seeds)
    agg_path = OUT_ROOT / "aggregated_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"  aggregated_results.json → {agg_path}")
    write_summary(agg, seeds)

    print(f"\n[完成] Stage II Fast-Publish 结果已输出到 {OUT_ROOT}")
    print(f"  请审阅 summary.md 和 aggregated_results.json")


if __name__ == "__main__":
    main()
