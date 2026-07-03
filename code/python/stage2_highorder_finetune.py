# -*- coding: utf-8 -*-
"""
stage2_highorder_finetune.py — High-Order-Aware Fine-Tuning Experiments

Three fine-tuning variants at 5% real data, each × 5 seeds (2026–2030):
  ho_weighted_5pct:  high-order class-weighted CE (weights [1,1,1,1.5,2.5,2.0])
  ho_balanced_5pct:  high-order-heavy batch sampling (WeightedRandomSampler)
  ho_replay_5pct:    real:sim replay fine-tune (real:sim = 2:1, sim data mixed in)

All variants start from the existing S2 SimOnly pretrained checkpoint and
fine-tune on the 5% A-train subset, directly comparable to s3_ft_5pct (84.4%).

可恢复: 每个 run 检查 metrics.json 是否存在，存在则跳过。
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import STAGE2_ROOT, ensure_dir, now_str
from train_mrf_lowmem import set_seed, MOD_NAMES, save_training_curve
from stage2_round1 import (
    train_m1, train_m1_joint,
    HP_FINETUNE, MODEL_MODE, STATS_DIM, NUM_CLASSES,
    load_existing_metrics, save_result,
)
from stage2_fast_publish import (
    build_real_cache, merge_sim_caches, get_indices,
    subset_indices, eval_on_test,
    OUT_ROOT, ALL_SEEDS,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
HO_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.5, 2.5, 2.0], dtype=torch.float32)
RATIO_5PCT = 0.05
RATIO_LABEL = "5pct"

HO_CONFIGS = ["ho_weighted_5pct", "ho_balanced_5pct", "ho_replay_5pct"]


# ─────────────────────────────────────────────────────────
# Method 1: High-order class-weighted CE
# ─────────────────────────────────────────────────────────
def run_ho_weighted(seed: int, sim_cache: dict, real_cache: dict, idx: dict,
                    device, s2_ckpt_path: Optional[Path] = None) -> dict:
    config = "ho_weighted_5pct"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [HO-weighted-5pct] seed={seed}")
    ensure_dir(out_dir)

    pretrained_ckpt = None
    if s2_ckpt_path is not None and s2_ckpt_path.exists():
        pretrained_ckpt = torch.load(str(s2_ckpt_path), map_location=device, weights_only=False)
        print(f"    Loaded S2 pretrain (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")
    else:
        print(f"    [WARNING] S2 checkpoint not found: {s2_ckpt_path}")

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    train_subset = subset_indices(idx["a_train"], y, turbs, RATIO_5PCT, seed)

    if pretrained_ckpt is not None:
        stats_mean = pretrained_ckpt["stats_mean"]
        stats_std = pretrained_ckpt["stats_std"]
    else:
        from stage2_round1 import compute_stats_mean_std
        stats_mean, stats_std = compute_stats_mean_std(real_cache["X_stats"], train_subset)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_subset, idx["a_val"],
        stats_mean, stats_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=pretrained_ckpt,
        log_prefix=f"[HO-wt/s{seed}] ",
        class_weights_override=HO_WEIGHTS,
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")

    result = {
        "stage": "HO", "config": config, "seed": seed, "ratio": RATIO_5PCT,
        "mode": "highorder_weighted_ce",
        "A_test": a_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# Method 2: High-order-heavy batch sampling
# ─────────────────────────────────────────────────────────
def run_ho_balanced(seed: int, sim_cache: dict, real_cache: dict, idx: dict,
                    device, s2_ckpt_path: Optional[Path] = None) -> dict:
    config = "ho_balanced_5pct"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [HO-balanced-5pct] seed={seed}")
    ensure_dir(out_dir)

    pretrained_ckpt = None
    if s2_ckpt_path is not None and s2_ckpt_path.exists():
        pretrained_ckpt = torch.load(str(s2_ckpt_path), map_location=device, weights_only=False)
        print(f"    Loaded S2 pretrain (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")
    else:
        print(f"    [WARNING] S2 checkpoint not found: {s2_ckpt_path}")

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    train_subset = subset_indices(idx["a_train"], y, turbs, RATIO_5PCT, seed)

    if pretrained_ckpt is not None:
        stats_mean = pretrained_ckpt["stats_mean"]
        stats_std = pretrained_ckpt["stats_std"]
    else:
        from stage2_round1 import compute_stats_mean_std
        stats_mean, stats_std = compute_stats_mean_std(real_cache["X_stats"], train_subset)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1(
        real_cache, train_subset, idx["a_val"],
        stats_mean, stats_std, HP_FINETUNE, ckpt_path, device,
        pretrained_ckpt=pretrained_ckpt,
        log_prefix=f"[HO-bal/s{seed}] ",
        sampler_mode="highorder_heavy",
    )
    save_training_curve(history, out_dir / "training_curve.png")

    a_test = eval_on_test(model, real_cache, idx["a_test"], stats_mean, stats_std, device, out_dir, "A_test")

    result = {
        "stage": "HO", "config": config, "seed": seed, "ratio": RATIO_5PCT,
        "mode": "highorder_balanced_batch",
        "A_test": a_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# Method 3: Real:sim replay fine-tune (real:sim = 2:1)
# ─────────────────────────────────────────────────────────
def run_ho_replay(seed: int, sim_cache: dict, real_cache: dict, idx: dict,
                  device, s2_ckpt_path: Optional[Path] = None) -> dict:
    config = "ho_replay_5pct"
    out_dir = OUT_ROOT / f"seed_{seed}" / config
    existing = load_existing_metrics(out_dir)
    if existing is not None:
        print(f"  [SKIP] {config} seed={seed}")
        return existing

    print(f"\n  [HO-replay-5pct] seed={seed}")
    ensure_dir(out_dir)

    pretrained_ckpt = None
    if s2_ckpt_path is not None and s2_ckpt_path.exists():
        pretrained_ckpt = torch.load(str(s2_ckpt_path), map_location=device, weights_only=False)
        print(f"    Loaded S2 pretrain (best_val_acc={pretrained_ckpt.get('best_val_acc', '?')})")
    else:
        print(f"    [WARNING] S2 checkpoint not found: {s2_ckpt_path}")

    y = real_cache["y"]
    turbs = real_cache["turbs"]
    train_subset = subset_indices(idx["a_train"], y, turbs, RATIO_5PCT, seed)

    # Use S2 pretrained stats for normalization
    if pretrained_ckpt is not None:
        stats_mean = pretrained_ckpt["stats_mean"]
        stats_std = pretrained_ckpt["stats_std"]
    else:
        from stage2_round1 import compute_stats_mean_std
        stats_mean, stats_std = compute_stats_mean_std(real_cache["X_stats"], train_subset)

    # Build combined sim + real cache (following run_s4_joint pattern)
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

    # train_idx = sim_train + 5% real subset (offset by n_sim)
    train_idx = np.concatenate([
        sim_train_idx_local,
        n_sim + train_subset,
    ], axis=0)
    val_idx = (n_sim + idx["a_val"]).astype(np.int64)

    ckpt_path = out_dir / "best_model.pt"
    model, history = train_m1_joint(
        combined, train_idx, val_idx,
        stats_mean, stats_std, HP_FINETUNE, ckpt_path, device,
        mode="replay", log_prefix=f"[HO-rep/s{seed}] ",
        pretrained_ckpt=pretrained_ckpt,
    )
    save_training_curve(history, out_dir / "training_curve.png")

    # Evaluate on A-test (offset indices into combined cache)
    a_test_idx = (n_sim + idx["a_test"]).astype(np.int64)
    eval_cache = {
        "X_cdm": combined_X_cdm, "X_iq": combined_X_iq,
        "X_stats": combined_X_stats, "y": combined_y,
        "turbs": np.concatenate([sim_cache["turbs"], real_cache["turbs"]], axis=0),
    }
    a_test = eval_on_test(model, eval_cache, a_test_idx, stats_mean, stats_std, device, out_dir, "A_test")

    result = {
        "stage": "HO", "config": config, "seed": seed, "ratio": RATIO_5PCT,
        "mode": "highorder_replay_finetune",
        "A_test": a_test,
        "best_val_acc": max(h.get("val_acc", 0) for h in history) if history else 0,
    }
    save_result(result, out_dir)
    return result


# ─────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────
def aggregate_ho_results(seeds: list[int]) -> dict:
    """Aggregate high-order finetune results across seeds."""
    agg = {}
    for config in HO_CONFIGS:
        per_seed = []
        for seed in seeds:
            p = OUT_ROOT / f"seed_{seed}" / config / "metrics.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    per_seed.append(json.load(f))
        if not per_seed:
            continue

        a_tests = [r.get("A_test", {}) for r in per_seed]
        metrics = {}
        for key in ["overall_acc", "macro_f1", "high_order_acc",
                     "acc_128qam", "acc_256qam"]:
            vals = [float(t.get(key, 0.0)) * 100 for t in a_tests]
            metrics[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": vals,
            }
        # per-class accuracy
        per_class = {}
        for cls in MOD_NAMES:
            vals = [float(t.get("per_class_acc", {}).get(cls, 0.0)) * 100 for t in a_tests]
            per_class[cls] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        metrics["per_class_acc"] = per_class

        agg[config] = {
            "n_seeds": len(per_seed),
            "seeds": seeds,
            "metrics": metrics,
        }
    return agg


def write_ho_summary(agg: dict, seeds: list[int]):
    """Write markdown summary of high-order finetune results."""
    summary_path = OUT_ROOT / "summary_ho.md"
    lines = ["# High-Order-Aware Fine-Tuning Results\n"]
    lines.append(f"Seeds: {seeds}\n")
    lines.append("All variants use 5% A-train subset + S2 SimOnly pretrained checkpoint.\n")
    lines.append("Baseline: s3_ft_5pct = 84.4 ± 1.8% (from aggregated_results.json)\n\n")

    lines.append("| Config | Overall | Macro-F1 | High-order | 128QAM | 256QAM |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for config in HO_CONFIGS:
        if config not in agg:
            continue
        m = agg[config]["metrics"]
        row = f"| {config} | "
        for key in ["overall_acc", "macro_f1", "high_order_acc", "acc_128qam", "acc_256qam"]:
            row += f"{m[key]['mean']:.1f}±{m[key]['std']:.1f} | "
        lines.append(row[:-2])

    lines.append("\n## Per-class accuracy\n")
    lines.append("| Config | QPSK | 16QAM | 32QAM | 64QAM | 128QAM | 256QAM |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for config in HO_CONFIGS:
        if config not in agg:
            continue
        pc = agg[config]["metrics"]["per_class_acc"]
        row = f"| {config} | "
        for cls in MOD_NAMES:
            row += f"{pc[cls]['mean']:.1f}±{pc[cls]['std']:.1f} | "
        lines.append(row[:-2])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  summary_ho.md → {summary_path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else ALL_SEEDS
    print(f"=== High-Order-Aware Fine-Tuning ===")
    print(f"Seeds: {seeds}")
    print(f"Output: {OUT_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build caches
    print(f"\n=== Building caches ===")
    real_cache = build_real_cache()
    sim_cache = merge_sim_caches()
    idx = get_indices(real_cache)

    # Run experiments
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed = {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        s2_ckpt = OUT_ROOT / f"seed_{seed}" / "s2_simonly" / "best_model.pt"

        run_ho_weighted(seed, sim_cache, real_cache, idx, device, s2_ckpt_path=s2_ckpt)
        run_ho_balanced(seed, sim_cache, real_cache, idx, device, s2_ckpt_path=s2_ckpt)
        run_ho_replay(seed, sim_cache, real_cache, idx, device, s2_ckpt_path=s2_ckpt)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"  汇总 {len(seeds)} seeds × 3 configs")
    print(f"{'='*60}")
    agg = aggregate_ho_results(seeds)
    agg_path = OUT_ROOT / "aggregated_results_ho.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"  aggregated_results_ho.json → {agg_path}")
    write_ho_summary(agg, seeds)

    print(f"\n[完成] High-Order Fine-Tuning 结果已输出到 {OUT_ROOT}")
    print(f"  请审阅 summary_ho.md 和 aggregated_results_ho.json")


if __name__ == "__main__":
    main()
