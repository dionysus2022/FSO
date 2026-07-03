# -*- coding: utf-8 -*-
"""
make_ab_domain_split.py — A/B 域划分

Dataset-A (in-domain): train/val/test = 70/10/20 (file-level, stratified by mod+turb)
Dataset-B (out-of-domain): finetune/test = 20/80 (file-level, stratified by mod+turb)

输出:
  - ab_domain_split.csv (1620 rows, domain + split columns)
  - ab_distribution_stats.json (PAPR, amp_mean, phase_concentration, amp_std per domain/turb)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
from stage2_common import FROZEN_SPLIT_CSV, STAGE2_ROOT, now_str, ensure_dir

OUT_DIR = STAGE2_ROOT / "fast_publish"
OUT_CSV = OUT_DIR / "ab_domain_split.csv"
OUT_STATS = OUT_DIR / "ab_distribution_stats.json"

SEED = 2026


def file_level_split(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per (mod_label, turb_name) file-level split. Returns df with 'split' column."""
    rng = np.random.default_rng(seed)
    result_rows = []
    for (mod, turb), grp in df.groupby(["mod_label", "turb_name"]):
        files = sorted(grp["rx_file"].unique())
        rng.shuffle(files)
        tr_files = set(files[:n_train])
        va_files = set(files[n_train:n_train + n_val])
        te_files = set(files[n_train + n_val:n_train + n_val + n_test])
        for _, row in grp.iterrows():
            r = row.to_dict()
            f = row["rx_file"]
            if f in tr_files:
                r["split"] = "train"
            elif f in va_files:
                r["split"] = "val"
            elif f in te_files:
                r["split"] = "test"
            else:
                r["split"] = "test"
            result_rows.append(r)
    return pd.DataFrame(result_rows)


def file_level_split_b(
    df: pd.DataFrame,
    n_finetune: int,
    n_test: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per (mod_label, turb_name) file-level split for B domain."""
    rng = np.random.default_rng(seed)
    result_rows = []
    for (mod, turb), grp in df.groupby(["mod_label", "turb_name"]):
        files = sorted(grp["rx_file"].unique())
        rng.shuffle(files)
        ft_files = set(files[:n_finetune])
        te_files = set(files[n_finetune:n_finetune + n_test])
        for _, row in grp.iterrows():
            r = row.to_dict()
            f = row["rx_file"]
            if f in ft_files:
                r["split"] = "finetune"
            elif f in te_files:
                r["split"] = "test"
            else:
                r["split"] = "test"
            result_rows.append(r)
    return pd.DataFrame(result_rows)


def check_leakage(df: pd.DataFrame, splits: list[str]):
    """Assert no rx_file overlap between splits."""
    file_sets = {}
    for s in splits:
        file_sets[s] = set(df[df["split"] == s]["rx_file"].unique())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = file_sets[splits[i]] & file_sets[splits[j]]
            assert len(overlap) == 0, f"Leakage between {splits[i]} and {splits[j]}: {overlap}"
    print(f"  [OK] No leakage between {splits}")


def compute_distribution_stats(df: pd.DataFrame) -> dict:
    """Compute distribution statistics per domain per turb."""
    stats = {}
    features = ["papr_db", "amp_mean", "amp_std", "phase_concentration", "phase_diff_std", "iq_corr"]
    for domain in ["A", "B"]:
        d = df[df["dataset_source"] == domain]
        stats[domain] = {}
        for turb in ["weak", "strong"]:
            dt = d[d["turb_name"] == turb]
            stats[domain][turb] = {}
            for feat in features:
                vals = dt[feat].dropna().values
                stats[domain][turb][feat] = {
                    "mean": float(np.mean(vals)) if len(vals) > 0 else None,
                    "std": float(np.std(vals)) if len(vals) > 0 else None,
                    "median": float(np.median(vals)) if len(vals) > 0 else None,
                    "n": int(len(vals)),
                }
    return stats


def main():
    ensure_dir(OUT_DIR)
    print(f"[1/4] Loading frozen split: {FROZEN_SPLIT_CSV}")
    df = pd.read_csv(FROZEN_SPLIT_CSV)
    print(f"  Total: {len(df)} rows (A={len(df[df.dataset_source=='A'])}, B={len(df[df.dataset_source=='B'])})")

    print(f"\n[2/4] Splitting Dataset-A (70/10/20, file-level)...")
    df_a = df[df["dataset_source"] == "A"].copy()
    # 25 files per (mod, turb) → train 18 / val 2 / test 5
    df_a_split = file_level_split(df_a, n_train=18, n_val=2, n_test=5, seed=SEED)
    a_counts = df_a_split["split"].value_counts().to_dict()
    print(f"  A: {a_counts}")
    check_leakage(df_a_split, ["train", "val", "test"])

    print(f"\n[3/4] Splitting Dataset-B (20/80 finetune/test, file-level)...")
    df_b = df[df["dataset_source"] == "B"].copy()
    # 20 files per (mod, turb) → finetune 4 / test 16
    df_b_split = file_level_split_b(df_b, n_finetune=4, n_test=16, seed=SEED)
    b_counts = df_b_split["split"].value_counts().to_dict()
    print(f"  B: {b_counts}")
    check_leakage(df_b_split, ["finetune", "test"])

    print(f"\n[4/4] Merging and saving...")
    df_merged = pd.concat([df_a_split, df_b_split], ignore_index=True)
    df_merged["domain"] = df_merged["dataset_source"]

    # Verify
    print(f"\n  Verification:")
    for domain in ["A", "B"]:
        d = df_merged[df_merged["domain"] == domain]
        for s in sorted(d["split"].unique()):
            ds = d[d["split"] == s]
            print(f"    {domain}/{s}: {len(ds)} frames, {ds['rx_file'].nunique()} files, "
                  f"{ds['mod_name'].nunique()} mods, {ds['turb_name'].nunique()} turbs")

    # Per (mod, turb, domain, split) counts
    print(f"\n  Per (domain, split, turb) frame counts:")
    for (domain, split, turb), grp in df_merged.groupby(["domain", "split", "turb_name"]):
        print(f"    {domain}/{split}/{turb}: {len(grp)}")

    df_merged.to_csv(OUT_CSV, index=False)
    print(f"\n  [saved] {OUT_CSV}")

    # Distribution stats
    stats = compute_distribution_stats(df_merged)
    stats["_meta"] = {
        "generated": now_str(),
        "total_frames": len(df_merged),
        "A_frames": len(df_merged[df_merged.domain == "A"]),
        "B_frames": len(df_merged[df_merged.domain == "B"]),
        "A_split": a_counts,
        "B_split": b_counts,
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {OUT_STATS}")

    print(f"\n[完成] A/B 域划分已生成到 {OUT_DIR}")


if __name__ == "__main__":
    main()
