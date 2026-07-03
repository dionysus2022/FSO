#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_all_models_by_source.py
统一评估所有模型变体（M0-M3）在合并测试集上的表现，按 dataset_source 拆分。
输出每类调制准确率和 A-B Gap。
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_RESULTS = Path(r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results")
SEEDS = [2026, 2027, 2028, 2029, 2030]
TURBS = ["weak", "strong"]
MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]

# 模型配置：(显示名, 模型目录名, 模型模式, 子目录后缀)
MODELS = [
    ("M0-CDM-Stats",            "recognition_mrf_lowmem_merged",   "cdm_stats",              "cdm_stats"),
    ("M1-CDM-Stats-Radial",     "recognition_mrf_lowmem_radial",   "cdm_stats_radial",       "cdm_stats_radial"),
    ("M2-CDM-Stats-Rad-Ang",    "recognition_mrf_lowmem_radial_angle", "cdm_stats_radial_angle", "cdm_stats_radial_angle"),
    ("M3-CDM-Stats-RAP",        "recognition_mrf_lowmem_rap",      "cdm_stats_rap",           "cdm_stats_rap"),
    ("M4-RAP-Aug",              "recognition_mrf_lowmem_rap_aug",  "cdm_stats_rap",           "cdm_stats_rap"),
    ("M5-RAP-MMD",              "recognition_mrf_lowmem_rap_mmd",  "cdm_stats_rap",           "cdm_stats_rap"),
    ("M6-RAP-CORAL",            "recognition_mrf_lowmem_rap_coral","cdm_stats_rap",           "cdm_stats_rap"),
]

all_results = []

for (display, dir_name, model_mode, subdir) in MODELS:
    base_out = BASE_RESULTS / dir_name
    if not base_out.exists():
        print(f"[SKIP] {display}: directory not found {base_out}")
        continue

    for seed in SEEDS:
        seed_dir = base_out / f"seed_{seed}"
        for turb in TURBS:
            turb_dir = seed_dir / f"{turb}_{subdir}"
            split_csv = turb_dir / "split.csv"
            pred_csv = turb_dir / f"{turb}_{subdir}_test_predictions.csv"
            if not split_csv.exists() or not pred_csv.exists():
                continue
            sp = pd.read_csv(split_csv)
            sp_test = sp[sp["split"] == "test"].reset_index(drop=True)
            pred = pd.read_csv(pred_csv).reset_index(drop=True)
            if len(sp_test) != len(pred):
                continue
            pred["dataset_source"] = sp_test["dataset_source"].values
            pred["mod_name"] = sp_test["mod_name"].values
            pred["correct"] = (pred["y_true"] == pred["y_pred"]).astype(int)
            for src in ["A", "B", "ALL"]:
                sub = pred if src == "ALL" else pred[pred["dataset_source"] == src]
                if len(sub) == 0:
                    continue
                acc = sub["correct"].mean()
                per_class = {}
                for m in MOD_NAMES:
                    ms = sub[sub["mod_name"] == m]
                    per_class[m] = float(ms["correct"].mean()) if len(ms) > 0 else float("nan")
                all_results.append({
                    "model": display, "seed": seed, "turb": turb, "source": src,
                    "n": len(sub), "acc": acc, "per_class": per_class,
                })

# 汇总打印
print("=" * 80)
print("All Models: Test Accuracy by Source (5-seed mean±std)")
print("=" * 80)

summary_rows = []
for (display, _, _, _) in MODELS:
    for turb in TURBS:
        for src in ["A", "B", "ALL"]:
            sub = [r for r in all_results if r["model"] == display and r["turb"] == turb and r["source"] == src]
            if len(sub) == 0:
                continue
            accs = np.array([r["acc"] for r in sub])
            pcm = {}; pcs = {}
            for m in MOD_NAMES:
                vals = np.array([r["per_class"][m] for r in sub if not np.isnan(r["per_class"][m])])
                if len(vals) > 0:
                    pcm[m] = float(np.mean(vals)); pcs[m] = float(np.std(vals))
            summary_rows.append({
                "model": display, "turb": turb, "source": src,
                "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
                "per_class_mean": pcm, "per_class_std": pcs,
            })

# 打印汇总表
for turb in TURBS:
    print(f"\n{'='*80}")
    print(f"  {turb.upper()} TURBULENCE")
    print(f"{'='*80}")
    for src in ["A", "B", "ALL"]:
        print(f"\n  Source={src}:")
        header = f"  {'Model':<26s} {'Acc':>10s}"
        for m in MOD_NAMES:
            header += f" {m:>8s}"
        print(header)
        print("  " + "-" * (26 + 10 + 8*6))
        for row in summary_rows:
            if row["turb"] != turb or row["source"] != src:
                continue
            line = f"  {row['model']:<26s} {row['acc_mean']*100:8.2f}±{row['acc_std']*100:.1f}"
            for m in MOD_NAMES:
                if m in row["per_class_mean"]:
                    line += f" {row['per_class_mean'][m]*100:8.1f}"
                else:
                    line += f" {'N/A':>8s}"
            print(line)

# A-B Gap
print(f"\n{'='*80}")
print("  A-B GAP (Accuracy difference)")
print(f"{'='*80}")
header = f"  {'Model':<26s} {'Weak Gap':>10s} {'Strong Gap':>12s}"
print(header)
print("  " + "-" * 50)
for (display, _, _, _) in MODELS:
    gaps = {}
    for turb in TURBS:
        a_row = [r for r in summary_rows if r["model"]==display and r["turb"]==turb and r["source"]=="A"]
        b_row = [r for r in summary_rows if r["model"]==display and r["turb"]==turb and r["source"]=="B"]
        if a_row and b_row:
            gaps[turb] = (a_row[0]["acc_mean"] - b_row[0]["acc_mean"]) * 100
    line = f"  {display:<26s}"
    line += f" {gaps.get('weak', float('nan')):8.2f}%"
    line += f" {gaps.get('strong', float('nan')):10.2f}%"
    print(line)

# High-order QAM metric
print(f"\n{'='*80}")
print("  HIGH-ORDER QAM ACCURACY = mean(64QAM, 128QAM, 256QAM)")
print(f"{'='*80}")
header = f"  {'Model':<26s} {'Weak-HO':>10s} {'Strong-HO':>12s}"
print(header)
print("  " + "-" * 50)
for (display, _, _, _) in MODELS:
    ho = {}
    for turb in TURBS:
        all_row = [r for r in summary_rows if r["model"]==display and r["turb"]==turb and r["source"]=="ALL"]
        if all_row:
            pcm = all_row[0]["per_class_mean"]
            vals = [pcm.get(m, 0) for m in ["64QAM", "128QAM", "256QAM"]]
            ho[turb] = np.mean(vals) * 100
    line = f"  {display:<26s}"
    line += f" {ho.get('weak', float('nan')):8.2f}%"
    line += f" {ho.get('strong', float('nan')):10.2f}%"
    print(line)

# 保存
out_dir = BASE_RESULTS.parent / "results" / "exp1_all_models_eval"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "all_models_eval.json", "w", encoding="utf-8") as f:
    json.dump({"results": all_results, "summary": summary_rows}, f, indent=2, ensure_ascii=False, default=float)

flat_rows = []
for r in summary_rows:
    row = {"model": r["model"], "turb": r["turb"], "source": r["source"],
           "acc_mean": r["acc_mean"], "acc_std": r["acc_std"]}
    for m in MOD_NAMES:
        row[f"{m}_mean"] = r["per_class_mean"].get(m, float("nan"))
        row[f"{m}_std"] = r["per_class_std"].get(m, float("nan"))
    flat_rows.append(row)
pd.DataFrame(flat_rows).to_csv(out_dir / "all_models_eval.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved to {out_dir}")
