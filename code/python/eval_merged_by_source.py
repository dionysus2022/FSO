#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_merged_by_source.py
评估合并训练模型在测试集上的表现，按 dataset_source (A/B) 拆分准确率。
关联 split.csv（含 dataset_source）和 test_predictions.csv。
输出 5-seed mean±std 的分来源准确率。
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_OUT = Path(r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem_merged")
SEEDS = [2026, 2027, 2028, 2029, 2030]
TURBS = ["weak", "strong"]
MODEL = "cdm_stats"

MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]

results = []  # rows: seed, turb, source, acc, per-class acc...

for seed in SEEDS:
    seed_dir = BASE_OUT / f"seed_{seed}"
    for turb in TURBS:
        turb_dir = seed_dir / f"{turb}_{MODEL}"
        split_csv = turb_dir / "split.csv"
        pred_csv = turb_dir / f"{turb}_{MODEL}_test_predictions.csv"

        if not split_csv.exists() or not pred_csv.exists():
            print(f"[WARN] missing files for seed={seed} turb={turb}")
            continue

        sp = pd.read_csv(split_csv)
        sp_test = sp[sp["split"] == "test"].reset_index(drop=True)
        pred = pd.read_csv(pred_csv).reset_index(drop=True)

        # 行数对齐检查
        if len(sp_test) != len(pred):
            print(f"[WARN] seed={seed} turb={turb}: split_test={len(sp_test)} vs pred={len(pred)}")
            continue

        # 关联 dataset_source
        pred["dataset_source"] = sp_test["dataset_source"].values
        pred["mod_name"] = sp_test["mod_name"].values
        pred["rx_file"] = sp_test["rx_file"].values
        pred["correct"] = (pred["y_true"] == pred["y_pred"]).astype(int)

        # 总体
        for src in ["A", "B", "ALL"]:
            sub = pred if src == "ALL" else pred[pred["dataset_source"] == src]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean()
            # per-class
            per_class = {}
            for m in MOD_NAMES:
                ms = sub[sub["mod_name"] == m]
                per_class[m] = float(ms["correct"].mean()) if len(ms) > 0 else float("nan")
            results.append({
                "seed": seed, "turb": turb, "source": src,
                "n": len(sub), "acc": acc, "per_class": per_class,
            })
            print(f"seed={seed} turb={turb} src={src}: n={len(sub)} acc={acc*100:.2f}%")

# 汇总
print("\n" + "=" * 70)
print("Merged AB Training: Test Accuracy by Source (5-seed mean±std)")
print("=" * 70)

summary_rows = []
for turb in TURBS:
    for src in ["A", "B", "ALL"]:
        sub = [r for r in results if r["turb"] == turb and r["source"] == src]
        if len(sub) == 0:
            continue
        accs = np.array([r["acc"] for r in sub])
        per_class_means = {}
        per_class_stds = {}
        for m in MOD_NAMES:
            vals = np.array([r["per_class"][m] for r in sub if not np.isnan(r["per_class"][m])])
            if len(vals) > 0:
                per_class_means[m] = float(np.mean(vals))
                per_class_stds[m] = float(np.std(vals))
        summary_rows.append({
            "turb": turb, "source": src,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "per_class_mean": per_class_means,
            "per_class_std": per_class_stds,
        })
        print(f"\n{turb} | source={src}: acc = {accs.mean()*100:.2f} ± {accs.std()*100:.2f}% (n_seeds={len(sub)})")
        for m in MOD_NAMES:
            if m in per_class_means:
                print(f"    {m:8s}: {per_class_means[m]*100:.2f} ± {per_class_stds[m]*100:.2f}%")

# 保存
out_dir = BASE_OUT.parent / "results" / "merged_ab_eval_2026.06.29"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "merged_ab_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "summary": summary_rows}, f, indent=2, ensure_ascii=False)

# 也保存为易读的 CSV
flat_rows = []
for r in summary_rows:
    row = {"turb": r["turb"], "source": r["source"],
           "acc_mean": r["acc_mean"], "acc_std": r["acc_std"]}
    for m in MOD_NAMES:
        row[f"{m}_mean"] = r["per_class_mean"].get(m, float("nan"))
        row[f"{m}_std"] = r["per_class_std"].get(m, float("nan"))
    flat_rows.append(row)
pd.DataFrame(flat_rows).to_csv(out_dir / "merged_ab_eval_summary.csv", index=False, encoding="utf-8-sig")

print(f"\nSaved to {out_dir}")
