#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_5seed_train_rap.py
实验1：特征改进 M2 (Radial-Angle) + M3 (RAP) 的 5-seed 训练。
合并数据集 (A+B)，分层文件级划分。
"""
import os
import json
import subprocess
import time
import numpy as np
import pandas as pd

PYTHON = r"C:\Users\26081\AppData\Local\Programs\Python\Python311\python.exe"
TRAIN_SCRIPT = r"D:\Project_code\FSO_research\code\python\train_mrf_lowmem.py"
DATA_ROOT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\merged_AB"

SEEDS = [2026, 2027, 2028, 2029, 2030]
EPOCHS = "60"
BATCH_SIZE = "32"
PATIENCE = "10"
LR = "0.001"

# M2: CDM-Stats-Radial-Angle (stats 16 + radial 32 + angle 32 = 80)
# M3: CDM-Stats-RAP (stats 16 + radial 32 + angle 32 + polar 64 = 144)
EXPERIMENTS = [
    {"model": "cdm_stats_radial_angle", "out_base": "recognition_mrf_lowmem_radial_angle"},
    {"model": "cdm_stats_rap",          "out_base": "recognition_mrf_lowmem_rap"},
]

BASE_RESULTS = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results"

all_results = []

for exp in EXPERIMENTS:
    model = exp["model"]
    base_out = os.path.join(BASE_RESULTS, exp["out_base"])
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {model}")
    print(f"# Output: {base_out}")
    print(f"{'#'*70}")

    for seed in SEEDS:
        out_dir = os.path.join(base_out, f"seed_{seed}")
        print(f"\n{'='*60}")
        print(f"Training seed = {seed} ({model})")
        print(f"{'='*60}")

        cmd = [
            PYTHON, TRAIN_SCRIPT,
            "--data-root", DATA_ROOT,
            "--out", out_dir,
            "--model", model,
            "--epochs", EPOCHS,
            "--batch-size", BATCH_SIZE,
            "--patience", PATIENCE,
            "--lr", LR,
            "--seed", str(seed),
            "--rebuild-cache",
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        elapsed = time.time() - t0

        try:
            out = result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout
            print(out.encode('ascii', 'replace').decode('ascii'))
        except Exception:
            pass

        if result.returncode != 0:
            print(f"[ERROR] {model} seed {seed} failed!")
            all_results.append({'model': model, 'seed': seed, 'status': 'failed'})
            continue

        sp = os.path.join(out_dir, f"summary_all_turbulence_{model}.csv")
        if os.path.exists(sp):
            df = pd.read_csv(sp)
            for _, row in df.iterrows():
                all_results.append({
                    'model': model,
                    'seed': seed,
                    'turb': row['turb_name'],
                    'acc': float(row['test_accuracy']),
                    'f1': float(row['test_f1_macro']),
                })
                print(f"  {row['turb_name']}: acc={float(row['test_accuracy'])*100:.2f}%")

# 汇总
print("\n" + "=" * 70)
print("All Experiments Summary")
print("=" * 70)

df_res = pd.DataFrame([r for r in all_results if 'turb' in r])
for model in [e["model"] for e in EXPERIMENTS]:
    for turb in ["weak", "strong"]:
        sub = df_res[(df_res['model'] == model) & (df_res['turb'] == turb)]
        if len(sub) > 0:
            print(f"\n{model} | {turb}: acc = {sub['acc'].mean()*100:.2f} ± {sub['acc'].std()*100:.2f}%")
            for _, r in sub.iterrows():
                print(f"  seed {r['seed']}: acc={r['acc']*100:.2f}%")

summary_path = os.path.join(BASE_RESULTS, "exp1_rap_training_summary.csv")
df_res.to_csv(summary_path, index=False)
print(f"\nSaved to {summary_path}")
print("\nDone.")
