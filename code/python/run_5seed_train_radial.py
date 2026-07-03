#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_5seed_train_radial.py
合并数据集 (A+B) 的 5-seed cdm_stats_radial 训练。
对比基线 cdm_stats，验证径向幅度直方图对高阶 QAM 的提升。
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
BASE_OUT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem_radial"

SEEDS = [2026, 2027, 2028, 2029, 2030]
MODEL = "cdm_stats_radial"
EPOCHS = "60"
BATCH_SIZE = "32"
PATIENCE = "10"
LR = "0.001"

all_results = []

for seed in SEEDS:
    out_dir = os.path.join(BASE_OUT, f"seed_{seed}")
    print(f"\n{'='*70}")
    print(f"Training seed = {seed} (cdm_stats_radial, merged AB)")
    print(f"{'='*70}")

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--data-root", DATA_ROOT,
        "--out", out_dir,
        "--model", MODEL,
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
        out = result.stdout[-2500:] if len(result.stdout) > 2500 else result.stdout
        print(out.encode('ascii', 'replace').decode('ascii'))
    except Exception:
        pass

    if result.returncode != 0:
        print(f"[ERROR] seed {seed} failed! returncode={result.returncode}")
        all_results.append({'seed': seed, 'status': 'failed'})
        continue

    sp = os.path.join(out_dir, f"summary_all_turbulence_{MODEL}.csv")
    if os.path.exists(sp):
        df = pd.read_csv(sp)
        for _, row in df.iterrows():
            all_results.append({
                'seed': seed,
                'turb': row['turb_name'],
                'acc': float(row['test_accuracy']),
                'f1': float(row['test_f1_macro']),
            })
            print(f"  {row['turb_name']}: acc={float(row['test_accuracy'])*100:.2f}%, f1={float(row['test_f1_macro']):.4f}")

print("\n" + "=" * 70)
print("5-Seed Training Summary (cdm_stats_radial, Merged AB)")
print("=" * 70)

df_res = pd.DataFrame([r for r in all_results if 'turb' in r])
for turb in ["weak", "strong"]:
    sub = df_res[df_res['turb'] == turb]
    if len(sub) > 0:
        print(f"\n{turb}: acc = {sub['acc'].mean()*100:.2f} ± {sub['acc'].std()*100:.2f}%")
        for _, r in sub.iterrows():
            print(f"  seed {r['seed']}: acc={r['acc']*100:.2f}%")

summary_path = os.path.join(BASE_OUT, "5seed_training_summary_radial.csv")
df_res.to_csv(summary_path, index=False)
print(f"\nSaved to {summary_path}")
