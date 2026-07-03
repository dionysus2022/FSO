#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_5seed_train_merged.py
合并数据集 (Dataset-A + Dataset-B) 的 5-seed 训练。
使用分层文件级划分：每个 split 都包含 A 和 B 的样本。
输出到 recognition_mrf_lowmem_merged/seed_XXXX/{weak,strong}_cdm_stats/
"""
import os
import sys
import json
import subprocess
import time

PYTHON = r"C:\Users\26081\AppData\Local\Programs\Python\Python311\python.exe"
TRAIN_SCRIPT = r"D:\Project_code\FSO_research\code\python\train_mrf_lowmem.py"
DATA_ROOT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\merged_AB"
BASE_OUT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem_merged"

SEEDS = [2026, 2027, 2028, 2029, 2030]
MODEL = "cdm_stats"
EPOCHS = "60"
BATCH_SIZE = "32"
PATIENCE = "10"
LR = "0.001"

all_results = []

for seed in SEEDS:
    out_dir = os.path.join(BASE_OUT, f"seed_{seed}")
    print(f"\n{'='*70}")
    print(f"Training seed = {seed} (merged AB dataset)")
    print(f"Output: {out_dir}")
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
        out = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
        print(out.encode('ascii', 'replace').decode('ascii'))
    except Exception:
        pass

    if result.returncode != 0:
        print(f"[ERROR] seed {seed} failed! returncode={result.returncode}")
        try:
            print(result.stderr[-1500:].encode('ascii', 'replace').decode('ascii'))
        except Exception:
            pass
        all_results.append({'seed': seed, 'status': 'failed', 'elapsed': elapsed})
        continue

    sp = os.path.join(out_dir, f"summary_all_turbulence_{MODEL}.csv")
    if os.path.exists(sp):
        import pandas as pd
        df = pd.read_csv(sp)
        for _, row in df.iterrows():
            all_results.append({
                'seed': seed,
                'turb': row['turb_name'],
                'acc': float(row['test_accuracy']),
                'f1': float(row['test_f1_macro']),
                'elapsed': elapsed,
            })
            print(f"  {row['turb_name']}: acc={float(row['test_accuracy'])*100:.2f}%, f1={float(row['test_f1_macro']):.4f}")
    else:
        print(f"  [WARN] summary file not found: {sp}")

# 汇总
print("\n" + "=" * 70)
print("5-Seed Training Summary (Merged AB Dataset)")
print("=" * 70)

import pandas as pd
df_res = pd.DataFrame([r for r in all_results if 'turb' in r])

if len(df_res) > 0:
    for turb in ["weak", "strong"]:
        sub = df_res[df_res['turb'] == turb]
        if len(sub) > 0:
            acc_mean = sub['acc'].mean() * 100
            acc_std = sub['acc'].std() * 100
            f1_mean = sub['f1'].mean()
            f1_std = sub['f1'].std()
            print(f"\n{turb}:")
            print(f"  Accuracy: {acc_mean:.2f} +/- {acc_std:.2f}%")
            print(f"  F1 Macro: {f1_mean:.4f} +/- {f1_std:.4f}")
            for _, r in sub.iterrows():
                print(f"    seed {r['seed']}: acc={r['acc']*100:.2f}%, f1={r['f1']:.4f}")

    summary_path = os.path.join(BASE_OUT, "5seed_training_summary_merged.csv")
    df_res.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")

print("\nDone.")
