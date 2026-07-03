"""
5-Fold 文件级交叉验证 — cdm_stats 模型
对 weak 和 strong 各跑 5 折，汇总 mean +/- std
"""
import os, sys, json, subprocess
import numpy as np
import pandas as pd

PYTHON = r"C:\Users\26081\AppData\Local\Programs\Python\Python311\python.exe"
TRAIN_SCRIPT = r"D:\Project_code\FSO_research\code\python\train_mrf_lowmem.py"
DATA_ROOT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\2026.06.28"
BASE_OUT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem"

SEED = 2026
N_FOLDS = 5
MODEL = "cdm_stats"

results = []

for fold in range(N_FOLDS):
    out_dir = os.path.join(BASE_OUT, f"5fold_cdm_stats", f"fold_{fold}")
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{N_FOLDS} -> {out_dir}")
    print(f"{'='*60}")

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--data-root", DATA_ROOT,
        "--out", out_dir,
        "--model", MODEL,
        "--epochs", "60",
        "--batch-size", "32",
        "--patience", "10",
        "--lr", "0.001",
        "--seed", str(SEED),
        "--fold", str(fold),
        "--n-folds", str(N_FOLDS),
        "--rebuild-cache",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    # Safe print (avoid GBK encoding issues on Windows console)
    try:
        out = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        print(out.encode('ascii', 'replace').decode('ascii'))
    except:
        pass
    if result.returncode != 0:
        print(f"[ERROR] Fold {fold} failed!")
        try:
            print(result.stderr[-1000:].encode('ascii', 'replace').decode('ascii'))
        except:
            pass
        continue

    # Read results
    for turb in ["weak", "strong"]:
        sp = os.path.join(out_dir, f"summary_all_turbulence_{MODEL}.csv")
        if os.path.exists(sp):
            df = pd.read_csv(sp)
            row = df[df['turb_name'] == turb]
            if len(row) > 0:
                results.append({
                    'fold': fold,
                    'turb': turb,
                    'acc': float(row['test_accuracy'].values[0]),
                    'f1': float(row['test_f1_macro'].values[0]),
                })
                print(f"  {turb}: acc={float(row['test_accuracy'].values[0])*100:.2f}%, f1={float(row['test_f1_macro'].values[0]):.4f}")

# Summary
print("\n" + "=" * 60)
print("5-Fold Cross Validation Summary")
print("=" * 60)

df_res = pd.DataFrame(results)
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
            print(f"    Fold {r['fold']}: acc={r['acc']*100:.2f}%, f1={r['f1']:.4f}")

# Save
df_res.to_csv(os.path.join(BASE_OUT, "5fold_cdm_stats", "summary.csv"), index=False)
print(f"\nSaved to {os.path.join(BASE_OUT, '5fold_cdm_stats', 'summary.csv')}")