#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exp23_aug_da.py
实验2 (RAP+增强) + 实验3 (RAP+MMD / RAP+CORAL) 的 5-seed 训练。
"""
import os
import subprocess
import time
import pandas as pd

PYTHON = r"C:\Users\26081\AppData\Local\Programs\Python\Python311\python.exe"
SCRIPT = r"D:\Project_code\FSO_research\code\python\train_mrf_aug_da.py"
DATA_ROOT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\merged_AB"
BASE_OUT = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results"

# 实验配置：(实验名, model, augment, da_method, out_dir)
EXPERIMENTS = [
    ("Exp2-RAP-Aug",   "cdm_stats_rap", True,  "none", "recognition_mrf_lowmem_rap_aug"),
    ("Exp3a-RAP-MMD",  "cdm_stats_rap", False, "mmd",  "recognition_mrf_lowmem_rap_mmd"),
    ("Exp3b-RAP-CORAL","cdm_stats_rap", False, "coral","recognition_mrf_lowmem_rap_coral"),
]

all_results = []

for name, model, augment, da_method, out_base in EXPERIMENTS:
    out_dir = os.path.join(BASE_OUT, out_base)
    print(f"\n{'#'*70}")
    print(f"# {name}: model={model} augment={augment} da={da_method}")
    print(f"{'#'*70}")

    cmd = [
        PYTHON, SCRIPT,
        "--data-root", DATA_ROOT,
        "--out", out_dir,
        "--model", model,
        "--seeds", "2026", "2027", "2028", "2029", "2030",
        "--epochs", "60",
        "--batch-size", "32",
        "--lr", "0.001",
        "--patience", "10",
        "--rebuild-cache",
        "--da-weight", "0.1",
    ]
    if augment:
        cmd.append("--augment")
    cmd.extend(["--da-method", da_method])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    elapsed = time.time() - t0

    try:
        out = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        print(out.encode('ascii', 'replace').decode('ascii'))
    except Exception:
        pass

    if result.returncode != 0:
        print(f"[ERROR] {name} failed!")
        all_results.append({"exp": name, "status": "failed"})
        continue

    # 读取 summary
    summary_path = os.path.join(BASE_OUT, f"summary_{model}_aug{int(augment)}_da{da_method}.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            all_results.append({
                "exp": name, "model": model, "augment": augment, "da": da_method,
                "seed": row["seed"], "turb": row["turb_name"],
                "acc": float(row["test_accuracy"]), "f1": float(row["test_f1_macro"]),
            })
            print(f"  {name} seed={row['seed']} {row['turb_name']}: acc={float(row['test_accuracy'])*100:.2f}%")

# 汇总
print("\n" + "=" * 70)
print("Exp2+3 Summary")
print("=" * 70)
df_res = pd.DataFrame([r for r in all_results if 'acc' in r])
for name, _, _, _, _ in EXPERIMENTS:
    for turb in ["weak", "strong"]:
        sub = df_res[(df_res['exp'] == name) & (df_res['turb'] == turb)]
        if len(sub) > 0:
            print(f"\n{name} | {turb}: acc = {sub['acc'].mean()*100:.2f} ± {sub['acc'].std()*100:.2f}%")

summary_path = os.path.join(BASE_OUT, "exp23_aug_da_summary.csv")
df_res.to_csv(summary_path, index=False)
print(f"\nSaved to {summary_path}")
