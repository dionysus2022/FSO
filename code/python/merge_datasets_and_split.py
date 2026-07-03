#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_datasets_and_split.py
合并 Dataset-A (2026.06.28) 和 Dataset-B (2026.06.29/eq) 的 manifest，
添加 dataset_source 列，并输出到合并目录。

合并后每类每湍流约 45 个 bin (A 25 + B 20)。
划分策略在 train_mrf_lowmem.py 的 group_split_by_rx_file 中按 dataset_source 分层执行：
  Train 70%: A 17 + B 14 = 31 bin
  Val 10%:   A 3  + B 2  = 5 bin
  Test 20%:  A 5  + B 4  = 9 bin
"""
import os
import shutil
from pathlib import Path
import pandas as pd

PRE_ROOT = Path(r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem")
MANI_A = PRE_ROOT / "2026.06.28" / "manifest_all.csv"
MANI_B = PRE_ROOT / "2026.06.29" / "eq" / "manifest_all.csv"
OUT_DIR = PRE_ROOT / "merged_AB"
OUT_MANI = OUT_DIR / "manifest_all.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(MANI_A)
    df_b = pd.read_csv(MANI_B)

    # 只保留成功样本
    df_a = df_a[df_a["status"].astype(str) == "ok"].copy()
    df_b = df_b[df_b["status"].astype(str) == "ok"].copy()

    # 添加 dataset_source 列
    df_a["dataset_source"] = "A"
    df_b["dataset_source"] = "B"

    # 对齐列：取两者交集，保证 concat 不产生 NaN
    common_cols = sorted(set(df_a.columns) & set(df_b.columns))
    df_a = df_a[common_cols]
    df_b = df_b[common_cols]

    df = pd.concat([df_a, df_b], ignore_index=True)

    # 重新分配 global_frame_id
    df["global_frame_id"] = range(1, len(df) + 1)

    # 保存
    df.to_csv(OUT_MANI, index=False, encoding="utf-8-sig")

    print(f"Dataset-A: {len(df_a)} frames")
    print(f"Dataset-B: {len(df_b)} frames")
    print(f"Merged:    {len(df)} frames")
    print(f"Saved to:  {OUT_MANI}")

    # 统计
    print("\n合并后每类每湍流的 bin 文件数：")
    g = df.groupby(["mod_name", "turb_name", "dataset_source"])["rx_file"].nunique().unstack(fill_value=0)
    g["total"] = g.sum(axis=1)
    print(g)

    print("\n合并后每类每湍流的帧数：")
    g2 = df.groupby(["mod_name", "turb_name", "dataset_source"]).size().unstack(fill_value=0)
    g2["total"] = g2.sum(axis=1)
    print(g2)


if __name__ == "__main__":
    main()
