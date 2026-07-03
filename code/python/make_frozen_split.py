# -*- coding: utf-8 -*-
"""
make_frozen_split.py — Step 2: 生成冻结真实测试集划分

用 group_split_by_rx_file(seed=2026) 对 merged_AB manifest 做文件级划分。
后续所有 Stage II 实验都用同一份 test，保证可比性。

输出：
    results/stage2/frozen_real_split.csv
    results/stage2/frozen_real_split_summary.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能 import stage2_common 和 train_mrf_lowmem
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage2_common import (
    REAL_DATA_ROOT,
    STAGE2_ROOT,
    FROZEN_SPLIT_CSV,
    FROZEN_SPLIT_SUMMARY,
    ensure_dir,
    now_str,
)
from train_mrf_lowmem import load_manifest, group_split_by_rx_file


def main():
    print(f"=== make_frozen_split.py  [{now_str()}] ===")
    ensure_dir(STAGE2_ROOT)

    # 1. 加载真实数据 manifest
    print(f"[1/5] 加载真实数据 manifest: {REAL_DATA_ROOT}")
    df = load_manifest(REAL_DATA_ROOT)
    print(f"  ok 行数: {len(df)}")
    print(f"  mod 分布: {df['mod_name'].value_counts().to_dict()}")
    print(f"  turb 分布: {df['turb_name'].value_counts().to_dict()}")
    print(f"  source 分布: {df['dataset_source'].value_counts().to_dict()}")

    # 2. 文件级划分
    print(f"\n[2/5] 调用 group_split_by_rx_file(seed=2026)")
    split_df = group_split_by_rx_file(df, seed=2026)

    # 3. 添加 uid
    print(f"\n[3/5] 添加 uid 列")
    split_df["uid"] = (
        split_df["dataset_source"].astype(str) + "_"
        + split_df["turb_name"].astype(str) + "_"
        + split_df["mod_name"].astype(str) + "_"
        + split_df["rx_file"].astype(str) + "_"
        + split_df["rx_frame_idx"].astype(str)
    )

    # 4. 验证无泄漏
    print(f"\n[4/5] 验证无 rx_file 泄漏")
    train_files = set(split_df[split_df["split"] == "train"]["rx_file"].unique())
    val_files = set(split_df[split_df["split"] == "val"]["rx_file"].unique())
    test_files = set(split_df[split_df["split"] == "test"]["rx_file"].unique())

    leak_train_test = train_files & test_files
    leak_val_test = val_files & test_files
    leak_train_val = train_files & val_files

    assert len(leak_train_test) == 0, f"train/test 泄漏: {len(leak_train_test)} files"
    assert len(leak_val_test) == 0, f"val/test 泄漏: {len(leak_val_test)} files"
    assert len(leak_train_val) == 0, f"train/val 泄漏: {len(leak_train_val)} files"
    print(f"  [OK] 无泄漏")
    print(f"  train files: {len(train_files)}, val files: {len(val_files)}, test files: {len(test_files)}")

    # 统计每 split 行数
    split_counts = split_df["split"].value_counts().to_dict()
    print(f"  行数: {split_counts}")
    assert split_counts.get("test", 0) > 0, "test 集为空"
    assert split_counts.get("train", 0) > 0, "train 集为空"

    # 验证 test 集覆盖
    test_df = split_df[split_df["split"] == "test"]
    test_mods = sorted(test_df["mod_label"].unique())
    test_turbs = sorted(test_df["turb_name"].unique())
    test_sources = sorted(test_df["dataset_source"].astype(str).unique())
    assert len(test_mods) == 6, f"test 缺调制: {test_mods}"
    assert len(test_turbs) == 2, f"test 缺湍流: {test_turbs}"
    assert len(test_sources) == 2, f"test 缺 source: {test_sources}"
    print(f"  test 覆盖: mods={test_mods}, turbs={test_turbs}, sources={test_sources}")

    # 5. 保存
    print(f"\n[5/5] 保存冻结划分")
    split_df.to_csv(FROZEN_SPLIT_CSV, index=False)
    print(f"  CSV: {FROZEN_SPLIT_CSV}  ({len(split_df)} rows)")

    # 保存 summary
    summary = {}
    for split_name in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split_name]
        grp = sub.groupby(["turb_name", "dataset_source", "mod_name"]).size()
        summary[split_name] = {}
        for (turb, src, mod), cnt in grp.items():
            key = f"{turb}/{src}/{mod}"
            summary[split_name][key] = int(cnt)
        summary[split_name]["_total"] = int(len(sub))
        summary[split_name]["_n_rx_files"] = int(sub["rx_file"].nunique())

    with open(FROZEN_SPLIT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {FROZEN_SPLIT_SUMMARY}")

    # 打印 test 集 mod×turb 概况
    print(f"\n=== Test 集 mod × turb ===")
    test_pivot = test_df.groupby(["turb_name", "mod_name"]).size().unstack(fill_value=0)
    print(test_pivot.to_string())

    print(f"\n[完成] 冻结划分已生成。test={len(test_df)} 行，后续所有 Stage II 实验都用此 test。")


if __name__ == "__main__":
    main()
