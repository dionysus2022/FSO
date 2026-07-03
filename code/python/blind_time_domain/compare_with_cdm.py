"""Create a factual CDM-CNN versus raw time-domain comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-sweep",
        type=Path,
        default=base / "training_time_domain_blind" / "sweep_summary.csv",
    )
    parser.add_argument(
        "--cdm-sc5",
        type=Path,
        default=base / "training_cdm_sc5_single_channel_cuda" / "metrics.json",
    )
    parser.add_argument(
        "--cdm-sc1",
        type=Path,
        default=base / "training_cdm_sc1_single_channel_cuda" / "metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "training_time_domain_blind" / "comparison_with_cdm.md",
    )
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def main() -> None:
    args = parse_args()
    with args.time_sweep.open(encoding="utf-8", newline="") as file:
        time_rows = list(csv.DictReader(file))
    time_rows.sort(key=lambda row: float(row["validation_frame_accuracy"]), reverse=True)
    sc5 = json.loads(args.cdm_sc5.read_text(encoding="utf-8"))
    sc1 = json.loads(args.cdm_sc1.read_text(encoding="utf-8"))

    lines = [
        "# CDM-CNN 与盲时域模型对比",
        "",
        "## 处理依赖",
        "",
        "| 方法 | FFT | 已知训练序列同步 | 信道估计/均衡 | 发送参考符号 |",
        "|---|---|---|---|---|",
        "| CDM-CNN | 是 | 是 | 是 | 分类输入不使用，但前端 DSP 使用前导 |",
        "| 时域 1D 模型 | 否 | 否 | 否 | 否 |",
        "",
        "时域方法的主要论点是处理链盲性更强，不预设其准确率一定高于 CDM。",
        "",
        "## CDM 现有结果",
        "",
        "| 方法 | 独立样本 validation | 帧级 validation | 帧数 |",
        "|---|---:|---:|---:|",
        (
            f"| CDM sc5 | {pct(sc5['final_best_checkpoint_subband_accuracy'])} "
            f"| {pct(sc5['frame_fused_validation_accuracy'])} "
            f"| {sc5['validation_frames']} |"
        ),
        (
            f"| CDM sc1 | {pct(sc1['final_best_checkpoint_subcarrier_accuracy'])} "
            f"| {pct(sc1['frame_fused_validation_accuracy'])} "
            f"| {sc1['validation_frames']} |"
        ),
        "",
        "## 时域实验结果",
        "",
        "| M | 模型 | 划分 | crop validation | 帧级 validation | 帧数 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in time_rows:
        lines.append(
            f"| {row['sequence_length']} | {row['model']} | "
            f"{row['train_frames']} → {row['validation_frames']} | "
            f"{pct(float(row['validation_crop_accuracy']))} | "
            f"{pct(float(row['validation_frame_accuracy']))} | "
            f"{row['validation_frame_count']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 两类方法必须使用相同原始帧和相同 frame-level 划分才可进行准确率对比。",
        "- crop、子载波和子带不是相互独立的统计样本，主要结论应基于帧级指标。",
        "- validation 用于模型选择时不能称为最终 test；正式结论需要独立采集 session。",
        "- 原始时域模型可能学习帧结构或设备时序特征，应通过跨日期和随机采集顺序验证。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

