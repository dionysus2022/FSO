"""Run sequence-length, model, and frame-split experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=base / "dataset_time_domain_blind"
    )
    parser.add_argument(
        "--output-root", type=Path, default=base / "training_time_domain_blind"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["cnn1d", "resnet1d", "resnet_se"],
        default=["cnn1d", "resnet1d", "resnet_se"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["1-60:61-75", "1-50:51-75", "16-75:1-15"],
        help="Each split is TRAIN_FRAMES:VAL_FRAMES.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace(",", "_").replace("-", "to")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train.py")
    rows = []
    for length in args.lengths:
        dataset_dir = args.dataset_root / f"M_{length}"
        if not (dataset_dir / "manifest.csv").is_file():
            raise FileNotFoundError(dataset_dir / "manifest.csv")
        for model in args.models:
            for split_index, split in enumerate(args.splits, start=1):
                train_frames, val_frames = split.split(":", 1)
                run_name = (
                    f"M_{length}_{model}_split{split_index}_"
                    f"train_{safe_name(train_frames)}_val_{safe_name(val_frames)}"
                )
                output_dir = args.output_root / run_name
                command = [
                    sys.executable,
                    str(train_script),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(output_dir),
                    "--model",
                    model,
                    "--train-frames",
                    train_frames,
                    "--val-frames",
                    val_frames,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                ]
                print("RUN:", " ".join(command), flush=True)
                subprocess.run(command, check=True)
                metrics = json.loads(
                    (output_dir / "metrics.json").read_text(encoding="utf-8")
                )
                rows.append(
                    {
                        "run_name": run_name,
                        "sequence_length": length,
                        "model": model,
                        "split_index": split_index,
                        "train_frames": train_frames,
                        "validation_frames": val_frames,
                        "validation_crop_accuracy": metrics["validation"][
                            "crop_accuracy"
                        ],
                        "validation_frame_accuracy": metrics["validation"][
                            "frame_accuracy"
                        ],
                        "validation_crop_count": metrics["validation"]["crop_count"],
                        "validation_frame_count": metrics["validation"]["frame_count"],
                        "best_epoch": metrics["best_epoch"],
                    }
                )
                with (args.output_root / "sweep_summary.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == "__main__":
    main()

