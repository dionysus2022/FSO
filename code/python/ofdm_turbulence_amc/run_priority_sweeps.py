"""Batch launcher for the prioritized P0-P4 AMC experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=base / "simulation_priority_sweeps")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--subcarriers", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["raw_iq", "multiscale_iq", "iq_amp_phase", "gaf_gadf"],
    )
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--curriculum-clean-epochs", type=int, default=5)
    parser.add_argument("--curriculum-weak-epochs", type=int, default=5)
    parser.add_argument("--curriculum-full-turbulence-epochs", type=int, default=5)
    parser.add_argument("--snrs", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument("--gaf-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, length: int, subcarriers: int, representation: str) -> list[str]:
    output_dir = args.output_root / f"M{length}_N{subcarriers}_{representation}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_experiment.py")),
        "--output-dir",
        str(output_dir),
        "--sequence-length",
        str(length),
        "--n-subcarriers",
        str(subcarriers),
        "--representation",
        representation,
        "--train-examples",
        str(args.train_examples),
        "--val-examples",
        str(args.val_examples),
        "--test-examples",
        str(args.test_examples),
        "--epochs",
        str(args.epochs),
        "--curriculum-clean-epochs",
        str(args.curriculum_clean_epochs),
        "--curriculum-weak-epochs",
        str(args.curriculum_weak_epochs),
        "--curriculum-full-turbulence-epochs",
        str(args.curriculum_full_turbulence_epochs),
        "--batch-size",
        str(args.batch_size),
        "--gaf-size",
        str(args.gaf_size),
        "--allow-overwrite",
    ]
    command.extend(["--snrs", *[str(value) for value in args.snrs]])
    return command


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        for subcarriers in args.subcarriers:
            for representation in args.representations:
                command = build_command(args, length, subcarriers, representation)
                print(" ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
