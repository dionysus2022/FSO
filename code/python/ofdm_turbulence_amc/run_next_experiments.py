"""Run the approved next-stage experiments sequentially without overwriting old results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RESULTS_ROOT = Path(
    r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
    r"\OptDSP_lite-master\2_Data_Results"
)
OLD_ROOT = RESULTS_ROOT / "formal_experiments_M16384_N16_20260624"
NEW_ROOT = RESULTS_ROOT / "next_experiments_20260625"
HERE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def no_turbulence() -> None:
    models = (
        ("M8192_N16_raw_iq", "raw_iq"),
        ("M8192_N16_multiscale_iq", "multiscale_iq"),
        ("M8192_N16_iq_amp_phase", "iq_amp_phase"),
    )
    for folder, representation in models:
        source = OLD_ROOT / folder
        output = NEW_ROOT / "no_turbulence" / representation
        if (output / "metrics_no_turbulence.json").exists():
            print(f"SKIP completed: {output}", flush=True)
            continue
        run(
            [
                sys.executable,
                str(HERE / "evaluate_no_turbulence.py"),
                "--checkpoint",
                str(source / "best_model.pt"),
                "--source-results-dir",
                str(source),
                "--output-dir",
                str(output),
                "--test-examples",
                "300",
                "--batch-size",
                "16",
                "--seed",
                "42",
            ]
        )


def train_multiscale(length: int, batch_size: int) -> None:
    output = NEW_ROOT / f"M{length}_N16_multiscale_iq"
    if (output / "metrics.json").exists():
        print(f"SKIP completed: {output}", flush=True)
        return
    run(
        [
            sys.executable,
            str(HERE / "run_experiment.py"),
            "--output-dir",
            str(output),
            "--sequence-length",
            str(length),
            "--n-subcarriers",
            "16",
            "--representation",
            "multiscale_iq",
            "--train-examples",
            "1000",
            "--val-examples",
            "200",
            "--test-examples",
            "300",
            "--curriculum-clean-epochs",
            "5",
            "--curriculum-weak-epochs",
            "5",
            "--curriculum-full-turbulence-epochs",
            "5",
            "--epochs",
            "20",
            "--batch-size",
            str(batch_size),
            "--snrs",
            "-10",
            "-5",
            "0",
            "5",
            "10",
            "15",
            "20",
            "--seed",
            "42",
        ]
    )


def main() -> None:
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    no_turbulence()
    train_multiscale(16384, 16)
    train_multiscale(32768, 8)
    print("Base sequence completed. Multi-window voting is the next implementation step.", flush=True)


if __name__ == "__main__":
    main()
