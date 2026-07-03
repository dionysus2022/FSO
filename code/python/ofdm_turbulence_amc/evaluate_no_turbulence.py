"""Evaluate trained AMC checkpoints under AWGN-only (no turbulence) conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

try:
    from .run_experiment import build_model, evaluate, save_predictions, set_seed, wrap_representation
    from .simulation import MODULATIONS, OFDMTurbulenceDataset, SimulationConfig
except ImportError:
    from run_experiment import build_model, evaluate, save_predictions, set_seed, wrap_representation
    from simulation import MODULATIONS, OFDMTurbulenceDataset, SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-examples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confusion-snr", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["turbulence", "snr", "accuracy", "samples"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for snr in sorted({record["snr"] for record in records}):
        selected = [record for record in records if np.isclose(record["snr"], snr)]
        rows.append(
            {
                "turbulence": "none",
                "snr": snr,
                "accuracy": float(np.mean([r["label"] == r["prediction"] for r in selected])),
                "samples": len(selected),
            }
        )
    return rows


def plot_no_turbulence(output_dir: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [row["snr"] for row in rows],
        [100 * row["accuracy"] for row in rows],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Classification accuracy (%)",
        title="Six-class OFDM AMC without turbulence (AWGN only)",
        ylim=(0, 102),
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_snr_no_turbulence.png", dpi=220)
    plt.close(fig)


def plot_joint(output_dir: Path, no_turbulence_rows: list[dict], source_csv: Path) -> None:
    groups: dict[str, list[dict]] = {"none": no_turbulence_rows}
    with source_csv.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups.setdefault(row["turbulence"], []).append(
                {"snr": float(row["snr"]), "accuracy": float(row["accuracy"])}
            )
    colors = {
        "none": "#1f77b4",
        "weak": "#2ca02c",
        "moderate": "#ff7f0e",
        "strong": "#d62728",
    }
    labels = {"none": "No turbulence", "weak": "Weak", "moderate": "Moderate", "strong": "Strong"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for turbulence in ("none", "weak", "moderate", "strong"):
        rows = sorted(groups.get(turbulence, []), key=lambda row: row["snr"])
        if rows:
            ax.plot(
                [row["snr"] for row in rows],
                [100 * row["accuracy"] for row in rows],
                marker="o",
                linewidth=2,
                color=colors[turbulence],
                label=labels[turbulence],
            )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Classification accuracy (%)",
        title="Six-class OFDM AMC: AWGN and turbulence comparison",
        ylim=(0, 102),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_snr_four_channel_conditions.png", dpi=220)
    plt.close(fig)


def plot_confusion(output_dir: Path, records: list[dict], target_snr: float) -> None:
    available = sorted({record["snr"] for record in records})
    resolved = min(available, key=lambda value: abs(value - target_snr))
    selected = [record for record in records if np.isclose(record["snr"], resolved)]
    matrix = confusion_matrix(
        [record["label"] for record in selected],
        [record["prediction"] for record in selected],
        labels=range(len(MODULATIONS)),
        normalize="true",
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=MODULATIONS).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False
    )
    ax.set_title(f"No turbulence, SNR={resolved:g} dB")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_none_snr_{resolved:g}dB.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**checkpoint["args"])
    saved_args.batch_size = args.batch_size
    saved_args.num_workers = args.num_workers
    config_data = checkpoint["simulation_config"]
    config = config_data if isinstance(config_data, SimulationConfig) else SimulationConfig(**config_data)
    dataset = OFDMTurbulenceDataset(
        list(saved_args.snrs),
        ["none"],
        args.test_examples,
        args.seed + 4000,
        config,
        cache=True,
    )
    dataset = wrap_representation(dataset, saved_args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(saved_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    records = evaluate(model, loader, device)
    rows = summarize(records)
    save_predictions(args.output_dir / "test_predictions_no_turbulence.csv", records)
    write_rows(args.output_dir / "accuracy_vs_snr_no_turbulence.csv", rows)
    plot_no_turbulence(args.output_dir, rows)
    plot_joint(args.output_dir, rows, args.source_results_dir / "accuracy_vs_snr.csv")
    plot_confusion(args.output_dir, records, args.confusion_snr)
    overall = float(np.mean([record["label"] == record["prediction"] for record in records]))
    with (args.output_dir / "metrics_no_turbulence.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "representation": saved_args.representation,
                "sequence_length": config.sequence_length,
                "n_subcarriers": config.n_subcarriers,
                "test_examples_per_condition": args.test_examples,
                "overall_accuracy": overall,
                "accuracy_vs_snr": rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Completed {saved_args.representation}: no-turbulence accuracy={overall:.4f}")


if __name__ == "__main__":
    main()
