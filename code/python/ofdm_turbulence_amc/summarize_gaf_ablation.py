"""Summarize the fair 20 dB raw-IQ/GAF representation ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\Project_code\FSO_research\results")
EXPERIMENTS = {
    "Raw IQ\nResNet-SE": ROOT / "ablation_raw_iq_N16_M16384_snr20_fair",
    "GAF 64\n2D CNN": ROOT / "ablation_gaf64_N16_M16384_snr20",
    "GAF 128\n2D CNN": ROOT / "ablation_gaf128_N16_M16384_snr20",
    "GAF 256\n2D CNN": ROOT / "ablation_gaf256_N16_M16384_snr20",
}
TURBULENCE = ("weak", "moderate", "strong")


def main() -> None:
    rows = []
    for method, directory in EXPERIMENTS.items():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        by_turbulence = {
            row["turbulence"]: row["accuracy"] for row in metrics["accuracy_vs_snr"]
        }
        for turbulence in TURBULENCE:
            rows.append(
                {
                    "method": method.replace("\n", " "),
                    "turbulence": turbulence,
                    "snr_db": 20,
                    "accuracy": by_turbulence[turbulence],
                    "test_samples": next(
                        row["samples"]
                        for row in metrics["accuracy_vs_snr"]
                        if row["turbulence"] == turbulence
                    ),
                }
            )

    output_dir = ROOT / "ablation_gaf_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gaf_ablation_snr20.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(EXPERIMENTS))
    width = 0.24
    colors = {"weak": "#2ca02c", "moderate": "#ff7f0e", "strong": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, turbulence in enumerate(TURBULENCE):
        values = [
            100
            * next(
                row["accuracy"]
                for row in rows
                if row["method"] == method.replace("\n", " ")
                and row["turbulence"] == turbulence
            )
            for method in EXPERIMENTS
        ]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=turbulence.capitalize(),
            color=colors[turbulence],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(
        xticks=x,
        xticklabels=list(EXPERIMENTS),
        ylabel="Classification accuracy (%)",
        title="Raw-IQ versus GAF representation ablation at SNR=20 dB",
        ylim=(0, 62),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gaf_ablation_snr20.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

