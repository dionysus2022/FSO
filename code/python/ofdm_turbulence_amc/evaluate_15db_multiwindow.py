"""Sequence-level multi-window evaluation for the 15 dB target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

try:
    from .cp_fft_features import (
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
    )
    from .simulation import MODULATIONS, SimulationConfig, generate_sequence_with_metadata
except ImportError:
    from cp_fft_features import (
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
    )
    from simulation import MODULATIONS, SimulationConfig, generate_sequence_with_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3042)
    return parser.parse_args()


def window_features(window: np.ndarray, config: SimulationConfig) -> np.ndarray:
    sync = estimate_cp_synchronization(window, config)
    points, noise_variance, symbol_count = recover_subcarriers(
        window, config, sync.boundary_offset, sync.cfo, True
    )
    features, _ = extract_constellation_features(
        points, noise_variance, sync.metric, sync.cfo, symbol_count
    )
    return features


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(
        args.source_dir / "extra_trees_direct" / "model.joblib"
    )
    base_config = SimulationConfig(
        sequence_length=8192,
        n_subcarriers=16,
        cp_length=16,
        cfo_max=0.0,
        sto_max=0,
        phase_max=np.pi,
    )
    maximum_windows = max(args.windows)
    long_config = SimulationConfig(
        sequence_length=8192 * maximum_windows,
        n_subcarriers=16,
        cp_length=16,
        cfo_max=0.0,
        sto_max=0,
        phase_max=np.pi,
    )
    labels = []
    all_window_features = []
    for label in range(len(MODULATIONS)):
        for sample_index in range(args.test_examples):
            sequence, _ = generate_sequence_with_metadata(
                label, 15.0, "none", sample_index, args.seed, long_config
            )
            sequence_features = []
            for window_index in range(maximum_windows):
                start = window_index * 8192
                features = window_features(
                    sequence[:, start : start + 8192], base_config
                )
                sequence_features.append(features)
            all_window_features.append(np.stack(sequence_features))
            labels.append(label)
            completed = len(labels)
            if completed % 250 == 0:
                print(
                    f"{completed}/{args.test_examples * len(MODULATIONS)} sequences",
                    flush=True,
                )

    labels_array = np.asarray(labels, dtype=np.int64)
    feature_array = np.stack(all_window_features)
    flat_probabilities = model.predict_proba(
        feature_array.reshape(-1, feature_array.shape[-1])
    )
    probabilities = flat_probabilities.reshape(
        feature_array.shape[0], feature_array.shape[1], -1
    )
    results = []
    for count in args.windows:
        prediction = np.argmax(np.mean(probabilities[:, :count, :], axis=1), axis=1)
        accuracy = accuracy_score(labels_array, prediction)
        errors = int(np.sum(labels_array != prediction))
        results.append(
            {
                "windows": count,
                "accuracy": float(accuracy),
                "errors": errors,
                "sequences": int(labels_array.size),
            }
        )
        np.savez_compressed(
            args.output_dir / f"predictions_K{count}.npz",
            true_class=labels_array,
            predicted_class=prediction,
        )
        matrix = confusion_matrix(
            labels_array,
            prediction,
            labels=range(len(MODULATIONS)),
            normalize="true",
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(matrix, display_labels=MODULATIONS).plot(
            ax=ax, cmap="Blues", values_format=".3f", colorbar=False
        )
        ax.set_title(f"15 dB sequence-level probability averaging, K={count}")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"confusion_matrix_K{count}.png", dpi=220)
        plt.close(fig)
        print(f"K={count}: accuracy={accuracy:.6f}, errors={errors}", flush=True)

    metrics = {
        "task": "15 dB sequence-level multi-window probability averaging",
        "source_model": str(
            args.source_dir / "extra_trees_direct" / "model.joblib"
        ),
        "condition": "N=16, CP=16, CFO=0, STO=0, no turbulence",
        "window_length": 8192,
        "test_examples_per_class": args.test_examples,
        "results": results,
        "windows_are_not_counted_as_independent_samples": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
