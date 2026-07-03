"""Targeted 15 dB experiment with enhanced constellation features and hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

try:
    from .cp_fft_features import (
        SynchronizationResult,
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
        synchronization_errors,
    )
    from .simulation import MODULATIONS, SimulationConfig, generate_sequence_with_metadata
except ImportError:
    from cp_fft_features import (
        SynchronizationResult,
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
        synchronization_errors,
    )
    from simulation import MODULATIONS, SimulationConfig, generate_sequence_with_metadata


def parse_args():
    root = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cp_fft_15db_target_N16_CP16_20260625",
    )
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def extract_split(name, count, seed, args, config):
    path = args.output_dir / f"features_{name}.npz"
    if path.exists() and not args.allow_overwrite:
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    rows = []
    labels = []
    boundary_errors = []
    cfo_errors = []
    started = time.time()
    feature_names = None
    for label in range(len(MODULATIONS)):
        for sample_index in range(count):
            sequence, metadata = generate_sequence_with_metadata(
                label, 15.0, "none", sample_index, seed, config
            )
            sync = estimate_cp_synchronization(sequence, config)
            points, noise_variance, symbol_count = recover_subcarriers(
                sequence, config, sync.boundary_offset, sync.cfo, True
            )
            row, feature_names = extract_constellation_features(
                points, noise_variance, sync.metric, sync.cfo, symbol_count
            )
            rows.append(row)
            labels.append(label)
            boundary_error, cfo_error = synchronization_errors(
                sync,
                int(metadata["symbol_boundary_offset"]),
                float(metadata["cfo"]),
                config.n_subcarriers + config.cp_length,
            )
            boundary_errors.append(boundary_error)
            cfo_errors.append(cfo_error)
            completed = len(rows)
            total = count * len(MODULATIONS)
            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - started
                print(
                    f"{name}: {completed}/{total}, "
                    f"{completed / max(elapsed, 1e-6):.1f} samples/s",
                    flush=True,
                )
    output = {
        "X": np.stack(rows),
        "y": np.asarray(labels, dtype=np.int64),
        "boundary_error": np.asarray(boundary_errors, dtype=np.int64),
        "cfo_error": np.asarray(cfo_errors, dtype=np.float32),
        "feature_names": np.asarray(feature_names, dtype="U80"),
    }
    np.savez_compressed(path, **output)
    return output


def make_extra_trees(args, seed_offset=0):
    return ExtraTreesClassifier(
        n_estimators=args.trees,
        min_samples_leaf=1,
        max_features=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed + seed_offset,
    )


class HierarchicalClassifier:
    """Coarse groups, then dedicated 32/128 and 64/256 pair classifiers."""

    GROUPS = ((0,), (1,), (2, 4), (3, 5))

    def __init__(self, args):
        self.coarse = make_extra_trees(args, 10)
        self.pair_32_128 = make_extra_trees(args, 20)
        self.pair_64_256 = make_extra_trees(args, 30)

    @staticmethod
    def group_labels(y):
        mapping = np.asarray([0, 1, 2, 3, 2, 3])
        return mapping[y]

    def fit(self, X, y):
        self.coarse.fit(X, self.group_labels(y))
        mask_a = np.isin(y, [2, 4])
        mask_b = np.isin(y, [3, 5])
        self.pair_32_128.fit(X[mask_a], y[mask_a])
        self.pair_64_256.fit(X[mask_b], y[mask_b])
        return self

    def predict(self, X):
        groups = self.coarse.predict(X)
        output = np.empty(groups.size, dtype=np.int64)
        output[groups == 0] = 0
        output[groups == 1] = 1
        mask_a = groups == 2
        mask_b = groups == 3
        output[mask_a] = self.pair_32_128.predict(X[mask_a])
        output[mask_b] = self.pair_64_256.predict(X[mask_b])
        return output


def save_confusion(path, true, predicted, title):
    matrix = confusion_matrix(
        true, predicted, labels=range(len(MODULATIONS)), normalize="true"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=MODULATIONS).plot(
        ax=ax, cmap="Blues", values_format=".3f", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{metrics_path} exists.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        sequence_length=8192,
        n_subcarriers=16,
        cp_length=16,
        cfo_max=0.0,
        sto_max=0,
        phase_max=np.pi,
    )
    train = extract_split("train", args.train_examples, args.seed + 1000, args, config)
    val = extract_split("val", args.val_examples, args.seed + 2000, args, config)
    test = extract_split("test", args.test_examples, args.seed + 3000, args, config)

    models = {
        "extra_trees_direct": make_extra_trees(args),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=args.seed,
        ),
        "extra_trees_hierarchical": HierarchicalClassifier(args),
    }
    results = []
    for name, model in models.items():
        model.fit(train["X"], train["y"])
        val_prediction = model.predict(val["X"])
        test_prediction = model.predict(test["X"])
        val_accuracy = accuracy_score(val["y"], val_prediction)
        test_accuracy = accuracy_score(test["y"], test_prediction)
        results.append(
            {
                "model": name,
                "validation_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "test_errors": int(np.sum(test["y"] != test_prediction)),
                "test_samples": int(test["y"].size),
            }
        )
        model_dir = args.output_dir / name
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        np.savez_compressed(
            model_dir / "test_predictions.npz",
            true_class=test["y"],
            predicted_class=test_prediction,
        )
        save_confusion(
            model_dir / "confusion_matrix_15dB.png",
            test["y"],
            test_prediction,
            f"{name}, blind CP-FFT+Merge, 15 dB",
        )
        print(
            f"{name}: val={val_accuracy:.5f}, test={test_accuracy:.5f}, "
            f"errors={np.sum(test['y'] != test_prediction)}",
            flush=True,
        )

    with (args.output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metrics = {
        "task": "targeted 15 dB six-class AMC",
        "condition": "N=16, CP=16, CFO=0, STO=0, no turbulence",
        "modulations": MODULATIONS,
        "train_examples_per_class": args.train_examples,
        "val_examples_per_class": args.val_examples,
        "test_examples_per_class": args.test_examples,
        "test_samples": int(test["y"].size),
        "feature_count": int(train["X"].shape[1]),
        "feature_names": train["feature_names"].tolist(),
        "exact_boundary_rate": float(np.mean(test["boundary_error"] == 0)),
        "mean_absolute_cfo_error": float(np.mean(test["cfo_error"])),
        "results": results,
        "claim_scope": (
            "100% is only valid if zero errors occur on this independent ideal-"
            "condition test set; it does not imply robustness to short CP, "
            "nonzero CFO/STO, turbulence, or lower SNR."
        ),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
