"""Compare Oracle FFT, blind CP-FFT, and blind CP-FFT with CP merging."""

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

try:
    from .cp_fft_features import (
        SynchronizationResult,
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
        synchronization_errors,
    )
    from .simulation import (
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        SimulationConfig,
        generate_sequence_with_metadata,
    )
except ImportError:
    from cp_fft_features import (
        SynchronizationResult,
        estimate_cp_synchronization,
        extract_constellation_features,
        recover_subcarriers,
        synchronization_errors,
    )
    from simulation import (
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        SimulationConfig,
        generate_sequence_with_metadata,
    )


MODES = (
    "oracle_fft",
    "oracle_fft_merge",
    "blind_cp_fft",
    "blind_cp_fft_merge",
)


def parse_args() -> argparse.Namespace:
    root = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cp_fft_baselines_N16_CP16_20260625",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--cfo-max", type=float, default=0.0)
    parser.add_argument("--sto-max", type=int, default=0)
    parser.add_argument("--snrs", type=float, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--turbulence",
        nargs="+",
        choices=list(TRAINING_TURBULENCE_LEVELS),
        default=["none"],
    )
    parser.add_argument("--train-examples", type=int, default=300)
    parser.add_argument("--val-examples", type=int, default=100)
    parser.add_argument("--test-examples", type=int, default=200)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confusion-snr", type=float, default=10.0)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def records(args: argparse.Namespace, examples: int):
    return [
        (class_index, snr, turbulence, sample_index)
        for turbulence in args.turbulence
        for snr in args.snrs
        for class_index in range(len(MODULATIONS))
        for sample_index in range(examples)
    ]


def extract_split(
    split: str,
    split_records,
    split_seed: int,
    args: argparse.Namespace,
    config: SimulationConfig,
):
    cache = args.output_dir / f"features_{split}.npz"
    if cache.exists() and not args.allow_overwrite:
        data = np.load(cache, allow_pickle=False)
        return {key: data[key] for key in data.files}

    features = {mode: [] for mode in MODES}
    labels = []
    snrs = []
    turbulence_values = []
    boundary_errors = []
    cfo_errors = []
    sync_metrics = []
    feature_names = None
    started = time.time()

    for index, (label, snr, turbulence, sample_index) in enumerate(
        split_records, start=1
    ):
        sequence, metadata = generate_sequence_with_metadata(
            label, snr, turbulence, sample_index, split_seed, config
        )
        blind = estimate_cp_synchronization(sequence, config)
        oracle = SynchronizationResult(
            int(metadata["symbol_boundary_offset"]),
            float(metadata["cfo"]),
            1.0,
        )
        boundary_error, cfo_error = synchronization_errors(
            blind,
            int(metadata["symbol_boundary_offset"]),
            float(metadata["cfo"]),
            config.n_subcarriers + config.cp_length,
        )
        boundary_errors.append(boundary_error)
        cfo_errors.append(cfo_error)
        sync_metrics.append(blind.metric)

        mode_settings = {
            "oracle_fft": (oracle, False),
            "oracle_fft_merge": (oracle, True),
            "blind_cp_fft": (blind, False),
            "blind_cp_fft_merge": (blind, True),
        }
        for mode, (synchronization, merge_cp) in mode_settings.items():
            points, noise_variance, symbol_count = recover_subcarriers(
                sequence,
                config,
                synchronization.boundary_offset,
                synchronization.cfo,
                merge_cp,
            )
            row, names = extract_constellation_features(
                points,
                noise_variance,
                synchronization.metric,
                synchronization.cfo,
                symbol_count,
            )
            features[mode].append(row)
            feature_names = names

        labels.append(label)
        snrs.append(snr)
        turbulence_values.append(turbulence)
        if index % 500 == 0 or index == len(split_records):
            elapsed = time.time() - started
            print(
                f"{split}: {index}/{len(split_records)}, "
                f"{index / max(elapsed, 1e-6):.1f} samples/s",
                flush=True,
            )

    output = {
        **{f"X_{mode}": np.stack(rows) for mode, rows in features.items()},
        "y": np.asarray(labels, dtype=np.int64),
        "snr": np.asarray(snrs, dtype=np.float32),
        "turbulence": np.asarray(turbulence_values, dtype="U16"),
        "boundary_error": np.asarray(boundary_errors, dtype=np.int64),
        "cfo_error": np.asarray(cfo_errors, dtype=np.float32),
        "sync_metric": np.asarray(sync_metrics, dtype=np.float32),
        "feature_names": np.asarray(feature_names, dtype="U64"),
    }
    np.savez_compressed(cache, **output)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(mode, labels, predictions, snrs, turbulence):
    rows = []
    for level in sorted(np.unique(turbulence)):
        for snr in sorted(np.unique(snrs)):
            mask = (turbulence == level) & np.isclose(snrs, snr)
            rows.append(
                {
                    "mode": mode,
                    "turbulence": str(level),
                    "snr": float(snr),
                    "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                    "samples": int(np.sum(mask)),
                }
            )
    return rows


def plot_curves(path: Path, all_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    markers = {
        "oracle_fft": "o",
        "oracle_fft_merge": "D",
        "blind_cp_fft": "s",
        "blind_cp_fft_merge": "^",
    }
    for mode in MODES:
        for level in sorted({row["turbulence"] for row in all_rows}):
            selected = [
                row
                for row in all_rows
                if row["mode"] == mode and row["turbulence"] == level
            ]
            ax.plot(
                [row["snr"] for row in selected],
                [100 * row["accuracy"] for row in selected],
                marker=markers[mode],
                linewidth=2,
                label=f"{mode} / {level}",
            )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Classification accuracy (%)",
        title="Oracle and blind CP-FFT constellation-feature baselines",
        ylim=(0, 102),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_confusions(
    output_dir: Path,
    mode: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    snrs: np.ndarray,
    turbulence: np.ndarray,
    target_snr: float,
) -> None:
    resolved_snr = min(np.unique(snrs), key=lambda value: abs(value - target_snr))
    for level in sorted(np.unique(turbulence)):
        mask = (turbulence == level) & np.isclose(snrs, resolved_snr)
        matrix = confusion_matrix(
            labels[mask],
            predictions[mask],
            labels=range(len(MODULATIONS)),
            normalize="true",
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(matrix, display_labels=MODULATIONS).plot(
            ax=ax, cmap="Blues", values_format=".2f", colorbar=False
        )
        ax.set_title(f"{mode}, {level}, SNR={resolved_snr:g} dB")
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"confusion_matrix_{mode}_{level}_snr_{resolved_snr:g}dB.png",
            dpi=220,
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{metrics_path} exists; use a new output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        sequence_length=args.sequence_length,
        n_subcarriers=args.n_subcarriers,
        cp_length=args.cp_length,
        cfo_max=args.cfo_max,
        sto_max=args.sto_max,
        phase_max=np.pi,
    )
    split_specs = {
        "train": (records(args, args.train_examples), args.seed + 1000),
        "val": (records(args, args.val_examples), args.seed + 2000),
        "test": (records(args, args.test_examples), args.seed + 3000),
    }
    datasets = {
        split: extract_split(split, rows, seed, args, config)
        for split, (rows, seed) in split_specs.items()
    }

    model_rows = []
    condition_output = []
    for mode in MODES:
        model = ExtraTreesClassifier(
            n_estimators=args.trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(datasets["train"][f"X_{mode}"], datasets["train"]["y"])
        val_prediction = model.predict(datasets["val"][f"X_{mode}"])
        test_prediction = model.predict(datasets["test"][f"X_{mode}"])
        val_accuracy = accuracy_score(datasets["val"]["y"], val_prediction)
        test_accuracy = accuracy_score(datasets["test"]["y"], test_prediction)
        model_rows.append(
            {
                "mode": mode,
                "validation_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
            }
        )
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        joblib.dump(model, mode_dir / "model.joblib")
        np.savez_compressed(
            mode_dir / "test_predictions.npz",
            true_class=datasets["test"]["y"],
            predicted_class=test_prediction,
            snr=datasets["test"]["snr"],
            turbulence=datasets["test"]["turbulence"],
        )
        rows = condition_rows(
            mode,
            datasets["test"]["y"],
            test_prediction,
            datasets["test"]["snr"],
            datasets["test"]["turbulence"],
        )
        condition_output.extend(rows)
        write_csv(mode_dir / "accuracy_vs_snr.csv", rows)
        plot_confusions(
            mode_dir,
            mode,
            datasets["test"]["y"],
            test_prediction,
            datasets["test"]["snr"],
            datasets["test"]["turbulence"],
            args.confusion_snr,
        )
        print(
            f"{mode}: val={val_accuracy:.4f}, test={test_accuracy:.4f}",
            flush=True,
        )

    write_csv(args.output_dir / "model_comparison.csv", model_rows)
    write_csv(args.output_dir / "accuracy_vs_snr_all_modes.csv", condition_output)
    plot_curves(args.output_dir / "accuracy_vs_snr_all_modes.png", condition_output)

    test = datasets["test"]
    symbol_length = config.n_subcarriers + config.cp_length
    sync_summary = {
        "exact_boundary_rate": float(np.mean(test["boundary_error"] == 0)),
        "within_one_sample_rate": float(np.mean(test["boundary_error"] <= 1)),
        "mean_boundary_error_samples": float(np.mean(test["boundary_error"])),
        "mean_absolute_cfo_error": float(np.mean(test["cfo_error"])),
        "p95_absolute_cfo_error": float(np.quantile(test["cfo_error"], 0.95)),
        "mean_cp_metric": float(np.mean(test["sync_metric"])),
        "symbol_length": symbol_length,
    }
    metrics = {
        "task": "weak-prior OFDM AMC using blind CP synchronization and FFT constellation features",
        "modulations": MODULATIONS,
        "modes": MODES,
        "config": config.__dict__,
        "snrs_db": args.snrs,
        "turbulence_levels": args.turbulence,
        "train_examples_per_condition": args.train_examples,
        "val_examples_per_condition": args.val_examples,
        "test_examples_per_condition": args.test_examples,
        "feature_count": int(datasets["train"]["feature_names"].size),
        "feature_names": datasets["train"]["feature_names"].tolist(),
        "synchronization": sync_summary,
        "model_results": model_rows,
        "evaluation_unit": "independent randomly generated received IQ sequence",
        "oracle_metadata_used_only_by": "oracle_fft",
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
