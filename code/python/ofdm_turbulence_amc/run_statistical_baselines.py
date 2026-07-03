"""Explicit statistical-feature baselines for six-class time-domain OFDM AMC.

The script uses the same deterministic simulator and split seeds as
``run_experiment.py``. It does not overwrite deep-learning results.
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from .simulation import (
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        TURBULENCE_LEVELS,
        SimulationConfig,
        generate_sequence,
    )
except ImportError:
    from simulation import (
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        TURBULENCE_LEVELS,
        SimulationConfig,
        generate_sequence,
    )


ALL_EVAL_TURBULENCE = ("none",) + TURBULENCE_LEVELS
FEATURE_GROUPS = (
    "amplitude_moments",
    "cumulants",
    "papr_histogram",
    "phase_difference",
    "spectral",
    "cp",
)


def parse_args() -> argparse.Namespace:
    results_root = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=results_root
        / "statistical_feature_baselines_M8192_N16_20260625",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--snrs", type=float, nargs="+", default=list(range(-10, 21, 5)))
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confusion-snr", type=float, default=10.0)
    parser.add_argument("--histogram-bins", type=int, default=24)
    parser.add_argument("--phase-bins", type=int, default=16)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["svm", "random_forest", "xgboost"],
        default=["svm", "random_forest", "xgboost"],
    )
    parser.add_argument(
        "--skip-group-ablation",
        action="store_true",
        help="Skip one-feature-group-at-a-time Random Forest diagnostics.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _safe_log(value: float, eps: float = 1e-12) -> float:
    return float(np.log(max(float(value), eps)))


def extract_features(
    sequence: np.ndarray,
    config: SimulationConfig,
    histogram_bins: int,
    phase_bins: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract phase-offset-invariant and mostly scale-robust signal statistics."""
    eps = 1e-12
    x = sequence[0].astype(np.float64) + 1j * sequence[1].astype(np.float64)
    x = x - np.mean(x)
    amplitude = np.abs(x)
    power = amplitude * amplitude
    m2 = np.mean(power) + eps
    m4 = np.mean(power**2) + eps
    m6 = np.mean(power**3) + eps

    values: list[float] = []
    names: list[str] = []
    groups: list[str] = []

    def add(name: str, value: float, group: str) -> None:
        names.append(name)
        values.append(float(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)))
        groups.append(group)

    # Preserve the requested raw moments, then add scale-independent ratios.
    add("log_amplitude_moment_2", _safe_log(m2), "amplitude_moments")
    add("log_amplitude_moment_4", _safe_log(m4), "amplitude_moments")
    add("log_amplitude_moment_6", _safe_log(m6), "amplitude_moments")
    add("normalized_amplitude_moment_4", m4 / (m2**2), "amplitude_moments")
    add("normalized_amplitude_moment_6", m6 / (m2**3), "amplitude_moments")

    # Complex cumulants. Magnitudes remove the unknown constant phase offset.
    ex2 = np.mean(x**2)
    ex4 = np.mean(x**4)
    ex6 = np.mean(x**6)
    c40 = ex4 - 3.0 * ex2**2
    c42 = m4 - np.abs(ex2) ** 2 - 2.0 * m2**2
    c60 = ex6 - 15.0 * ex4 * ex2 + 30.0 * ex2**3
    add("normalized_abs_c40", np.abs(c40) / (m2**2), "cumulants")
    add("normalized_abs_c42", np.abs(c42) / (m2**2), "cumulants")
    add("normalized_abs_c60", np.abs(c60) / (m2**3), "cumulants")

    add("papr", np.max(power) / m2, "papr_histogram")
    normalized_amplitude = amplitude / np.sqrt(m2)
    amp_hist, _ = np.histogram(
        normalized_amplitude,
        bins=histogram_bins,
        range=(0.0, 4.0),
        density=False,
    )
    amp_hist = amp_hist.astype(np.float64) / max(amplitude.size, 1)
    overflow = np.mean(normalized_amplitude >= 4.0)
    for index, value in enumerate(amp_hist):
        add(f"amplitude_histogram_{index:02d}", value, "papr_histogram")
    add("amplitude_histogram_overflow", overflow, "papr_histogram")

    phase_difference = np.angle(x[1:] * np.conj(x[:-1]))
    for harmonic in range(1, 7):
        circular = np.mean(np.exp(1j * harmonic * phase_difference))
        add(
            f"phase_difference_circular_abs_{harmonic}",
            np.abs(circular),
            "phase_difference",
        )
    phase_hist, _ = np.histogram(
        phase_difference,
        bins=phase_bins,
        range=(-np.pi, np.pi),
        density=False,
    )
    phase_hist = phase_hist.astype(np.float64) / max(phase_difference.size, 1)
    for index, value in enumerate(phase_hist):
        add(f"phase_difference_histogram_{index:02d}", value, "phase_difference")

    spectrum = np.fft.fft(x)
    psd = np.abs(spectrum) ** 2 + eps
    flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)
    add("spectral_flatness", flatness, "spectral")

    # Blind CP evidence: global lag-N correlation plus a boundary-search peak.
    lag = config.n_subcarriers
    delayed_a = x[:-lag]
    delayed_b = x[lag:]
    lag_corr = np.abs(np.vdot(delayed_a, delayed_b)) / np.sqrt(
        np.vdot(delayed_a, delayed_a).real
        * np.vdot(delayed_b, delayed_b).real
        + eps
    )
    add("cp_global_lag_correlation", lag_corr, "cp")

    symbol_length = config.n_subcarriers + config.cp_length
    offset_scores = []
    for offset in range(symbol_length):
        starts = np.arange(offset, x.size - symbol_length + 1, symbol_length)
        if starts.size == 0:
            offset_scores.append(0.0)
            continue
        cp_indices = (starts[:, None] + np.arange(config.cp_length)).reshape(-1)
        tail_indices = cp_indices + config.n_subcarriers
        first = x[cp_indices]
        second = x[tail_indices]
        score = np.abs(np.vdot(first, second)) / np.sqrt(
            np.vdot(first, first).real * np.vdot(second, second).real + eps
        )
        offset_scores.append(float(score))
    offset_scores_array = np.asarray(offset_scores)
    add("cp_boundary_search_peak", np.max(offset_scores_array), "cp")
    add("cp_boundary_search_mean", np.mean(offset_scores_array), "cp")
    add(
        "cp_boundary_peak_to_mean",
        np.max(offset_scores_array) / (np.mean(offset_scores_array) + eps),
        "cp",
    )

    return np.asarray(values, dtype=np.float32), names, groups


def build_records(
    snrs: list[float],
    turbulence_levels: tuple[str, ...],
    examples_per_condition: int,
) -> list[tuple[int, float, str, int]]:
    return [
        (modulation_index, snr, turbulence, sample_index)
        for turbulence in turbulence_levels
        for snr in snrs
        for modulation_index in range(len(MODULATIONS))
        for sample_index in range(examples_per_condition)
    ]


def extract_split(
    split: str,
    records: list[tuple[int, float, str, int]],
    seed: int,
    config: SimulationConfig,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    cache_path = args.output_dir / f"features_{split}.npz"
    if cache_path.exists() and not args.allow_overwrite:
        cached = np.load(cache_path, allow_pickle=False)
        return {key: cached[key] for key in cached.files}

    feature_rows = []
    labels = []
    snrs = []
    turbulence = []
    feature_names = None
    feature_groups = None
    started = time.time()
    for index, record in enumerate(records, start=1):
        modulation_index, snr, level, sample_index = record
        sequence = generate_sequence(
            modulation_index, snr, level, sample_index, seed, config
        )
        row, names, groups = extract_features(
            sequence, config, args.histogram_bins, args.phase_bins
        )
        if feature_names is None:
            feature_names = names
            feature_groups = groups
        feature_rows.append(row)
        labels.append(modulation_index)
        snrs.append(snr)
        turbulence.append(level)
        if index % 1000 == 0 or index == len(records):
            elapsed = time.time() - started
            print(
                f"{split}: {index}/{len(records)} samples, "
                f"{index / max(elapsed, 1e-6):.1f} samples/s",
                flush=True,
            )

    output = {
        "X": np.stack(feature_rows),
        "y": np.asarray(labels, dtype=np.int64),
        "snr": np.asarray(snrs, dtype=np.float32),
        "turbulence": np.asarray(turbulence, dtype="U16"),
        "feature_names": np.asarray(feature_names, dtype="U64"),
        "feature_groups": np.asarray(feature_groups, dtype="U32"),
    }
    np.savez_compressed(cache_path, **output)
    return output


def make_model(name: str, args: argparse.Namespace):
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=10000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        return XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(MODULATIONS),
            eval_metric="mlogloss",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    raise ValueError(name)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(
    model_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    snrs: np.ndarray,
    turbulence: np.ndarray,
) -> list[dict]:
    rows = []
    for level in ALL_EVAL_TURBULENCE:
        for snr in sorted(np.unique(snrs)):
            mask = (turbulence == level) & np.isclose(snrs, snr)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "model": model_name,
                    "turbulence": level,
                    "snr": float(snr),
                    "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                    "samples": int(np.sum(mask)),
                }
            )
    return rows


def plot_snr_curves(output_dir: Path, model_name: str, rows: list[dict]) -> None:
    colors = {
        "none": "#1f77b4",
        "weak": "#2ca02c",
        "moderate": "#ff7f0e",
        "strong": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for level in ALL_EVAL_TURBULENCE:
        selected = [row for row in rows if row["turbulence"] == level]
        ax.plot(
            [row["snr"] for row in selected],
            [100.0 * row["accuracy"] for row in selected],
            marker="o",
            linewidth=2,
            label=level.capitalize(),
            color=colors[level],
        )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Classification accuracy (%)",
        title=f"Statistical features + {model_name}",
        ylim=(0, 102),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"accuracy_vs_snr_{model_name}.png", dpi=220)
    plt.close(fig)


def plot_confusions(
    output_dir: Path,
    model_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    snrs: np.ndarray,
    turbulence: np.ndarray,
    target_snr: float,
) -> None:
    resolved_snr = min(np.unique(snrs), key=lambda value: abs(value - target_snr))
    for level in ALL_EVAL_TURBULENCE:
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
        ax.set_title(f"{model_name}, {level}, SNR={resolved_snr:g} dB")
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"confusion_matrix_{model_name}_{level}_snr_{resolved_snr:g}dB.png",
            dpi=220,
        )
        plt.close(fig)


def run_group_ablation(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for group in FEATURE_GROUPS:
        mask = train["feature_groups"] == group
        model = RandomForestClassifier(
            n_estimators=max(200, args.rf_trees // 2),
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        model.fit(train["X"][:, mask], train["y"])
        val_prediction = model.predict(val["X"][:, mask])
        test_prediction = model.predict(test["X"][:, mask])
        rows.append(
            {
                "feature_group": group,
                "feature_count": int(np.sum(mask)),
                "validation_accuracy": float(
                    accuracy_score(val["y"], val_prediction)
                ),
                "test_accuracy": float(accuracy_score(test["y"], test_prediction)),
            }
        )
    write_rows(args.output_dir / "feature_group_ablation.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(rows))
    ax.bar(
        positions - 0.18,
        [100 * row["validation_accuracy"] for row in rows],
        width=0.36,
        label="Validation",
    )
    ax.bar(
        positions + 0.18,
        [100 * row["test_accuracy"] for row in rows],
        width=0.36,
        label="Test",
    )
    ax.set_xticks(positions, [row["feature_group"] for row in rows], rotation=25)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Information-location ablation by explicit feature group")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature_group_ablation.png", dpi=220)
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    protected = args.output_dir / "metrics.json"
    if protected.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"{protected} already exists. Use a new output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        sequence_length=args.sequence_length,
        n_subcarriers=args.n_subcarriers,
        cp_length=args.cp_length,
    )

    split_specs = {
        "train": (
            build_records(
                args.snrs,
                tuple(TRAINING_TURBULENCE_LEVELS),
                args.train_examples,
            ),
            args.seed + 1000,
        ),
        "val": (
            build_records(args.snrs, ALL_EVAL_TURBULENCE, args.val_examples),
            args.seed + 2000,
        ),
        "test": (
            build_records(args.snrs, ALL_EVAL_TURBULENCE, args.test_examples),
            args.seed + 3000,
        ),
    }
    datasets = {
        split: extract_split(split, records, seed, config, args)
        for split, (records, seed) in split_specs.items()
    }

    ablation_rows = []
    if not args.skip_group_ablation:
        ablation_rows = run_group_ablation(
            datasets["train"], datasets["val"], datasets["test"], args
        )

    result_rows = []
    condition_output_rows = []
    skipped_models = []
    for model_name in args.models:
        model = make_model(model_name, args)
        if model is None:
            skipped_models.append(
                {
                    "model": model_name,
                    "reason": "package not installed in the active interpreter",
                }
            )
            print(f"SKIP {model_name}: package is not installed", flush=True)
            continue
        started = time.time()
        model.fit(datasets["train"]["X"], datasets["train"]["y"])
        val_prediction = model.predict(datasets["val"]["X"])
        test_prediction = model.predict(datasets["test"]["X"])
        elapsed = time.time() - started
        val_accuracy = accuracy_score(datasets["val"]["y"], val_prediction)
        test_accuracy = accuracy_score(datasets["test"]["y"], test_prediction)
        result_rows.append(
            {
                "model": model_name,
                "validation_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "fit_and_predict_seconds": float(elapsed),
            }
        )
        model_dir = args.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        np.savez_compressed(
            model_dir / "test_predictions.npz",
            true_class=datasets["test"]["y"],
            predicted_class=test_prediction,
            snr=datasets["test"]["snr"],
            turbulence=datasets["test"]["turbulence"],
        )
        rows = condition_rows(
            model_name,
            datasets["test"]["y"],
            test_prediction,
            datasets["test"]["snr"],
            datasets["test"]["turbulence"],
        )
        condition_output_rows.extend(rows)
        write_rows(model_dir / "accuracy_vs_snr.csv", rows)
        plot_snr_curves(model_dir, model_name, rows)
        plot_confusions(
            model_dir,
            model_name,
            datasets["test"]["y"],
            test_prediction,
            datasets["test"]["snr"],
            datasets["test"]["turbulence"],
            args.confusion_snr,
        )
        if model_name == "random_forest":
            importances = model.feature_importances_
            importance_rows = sorted(
                [
                    {
                        "feature": str(name),
                        "group": str(group),
                        "importance": float(importance),
                    }
                    for name, group, importance in zip(
                        datasets["train"]["feature_names"],
                        datasets["train"]["feature_groups"],
                        importances,
                    )
                ],
                key=lambda row: row["importance"],
                reverse=True,
            )
            write_rows(model_dir / "feature_importance.csv", importance_rows)
        print(
            f"{model_name}: val={val_accuracy:.4f}, test={test_accuracy:.4f}",
            flush=True,
        )

    write_rows(args.output_dir / "model_comparison.csv", result_rows)
    write_rows(args.output_dir / "accuracy_vs_snr_all_models.csv", condition_output_rows)
    metrics = {
        "task": "explicit statistical-feature diagnosis for six-class OFDM AMC",
        "claim_tested": (
            "Locate high-order QAM separability in amplitude statistics, "
            "complex cumulants, phase differences, spectrum, CP, or their fusion."
        ),
        "modulations": MODULATIONS,
        "snrs_db": args.snrs,
        "train_turbulence": TRAINING_TURBULENCE_LEVELS,
        "evaluation_turbulence": ALL_EVAL_TURBULENCE,
        "sequence_length": args.sequence_length,
        "n_subcarriers": args.n_subcarriers,
        "cp_length": args.cp_length,
        "train_examples_per_condition": args.train_examples,
        "val_examples_per_condition": args.val_examples,
        "test_examples_per_condition": args.test_examples,
        "feature_names": datasets["train"]["feature_names"].tolist(),
        "feature_groups": datasets["train"]["feature_groups"].tolist(),
        "group_ablation": ablation_rows,
        "model_results": result_rows,
        "skipped_models": skipped_models,
        "frame_level_accuracy_used": False,
        "evaluation_unit": "independent randomly generated time-domain IQ sequence",
    }
    protected.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
