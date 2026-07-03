# -*- coding: utf-8 -*-
"""
06_check_domain_gap_synth_vs_real.py

Purpose
-------
Check whether GNN-generated channels / synthetic modulation-recognition signals
are close to real experimental data.

This script supports two levels of checks:
1) Channel-level domain gap:
   real_channel_mat: gnn_channel_dataset.mat from MATLAB extraction
   gen_channel_npz : generated_channels_snr_5_20.npz from GNN generator

2) Signal-level domain gap, optional:
   synth_dir: synthetic modrec dataset directory containing modrec_synth_shard_*.npz
   real_dir : real modrec dataset directory containing npz shards in the same format

Main metrics
------------
- Subcarrier mean/std curve similarity
- Relative error of mean |H(k)| curve
- Correlation of mean |H(k)| curve
- Feature MMD between real and generated samples
- Optional binary domain-classifier accuracy/AUC if scikit-learn is installed

Usage example
-------------
python 06_check_domain_gap_synth_vs_real.py ^
  --real_channel_mat "D:/.../gnn_channel_dataset/gnn_channel_dataset.mat" ^
  --gen_channel_npz "D:/.../gnn_channel_model/generated/generated_channels_snr_5_20.npz" ^
  --synth_dir "D:/.../modrec_dataset_gnn_5_20dB" ^
  --out_dir "D:/.../domain_gap_report"

After you have converted the second real acquisition to the same Y_iq npz format,
add:
  --real_dir "D:/.../real_modrec_dataset_second_acq"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import h5py
except Exception:
    h5py = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_complex_array(x: np.ndarray) -> np.ndarray:
    """Convert common real/imag storage formats to complex ndarray."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return x.astype(np.complex64)

    # MATLAB / numpy possible format: [..., 2] = real/imag
    if x.ndim >= 2 and x.shape[-1] == 2:
        return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)

    # Possible format: [2, N, K, ...] = real/imag
    if x.ndim >= 2 and x.shape[0] == 2:
        return (x[0, ...] + 1j * x[1, ...]).astype(np.complex64)

    # If only magnitude/SNR is available, keep as real complex with zero imaginary.
    return x.astype(np.float32).astype(np.complex64)


def standardize_H_shape(H: np.ndarray) -> np.ndarray:
    """
    Return H as [N, K] complex array.
    If H is [N, K, T], average over time/symbol dimension.
    If H is [K, N], transpose if K looks smaller than N.
    """
    H = to_complex_array(np.asarray(H))
    H = np.squeeze(H)

    if H.ndim == 1:
        H = H[None, :]
    elif H.ndim == 2:
        # Heuristic: samples usually larger than subcarriers. If first dimension looks like K,
        # and second looks like N, transpose.
        if H.shape[0] < H.shape[1] and H.shape[0] <= 512 and H.shape[1] > 512:
            H = H.T
    elif H.ndim == 3:
        # Heuristic: choose sample axis as largest or first axis. Most expected: [N, K, T].
        # Average over the shortest non-sample time axis if needed.
        if H.shape[0] <= 512 and H.shape[1] > 512:
            # possible [K, N, T]
            H = np.transpose(H, (1, 0, 2))
        # Now expected [N, K, T]
        H = np.nanmean(H, axis=2)
    else:
        # Flatten all trailing dimensions except sample, keeping first dimension as N.
        H = H.reshape(H.shape[0], -1)

    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return H.astype(np.complex64)


def load_mat_any(path: Path) -> Dict[str, np.ndarray]:
    """Load MATLAB v7 or v7.3 mat into a plain dict."""
    if not path.exists():
        raise FileNotFoundError(path)

    data: Dict[str, np.ndarray] = {}

    if sio is not None:
        try:
            raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            for k, v in raw.items():
                if not k.startswith("__"):
                    try:
                        data[k] = np.asarray(v)
                    except Exception:
                        pass
            if data:
                return data
        except NotImplementedError:
            pass
        except Exception:
            pass

    if h5py is None:
        raise RuntimeError("Cannot read this .mat file. Install scipy or h5py.")

    with h5py.File(str(path), "r") as f:
        def visit(name, obj):
            if hasattr(obj, "shape"):
                try:
                    arr = np.array(obj)
                    # h5py reads MATLAB arrays with reversed dimensions often.
                    if arr.ndim >= 2:
                        arr = np.transpose(arr)
                    data[name.split("/")[-1]] = arr
                except Exception:
                    pass
        f.visititems(visit)
    return data


def pick_H_from_mat(data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    candidates = [
        "H_res", "H_real", "H_hat", "H_all", "H", "channel", "channels",
        "H_dataset", "H_frames", "H_sc", "H_eff"
    ]
    for key in candidates:
        if key in data:
            H = standardize_H_shape(data[key])
            if H.size > 0 and H.shape[1] >= 4:
                return key, H

    # fallback: choose largest complex-like numeric array
    best_key, best_arr, best_size = None, None, -1
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.dtype.kind not in "fcui" and not np.iscomplexobj(arr):
            continue
        if arr.size > best_size and arr.ndim >= 2:
            best_key, best_arr, best_size = k, arr, arr.size
    if best_key is None:
        raise KeyError("No channel-like array found in mat file. Please check variable name.")
    return best_key, standardize_H_shape(best_arr)


def load_generated_H(npz_path: Path) -> Tuple[str, np.ndarray, Dict[str, np.ndarray]]:
    z = np.load(str(npz_path), allow_pickle=True)
    keys = list(z.keys())
    candidates = ["H_gen", "H", "channels", "generated_H", "H_generated"]
    for k in candidates:
        if k in z:
            return k, standardize_H_shape(z[k]), {kk: z[kk] for kk in keys}
    # fallback: largest array
    best_key = max(keys, key=lambda k: np.asarray(z[k]).size)
    return best_key, standardize_H_shape(z[best_key]), {kk: z[kk] for kk in keys}


def sample_rows(x: np.ndarray, max_samples: int, seed: int = 2026) -> np.ndarray:
    if x.shape[0] <= max_samples:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    return x[idx]


def channel_features(H: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Extract compact per-sample features from [N,K] complex channel."""
    H = standardize_H_shape(H)
    mag = np.abs(H).astype(np.float64)
    phase_step = np.angle(H[:, 1:] * np.conj(H[:, :-1])) if H.shape[1] > 1 else np.zeros((H.shape[0], 1))
    diff_mag = np.diff(mag, axis=1) if H.shape[1] > 1 else np.zeros((H.shape[0], 1))

    def safe_mean(a, axis=1): return np.nanmean(a, axis=axis)
    def safe_std(a, axis=1): return np.nanstd(a, axis=axis)
    def safe_pct(a, q): return np.nanpercentile(a, q, axis=1)

    adj_corr = []
    for row in mag:
        if row.size < 3 or np.std(row[:-1]) < 1e-12 or np.std(row[1:]) < 1e-12:
            adj_corr.append(0.0)
        else:
            adj_corr.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
    adj_corr = np.asarray(adj_corr)

    mean_mag = safe_mean(mag)
    std_mag = safe_std(mag)
    feat = np.stack([
        mean_mag,
        std_mag,
        std_mag / (mean_mag + 1e-12),
        safe_pct(mag, 10),
        safe_pct(mag, 50),
        safe_pct(mag, 90),
        safe_mean(np.abs(diff_mag)),
        safe_std(diff_mag),
        safe_mean(np.abs(phase_step)),
        safe_std(phase_step),
        adj_corr,
    ], axis=1)
    names = [
        "mag_mean", "mag_std", "mag_cv", "mag_p10", "mag_p50", "mag_p90",
        "mag_absdiff_mean", "mag_diff_std", "phase_step_abs_mean", "phase_step_std", "adjacent_mag_corr"
    ]
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32), names


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_samples: int = 3000) -> float:
    x = sample_rows(np.asarray(x, dtype=np.float64), max_samples, seed=1)
    y = sample_rows(np.asarray(y, dtype=np.float64), max_samples, seed=2)
    z = np.vstack([x, y])
    # Standardize to make dimensions comparable.
    mu = z.mean(axis=0, keepdims=True)
    sd = z.std(axis=0, keepdims=True) + 1e-9
    x = (x - mu) / sd
    y = (y - mu) / sd

    # Median heuristic on a subset.
    z = sample_rows(np.vstack([x, y]), min(1000, x.shape[0] + y.shape[0]), seed=3)
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (2.0 * med + 1e-12)

    def kernel(a, b):
        dist2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * dist2)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    # Biased MMD is stable and sufficient for monitoring.
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def domain_classifier_report(x_real: np.ndarray, x_gen: np.ndarray, seed: int = 2026) -> Dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {"available": 0.0, "error": str(e)}

    n = min(x_real.shape[0], x_gen.shape[0], 5000)
    x_real = sample_rows(x_real, n, seed=seed)
    x_gen = sample_rows(x_gen, n, seed=seed + 1)
    X = np.vstack([x_real, x_gen])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    try:
        prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob))
    except Exception:
        auc = float("nan")
    return {
        "available": 1.0,
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": auc,
        "n_per_domain": float(n),
    }


def curve_metrics(real_curve: np.ndarray, gen_curve: np.ndarray) -> Dict[str, float]:
    K = min(real_curve.size, gen_curve.size)
    a = real_curve[:K].astype(np.float64)
    b = gen_curve[:K].astype(np.float64)
    rel_err = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.mean(np.abs(a - b)))
    return {"relative_l2_error": rel_err, "corr": corr, "mae": mae}


def save_dict_csv(path: Path, d: Dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in d.items():
            writer.writerow([k, v])


def save_feature_summary(path: Path, feat_real: np.ndarray, feat_gen: np.ndarray, names: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "real_mean", "real_std", "gen_mean", "gen_std", "abs_mean_gap", "relative_mean_gap"])
        for i, name in enumerate(names):
            rm, rs = float(np.mean(feat_real[:, i])), float(np.std(feat_real[:, i]))
            gm, gs = float(np.mean(feat_gen[:, i])), float(np.std(feat_gen[:, i]))
            gap = abs(rm - gm)
            rel = gap / (abs(rm) + 1e-12)
            writer.writerow([name, rm, rs, gm, gs, gap, rel])


def plot_channel_curves(out_dir: Path, H_real: np.ndarray, H_gen: np.ndarray) -> None:
    if plt is None:
        return
    K = min(H_real.shape[1], H_gen.shape[1])
    mag_real = np.abs(H_real[:, :K])
    mag_gen = np.abs(H_gen[:, :K])
    mu_r, sd_r = mag_real.mean(axis=0), mag_real.std(axis=0)
    mu_g, sd_g = mag_gen.mean(axis=0), mag_gen.std(axis=0)
    x = np.arange(K)

    plt.figure(figsize=(10, 5))
    plt.plot(x, mu_r, label="real mean |H(k)|")
    plt.plot(x, mu_g, label="generated mean |H(k)|")
    plt.xlabel("subcarrier index")
    plt.ylabel("mean magnitude")
    plt.title("Mean subcarrier channel magnitude curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "channel_mean_mag_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, sd_r, label="real std |H(k)|")
    plt.plot(x, sd_g, label="generated std |H(k)|")
    plt.xlabel("subcarrier index")
    plt.ylabel("std magnitude")
    plt.title("Std of subcarrier channel magnitude curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "channel_std_mag_curve.png", dpi=200)
    plt.close()

    # Histograms of compact features are written as one figure per important feature.
    feat_r, names = channel_features(H_real[:, :K])
    feat_g, _ = channel_features(H_gen[:, :K])
    for name in ["mag_mean", "mag_std", "mag_cv", "mag_absdiff_mean", "phase_step_std", "adjacent_mag_corr"]:
        if name not in names:
            continue
        i = names.index(name)
        plt.figure(figsize=(7, 4))
        plt.hist(feat_r[:, i], bins=50, alpha=0.55, density=True, label="real")
        plt.hist(feat_g[:, i], bins=50, alpha=0.55, density=True, label="generated")
        plt.xlabel(name)
        plt.ylabel("density")
        plt.title(f"Feature distribution: {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{name}.png", dpi=200)
        plt.close()


def load_modrec_shards(data_dir: Path, max_samples: int = 10000, seed: int = 2026) -> Optional[np.ndarray]:
    if data_dir is None or not data_dir.exists():
        return None
    files = sorted(list(data_dir.glob("*.npz")))
    files = [p for p in files if "shard" in p.name.lower() or "modrec" in p.name.lower()]
    if not files:
        return None

    Ys = []
    total = 0
    for p in files:
        z = np.load(str(p), allow_pickle=True)
        if "Y_iq" not in z:
            continue
        Y = np.asarray(z["Y_iq"])
        if Y.ndim != 4:
            continue
        Ys.append(Y)
        total += Y.shape[0]
        if total >= max_samples * 2:
            break
    if not Ys:
        return None
    Y = np.concatenate(Ys, axis=0)
    if Y.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Y.shape[0], size=max_samples, replace=False)
        Y = Y[idx]
    return Y


def signal_features_from_Yiq(Y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Y_iq [N,2,K,T] -> compact features."""
    Y = np.asarray(Y)
    if Y.ndim != 4 or Y.shape[1] != 2:
        raise ValueError("Expected Y_iq shape [N,2,K,T].")
    C = Y[:, 0] + 1j * Y[:, 1]
    amp = np.abs(C)
    pow_sc = np.mean(np.abs(C) ** 2, axis=2)  # [N,K]
    phase_step_k = np.angle(C[:, 1:, :] * np.conj(C[:, :-1, :])) if C.shape[1] > 1 else np.zeros((C.shape[0], 1, C.shape[2]))
    feat = np.stack([
        np.mean(amp, axis=(1, 2)),
        np.std(amp, axis=(1, 2)),
        np.percentile(amp, 10, axis=(1, 2)),
        np.percentile(amp, 50, axis=(1, 2)),
        np.percentile(amp, 90, axis=(1, 2)),
        np.mean(pow_sc, axis=1),
        np.std(pow_sc, axis=1),
        np.mean(np.abs(phase_step_k), axis=(1, 2)),
        np.std(phase_step_k, axis=(1, 2)),
    ], axis=1)
    names = [
        "amp_mean", "amp_std", "amp_p10", "amp_p50", "amp_p90",
        "subcarrier_power_mean", "subcarrier_power_std", "phase_step_k_abs_mean", "phase_step_k_std"
    ]
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32), names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_channel_mat", type=str, required=True, help="gnn_channel_dataset.mat from real experimental extraction")
    parser.add_argument("--gen_channel_npz", type=str, required=True, help="generated_channels_snr_5_20.npz")
    parser.add_argument("--synth_dir", type=str, default="", help="optional synthetic modrec npz directory")
    parser.add_argument("--real_dir", type=str, default="", help="optional real modrec npz directory in the same Y_iq format")
    parser.add_argument("--out_dir", type=str, required=True, help="output report directory")
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mkdir(out_dir)

    real_data = load_mat_any(Path(args.real_channel_mat))
    real_key, H_real = pick_H_from_mat(real_data)
    gen_key, H_gen, gen_npz = load_generated_H(Path(args.gen_channel_npz))

    # Align subcarrier count and sample maximum.
    K = min(H_real.shape[1], H_gen.shape[1])
    H_real = sample_rows(H_real[:, :K], args.max_samples, seed=10)
    H_gen = sample_rows(H_gen[:, :K], args.max_samples, seed=11)

    feat_real, feat_names = channel_features(H_real)
    feat_gen, _ = channel_features(H_gen)

    mean_curve_real = np.abs(H_real).mean(axis=0)
    mean_curve_gen = np.abs(H_gen).mean(axis=0)
    std_curve_real = np.abs(H_real).std(axis=0)
    std_curve_gen = np.abs(H_gen).std(axis=0)

    report: Dict[str, object] = {}
    report["real_channel_key"] = real_key
    report["generated_channel_key"] = gen_key
    report["real_channel_shape_used"] = str(H_real.shape)
    report["generated_channel_shape_used"] = str(H_gen.shape)

    mc = curve_metrics(mean_curve_real, mean_curve_gen)
    sc = curve_metrics(std_curve_real, std_curve_gen)
    report.update({f"mean_mag_curve_{k}": v for k, v in mc.items()})
    report.update({f"std_mag_curve_{k}": v for k, v in sc.items()})
    report["channel_feature_mmd2"] = rbf_mmd2(feat_real, feat_gen)

    clf = domain_classifier_report(feat_real, feat_gen)
    for k, v in clf.items():
        report[f"channel_domain_classifier_{k}"] = v

    save_dict_csv(out_dir / "channel_domain_gap_metrics.csv", report)
    save_feature_summary(out_dir / "channel_feature_summary.csv", feat_real, feat_gen, feat_names)
    plot_channel_curves(out_dir, H_real, H_gen)

    np.savez_compressed(
        out_dir / "channel_curves_used.npz",
        mean_curve_real=mean_curve_real,
        mean_curve_gen=mean_curve_gen,
        std_curve_real=std_curve_real,
        std_curve_gen=std_curve_gen,
        feat_real=feat_real,
        feat_gen=feat_gen,
        feature_names=np.asarray(feat_names),
    )

    # Optional signal-level comparison.
    signal_report = {}
    if args.synth_dir and args.real_dir:
        Y_synth = load_modrec_shards(Path(args.synth_dir), max_samples=args.max_samples, seed=20)
        Y_real = load_modrec_shards(Path(args.real_dir), max_samples=args.max_samples, seed=21)
        if Y_synth is not None and Y_real is not None:
            sf_synth, sf_names = signal_features_from_Yiq(Y_synth)
            sf_real, _ = signal_features_from_Yiq(Y_real)
            signal_report["synth_Y_shape_used"] = str(Y_synth.shape)
            signal_report["real_Y_shape_used"] = str(Y_real.shape)
            signal_report["signal_feature_mmd2"] = rbf_mmd2(sf_real, sf_synth)
            s_clf = domain_classifier_report(sf_real, sf_synth)
            for k, v in s_clf.items():
                signal_report[f"signal_domain_classifier_{k}"] = v
            save_dict_csv(out_dir / "signal_domain_gap_metrics.csv", signal_report)
            save_feature_summary(out_dir / "signal_feature_summary.csv", sf_real, sf_synth, sf_names)
        else:
            signal_report["error"] = "Could not load Y_iq shards from synth_dir and/or real_dir."
            save_dict_csv(out_dir / "signal_domain_gap_metrics.csv", signal_report)

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump({"channel_report": report, "signal_report": signal_report}, f, indent=2, ensure_ascii=False)

    print("[OK] Domain gap report saved to:", out_dir)
    print("Important files:")
    print(" - channel_domain_gap_metrics.csv")
    print(" - channel_feature_summary.csv")
    print(" - channel_mean_mag_curve.png")
    print(" - channel_std_mag_curve.png")
    print(" - report.json")
    if args.synth_dir and args.real_dir:
        print(" - signal_domain_gap_metrics.csv")
        print(" - signal_feature_summary.csv")


if __name__ == "__main__":
    main()
