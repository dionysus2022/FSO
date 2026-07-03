#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_audit_synthetic_modrec_dataset.py

Purpose
-------
Audit the synthetic modulation-recognition dataset generated from GNN channels.
This script is designed for the situation where training accuracy is above random
but class behavior is abnormal, e.g. 16QAM is never predicted or 256QAM is too easy.

It checks:
1) shard fields and shapes
2) label / split / SNR / turbulence distributions
3) NaN / Inf / zero samples
4) per-class power and amplitude statistics
5) whether simple energy/moment features can predict modulation labels
6) whether 16QAM has abnormal feature overlap with other QAM classes
7) diagnostic plots for amplitude histograms and example IQ scatter clouds

Input directory should contain files like:
  modrec_synth_shard_0000.npz
  manifest.csv
  dataset_info.json

Each npz shard is expected to contain at least:
  Y_iq: [N, 2, K, T]
  mod_label: [N]
  target_snr_db: [N]
Optional:
  mod_name, turb_label, split_label, actual_snr_db, channel_index

Author: ChatGPT
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


DEFAULT_MODS = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def as_str_list(x):
    """Convert loaded numpy string/object arrays to Python str list."""
    if x is None:
        return None
    arr = np.asarray(x)
    out = []
    for v in arr.reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def load_mod_names(data_dir, cli_mods=None):
    if cli_mods:
        return [m.strip() for m in cli_mods.split(",") if m.strip()]
    info_path = Path(data_dir) / "dataset_info.json"
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            for key in ["mods", "mod_list", "classes", "class_names"]:
                if key in info and isinstance(info[key], list):
                    return [str(x) for x in info[key]]
            if "label_map" in info and isinstance(info["label_map"], dict):
                inv = {int(v): k for k, v in info["label_map"].items()}
                return [inv[i] for i in sorted(inv.keys())]
        except Exception:
            pass
    return DEFAULT_MODS


def list_shards(data_dir):
    pats = [
        "modrec_synth_shard_*.npz",
        "*.npz",
    ]
    found = []
    for pat in pats:
        found = sorted(glob.glob(str(Path(data_dir) / pat)))
        if found:
            break
    return found


def sample_from_shards(shards, max_per_class=3000, seed=1234, prefer_split=None):
    """
    Reservoir-like balanced sampling by class from shards.
    prefer_split: None, 0, 1, etc. If split_label exists, only take that split.
    """
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    meta_buckets = defaultdict(list)
    global_stats = {
        "num_shards": len(shards),
        "total_samples_seen": 0,
        "fields": set(),
        "shape_examples": [],
        "nan_count": 0,
        "inf_count": 0,
        "zero_rms_count": 0,
        "label_counter": Counter(),
        "split_counter": Counter(),
        "snr_counter": Counter(),
        "turb_counter": Counter(),
        "per_shard": [],
    }

    for sp in shards:
        try:
            z = np.load(sp, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] failed loading {sp}: {e}")
            continue
        keys = list(z.keys())
        global_stats["fields"].update(keys)
        if "Y_iq" not in z or "mod_label" not in z:
            print(f"[WARN] skip {sp}, missing Y_iq or mod_label. keys={keys}")
            continue
        Y = z["Y_iq"]
        y = z["mod_label"].astype(int).reshape(-1)
        N = len(y)
        global_stats["total_samples_seen"] += N
        global_stats["shape_examples"].append([os.path.basename(sp), list(Y.shape)])
        global_stats["label_counter"].update(y.tolist())

        split = z["split_label"].astype(int).reshape(-1) if "split_label" in z else np.full(N, -1, dtype=int)
        snr = z["target_snr_db"].astype(float).reshape(-1) if "target_snr_db" in z else np.full(N, np.nan)
        turb = z["turb_label"].astype(int).reshape(-1) if "turb_label" in z else np.full(N, -1, dtype=int)
        actual_snr = z["actual_snr_db"].astype(float).reshape(-1) if "actual_snr_db" in z else np.full(N, np.nan)

        global_stats["split_counter"].update(split.tolist())
        global_stats["snr_counter"].update([int(round(v)) for v in snr[np.isfinite(snr)].tolist()])
        global_stats["turb_counter"].update(turb.tolist())

        take_mask = np.ones(N, dtype=bool)
        if prefer_split is not None and "split_label" in z:
            take_mask &= (split == int(prefer_split))
        idx_all = np.nonzero(take_mask)[0]
        # shuffle inside shard to avoid deterministic early examples only
        rng.shuffle(idx_all)

        for idx in idx_all:
            cls = int(y[idx])
            if len(buckets[cls]) >= max_per_class:
                # probabilistic replacement to avoid only early shards
                if rng.random() > 0.01:
                    continue
                ridx = int(rng.integers(0, max_per_class))
                replace = True
            else:
                ridx = None
                replace = False

            sample = np.asarray(Y[idx], dtype=np.float32)
            if not np.isfinite(sample).all():
                global_stats["nan_count"] += int(np.isnan(sample).sum())
                global_stats["inf_count"] += int(np.isinf(sample).sum())
                continue
            rms = float(np.sqrt(np.mean(sample * sample)))
            if rms == 0 or not np.isfinite(rms):
                global_stats["zero_rms_count"] += 1
                continue
            meta = {
                "label": cls,
                "split": int(split[idx]),
                "target_snr_db": float(snr[idx]),
                "actual_snr_db": float(actual_snr[idx]),
                "turb_label": int(turb[idx]),
                "shard": os.path.basename(sp),
                "index_in_shard": int(idx),
            }
            if replace:
                buckets[cls][ridx] = sample
                meta_buckets[cls][ridx] = meta
            else:
                buckets[cls].append(sample)
                meta_buckets[cls].append(meta)

        global_stats["per_shard"].append({
            "shard": os.path.basename(sp),
            "N": int(N),
            "Y_shape": list(Y.shape),
            "labels": dict(Counter(y.tolist())),
        })
        z.close()

    X_list, y_list, meta_list = [], [], []
    for cls in sorted(buckets.keys()):
        X_list.extend(buckets[cls])
        y_list.extend([cls] * len(buckets[cls]))
        meta_list.extend(meta_buckets[cls])
    if not X_list:
        raise RuntimeError("No samples collected. Check data_dir and shard fields.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, meta_list, global_stats


def feature_extract(Y):
    """Y: [N,2,K,T]. Return feature matrix and feature names."""
    I = Y[:, 0]
    Q = Y[:, 1]
    C = I + 1j * Q
    A = np.abs(C)
    P = A ** 2
    # flatten per sample
    Af = A.reshape(A.shape[0], -1)
    Pf = P.reshape(P.shape[0], -1)
    If = I.reshape(I.shape[0], -1)
    Qf = Q.reshape(Q.shape[0], -1)

    eps = 1e-12
    mean_abs = Af.mean(axis=1)
    std_abs = Af.std(axis=1)
    rms_complex = np.sqrt(Pf.mean(axis=1))
    rms_iq = np.sqrt((If**2 + Qf**2).mean(axis=1) / 2.0)
    power = Pf.mean(axis=1)
    q10 = np.quantile(Af, 0.10, axis=1)
    q25 = np.quantile(Af, 0.25, axis=1)
    q50 = np.quantile(Af, 0.50, axis=1)
    q75 = np.quantile(Af, 0.75, axis=1)
    q90 = np.quantile(Af, 0.90, axis=1)
    q95 = np.quantile(Af, 0.95, axis=1)
    # shape moments
    centered = Af - mean_abs[:, None]
    m2 = np.mean(centered**2, axis=1) + eps
    skew_abs = np.mean(centered**3, axis=1) / (m2 ** 1.5)
    kurt_abs = np.mean(centered**4, axis=1) / (m2 ** 2)
    # I/Q balance
    i_mean = If.mean(axis=1)
    q_mean = Qf.mean(axis=1)
    i_std = If.std(axis=1)
    q_std = Qf.std(axis=1)
    iq_corr = np.mean((If - i_mean[:, None]) * (Qf - q_mean[:, None]), axis=1) / (i_std * q_std + eps)
    # phase difference statistics along subcarrier dimension
    phase = np.angle(C)
    dphi_k = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    dphi_t = np.angle(np.exp(1j * np.diff(phase, axis=2))) if phase.shape[2] > 1 else np.zeros_like(phase[:, :, :1])
    dphi_k_std = dphi_k.reshape(dphi_k.shape[0], -1).std(axis=1)
    dphi_t_std = dphi_t.reshape(dphi_t.shape[0], -1).std(axis=1)

    F = np.stack([
        mean_abs, std_abs, rms_complex, rms_iq, power,
        q10, q25, q50, q75, q90, q95,
        skew_abs, kurt_abs,
        i_mean, q_mean, i_std, q_std, iq_corr,
        dphi_k_std, dphi_t_std,
    ], axis=1).astype(np.float32)
    names = [
        "mean_abs", "std_abs", "rms_complex", "rms_iq", "power",
        "abs_q10", "abs_q25", "abs_q50", "abs_q75", "abs_q90", "abs_q95",
        "skew_abs", "kurt_abs",
        "i_mean", "q_mean", "i_std", "q_std", "iq_corr",
        "dphi_k_std", "dphi_t_std",
    ]
    return F, names


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_by_class(F, y, feat_names, mod_names):
    rows = []
    for cls in sorted(np.unique(y).tolist()):
        mask = (y == cls)
        row = {
            "label": int(cls),
            "mod_name": mod_names[cls] if cls < len(mod_names) else str(cls),
            "n": int(mask.sum()),
        }
        for j, name in enumerate(feat_names):
            vals = F[mask, j]
            row[f"{name}_mean"] = float(np.mean(vals))
            row[f"{name}_std"] = float(np.std(vals))
            row[f"{name}_min"] = float(np.min(vals))
            row[f"{name}_max"] = float(np.max(vals))
        rows.append(row)
    return rows


def snr_class_distribution(meta, y, mod_names):
    rows = []
    counter = Counter()
    for m, lab in zip(meta, y):
        snr = m.get("target_snr_db", np.nan)
        if np.isfinite(snr):
            snr_i = int(round(float(snr)))
        else:
            snr_i = -999
        counter[(int(lab), snr_i)] += 1
    for (lab, snr), n in sorted(counter.items()):
        rows.append({
            "label": lab,
            "mod_name": mod_names[lab] if lab < len(mod_names) else str(lab),
            "target_snr_db": snr,
            "n": int(n),
        })
    return rows


def run_probe(F, y, out_dir, mod_names, feature_subset=None, prefix="probe"):
    if not HAS_SKLEARN:
        return {"available": False, "reason": "sklearn not installed"}
    if feature_subset is not None:
        F_use = F[:, feature_subset]
        used_names = feature_subset
    else:
        F_use = F
        used_names = None
    # stratified split can fail if tiny class; handle fallback
    try:
        Xtr, Xte, ytr, yte = train_test_split(F_use, y, test_size=0.35, random_state=42, stratify=y)
    except Exception:
        Xtr, Xte, ytr, yte = train_test_split(F_use, y, test_size=0.35, random_state=42)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    cm = confusion_matrix(yte, pred, labels=list(range(len(mod_names))))
    np.savetxt(Path(out_dir) / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    try:
        rep = classification_report(yte, pred, labels=list(range(len(mod_names))), target_names=mod_names, output_dict=True, zero_division=0)
        with open(Path(out_dir) / f"{prefix}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {"available": True, "accuracy": acc, "feature_subset": used_names}


def plot_diagnostics(Y, y, F, feat_names, mod_names, meta, out_dir, max_points=6000):
    if not HAS_MPL:
        return
    out_dir = Path(out_dir)
    # amplitude histogram by class
    plt.figure(figsize=(10, 6))
    for cls in sorted(np.unique(y)):
        idx = np.nonzero(y == cls)[0]
        if len(idx) == 0:
            continue
        idx = idx[: min(len(idx), 300)]
        C = Y[idx, 0] + 1j * Y[idx, 1]
        A = np.abs(C).reshape(-1)
        # subsample for plotting
        if A.size > max_points:
            rng = np.random.default_rng(100 + int(cls))
            A = rng.choice(A, size=max_points, replace=False)
        plt.hist(A, bins=80, density=True, alpha=0.35, label=mod_names[int(cls)] if int(cls) < len(mod_names) else str(cls))
    plt.xlabel("|Y|")
    plt.ylabel("density")
    plt.title("Amplitude distribution by modulation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_hist_by_mod.png", dpi=180)
    plt.close()

    # power feature boxplot
    feature_to_plot = ["mean_abs", "std_abs", "rms_complex", "abs_q50", "abs_q90", "kurt_abs"]
    for name in feature_to_plot:
        if name not in feat_names:
            continue
        j = feat_names.index(name)
        data = [F[y == cls, j] for cls in sorted(np.unique(y))]
        labels = [mod_names[int(cls)] if int(cls) < len(mod_names) else str(cls) for cls in sorted(np.unique(y))]
        plt.figure(figsize=(9, 5))
        bp = plt.boxplot(data, showfliers=False)
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=30)
        plt.ylabel(name)
        plt.title(f"{name} by modulation")
        plt.tight_layout()
        plt.savefig(out_dir / f"feature_box_{name}.png", dpi=180)
        plt.close()

    # IQ scatter examples, high-SNR preferred
    for cls in sorted(np.unique(y)):
        idx_all = np.nonzero(y == cls)[0]
        if len(idx_all) == 0:
            continue
        # prefer high snr sample
        idx_sorted = sorted(idx_all.tolist(), key=lambda ii: meta[ii].get("target_snr_db", -999), reverse=True)
        ii = idx_sorted[0]
        C = (Y[ii, 0] + 1j * Y[ii, 1]).reshape(-1)
        if C.size > max_points:
            rng = np.random.default_rng(999 + int(cls))
            C = rng.choice(C, size=max_points, replace=False)
        plt.figure(figsize=(5, 5))
        plt.scatter(C.real, C.imag, s=2, alpha=0.35)
        plt.axis("equal")
        title = f"{mod_names[int(cls)] if int(cls) < len(mod_names) else cls}, SNR={meta[ii].get('target_snr_db', np.nan):.1f} dB"
        plt.title(title)
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.tight_layout()
        plt.savefig(out_dir / f"iq_scatter_example_class_{int(cls)}.png", dpi=180)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing modrec_synth_shard_*.npz")
    ap.add_argument("--out_dir", required=True, help="Output report directory")
    ap.add_argument("--mods", default=None, help="Comma-separated mod names, default read from dataset_info.json or standard order")
    ap.add_argument("--max_per_class", type=int, default=3000, help="Max samples per class to audit")
    ap.add_argument("--prefer_split", type=int, default=None, help="Only audit split_label if field exists, e.g. 1 for val")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    safe_mkdir(args.out_dir)
    mod_names = load_mod_names(args.data_dir, args.mods)
    shards = list_shards(args.data_dir)
    if not shards:
        raise FileNotFoundError(f"No npz shards found in {args.data_dir}")

    print(f"[1] Found {len(shards)} shard(s)")
    print(f"[2] Mod order: {mod_names}")
    X, y, meta, stats = sample_from_shards(
        shards,
        max_per_class=args.max_per_class,
        seed=args.seed,
        prefer_split=args.prefer_split,
    )
    print(f"[3] Collected audit samples: {X.shape}, labels={dict(Counter(y.tolist()))}")

    # Convert set/counter to json-friendly
    stats_json = dict(stats)
    stats_json["fields"] = sorted(list(stats_json["fields"]))
    stats_json["label_counter"] = {str(k): int(v) for k, v in stats_json["label_counter"].items()}
    stats_json["split_counter"] = {str(k): int(v) for k, v in stats_json["split_counter"].items()}
    stats_json["snr_counter"] = {str(k): int(v) for k, v in stats_json["snr_counter"].items()}
    stats_json["turb_counter"] = {str(k): int(v) for k, v in stats_json["turb_counter"].items()}
    stats_json["mod_names"] = mod_names
    with open(Path(args.out_dir) / "dataset_global_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)

    # Feature extraction
    F, feat_names = feature_extract(X)
    np.savez_compressed(Path(args.out_dir) / "audit_sample_features.npz", F=F, y=y, feature_names=np.array(feat_names, dtype=object))

    rows = summarize_by_class(F, y, feat_names, mod_names)
    # Build fieldnames dynamically
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(Path(args.out_dir) / "feature_summary_by_mod.csv", rows, fieldnames)

    snr_rows = snr_class_distribution(meta, y, mod_names)
    write_csv(Path(args.out_dir) / "sample_count_by_mod_snr.csv", snr_rows, ["label", "mod_name", "target_snr_db", "n"])

    # Feature leakage probes
    results = {}
    # all handcrafted features
    results["all_handcrafted_features"] = run_probe(F, y, args.out_dir, mod_names, None, "probe_all_features")
    # energy-only features
    energy_names = ["mean_abs", "std_abs", "rms_complex", "rms_iq", "power", "abs_q50", "abs_q90"]
    energy_idx = [feat_names.index(n) for n in energy_names if n in feat_names]
    results["energy_amplitude_features_only"] = run_probe(F, y, args.out_dir, mod_names, energy_idx, "probe_energy_features")
    # phase-only features
    phase_names = ["iq_corr", "dphi_k_std", "dphi_t_std"]
    phase_idx = [feat_names.index(n) for n in phase_names if n in feat_names]
    if phase_idx:
        results["phase_features_only"] = run_probe(F, y, args.out_dir, mod_names, phase_idx, "probe_phase_features")

    # Pairwise distance of class mean features, useful for 16QAM overlap
    class_means = []
    labels_sorted = sorted(np.unique(y).tolist())
    for cls in labels_sorted:
        class_means.append(F[y == cls].mean(axis=0))
    class_means = np.stack(class_means, axis=0)
    # standardize feature dimensions before distance
    cm_std = (class_means - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)
    D = np.sqrt(((cm_std[:, None, :] - cm_std[None, :, :]) ** 2).sum(axis=2))
    np.savetxt(Path(args.out_dir) / "class_mean_feature_distance.csv", D, delimiter=",", fmt="%.6f")

    # Save readable pairwise rows
    pair_rows = []
    for i, a in enumerate(labels_sorted):
        for j, b in enumerate(labels_sorted):
            if j <= i:
                continue
            pair_rows.append({
                "label_a": int(a), "mod_a": mod_names[int(a)] if int(a) < len(mod_names) else str(a),
                "label_b": int(b), "mod_b": mod_names[int(b)] if int(b) < len(mod_names) else str(b),
                "standardized_feature_distance": float(D[i, j]),
            })
    write_csv(Path(args.out_dir) / "class_pair_feature_distance.csv", pair_rows,
              ["label_a", "mod_a", "label_b", "mod_b", "standardized_feature_distance"])

    # Plots
    plot_diagnostics(X, y, F, feat_names, mod_names, meta, args.out_dir)

    # Decision hints
    hints = []
    energy_acc = results.get("energy_amplitude_features_only", {}).get("accuracy", None)
    all_acc = results.get("all_handcrafted_features", {}).get("accuracy", None)
    if energy_acc is not None:
        if energy_acc > 0.45:
            hints.append("WARNING: energy/amplitude-only features classify modulation too well; synthetic data may contain power leakage or unequal constellation normalization.")
        elif energy_acc > 0.30:
            hints.append("CAUTION: energy/amplitude features have nontrivial predictive power; check per-mod RMS and constellation normalization.")
        else:
            hints.append("OK: energy/amplitude-only probe is not extremely strong.")
    if all_acc is not None and all_acc < 0.25:
        hints.append("WARNING: simple distribution features barely separate classes; the synthetic data may be too distorted or over-normalized.")
    # 16QAM distance hint
    if 1 in labels_sorted:
        idx16 = labels_sorted.index(1)
        near = []
        for j, lab in enumerate(labels_sorted):
            if j != idx16:
                near.append((float(D[idx16, j]), lab))
        near.sort()
        hints.append("16QAM nearest classes by handcrafted feature mean: " + ", ".join([
            f"{mod_names[int(lab)] if int(lab) < len(mod_names) else lab}:dist={dist:.3f}" for dist, lab in near[:3]
        ]))

    report = {
        "audit_samples_shape": list(X.shape),
        "mod_names": mod_names,
        "label_counts_in_audit_sample": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        "probe_results": results,
        "hints": hints,
        "output_files": [
            "dataset_global_stats.json",
            "feature_summary_by_mod.csv",
            "sample_count_by_mod_snr.csv",
            "probe_all_features_confusion_matrix.csv",
            "probe_energy_features_confusion_matrix.csv",
            "probe_all_features_classification_report.json",
            "probe_energy_features_classification_report.json",
            "class_pair_feature_distance.csv",
            "amplitude_hist_by_mod.png",
            "feature_box_*.png",
            "iq_scatter_example_class_*.png",
        ]
    }
    with open(Path(args.out_dir) / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== AUDIT SUMMARY ===")
    print(json.dumps({"probe_results": results, "hints": hints}, ensure_ascii=False, indent=2))
    print(f"\n[Done] Report saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
