from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(r"D:\Project_code\FSO_research")
FAST = ROOT / "results" / "stage2" / "fast_publish"
OUT = ROOT / "results" / "docs" / "b128_diagnostics"
MODS = ["64QAM", "128QAM", "256QAM"]
EXPECTED = {"64QAM": 3, "128QAM": 4, "256QAM": 5}


def scalar_text(x):
    return str(np.asarray(x).squeeze())


def audit(df):
    errors = []
    checked = 0
    for row in df.itertuples(index=False):
        path = Path(row.out_mat)
        if not path.exists():
            errors.append({"type": "missing", "path": str(path)})
            continue
        mat = loadmat(path, variable_names=["mod_name", "mod_label", "turb_name", "rx_file"])
        checked += 1
        checks = {
            "csv_map": int(row.mod_label) == EXPECTED[row.mod_name],
            "mat_name": scalar_text(mat["mod_name"]) == row.mod_name,
            "mat_label": int(np.asarray(mat["mod_label"]).squeeze()) == int(row.mod_label),
            "mat_turb": scalar_text(mat["turb_name"]) == row.turb_name,
            "path_mod": row.mod_name.lower() in str(path).lower(),
            "rx_path_mod": row.mod_name.lower() in str(row.rx_file).lower(),
            "path_turb": row.turb_name.lower() in str(path).lower(),
        }
        for name, ok in checks.items():
            if not ok:
                errors.append({"type": name, "path": str(path), "csv_mod": row.mod_name,
                               "csv_label": int(row.mod_label), "csv_turb": row.turb_name})
    return {"checked": checked, "errors": errors, "passed": not errors}


def plot_constellations(df):
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    summaries = {}
    for ri, (domain, turb) in enumerate([("A", "weak"), ("A", "strong"),
                                         ("B", "weak"), ("B", "strong")]):
        for ci, mod in enumerate(MODS):
            rows = df[(df.dataset_source == domain) & (df.turb_name == turb) &
                      (df.mod_name == mod)].sort_values(["rx_file", "rx_frame_idx"]).head(20)
            points = []
            for path in rows.out_mat:
                x = loadmat(path, variable_names=["rx_sc"])["rx_sc"].reshape(-1)
                x = x - x.mean()
                x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
                points.append(x)
            x = np.concatenate(points)
            # Deterministic thinning keeps the figure readable.
            shown = x[::max(1, len(x) // 12000)]
            ax = axes[ri, ci]
            ax.scatter(shown.real, shown.imag, s=1, alpha=0.12, rasterized=True)
            ax.set_title(f"{domain}-{turb}-{mod} (20 frames)")
            ax.set_aspect("equal")
            ax.set_xlim(-2.3, 2.3)
            ax.set_ylim(-2.3, 2.3)
            ax.grid(alpha=0.15)
            amp = np.abs(x)
            summaries[f"{domain}_{turb}_{mod}"] = {
                "frames": 20, "amp_mean": float(amp.mean()), "amp_std": float(amp.std()),
                "phase_concentration": float(abs(np.mean(np.exp(1j * np.angle(x))))),
                "papr_db": float(10 * np.log10(np.max(amp ** 2) / np.mean(amp ** 2))),
            }
    fig.savefig(OUT / "ab_highorder_constellations_20frames.png", dpi=220)
    plt.close(fig)
    return summaries


def fit_eval(X, y, train, test, name):
    model = make_pipeline(StandardScaler(), SVC(C=5.0, gamma="scale", class_weight="balanced"))
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    cm = confusion_matrix(y[test], pred, labels=[3, 4, 5])
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {"name": name, "n_train": int(len(train)), "n_test": int(len(test)),
            "accuracy": float(accuracy_score(y[test], pred)),
            "macro_f1": float(f1_score(y[test], pred, average="macro")),
            "recall_64": float(recalls[0]), "recall_128": float(recalls[1]),
            "recall_256": float(recalls[2]), "confusion_matrix": cm.tolist()}


def diagnostics(cache):
    X = cache["X_stats"].astype(np.float64)
    y = cache["y"].astype(int)
    source = cache["sources"].astype(str)
    split = cache["split"].astype(str)
    high = np.isin(y, [3, 4, 5])
    a_train = np.where(high & (source == "A") & (split == "train"))[0]
    a_test = np.where(high & (source == "A") & (split == "test"))[0]
    b_train = np.where(high & (source == "B") & (split == "finetune"))[0]
    b_test = np.where(high & (source == "B") & (split == "test"))[0]
    return [
        fit_eval(X, y, a_train, a_test, "A-train -> A-test"),
        fit_eval(X, y, a_train, b_test, "A-train -> B-test"),
        fit_eval(X, y, b_train, b_test, "B-finetune-pool -> B-test"),
        fit_eval(X, y, b_train, a_test, "B-finetune-pool -> A-test"),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FAST / "ab_domain_split.csv")
    selected = df[df.mod_name.isin(MODS)].copy()
    cache = np.load(FAST / "real_cache_ab.npz", allow_pickle=True)
    result = {
        "label_map": EXPECTED,
        "audit": audit(selected),
        "constellation_summary": plot_constellations(selected),
        "three_class_stats_svm": diagnostics(cache),
        "scope": "Diagnostic SVM on the existing 48-D statistics; not the paper's CDM-Stats-Radial network.",
    }
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
