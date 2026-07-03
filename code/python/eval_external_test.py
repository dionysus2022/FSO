#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_external_test.py

外部测试评估脚本：
  - 加载 Dataset-A 训练好的模型 (5 seeds)
  - 在 Dataset-B (独立采集数据) 上测试
  - 报告 5-seed mean±std

用法:
  python eval_external_test.py \
    --data-root "D:\...\preprocessed_uniform_qam_rx_lowmem\2026.06.29\eq" \
    --model-root "D:\...\recognition_mrf_lowmem" \
    --seeds 2026 2027 2028 2029 2030 \
    --out-dir "D:\...\results\external_test"
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import scipy.io as sio
except Exception:
    raise RuntimeError("需要 scipy: pip install scipy")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

# 从 train_mrf_lowmem 导入模型定义和工具函数
sys.path.insert(0, str(Path(__file__).parent))
from train_mrf_lowmem import (
    MOD_NAMES,
    MOD_TO_LABEL,
    LABEL_TO_MOD,
    load_sample_from_mat,
    MultiRepNet,
    make_cdm_from_rx_sc,
    make_blind_stats,
    rx_sc_to_iq_channels,
)

# ========================= 数据加载 =========================

def load_external_dataset(data_root: Path, turb_name: str) -> Dict[str, np.ndarray]:
    """
    从预处理输出目录加载外部测试数据。
    不依赖 manifest，直接扫描 .mat 文件。
    """
    cdm_list, stats_list, y_list, path_list = [], [], [], []

    for mod_name in MOD_NAMES:
        mod_dir = data_root / mod_name / turb_name
        if not mod_dir.exists():
            print(f"  [skip] {mod_name}/{turb_name} 目录不存在: {mod_dir}")
            continue

        mat_files = sorted(mod_dir.rglob("frame_*.mat"))
        print(f"  {mod_name}/{turb_name}: {len(mat_files)} frames")

        for mat_path in mat_files:
            try:
                s = load_sample_from_mat(mat_path, use_iq=False)
                if s["label"] < 0:
                    # 从路径推断 label
                    label = MOD_TO_LABEL.get(mod_name, -1)
                    if label < 0:
                        continue
                    s["label"] = label

                cdm_list.append(s["cdm"])
                stats_list.append(s["stats"])
                y_list.append(s["label"])
                path_list.append(str(mat_path))
            except Exception as e:
                print(f"    [skip] {mat_path.name}: {e}")

    if len(y_list) == 0:
        raise RuntimeError(f"没有加载到任何样本 for turb={turb_name}")

    X_cdm = np.stack(cdm_list, axis=0).astype(np.float32)
    X_stats = np.stack(stats_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    print(f"  总计: {len(y)} samples, X_cdm={X_cdm.shape}, X_stats={X_stats.shape}")
    print(f"  类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return {
        "X_cdm": X_cdm,
        "X_stats": X_stats,
        "y": y,
        "paths": path_list,
    }


def normalize_stats(X_stats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """用训练集的 mean/std 归一化统计特征"""
    X = X_stats.copy().astype(np.float32)
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1.0

    # 处理 NaN
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(mean, inds[1])

    X = (X - mean) / std_safe
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


class ExtDataset(Dataset):
    def __init__(self, X_cdm, X_stats, y):
        self.X_cdm = X_cdm
        self.X_stats = X_stats
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_cdm[idx]),
            torch.zeros(1, 1, dtype=torch.float32),  # dummy IQ
            torch.from_numpy(self.X_stats[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ========================= 模型加载 =========================

def load_model_and_stats(model_dir: Path, device: torch.device):
    """加载模型 checkpoint 和归一化参数"""
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    stats_mean = np.asarray(ckpt["stats_mean"], dtype=np.float32).reshape(-1)
    stats_std = np.asarray(ckpt["stats_std"], dtype=np.float32).reshape(-1)
    stats_std[stats_std < 1e-6] = 1.0

    args = ckpt.get("args", {})
    model_mode = str(ckpt.get("model_mode", args.get("model", "cdm_stats")))
    dropout = float(args.get("dropout", 0.25))
    stats_dim = int(ckpt.get("stats_dim", len(stats_mean)))
    turb_name = str(ckpt.get("turb_name", ""))

    model = MultiRepNet(
        model_mode=model_mode,
        num_classes=6,
        iq_in_ch=5,
        stats_dim=stats_dim,
        dropout=dropout,
    ).to(device)

    state = ckpt["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # IQ 分支权重会被 missing，这是正常的（cdm_stats 模式不用 IQ）
        real_missing = [k for k in missing if "iq_branch" not in k]
        if real_missing:
            print(f"  [WARN] missing keys (non-IQ): {real_missing[:5]}")
        if unexpected:
            print(f"  [WARN] unexpected keys: {list(unexpected)[:5]}")

    model.eval()

    return model, stats_mean, stats_std, model_mode, turb_name


# ========================= 评估 =========================

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for cdm, iq, stats, y in loader:
        cdm = cdm.to(device, non_blocking=True)
        iq = iq.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        logits = model(cdm, iq, stats)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())

    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def compute_metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    if SKLEARN_OK:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    else:
        bal_acc = acc
        f1_macro = acc
        cm = np.zeros((6, 6), dtype=np.int64)
        for a, b in zip(y_true, y_pred):
            cm[int(a), int(b)] += 1

    # 每类准确率
    per_class_acc = {}
    for i, name in enumerate(MOD_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[name] = float((y_pred[mask] == i).mean())
        else:
            per_class_acc[name] = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "num_samples": int(len(y_true)),
    }


def save_confusion_matrix_plot(cm, labels, path: Path, title: str):
    if not MPL_OK:
        return
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ========================= 主流程 =========================

def main():
    ap = argparse.ArgumentParser(description="外部测试评估 (Dataset-A 模型 → Dataset-B)")
    ap.add_argument("--data-root", required=True,
                    help="Dataset-B 预处理输出根目录 (含 eq/ 子目录)")
    ap.add_argument("--model-root", required=True,
                    help="recognition_mrf_lowmem 根目录")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据目录: data_root/eq/ 或 data_root (如果直接指向 eq)
    eq_root = Path(args.data_root)
    if (eq_root / "eq").exists():
        eq_root = eq_root / "eq"
    print(f"Dataset-B 数据目录: {eq_root}")

    # 加载外部测试数据 (weak + strong)
    print("\n=== 加载 Dataset-B (weak) ===")
    data_weak = load_external_dataset(eq_root, "weak")
    print("\n=== 加载 Dataset-B (strong) ===")
    data_strong = load_external_dataset(eq_root, "strong")

    # 评估每个 seed
    all_results = {"weak": [], "strong": []}

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"Seed = {seed}")
        print(f"{'='*70}")

        for turb_name, data in [("weak", data_weak), ("strong", data_strong)]:
            model_dir = Path(args.model_root) / f"seed_{seed}" / f"{turb_name}_cdm_stats"
            if not model_dir.exists():
                print(f"  [skip] 模型目录不存在: {model_dir}")
                continue

            print(f"\n  --- {turb_name} | model: {model_dir} ---")

            model, stats_mean, stats_std, model_mode, ckpt_turb = load_model_and_stats(model_dir, device)
            print(f"  model_mode={model_mode}, turb={ckpt_turb}")

            # 归一化 stats
            X_stats_norm = normalize_stats(data["X_stats"], stats_mean, stats_std)

            # DataLoader
            ds = ExtDataset(data["X_cdm"], X_stats_norm, data["y"])
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            y_true, y_pred, probs = evaluate_model(model, loader, device)
            metrics = compute_metrics(y_true, y_pred)

            print(f"  Acc={metrics['accuracy']:.4f} | BalAcc={metrics['balanced_accuracy']:.4f} | F1={metrics['f1_macro']:.4f}")
            print(f"  Per-class: {metrics['per_class_accuracy']}")

            # 保存混淆矩阵
            cm_path = out_dir / f"cm_seed{seed}_{turb_name}.png"
            save_confusion_matrix_plot(
                metrics["confusion_matrix"], MOD_NAMES, cm_path,
                f"External Test | Seed {seed} | {turb_name}\nAcc={metrics['accuracy']:.4f}"
            )

            # 保存预测
            pred_df = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
                "true_name": [LABEL_TO_MOD[int(x)] for x in y_true],
                "pred_name": [LABEL_TO_MOD[int(x)] for x in y_pred],
            })
            for k, name in enumerate(MOD_NAMES):
                pred_df[f"prob_{name}"] = probs[:, k]
            pred_df.to_csv(out_dir / f"predictions_seed{seed}_{turb_name}.csv", index=False, encoding="utf-8-sig")

            all_results[turb_name].append({
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "per_class_accuracy": metrics["per_class_accuracy"],
                "num_samples": metrics["num_samples"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            })

    # 汇总 5-seed 统计
    print(f"\n{'='*70}")
    print("外部测试汇总 (5-seed mean±std)")
    print(f"{'='*70}")

    summary = {}
    for turb_name, results in all_results.items():
        if not results:
            continue
        accs = [r["accuracy"] for r in results]
        bal_accs = [r["balanced_accuracy"] for r in results]
        f1s = [r["f1_macro"] for r in results]

        summary[turb_name] = {
            "n_seeds": len(results),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "balacc_mean": float(np.mean(bal_accs)),
            "balacc_std": float(np.std(bal_accs, ddof=1)) if len(bal_accs) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "per_seed": results,
        }

        print(f"\n  {turb_name}:")
        print(f"    Accuracy:          {np.mean(accs):.4f} ± {np.std(accs, ddof=1) if len(accs) > 1 else 0:.4f}")
        print(f"    Balanced Accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs, ddof=1) if len(bal_accs) > 1 else 0:.4f}")
        print(f"    Macro-F1:          {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.4f}")
        for r in results:
            print(f"      seed {r['seed']}: acc={r['accuracy']:.4f}, f1={r['f1_macro']:.4f}")

        # 每类准确率 mean±std
        print(f"\n  {turb_name} 各类准确率:")
        for mod_name in MOD_NAMES:
            vals = [r["per_class_accuracy"].get(mod_name, float("nan")) for r in results]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                print(f"    {mod_name:8s}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}")

    # 保存 JSON
    with open(out_dir / "external_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存文本报告
    with open(out_dir / "external_test_report.txt", "w", encoding="utf-8") as f:
        f.write("外部测试评估报告 (External Test Report)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: Dataset-A 训练, 5 seeds ({args.seeds})\n")
        f.write(f"数据: Dataset-B (2026.06.29 采集)\n")
        f.write(f"  sub02 = weak turbulence\n")
        f.write(f"  sub04 = strong turbulence\n\n")

        for turb_name, stats in summary.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Turbulence = {turb_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Accuracy:          {stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}\n")
            f.write(f"Balanced Accuracy: {stats['balacc_mean']:.4f} ± {stats['balacc_std']:.4f}\n")
            f.write(f"Macro-F1:          {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}\n")
            f.write(f"Seeds: {stats['n_seeds']}\n\n")

            f.write("Per-seed results:\n")
            for r in stats["per_seed"]:
                f.write(f"  seed {r['seed']}: acc={r['accuracy']:.4f}, balacc={r['balanced_accuracy']:.4f}, f1={r['f1_macro']:.4f}\n")

            f.write(f"\nPer-class accuracy (mean±std):\n")
            for mod_name in MOD_NAMES:
                vals = [r["per_class_accuracy"].get(mod_name, float("nan")) for r in stats["per_seed"]]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    mean_v = np.mean(vals)
                    std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    f.write(f"  {mod_name:8s}: {mean_v:.4f} ± {std_v:.4f}\n")

    print(f"\n结果已保存到: {out_dir}")
    print(f"  - external_test_summary.json")
    print(f"  - external_test_report.txt")
    print(f"  - cm_seed*_{{weak,strong}}.png")
    print(f"  - predictions_seed*_{{weak,strong}}.csv")


if __name__ == "__main__":
    main()
