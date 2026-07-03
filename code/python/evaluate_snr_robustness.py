"""
evaluate_snr_robustness.py — SNR 鲁棒性评估 (支持 CNN / ViT / 联合模型)
======================================================================
输出:
  1. snr_robustness_curves.png          — SNR 鲁棒性对比曲线
  2. confusion_global.png               — 全局混淆矩阵
  3. confusion_turbulence.png            — 湍流分组混淆矩阵

用法: python evaluate_snr_robustness.py
"""

import os, sys, numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, SCRIPT_DIR)

from dataset import FSODataset, MODULATION_NAMES, TURBULENCE_GROUPS
from models import ConstellationCNN, ConstellationUNet, ConstellationViT

# ==================== 路径 ====================
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
MARKERS = ['o', 's', 'D', '^', 'v']
SNR_RANGE = list(range(-6, 21, 2))
BATCH_SIZE = 128

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 模型加载 ====================
def load_pipeline(mode="cascade", ckpt_dir=None):
    """
    mode="cascade": U-Net + CNN (Stage 1+2)
    mode="e2e":     U-Net + ViT 联合微调 (Stage 3)
    mode="baseline": 仅 CNN
    ckpt_dir: checkpoint 目录 (None 则使用全局 CKPT_DIR)
    """
    if ckpt_dir is None:
        ckpt_dir = CKPT_DIR

    if mode == "e2e":
        unet = ConstellationUNet(1, 1, True).to(DEVICE)
        vit = ConstellationViT().to(DEVICE)
        path = os.path.join(ckpt_dir, "e2e_best.pth")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        vit.load_state_dict(ckpt["vit_state_dict"])
        unet.eval(); vit.eval()
        print(f"[模型] E2E (U-Net + ViT): {path}")
        return unet, vit, "e2e"

    elif mode == "cascade":
        unet = ConstellationUNet(1, 1, True).to(DEVICE)
        cnn = ConstellationCNN(NUM_CLASSES).to(DEVICE)
        p1 = os.path.join(ckpt_dir, "cnn_ideal_best.pth")
        p2 = os.path.join(ckpt_dir, "gnn_denoiser_best.pth")
        cnn.load_state_dict(torch.load(p1, map_location=DEVICE, weights_only=False)["model_state_dict"])
        unet.load_state_dict(torch.load(p2, map_location=DEVICE, weights_only=False)["model_state_dict"])
        unet.eval(); cnn.eval()
        print(f"[模型] Cascade (U-Net + CNN)")
        return unet, cnn, "cascade"

    elif mode == "baseline":
        cnn = ConstellationCNN(NUM_CLASSES).to(DEVICE)
        p1 = os.path.join(ckpt_dir, "cnn_ideal_best.pth")
        cnn.load_state_dict(torch.load(p1, map_location=DEVICE, weights_only=False)["model_state_dict"])
        cnn.eval()
        print(f"[模型] Baseline (CNN only)")
        return None, cnn, "baseline"


# ==================== 推理 ====================
@torch.no_grad()
def evaluate(models, loader):
    unet, cls, mode = models
    correct_base = defaultdict(lambda: defaultdict(int))
    total = defaultdict(lambda: defaultdict(int))
    correct_turb = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}
    total_turb = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}
    all_preds, all_labels, all_turb = [], [], []

    # 预解析湍流
    file_turb_map = []
    for f in loader.dataset.dataset.files if hasattr(loader.dataset, "dataset") else loader.dataset.files:
        fn = os.path.basename(f)
        t = next((t for t in TURBULENCE_GROUPS if t in fn), None)
        file_turb_map.append(t)
    if not file_turb_map:
        file_turb_map = [None] * len(loader.dataset)

    sample_idx = 0
    for distorted, ideal, labels, snr_values in loader:
        distorted = distorted.to(DEVICE)
        labels_g = labels.to(DEVICE)
        B = distorted.size(0)

        if mode == "baseline":
            logits = cls(distorted)
        elif mode == "cascade":
            repaired = unet(distorted)
            logits = cls(repaired)
        elif mode == "e2e":
            repaired = unet(distorted)
            logits = cls(repaired)

        preds = logits.argmax(dim=1)

        for i in range(B):
            if sample_idx >= len(file_turb_map): break
            s = int(snr_values[i].item())
            lbl = labels_g[i].item()
            p = preds[i].item()
            turb = file_turb_map[sample_idx]
            sample_idx += 1

            total[s][lbl] += 1
            if p == lbl: correct_base[s][lbl] += 1
            all_preds.append(p)
            all_labels.append(lbl)
            if turb:
                total_turb[turb][s][lbl] += 1
                if p == lbl: correct_turb[turb][s][lbl] += 1

    # 计算准确率
    acc_all = np.full((len(SNR_RANGE), NUM_CLASSES), np.nan)
    for i, snr in enumerate(SNR_RANGE):
        for cls in range(NUM_CLASSES):
            t = total[snr].get(cls, 0)
            if t > 0:
                acc_all[i, cls] = correct_base[snr].get(cls, 0) / t * 100

    return acc_all, all_preds, all_labels, correct_turb, total_turb


# ==================== 绘图 ====================
def plot_curves(acc_cascade, acc_e2e, acc_baseline, label_e2e):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if acc_baseline is not None:
        for j in range(NUM_CLASSES):
            axes[0].plot(SNR_RANGE, acc_baseline[:, j], color=COLORS[j], marker=MARKERS[j],
                         markersize=6, linewidth=1.8, label=CLASS_NAMES[j])
        axes[0].set_title("Baseline (CNN Only)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(fontsize=9, loc='lower right')

    if acc_e2e is not None:
        for j in range(NUM_CLASSES):
            axes[1].plot(SNR_RANGE, acc_e2e[:, j], color=COLORS[j], marker=MARKERS[j],
                         markersize=6, linewidth=1.8, label=CLASS_NAMES[j])
        axes[1].set_title(f"Proposed ({label_e2e})")
    elif acc_cascade is not None:
        for j in range(NUM_CLASSES):
            axes[1].plot(SNR_RANGE, acc_cascade[:, j], color=COLORS[j], marker=MARKERS[j],
                         markersize=6, linewidth=1.8, label=CLASS_NAMES[j])
        axes[1].set_title("Proposed (U-Net + CNN)")

    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9, loc='lower right')

    for ax in axes:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy (%)")

    fig.suptitle("FSO Modulation Recognition: SNR Robustness", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cm(all_preds, all_labels, title, fn):
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title} — Acc: {acc:.2f}%")
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/{fn}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] {fn}")
    return acc


# ==================== 主流程 ====================
def main():
    print(f"[设备] {DEVICE}")
    print(f"[数据] {DATA_DIR}")

    dataset = FSODataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}

    # --- Baseline ---
    print("\n--- Baseline (CNN only) ---")
    try:
        models = load_pipeline("baseline")
        acc, preds, labels, ct, tt = evaluate(models, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))
        results["baseline"] = (acc, preds, labels, ct, tt)
        plot_cm(preds, labels, "Baseline CNN", "confusion_global_baseline.png")
    except Exception as e:
        print(f"[跳过] Baseline: {e}")
        results["baseline"] = None

    # --- Cascade (U-Net + CNN) ---
    print("\n--- Cascade (U-Net + CNN) ---")
    try:
        models = load_pipeline("cascade")
        acc, preds, labels, ct, tt = evaluate(models, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))
        results["cascade"] = (acc, preds, labels, ct, tt)
        plot_cm(preds, labels, "U-Net + CNN Cascade", "confusion_global_cascade.png")
    except Exception as e:
        print(f"[跳过] Cascade: {e}")
        results["cascade"] = None

    # --- E2E (U-Net + ViT) ---
    print("\n--- E2E (U-Net + ViT) ---")
    try:
        models = load_pipeline("e2e")
        acc, preds, labels, ct, tt = evaluate(models, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))
        results["e2e"] = (acc, preds, labels, ct, tt)
        plot_cm(preds, labels, "U-Net + ViT (E2E)", "confusion_global_e2e.png")
    except Exception as e:
        print(f"[跳过] E2E: {e}")
        results["e2e"] = None

    # --- SNR 对比曲线 ---
    print("\n[绘图] SNR 鲁棒性曲线...")
    acc_b = results["baseline"][0] if results["baseline"] else None
    acc_c = results["cascade"][0] if results["cascade"] else None
    acc_e = results["e2e"][0] if results["e2e"] else None

    fig = plot_curves(acc_c, acc_e, acc_b, "U-Net + ViT (E2E)")
    fig.savefig(f"{SAVE_DIR}/snr_robustness_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] snr_robustness_curves.png")

    print("\n" + "=" * 60)
    print("  评估完成!")
    print(f"  输出目录: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
