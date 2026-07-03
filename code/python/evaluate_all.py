"""
evaluate_all.py — 统一评估脚本
===============================
输出:
  1. waterfall_accuracy_per_mod.png    — SNR 瀑布曲线 (Baseline vs Proposed)
  2. confusion_groups_gnn.png          — GNN+CNN 湍流分组混淆矩阵
  3. confusion_groups_baseline.png     — Baseline CNN 湍流分组混淆矩阵
  4. confusion_global_gnn.png          — GNN+CNN 全局混淆矩阵
  5. confusion_global_baseline.png     — Baseline CNN 全局混淆矩阵
  6. repair_visualization.png          — 畸变→修复→理想 对比可视化

用法: python evaluate_all.py data/raw/
"""

import os, sys, numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dataset import FSODataset, LABEL_MAPPING, MODULATION_NAMES, TURBULENCE_GROUPS
from models import ConstellationCNN, ConstellationUNet

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
MARKERS = ['o', 's', 'D', '^', 'v']
SNR_RANGE = list(range(-6, 21, 2))
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "figures")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")
CKPT_CNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "checkpoints", "cnn_ideal_best.pth")
CKPT_GNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "checkpoints", "gnn_denoiser_best.pth")
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 模型加载 ====================
def load_model(model, ckpt_path, name):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    print(f"[加载] {name}: {ckpt_path}")
    return model


print(f"[设备] {DEVICE}")
print(f"[数据] {DATA_DIR}")

dataset = FSODataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

cnn = load_model(ConstellationCNN(num_classes=NUM_CLASSES), CKPT_CNN, "CNN")
gnn = load_model(ConstellationUNet(in_channels=1, out_channels=1), CKPT_GNN, "U-Net")

# ==================== 1. 推理遍历 ====================
print("\n[推理] 收集所有样本预测...")

correct_baseline = defaultdict(int)
total_baseline = defaultdict(int)
correct_proposed = defaultdict(int)
total_proposed = defaultdict(int)

turb_cm_baseline = {g: ([], []) for g in TURBULENCE_GROUPS}
turb_cm_proposed = {g: ([], []) for g in TURBULENCE_GROUPS}
visual_samples = {"distorted": [], "repaired": [], "ideal": [], "labels": []}
all_preds_gnn, all_labels = [], []

@torch.no_grad()
def run():
    global all_preds_gnn, all_labels
    for distorted, ideal, labels, snr_values in loader:
        B = distorted.size(0)
        distorted = distorted.to(DEVICE)
        labels = labels.to(DEVICE)

        # Baseline: CNN direct
        logits_cnn = cnn(distorted)
        preds_cnn = logits_cnn.argmax(dim=1)

        # Proposed: U-Net + CNN
        repaired = gnn(distorted)
        logits_rep = cnn(repaired)
        preds_rep = logits_rep.argmax(dim=1)

        # === Waterfall stats ===
        for i in range(B):
            s = int(snr_values[i].item())
            lbl = int(labels[i].item())
            total_baseline[(s, lbl)] += 1
            total_proposed[(s, lbl)] += 1
            if preds_cnn[i].item() == lbl:
                correct_baseline[(s, lbl)] += 1
            if preds_rep[i].item() == lbl:
                correct_proposed[(s, lbl)] += 1

        # === Turbulence-group confusion matrix ===
        for i in range(B):
            turb = dataset._parse_filename(dataset.files[0])  # hack: need per-sample turbulence
            # Actually need to get turbulence from filename
        # (handled separately below)

        # === Global confusion matrix ===
        all_preds_gnn.extend(preds_rep.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # === Visual samples ===
        for i in range(B):
            lbl = labels[i].item()
            if len(visual_samples["distorted"]) < 9:
                if lbl in [0, 2, 4]:
                    visual_samples["distorted"].append(distorted[i].cpu())
                    visual_samples["repaired"].append(repaired[i].cpu())
                    visual_samples["ideal"].append(ideal[i])
                    visual_samples["labels"].append(lbl)

run()

# ==================== 2. Waterfall Plot ====================
print("[绘图] 1/5 瀑布曲线...")
acc_b = np.full((len(SNR_RANGE), NUM_CLASSES), np.nan)
acc_p = np.full((len(SNR_RANGE), NUM_CLASSES), np.nan)
for i, snr in enumerate(SNR_RANGE):
    for j in range(NUM_CLASSES):
        t = total_baseline.get((snr, j), 0)
        if t > 0:
            acc_b[i, j] = correct_baseline.get((snr, j), 0) / t * 100
            acc_p[i, j] = correct_proposed.get((snr, j), 0) / t * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, acc, title in zip(axes, [acc_b, acc_p], ["Baseline (CNN)", "Proposed (U-Net + CNN)"]):
    for j in range(NUM_CLASSES):
        ax.plot(SNR_RANGE, acc[:, j], color=COLORS[j], marker=MARKERS[j],
                markersize=6, linewidth=1.8, label=CLASS_NAMES[j])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(title); ax.set_ylim(0, 105); ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9, loc='lower right')
fig.suptitle("FSO Modulation Recognition Waterfall Curve", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/waterfall_accuracy_per_mod.png", dpi=200, bbox_inches="tight")
plt.close()

# ==================== 3. Global Confusion Matrix ====================
print("[绘图] 2/5 + 3/5 全局混淆矩阵...")
global_cm_gnn = confusion_matrix(all_labels, all_preds_gnn)

# 使用同一个 U-Net 生成 Baseline 预测
all_preds_baseline = []
for distorted, ideal, labels, snr_values in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    distorted = distorted.to(DEVICE)
    preds = cnn(distorted).argmax(dim=1)
    all_preds_baseline.extend(preds.cpu().tolist())

global_cm_baseline = confusion_matrix(all_labels, all_preds_baseline)

for title, cm, fn in [
    ("GNN+CNN", global_cm_gnn, "confusion_global_gnn.png"),
    ("Baseline CNN", global_cm_baseline, "confusion_global_baseline.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/{fn}", dpi=200, bbox_inches="tight")
    plt.close()

# ==================== 4. Turbulence-Grouped Confusion Matrices ====================
print("[绘图] 4/5 + 5/5 湍流分组混淆矩阵...")

@torch.no_grad()
def get_group_preds(use_gnn=True):
    results = {g: ([], []) for g in TURBULENCE_GROUPS}
    for distorted, ideal, labels, snr_values in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
        distorted = distorted.to(DEVICE)
        labels = labels.to(DEVICE)
        if use_gnn:
            repaired = gnn(distorted)
            preds = cnn(repaired).argmax(dim=1)
        else:
            preds = cnn(distorted).argmax(dim=1)

        for i in range(distorted.size(0)):
            # Parse turbulence from filename
            idx = len(results["weak"][0]) + len(results["moderate"][0]) + len(results["strong"][0])
            if idx >= len(dataset.files):
                break
            fname = os.path.basename(dataset.files[idx])
            for turb in TURBULENCE_GROUPS:
                if turb in fname:
                    results[turb][0].append(preds[i].item())
                    results[turb][1].append(labels[i].item())
                    break
        if sum(len(v[0]) for v in results.values()) >= len(dataset):
            break
    return results

# Use a simpler approach for turbulence groups
group_counts = {t: len([f for f in dataset.files if t in os.path.basename(f)]) for t in TURBULENCE_GROUPS}
print(f"  湍流分布: {group_counts}")

# Re-evaluate with per-turbulence tracking (使用 shuffle=False 保证索引一致)
grp_gnn = {g: {"preds": [], "labels": []} for g in TURBULENCE_GROUPS}
grp_cnn = {g: {"preds": [], "labels": []} for g in TURBULENCE_GROUPS}

# 新增: 按湍流 + SNR 的准确率统计 (用于湍流SNR曲线)
turb_snr_base = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}  # correct
turb_snr_base_t = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}  # total
turb_snr_prop = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}
turb_snr_prop_t = {g: defaultdict(lambda: defaultdict(int)) for g in TURBULENCE_GROUPS}

# 预先解析所有文件名的湍流类型, 与 DataLoader 顺序一致 (shuffle=False)
file_turb_map = []
for f in dataset.files:
    fn = os.path.basename(f)
    turb = next((t for t in TURBULENCE_GROUPS if t in fn), None)
    file_turb_map.append(turb)

turb_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
sample_idx = 0
for distorted, ideal, labels, snr_values in turb_loader:
    distorted = distorted.to(DEVICE)
    repaired = gnn(distorted)
    preds_gnn = cnn(repaired).argmax(dim=1).cpu().tolist()
    preds_cnn = cnn(distorted).argmax(dim=1).cpu().tolist()
    labels_cpu = labels.tolist()

    for i in range(distorted.size(0)):
        if sample_idx >= len(file_turb_map): break
        turb = file_turb_map[sample_idx]
        snr = int(snr_values[i].item())
        lbl = labels_cpu[i]
        sample_idx += 1
        if turb is None: continue
        grp_gnn[turb]["preds"].append(preds_gnn[i])
        grp_gnn[turb]["labels"].append(lbl)
        grp_cnn[turb]["preds"].append(preds_cnn[i])
        grp_cnn[turb]["labels"].append(lbl)

        # 统计按湍流+SNR的准确率
        turb_snr_base_t[turb][snr][lbl] += 1
        turb_snr_prop_t[turb][snr][lbl] += 1
        if preds_cnn[i] == lbl:
            turb_snr_base[turb][snr][lbl] += 1
        if preds_gnn[i] == lbl:
            turb_snr_prop[turb][snr][lbl] += 1

    if sample_idx >= len(file_turb_map):
        break

for data, title_prefix, fn in [
    (grp_gnn, "GNN+CNN", "confusion_groups_gnn.png"),
    (grp_cnn, "Baseline CNN", "confusion_groups_baseline.png"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    turbs = ["weak", "moderate", "strong"]
    titles = ["Weak", "Moderate", "Strong"]
    for ax, turb, ttl in zip(axes, turbs, titles):
        total = len(data[turb]["preds"])
        cm = confusion_matrix(data[turb]["labels"], data[turb]["preds"], labels=list(range(NUM_CLASSES)))
        acc = np.trace(cm) / max(cm.sum(), 1) * 100
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(f"{ttl}\nAcc: {acc:.1f}%")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"{title_prefix} — Per-Turbulence Confusion Matrices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/{fn}", dpi=200, bbox_inches="tight")
    plt.close()

# ==================== 5. Per-Turbulence SNR Curves ====================
print("[绘图] 5/7 湍流分组SNR曲线...")
turb_names = {"weak": "Weak", "moderate": "Moderate", "strong": "Strong"}
for g_idx, turb in enumerate(TURBULENCE_GROUPS):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, prefix in [(axes[0], turb_snr_base[turb], "Baseline"),
                              (axes[1], turb_snr_prop[turb], "Proposed")]:
        total_d = turb_snr_base_t[turb] if prefix == "Baseline" else turb_snr_prop_t[turb]
        for cls in range(NUM_CLASSES):
            snr_vals, acc_vals = [], []
            for snr in sorted(total_d.keys()):
                t = total_d[snr].get(cls, 0)
                c = data[snr].get(cls, 0)
                if t > 0:
                    snr_vals.append(snr)
                    acc_vals.append(c / t * 100)
            if snr_vals:
                ax.plot(snr_vals, acc_vals, color=COLORS[cls], marker=MARKERS[cls],
                        markersize=6, linewidth=1.8, label=CLASS_NAMES[cls])
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{prefix} - {turb_names[turb]}")
        ax.set_ylim(0, 105); ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(f"{turb_names[turb]} - SNR Curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/snr_curves_{turb}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] snr_curves_{turb}.png")

# ==================== 6. Repair Visualization ====================
print("[绘图] 6/6 修复可视化对比...")
vis_distorted = torch.stack(visual_samples["distorted"][:6])
vis_repaired = torch.stack(visual_samples["repaired"][:6])
vis_ideal = torch.stack(visual_samples["ideal"][:6])
vis_labels = visual_samples["labels"][:6]

fig, axes = plt.subplots(len(vis_labels), 3, figsize=(9, 2.5 * len(vis_labels)))
for i in range(len(vis_labels)):
    for j, (img, ttl) in enumerate([(vis_distorted[i], "Distorted"),
                                     (vis_repaired[i], "Repaired"),
                                     (vis_ideal[i], "Ideal")]):
        ax = axes[i, j] if len(vis_labels) > 1 else axes[j]
        im = ax.imshow(img.squeeze(0).numpy(), cmap='inferno', origin='upper')
        ax.set_title(f"{CLASS_NAMES[vis_labels[i]]} - {ttl}" if j == 0 else ttl)
        ax.axis('off')
fig.suptitle("Cascade Repair Visualization", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/repair_visualization.png", dpi=200, bbox_inches="tight")
plt.close()

# ==================== 打印汇总统计 ====================
print()
print("[Debug] 各 SNR x 调制格式 测试样本数:")
all_snrs = sorted(set(k[0] for k in total_baseline.keys()))
for snr in all_snrs:
    for cls in range(NUM_CLASSES):
        cnt = total_baseline.get((snr, cls), 0)
        mark = " <-- WARNING!" if cnt < 10 else ""
        print(f"  SNR {snr:>3d} dB  {CLASS_NAMES[cls]:<6s}: {cnt:>5d}{mark}")

print(f"\n{'='*60}")
print(f"  评估完成! 共 6 张图保存至 {SAVE_DIR}/")
print(f"{'='*60}")
print(f"  1. waterfall_accuracy_per_mod.png")
print(f"  2. confusion_global_gnn.png")
print(f"  3. confusion_global_baseline.png")
print(f"  4. confusion_groups_gnn.png")
print(f"  5. confusion_groups_baseline.png")
print(f"  6. repair_visualization.png")
print(f"{'='*60}")
