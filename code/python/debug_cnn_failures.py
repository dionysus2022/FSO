"""
debug_cnn_failures.py — Baseline CNN "错题本" 分析
===================================================
目的: 找出"弱湍流下预测错误、强湍流下预测正确"的样本,
      可视化对比，定位 CNN 失效的视觉原因。

用法: python debug_cnn_failures.py
输出: results/figures/cnn_failure_cases.png
"""

import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

# ==================== 路径配置 (从脚本位置推算) ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, SCRIPT_DIR)

from dataset import FSODataset, MODULATION_NAMES
from models import ConstellationCNN

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CKPT_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "cnn_ideal_best.pth")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
SAVE_PATH = os.path.join(SAVE_DIR, "cnn_failure_cases.png")

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
BATCH_SIZE = 128

# ==================== 加载模型 ====================
print("=" * 60)
print("  debug_cnn_failures.py — Baseline CNN 错题分析")
print("=" * 60)
print(f"[设备] {DEVICE}")

model = ConstellationCNN(num_classes=NUM_CLASSES)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print(f"[模型] 已加载: {CKPT_PATH}")

# ==================== 加载数据 ====================
print(f"[数据] {DATA_DIR}")
dataset = FSODataset(DATA_DIR)

# ==================== 推理并找"错题" ====================
# 收集: 每个样本的 (turb, modulation, 是否基线正确, 是否级联正确, SNR, 文件路径, distorted)
print("[推理] 收集错题样本...")

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

fail_weak_correct_strong = []  # 弱湍流预测错, 强湍流预测对

all_results = []

@torch.no_grad()
def run():
    for batch_idx, (distorted, ideal, labels, snr_values) in enumerate(loader):
        distorted = distorted.to(DEVICE)
        labels_gpu = labels.to(DEVICE)

        logits = model(distorted)
        preds = logits.argmax(dim=1)

        for i in range(distorted.size(0)):
            global_idx = batch_idx * BATCH_SIZE + i
            if global_idx >= len(dataset.files): break

            fname = os.path.basename(dataset.files[global_idx])
            from dataset import FILENAME_PATTERN
            m = FILENAME_PATTERN.search(fname)
            if not m: continue

            turb = m.group(1)
            mod = int(m.group(5))
            true_label = labels[i].item()
            pred_label = preds[i].item()
            snr = int(snr_values[i].item())

            all_results.append({
                "turb": turb, "mod": mod, "true": true_label,
                "pred": pred_label, "snr": snr,
                "correct": pred_label == true_label,
                "distorted": distorted[i].cpu().clone(),
                "fname": fname,
                "ideal": ideal[i],
            })

    print(f"[完成] 共查询 {len(all_results)} 个样本")

run()

# ==================== 筛选错题 ====================
# 条件: 弱湍流下预测错误
weak_failures = [r for r in all_results if r["turb"] == "weak" and not r["correct"]]
print(f"\n弱湍流错误样本: {len(weak_failures)}")

# 找一个强湍流的对应样本 (同调制 + 同SNR + 预测正确)
for wf in weak_failures:
    matches = [r for r in all_results
               if r["turb"] == "strong"
               and r["mod"] == wf["mod"]
               and r["snr"] == wf["snr"]
               and r["correct"]]
    if matches:
        fail_weak_correct_strong.append((wf, random.choice(matches)))
    if len(fail_weak_correct_strong) >= 9:
        break

print(f"找到错题配对 (weak 错误, strong 正确): {len(fail_weak_correct_strong)} 对")

if len(fail_weak_correct_strong) == 0:
    print("[警告] 未找到足够的错题配对，扩大搜索...")
    for wf in weak_failures:
        matches = [r for r in all_results
                   if r["turb"] != "weak"
                   and r["mod"] == wf["mod"]
                   and r["correct"]]
        if matches:
            fail_weak_correct_strong.append((wf, random.choice(matches)))
        if len(fail_weak_correct_strong) >= 6:
            break

# ==================== 可视化 ====================
N = min(6, len(fail_weak_correct_strong))
if N == 0:
    print("[退出] 无错题样本可展示")
    sys.exit(1)

print(f"\n[绘图] {N} 组错题对比...")
fig, axes = plt.subplots(N, 4, figsize=(14, 2.8 * N))

for row in range(N):
    wf, sf = fail_weak_correct_strong[row]

    # Column 0: Weak distorted (错误)
    ax0 = axes[row, 0] if N > 1 else axes[0]
    ax0.imshow(wf["distorted"].squeeze(0).numpy(), cmap='inferno', origin='upper')
    ax0.set_title(f"Weak | True:{CLASS_NAMES[wf['true']]} Pred:{CLASS_NAMES[wf['pred']]} SNR:{wf['snr']}dB", fontsize=9)
    ax0.axis('off')

    # Column 1: Weak ideal
    ax1 = axes[row, 1] if N > 1 else axes[1]
    ax1.imshow(wf["ideal"].squeeze(0).numpy(), cmap='inferno', origin='upper')
    ax1.set_title(f"Ideal: {CLASS_NAMES[wf['true']]}", fontsize=9)
    ax1.axis('off')

    # Column 2: Strong distorted (正确)
    ax2 = axes[row, 2] if N > 1 else axes[2]
    ax2.imshow(sf["distorted"].squeeze(0).numpy(), cmap='inferno', origin='upper')
    ax2.set_title(f"Strong | True:{CLASS_NAMES[sf['true']]} Pred:{CLASS_NAMES[sf['pred']]} SNR:{sf['snr']}dB", fontsize=9)
    ax2.axis('off')

    # Column 3: Strong ideal
    ax3 = axes[row, 3] if N > 1 else axes[3]
    ax3.imshow(sf["ideal"].squeeze(0).numpy(), cmap='inferno', origin='upper')
    ax3.set_title(f"Ideal: {CLASS_NAMES[sf['true']]}", fontsize=9)
    ax3.axis('off')

fig.suptitle("Baseline CNN Failure Analysis: Weak (Wrong) vs Strong (Correct)", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
print(f"[保存] {SAVE_PATH}")
plt.close()

# ==================== 统计打印 ====================
print("\n" + "=" * 60)
print("  [错题分布] Baseline CNN 弱湍流误分类统计")
print("=" * 60)

error_by_mod = defaultdict(int)
for r in all_results:
    if r["turb"] == "weak" and not r["correct"]:
        conf = (r["true"], r["pred"])
        error_by_mod[conf] += 1

if error_by_mod:
    print(f"  {'True→Pred':<18}  Count")
    print(f"  {'-'*18}  {'-'*6}")
    sorted_errors = sorted(error_by_mod.items(), key=lambda x: -x[1])
    for (t, p), c in sorted_errors[:10]:
        print(f"  {CLASS_NAMES[t]+'→'+CLASS_NAMES[p]:<18}  {c}")

print(f"\n  弱湍流总错误: {len(weak_failures)} / {sum(1 for r in all_results if r['turb']=='weak')}")
if sum(1 for r in all_results if r['turb'] == 'weak') > 0:
    print(f"  弱湍流准确率: {(1 - len(weak_failures) / sum(1 for r in all_results if r['turb']=='weak')) * 100:.1f}%")
print("=" * 60)
