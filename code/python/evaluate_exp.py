"""
evaluate_exp.py — 实验数据(exp_*)调制格式识别评估脚本
=========================================================
针对 train_exp_data.py 训练产生的 exp_* checkpoints，复现训练时的测试集划分，
为每个实验在 results/figures/<exp_name>/ 下生成:

  1. confusion_<exp>.png             — 5×5 混淆矩阵热力图
  2. per_class_accuracy_<exp>.png    — 5种调制格式识别率折线图

另在 figures 根目录生成多实验对比图:
  - exp_comparison_per_class.png     — 7个实验每类识别率叠加折线

用法:
  python evaluate_exp.py                       # 评估全部 7 个实验
  python evaluate_exp.py --exp_name exp_cnn    # 仅评估单个实验
  python evaluate_exp.py --no_comparison       # 跳过多实验对比图
"""

import os
import re
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as sio

# 项目内部导入
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E
from utils import get_device


# ================================================================
# 全局配置 (对齐 evaluate.py 风格)
# ================================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]   # 按比特数 1,2,4,6,8 升序
COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
MARKERS = ["o", "s", "D", "^", "v"]

# 项目根目录 (code/python -> 项目根)
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
FIG_ROOT = os.path.join(PROJECT_ROOT, "results", "figures")

# 全部需要评估的实验 (有 checkpoint、尚未可视化的)
ALL_EXPERIMENTS = [
    "exp_cnn",
    "exp_cnn_ft",
    "exp_vit",
    "exp_vit_ft",
    "exp_unet_resnet18",
    "exp_unet_resnet18_ft",
    "exp_weak_cnn",
]


# ================================================================
# 参数解析
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="exp_* 实验数据调制格式识别评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp_name", type=str, default=None,
                        help="单个实验名 (省略则评估全部 7 个)")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--no_comparison", action="store_true",
                        help="跳过多实验对比图")
    return parser.parse_args()


# ================================================================
# 实验数据集 (复现 train_exp_data.py 的 ExpDataset)
# ================================================================
class ExpDataset(Dataset):
    """加载 exp_frame_XXX_mod_Y[_win_Z].mat 文件"""
    FILENAME_PATTERN = re.compile(r'exp_frame_(\d+)_mod_(\d+)(?:_win_(\d+))?\.mat$')
    LABEL_MAPPING = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}

    def __init__(self, data_dir: str):
        self.files = []
        for f in glob.glob(os.path.join(data_dir, "*.mat")):
            m = self.FILENAME_PATTERN.search(os.path.basename(f))
            if m and int(m.group(2)) in self.LABEL_MAPPING:
                self.files.append(f)
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.files[idx])
        distorted = mat["Distorted_CDM"].astype(np.float32)
        ideal = mat["Ideal_CDM"].astype(np.float32)
        label = self.LABEL_MAPPING[int(np.squeeze(mat["Label_Bits"]))]
        distorted = torch.from_numpy(distorted).unsqueeze(0)
        ideal = torch.from_numpy(ideal).unsqueeze(0)
        return distorted, ideal, torch.tensor(label, dtype=torch.long)


# ================================================================
# 复现训练时的测试集划分 (与 train_exp_data.py:427-431 完全一致)
# ================================================================
def reproduce_test_indices(n_total: int, seed: int, split_ratio: float):
    """返回测试集在原始有序 files 列表中的索引 (后 split_ratio 之外的部分)。"""
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(n_total * split_ratio)
    return indices[n_train:]


# ================================================================
# 模型加载 (支持 3 种 ckpt 格式)
# ================================================================
def load_model_by_arch(exp_name, ckpt_path, device):
    """根据 model_arch 加载模型，返回 (model, forward_fn)。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    arch = config.get("model_arch", "cnn")

    def forward(x):
        return model(x)

    if arch == "cnn":
        model = ConstellationCNN(num_classes=5, dropout_rate=0.3, in_channels=1).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, forward, arch

    if arch == "unet_resnet18":
        model = UNet_ResNet18_E2E(num_classes=5, in_channels=1).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, forward, arch

    if arch == "vit":
        # 分离存储: unet_state_dict + vit_state_dict
        unet = ConstellationUNet(in_channels=1, out_channels=1, bilinear=True).to(device)
        vit = ConstellationViT(img_size=64, patch_size=16, in_channels=1, num_classes=5).to(device)
        unet.load_state_dict(ckpt["unet_state_dict"])
        vit.load_state_dict(ckpt["vit_state_dict"])
        unet.eval()
        vit.eval()

        def vit_forward(x):
            return vit(unet(x))

        return (unet, vit), vit_forward, arch

    raise ValueError(f"未知 model_arch: {arch}")


# ================================================================
# 推理
# ================================================================
@torch.no_grad()
def infer(model, forward_fn, loader, device):
    """返回 (all_preds, all_labels)。输入统一用 Distorted_CDM。"""
    all_preds, all_labels = [], []
    for distorted, ideal, labels in loader:
        distorted = distorted.to(device)
        logits = forward_fn(distorted)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def per_class_accuracy(preds, labels, num_classes=NUM_CLASSES):
    """返回每类识别率 (%) 列表。"""
    accs = []
    for c in range(num_classes):
        total = sum(1 for l in labels if l == c)
        correct = sum(1 for p, l in zip(preds, labels) if p == l == c)
        accs.append(correct / total * 100 if total > 0 else float("nan"))
    return accs


# ================================================================
# 绘图: 混淆矩阵
# ================================================================
def plot_confusion(preds, labels, exp_name, fig_dir):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    acc = np.trace(cm) / max(cm.sum(), 1) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{exp_name} — Acc: {acc:.2f}%")
    fig.tight_layout()

    fn = f"confusion_{exp_name}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")
    return acc


# ================================================================
# 绘图: 每类识别率折线 (X = 5 种调制格式)
# ================================================================
def plot_per_class_accuracy(accs, exp_name, fig_dir):
    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, accs, color="#2E86C1", marker="o", markersize=9,
            linewidth=2.2, label=exp_name)
    for xi, yi in zip(x, accs):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Modulation Format")
    ax.set_ylabel("Recognition Accuracy (%)")
    ax.set_title(f"{exp_name} — Per-Class Recognition Accuracy")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()

    fn = f"per_class_accuracy_{exp_name}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")


# ================================================================
# 绘图: 多实验对比 (每类识别率叠加)
# ================================================================
def plot_comparison(exp_results, fig_dir):
    """exp_results: {exp_name: [acc_per_class,...]}"""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(NUM_CLASSES)
    palette = plt.cm.tab10(np.linspace(0, 1, len(exp_results)))

    for i, (exp_name, accs) in enumerate(exp_results.items()):
        ax.plot(x, accs, color=palette[i], marker=MARKERS[i % len(MARKERS)],
                markersize=8, linewidth=2.0, label=exp_name)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Modulation Format")
    ax.set_ylabel("Recognition Accuracy (%)")
    ax.set_title("Experiment Comparison — Per-Class Recognition Accuracy")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    fig.tight_layout()

    fn = "exp_comparison_per_class.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  [对比图] {fn}")


# ================================================================
# 评估单个实验
# ================================================================
def evaluate_one(exp_name, args, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{exp_name}_best.pth")
    if not os.path.isfile(ckpt_path):
        print(f"[跳过] checkpoint 不存在: {ckpt_path}")
        return None

    # 从 checkpoint 读取训练配置 (data_dir / seed / split_ratio / model_arch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    data_dir = config.get("data_dir")
    seed = config.get("seed", 42)
    split_ratio = config.get("split_ratio", 0.7)
    stored_acc = ckpt.get("best_acc")
    arch = config.get("model_arch", "cnn")

    if not data_dir or not os.path.isdir(data_dir):
        print(f"[跳过] 数据目录不存在: {data_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  评估: {exp_name}  (arch={arch})")
    print(f"{'=' * 60}")
    print(f"  [数据] {data_dir}")
    print(f"  [权重] {ckpt_path}")
    print(f"  [训练报告] best_acc={stored_acc:.4f}" if stored_acc else "  [训练报告] best_acc=N/A")

    # 复现训练时的测试集划分
    full_dataset = ExpDataset(data_dir)
    n_total = len(full_dataset)
    if n_total == 0:
        print("[跳过] 数据集为空")
        return None
    test_indices = reproduce_test_indices(n_total, seed, split_ratio)
    test_ds = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  [测试集] {len(test_ds)} 样本 (复现 seed={seed}, split_ratio={split_ratio})")

    # 加载模型
    model, forward_fn, arch = load_model_by_arch(exp_name, ckpt_path, device)

    # 推理
    preds, labels = infer(model, forward_fn, test_loader, device)
    accs = per_class_accuracy(preds, labels)
    total_acc = sum(1 for p, l in zip(preds, labels) if p == l) / max(len(labels), 1) * 100

    print(f"  [结果] 总准确率: {total_acc:.2f}%")
    print(f"         每类识别率: " + " | ".join(f"{n}:{a:.1f}%" for n, a in zip(CLASS_NAMES, accs)))

    # 输出目录
    fig_dir = os.path.join(FIG_ROOT, exp_name)
    os.makedirs(fig_dir, exist_ok=True)

    # 绘图 (仅 2 张)
    plot_confusion(preds, labels, exp_name, fig_dir)
    plot_per_class_accuracy(accs, exp_name, fig_dir)

    print(f"  [完成] 图表保存至 {fig_dir}/")
    return accs


# ================================================================
# 主入口
# ================================================================
def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("  exp_* 实验数据调制格式识别评估")
    print("=" * 60)
    print(f"  [权重目录] {CKPT_DIR}")
    print(f"  [图表目录] {FIG_ROOT}")

    os.makedirs(FIG_ROOT, exist_ok=True)

    exp_list = [args.exp_name] if args.exp_name else ALL_EXPERIMENTS
    exp_results = {}
    for exp_name in exp_list:
        result = evaluate_one(exp_name, args, device)
        if result is not None:
            exp_results[exp_name] = result

    # 多实验对比图
    if not args.no_comparison and len(exp_results) >= 2:
        print(f"\n[绘图] 多实验每类识别率对比图 ({len(exp_results)} 个实验)...")
        plot_comparison(exp_results, FIG_ROOT)

    print(f"\n{'=' * 60}")
    print(f"  评估完成! 共评估 {len(exp_results)} 个实验")
    print(f"  图表保存至 {FIG_ROOT}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
