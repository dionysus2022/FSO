"""
evaluate_stage3.py — 阶段三：级联评估与可视化
========================================================
评估目标:
  验证 GNN→CNN 级联网络的端到端识别能力, 并可视化修复效果。

数据流:
  Distorted_CDM ──► ConstellationUNet (冻结) ──► Repaired_CDM
                                                       │
                                                       ▼
                                              ConstellationCNN (冻结)
                                                       │
                                                       ▼
                                                  预测分类 Logits

输出:
  - 终端打印: 总体准确率 (Accuracy) + 混淆矩阵 (Confusion Matrix)
  - 图表文件: evaluation_visualization.png (3组 1×3 对比图)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import accuracy_score, confusion_matrix

# 导入我们的自定义模块
from dataset import get_dataloaders, MODULATION_NAMES, LABEL_MAPPING
from models import ConstellationCNN, ConstellationUNet


# ================================================================
# 评估配置
# ================================================================
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw"),
    "batch_size": 64,
    "split_ratio": 0.8,
    "num_workers": 0,
    "seed": 42,

    "cnn_ckpt": "./checkpoints/cnn_ideal_best.pth",
    "gnn_ckpt": "./checkpoints/gnn_denoiser_best.pth",

    "num_visualize": 3,
    "colormap": "viridis",
    "save_fig": "./evaluation_visualization.png",
}


def set_seed(seed: int):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自动选择可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[设备] 使用 CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[设备] 使用 Apple MPS")
    else:
        device = torch.device("cpu")
        print("[设备] 使用 CPU")
    return device


def load_model_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device, model_name: str):
    """
    加载模型权重 (checkpoint 字典格式)

    参数:
      model:      模型实例
      ckpt_path:  .pth 文件路径
      device:     计算设备
      model_name: 模型名称 (用于日志)

    返回:
      加载权重后的模型 (eval 模式)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 {model_name} 的权重文件: '{ckpt_path}'\n"
            f"请先运行 train_stage1.py 和 train_stage2.py 完成训练。"
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # checkpoint 是字典, 从 "model_state_dict" 键中加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # 冻结模式

    # 打印加载信息
    if "epoch" in checkpoint:
        print(f"  └── 加载自 Epoch {checkpoint['epoch']}", end="")
        if "best_acc" in checkpoint:
            print(f", 最佳准确率: {checkpoint['best_acc']:.4f}", end="")
        if "best_loss" in checkpoint:
            print(f", 最佳损失: {checkpoint['best_loss']:.6f}", end="")
        print()

    return model


@torch.no_grad()
def cascade_evaluate(
    gnn: torch.nn.Module,
    cnn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    端到端级联评估: Distorted → GNN → Repaired → CNN → 预测

    参数:
      gnn:    ConstellationUNet (冻结, eval 模式)
      cnn:    ConstellationCNN  (冻结, eval 模式)
      loader: 测试集 DataLoader
      device: 计算设备

    返回:
      (all_preds, all_labels,  visual_samples)
      all_preds:      所有样本的预测标签   list[int]
      all_labels:     所有样本的真实标签   list[int]
      visual_samples: 用于可视化的样本字典  dict
    """
    all_preds = []
    all_labels = []

    # 用于存储可视化素材 (收集前 num_visualize 个 batch 中的样本)
    visual_samples = {
        "distorted": [],  # 原始畸变图
        "repaired": [],   # GNN 修复图
        "ideal": [],      # 理想参考图
        "preds": [],      # 预测标签
        "labels": [],     # 真实标签
    }

    for batch_idx, (distorted, ideal, labels, _) in enumerate(loader):
        # ---- 数据搬移到设备 ----
        # distorted: (B, 1, 64, 64) — 带湍流畸变的星座密度图
        # ideal:     (B, 1, 64, 64) — 纯净参考图 (仅用于可视化, 不参与推理)
        # labels:    (B,)          — 真实调制格式标签
        distorted = distorted.to(device)
        ideal = ideal.to(device)
        labels = labels.to(device)

        # ============================================================
        # 核心级联推理流程 (在 torch.no_grad() 下运行, 不记录梯度)
        # ============================================================

        # 步骤1: GNN 去噪修复
        # distorted (B,1,64,64) → repaired (B,1,64,64)  值域 [0, 1]
        repaired = gnn(distorted)

        # 步骤2: CNN 分类
        # repaired (B,1,64,64) → logits (B,5)
        logits = cnn(repaired)

        # 步骤3: 获取预测类别
        # logits (B,5) → argmax → preds (B,)
        preds = logits.argmax(dim=1)

        # ============================================================

        # 收集统计结果
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # 收集可视化样本 (仅前几个 batch, 攒够 num_visualize 个即停止收集)
        if len(visual_samples["distorted"]) < CONFIG["num_visualize"]:
            batch_size = distorted.size(0)
            for i in range(batch_size):
                if len(visual_samples["distorted"]) >= CONFIG["num_visualize"]:
                    break
                visual_samples["distorted"].append(distorted[i].cpu())
                visual_samples["repaired"].append(repaired[i].cpu())
                visual_samples["ideal"].append(ideal[i].cpu())
                visual_samples["preds"].append(preds[i].cpu().item())
                visual_samples["labels"].append(labels[i].cpu().item())

    return all_preds, all_labels, visual_samples


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None):
    """
    使用 matplotlib 绘制混淆矩阵

    参数:
      cm:          混淆矩阵 (n_classes × n_classes)
      class_names: 类别名称列表
      save_path:   保存路径 (None 则只显示不保存到独立文件)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # 使用 imshow 绘制热力图
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    # 坐标轴设置
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="真实标签 (True Label)",
        xlabel="预测标签 (Predicted Label)",
        title="混淆矩阵 — Confusion Matrix",
    )

    # 旋转 x 轴标签以免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个格子中标注数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()

    if save_path:
        # 保存混淆矩阵为单独的文件
        cm_path = save_path.replace(".png", "_cm.png")
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        print(f"[保存] 混淆矩阵图表已保存至: {cm_path}")


def plot_comparison(visual_samples: dict, class_names: list, save_path: str, cmap: str):
    """
    绘制 1×3 对比图: [原始畸变图(Distorted)] | [GNN修复图(Repaired)] | [理想参考图(Ideal)]

    参数:
      visual_samples: 包含 distorted / repaired / ideal / preds / labels 的字典
      class_names:    调制格式名称列表
      save_path:      图片保存路径
      cmap:           伪彩色映射名称 ('viridis' / 'jet' / 'plasma' ...)
    """
    num = len(visual_samples["distorted"])
    if num == 0:
        print("[可视化] 无可视化样本, 跳过绘图。")
        return

    # 创建画布: 每行3列 (Distorted | Repaired | Ideal), 共 num 行
    fig, axes = plt.subplots(num, 3, figsize=(12, 3.5 * num))

    # 如果 num=1, axes 是 1D, 需要升维便于统一索引
    if num == 1:
        axes = axes[np.newaxis, :]

    # 列标题
    col_titles = [
        "畸变图 (Distorted)",
        "GNN 修复图 (Repaired)",
        "理想参考图 (Ideal)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(num):
        # ---- 获取第 i 个样本的数据 ----
        # 每个 Tensor 形状为 (1, 64, 64), 用 .squeeze(0) 去掉通道维 → (64, 64)
        distorted_img = visual_samples["distorted"][i].squeeze(0).numpy()
        repaired_img = visual_samples["repaired"][i].squeeze(0).numpy()
        ideal_img = visual_samples["ideal"][i].squeeze(0).numpy()

        pred_label = visual_samples["preds"][i]
        true_label = visual_samples["labels"][i]

        # ---- 绘制三个子图 ----
        # 左: 原始畸变图
        im0 = axes[i, 0].imshow(distorted_img, cmap=cmap, origin="upper", aspect="equal")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_xlabel(f"预测={class_names[pred_label]}\n真实={class_names[true_label]}",
                              fontsize=10)

        # 中: GNN 修复图
        im1 = axes[i, 1].imshow(repaired_img, cmap=cmap, origin="upper", aspect="equal")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        # 判断预测是否正确, 用绿色/红色标注
        correct = "✓ 正确" if pred_label == true_label else "✗ 错误"
        color = "green" if pred_label == true_label else "red"
        axes[i, 1].set_xlabel(f"{correct}", fontsize=10, color=color, fontweight="bold")

        # 右: 理想参考图
        im2 = axes[i, 2].imshow(ideal_img, cmap=cmap, origin="upper", aspect="equal")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_xlabel(f"真实标签: {class_names[true_label]}", fontsize=10)

        # 为每行添加 colorbar (放在最右侧子图旁)
        for j, im in enumerate([im0, im1, im2]):
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    fig.suptitle(
        "GNN→CNN 级联修复与分类评估 — FSO 大气湍流畸变修复",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[保存] 可视化对比图已保存至: {save_path}")


def main():
    # ---- 初始化 ----
    set_seed(CONFIG["seed"])
    device = get_device()

    print("\n" + "=" * 60)
    print("  阶段三: GNN→CNN 级联评估与可视化")
    print("=" * 60)

    # ---- 1. 加载测试数据 ----
    print(f"\n[数据] 数据目录: {CONFIG['data_dir']}")
    _, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        split_ratio=CONFIG["split_ratio"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    # ---- 2. 加载冻结权重的 GNN 去噪器 (ConstellationUNet) ----
    print(f"\n[加载] 加载 GNN 去噪器权重: {CONFIG['gnn_ckpt']}")
    gnn = ConstellationUNet(in_channels=1, out_channels=1, bilinear=True)
    gnn = load_model_weights(gnn, CONFIG["gnn_ckpt"], device, "ConstellationUNet")
    print("[状态] ConstellationUNet 已冻结 (eval 模式)")

    # ---- 3. 加载冻结权重的 CNN 分类器 (ConstellationCNN) ----
    print(f"\n[加载] 加载 CNN 分类器权重: {CONFIG['cnn_ckpt']}")
    cnn = ConstellationCNN(num_classes=5)
    cnn = load_model_weights(cnn, CONFIG["cnn_ckpt"], device, "ConstellationCNN")
    print("[状态] ConstellationCNN 已冻结 (eval 模式)")

    # ---- 4. 端到端级联评估 ----
    print("\n" + "-" * 60)
    print("[评估] 开始端到端级联评估 (Distorted → GNN → Repaired → CNN → 预测)")
    print("-" * 60)

    all_preds, all_labels, visual_samples = cascade_evaluate(
        gnn, cnn, test_loader, device
    )

    total_samples = len(all_labels)
    print(f"[统计] 测试集总样本数: {total_samples}")

    # ---- 5. 计算总体准确率 ----
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"  🎯 级联网络端到端准确率 (Accuracy): {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*60}")

    # ---- 6. 计算并打印混淆矩阵 ----
    bits_to_name = {v: MODULATION_NAMES[k] for k, v in LABEL_MAPPING.items()}
    class_names = [bits_to_name[i] for i in range(5)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))

    print(f"\n{'='*60}")
    print("  混淆矩阵 (Confusion Matrix)")
    print(f"{'='*60}")

    # 表头
    header = "真实 \\ 预测" + "".join(f"{name:>10}" for name in class_names)
    print(header)
    print("-" * len(header))

    # 逐行打印
    for i, name in enumerate(class_names):
        row = f"{name:<12}" + "".join(f"{cm[i, j]:>10}" for j in range(5))
        print(row)

    # ---- 7. 各类别准确率 ----
    print(f"\n{'='*60}")
    print("  各类别识别准确率 (Per-Class Accuracy)")
    print(f"{'='*60}")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_acc = cm[i, i] / cm[i].sum()
            print(f"  {name:<8}: {per_class_acc:.4f}  ({per_class_acc*100:.1f}%)  "
                  f"[{cm[i,i]} / {cm[i].sum()}]")
        else:
            print(f"  {name:<8}: N/A (测试集中无此类样本)")

    # ---- 8. 绘制混淆矩阵热力图 ----
    plot_confusion_matrix(cm, class_names, save_path=CONFIG["save_fig"])

    # ---- 9. 绘制 1×3 可视化对比图 ----
    print(f"\n[可视化] 随机抽取 {len(visual_samples['distorted'])} 个样本进行对比绘图...")
    plot_comparison(
        visual_samples,
        class_names,
        save_path=CONFIG["save_fig"],
        cmap=CONFIG["colormap"],
    )

    print("\n" + "=" * 60)
    print("  评估完成! 请查看以下输出文件:")
    print(f"    1. {CONFIG['save_fig']}")
    print(f"    2. {CONFIG['save_fig'].replace('.png', '_cm.png')}")
    print("=" * 60)


if __name__ == "__main__":
    # 支持命令行指定数据目录: python evaluate_stage3.py ./my_data
    if len(sys.argv) > 1:
        CONFIG["data_dir"] = sys.argv[1]
    main()
