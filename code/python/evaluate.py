"""
evaluate.py — FSO 统一评估脚本，支持消融实验对比
=====================================================
根据 exp_name 自动识别模型架构并加载对应 checkpoint，
输出混淆矩阵、SNR 瀑布曲线、修复可视化对比图。

用法:
  # 评估单个实验
  python evaluate.py --exp_name cnn_ideal

  # 对比全部实验并输出汇总对比图
  python evaluate.py --compare_all
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# 项目内部导入
from dataset import FSODataset, MODULATION_NAMES, TURBULENCE_GROUPS
from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E, UNet_Swin_E2E
from utils import set_seed, get_device


# ================================================================
# 全局配置
# ================================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
MARKERS = ["o", "s", "D", "^", "v"]
SNR_RANGE = list(range(-6, 21, 2))


# ================================================================
# 参数解析
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSO 消融实验统一评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp_name", type=str, default=None,
                       help="实验名称 (自动加载 {save_dir}/{exp_name}_best.pth)")
    group.add_argument("--compare_all", action="store_true",
                       help="对比全部三个实验 (cnn_ideal / unet_denoiser / unet_vit_e2e)")

    parser.add_argument("--eval_data", type=str, default="./data",
                        help="评估数据目录")
    parser.add_argument("--save_dir", type=str, default="./results/checkpoints",
                        help="模型权重目录")
    parser.add_argument("--fig_dir", type=str, default="./results/figures",
                        help="图表保存目录")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


# ================================================================
# 模型加载
# ================================================================
def load_model_by_exp(exp_name: str, save_dir: str, device: torch.device):
    """
    根据实验名称自动加载对应模型。

    返回:
      (model_or_tuple, mode_str)
      mode_str ∈ {"cnn", "unet", "vit_e2e", "unet_resnet18_e2e", "unet_swin_e2e"}
    """
    best_path = os.path.join(save_dir, f"{exp_name}_best.pth")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # 从 checkpoint 推断架构
    if "swin_state_dict" in ckpt:
        from models import UNet_Swin_E2E
        in_channels = 2 if '_gaf' in exp_name else 1
        model = UNet_Swin_E2E(num_classes=NUM_CLASSES, in_channels=in_channels).to(device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        model.swin.load_state_dict(ckpt["swin_state_dict"])
        model.eval()
        print(f"[加载] {exp_name} — U-Net + Swin-T (E2E)")
        return (model.unet, model.swin), "unet_swin_e2e"
    elif "resnet_state_dict" in ckpt:
        # unet_resnet18_e2e 模式
        in_channels = 2 if '_gaf' in exp_name else 1
        model = UNet_ResNet18_E2E(num_classes=NUM_CLASSES, in_channels=in_channels).to(device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        model.resnet.load_state_dict(ckpt["resnet_state_dict"])
        model.eval()
        print(f"[加载] {exp_name} — U-Net + ResNet-18 (E2E)")
        return (model.unet, model.resnet), "unet_resnet18_e2e"
    elif "vit_state_dict" in ckpt:
        # vit/e2e 模式
        in_channels = 2 if '_gaf' in exp_name else 1
        unet = ConstellationUNet(in_channels, 1, True).to(device)
        vit = ConstellationViT(img_size=64, patch_size=16, in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
        unet.load_state_dict(ckpt["unet_state_dict"])
        vit.load_state_dict(ckpt["vit_state_dict"])
        unet.eval()
        vit.eval()
        print(f"[加载] {exp_name} — U-Net + ViT (E2E)")
        return (unet, vit), "vit_e2e"

    # 尝试从 model_state_dict 或直接 dict 加载
    sd = ckpt.get("model_state_dict", ckpt)

    # 通过键名判断是 CNN 还是 UNet
    # CNN: conv1/conv2/conv3/conv4/fc1/fc2
    # UNet: enc1/enc2/enc3/enc4/bottleneck/dec1/dec2/dec3/out_conv
    if any(k.startswith("enc") for k in sd.keys()):
        in_channels = 2 if '_gaf' in exp_name else 1
        model = ConstellationUNet(in_channels, 1, True).to(device)
        mode = "unet"
    else:
        in_channels = 2 if '_gaf' in exp_name else 1
        model = ConstellationCNN(NUM_CLASSES, in_channels=in_channels).to(device)
        mode = "cnn"

    model.load_state_dict(sd)
    model.eval()
    print(f"[加载] {exp_name} — {'ConstellationUNet' if mode == 'unet' else 'ConstellationCNN'}")
    return model, mode


# ================================================================
# 推理 & 统计
# ================================================================
@torch.no_grad()
def evaluate_model(model_or_tuple, mode, loader, device):
    """
    统一推理入口。

    返回:
      all_preds:   list[int] — 所有预测标签
      all_labels:  list[int] — 所有真实标签
      all_snrs:    list[int] — 对应 SNR
      all_turbs:   list[str] — 对应湍流强度 (None 表示未识别)
      repaired_imgs: list[Tensor] — U-Net 修复图 (仅 unet/vit 模式)
      ideal_imgs:  list[Tensor] — 理想参考图
      distorted_imgs: list[Tensor] — 畸变输入图
    """
    all_preds, all_labels, all_snrs, all_turbs = [], [], [], []
    repaired_imgs, ideal_imgs, distorted_imgs = [], [], []
    vis_snrs, vis_labels, vis_preds = [], [], []

    # 预解析湍流标签
    dataset = loader.dataset
    file_turb_map = []
    for f in (dataset.dataset.files if hasattr(dataset, "dataset") else dataset.files):
        fn = os.path.basename(f)
        t = next((t for t in TURBULENCE_GROUPS if t in fn), None)
        file_turb_map.append(t)

    sample_idx = 0
    has_denoiser = mode in ("unet", "vit_e2e", "unet_resnet18_e2e", "unet_swin_e2e")
    # 可视化样本：仅采集 强湍流 + SNR≤0dB 的每个目标调制格式各 1 个
    VIS_TARGET_CLASSES = {0, 2, 3, 4}  # BPSK, 16QAM, 64QAM, 256QAM
    vis_collected = set()

    for distorted, ideal, labels, snr_values in loader:
        distorted = distorted.to(device)
        labels_g = labels.to(device)
        B = distorted.size(0)

        if mode == "cnn":
            logits = model_or_tuple(distorted)
            repaired = None
        elif mode == "unet":
            repaired = model_or_tuple(distorted)
            logits = None
        elif mode in ("vit_e2e", "unet_resnet18_e2e", "unet_swin_e2e"):
            unet, cls = model_or_tuple
            repaired = unet(distorted)
            logits = cls(repaired)

        preds = logits.argmax(dim=1).cpu().tolist() if logits is not None else [0] * B

        for i in range(B):
            if sample_idx >= len(file_turb_map):
                break
            all_preds.append(preds[i])
            lbl = labels_g[i].item()
            all_labels.append(lbl)
            all_snrs.append(int(snr_values[i].item()))
            all_turbs.append(file_turb_map[sample_idx])
            sample_idx += 1

            # 可视化样本筛选：强湍流 + SNR≤0dB + 每个目标调制格式取 1 个
            turb = file_turb_map[sample_idx - 1]  # sample_idx 已自增
            snr_val = int(snr_values[i].item())
            if (has_denoiser and turb == "strong" and snr_val <= 0
                    and lbl in VIS_TARGET_CLASSES and lbl not in vis_collected):
                vis_collected.add(lbl)
                vis_labels.append(lbl)
                vis_preds.append(preds[i])
                distorted_imgs.append(distorted[i].cpu())
                repaired_imgs.append(repaired[i].cpu())
                ideal_imgs.append(ideal[i])
                vis_snrs.append(snr_val)

        if sample_idx >= len(file_turb_map):
            break

    n_vis = len(distorted_imgs)
    print(f"[可视化] 共采集 {n_vis} 个强湍流低SNR可视化样本")
    return all_preds, all_labels, all_snrs, all_turbs, \
           distorted_imgs, repaired_imgs, ideal_imgs, vis_snrs, vis_labels, vis_preds


# ================================================================
# 绘图函数
# ================================================================


def plot_waterfall(all_preds, all_labels, all_snrs, exp_name, fig_dir, suffix=""):
    """绘制 SNR-准确率瀑布曲线 (每类调制格式一条线)。"""
    snr_acc = {s: {c: {"correct": 0, "total": 0} for c in range(NUM_CLASSES)}
               for s in SNR_RANGE}
    for p, l, s in zip(all_preds, all_labels, all_snrs):
        if s in snr_acc:
            snr_acc[s][l]["total"] += 1
            if p == l:
                snr_acc[s][l]["correct"] += 1

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in range(NUM_CLASSES):
        x, y = [], []
        for s in sorted(snr_acc.keys()):
            t = snr_acc[s][c]["total"]
            if t > 0:
                x.append(s)
                y.append(snr_acc[s][c]["correct"] / t * 100)
        if x:
            ax.plot(x, y, color=COLORS[c], marker=MARKERS[c], markersize=6,
                    linewidth=1.8, label=CLASS_NAMES[c])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{exp_name} — SNR Waterfall")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    fn = f"waterfall_{exp_name}{suffix}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")


def plot_confusion_matrix(all_preds, all_labels, exp_name, fig_dir, suffix=""):
    """绘制全局混淆矩阵。"""
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    acc = np.trace(cm) / max(cm.sum(), 1) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{exp_name} — Acc: {acc:.2f}%")
    fig.tight_layout()

    fn = f"confusion_{exp_name}{suffix}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")
    return acc


def plot_turbulence_confusion(all_preds, all_labels, all_turbs, exp_name, fig_dir, suffix=""):
    """绘制按湍流强度分组的混淆矩阵 (3个子图)。"""
    grouped = {g: {"preds": [], "labels": []} for g in TURBULENCE_GROUPS}
    for p, l, t in zip(all_preds, all_labels, all_turbs):
        if t in grouped:
            grouped[t]["preds"].append(p)
            grouped[t]["labels"].append(l)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    turb_names = {"weak": "Weak", "moderate": "Moderate", "strong": "Strong"}
    for ax, turb in zip(axes, TURBULENCE_GROUPS):
        data = grouped[turb]
        total = len(data["preds"])
        if total == 0:
            ax.set_title(f"{turb_names[turb]}\nNo Data")
            continue
        cm = confusion_matrix(data["labels"], data["preds"], labels=list(range(NUM_CLASSES)))
        acc = np.trace(cm) / max(cm.sum(), 1) * 100
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(f"{turb_names[turb]}\nAcc: {acc:.1f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(f"{exp_name} — Per-Turbulence Confusion", fontweight="bold")
    fig.tight_layout()

    fn = f"confusion_groups_{exp_name}{suffix}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")


def plot_repair_visualization(distorted_imgs, repaired_imgs, ideal_imgs, all_labels,
                               all_preds, vis_snrs, exp_name, fig_dir, suffix=""):
    """绘制畸变→修复→理想对比图 (最多 6 行 × 3 列)。"""
    n = min(6, len(distorted_imgs))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(9, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        for j, (img, ttl) in enumerate([
            (distorted_imgs[i], "Distorted"),
            (repaired_imgs[i], "Repaired"),
            (ideal_imgs[i], "Ideal"),
        ]):
            ax = axes[i, j]
            # 双通道(GAF): 仅显示第0通道(CDM); 单通道: squeeze批次维
            disp = img[0] if img.size(0) == 2 else img.squeeze(0)
            ax.imshow(disp.numpy(), cmap="inferno", origin="upper")
            lbl_name = CLASS_NAMES[all_labels[i]] if i < len(all_labels) else "?"
            snr_val = vis_snrs[i] if i < len(vis_snrs) else "?"
            if j == 0:
                correct = all_preds[i] == all_labels[i] if i < len(all_preds) else True
                status = "✓" if correct else "✗"
                ax.set_title(f"{lbl_name} SNR={snr_val}dB - {ttl} [{status}]")
            else:
                ax.set_title(f"{ttl}  SNR={snr_val}dB")
            ax.axis("off")

    fig.suptitle(f"{exp_name} — Repair Visualization", fontweight="bold")
    fig.tight_layout()

    fn = f"repair_vis_{exp_name}{suffix}.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")


def plot_turbulence_snr_curves(all_preds, all_labels, all_snrs, all_turbs,
                                exp_name, fig_dir, suffix=""):
    """绘制按湍流+SNR 分组的准确率曲线 (每个湍流等级一张双轴图)。"""
    # 统计按湍流+SNR+类别
    stats = {g: {s: {c: {"correct": 0, "total": 0} for c in range(NUM_CLASSES)}
                 for s in SNR_RANGE} for g in TURBULENCE_GROUPS}
    for p, l, s, t in zip(all_preds, all_labels, all_snrs, all_turbs):
        if t in stats and s in stats[t]:
            stats[t][s][l]["total"] += 1
            if p == l:
                stats[t][s][l]["correct"] += 1

    turb_names = {"weak": "Weak", "moderate": "Moderate", "strong": "Strong"}
    for turb in TURBULENCE_GROUPS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in range(NUM_CLASSES):
            x, y = [], []
            for s in sorted(stats[turb].keys()):
                t = stats[turb][s][c]["total"]
                if t > 0:
                    x.append(s)
                    y.append(stats[turb][s][c]["correct"] / t * 100)
            if x:
                ax.plot(x, y, color=COLORS[c], marker=MARKERS[c],
                        markersize=6, linewidth=1.8, label=CLASS_NAMES[c])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{exp_name} — {turb_names[turb]} Turbulence")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()

        fn = f"snr_curves_{turb}_{exp_name}{suffix}.png"
        fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [图] {fn}")


# ================================================================
# 对比图: 将多个实验的混淆矩阵 & 瀑布曲线合并在同一张图
# ================================================================
def plot_comparison(exp_results, fig_dir):
    """绘制多实验对比图：全局混淆矩阵对比 + 瀑布曲线叠加。"""
    fig_dir = os.path.join(fig_dir, "..")  # 对比图放在 figures 根目录
    fig_dir = os.path.normpath(fig_dir)

    # ---- 对比瀑布曲线 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_name, (preds, labels, snrs, turbs, _, _, _, _, _, _) in exp_results.items():
        snr_acc = {s: {"correct": 0, "total": 0} for s in SNR_RANGE}
        for p, l, s in zip(preds, labels, snrs):
            if s in snr_acc:
                snr_acc[s]["total"] += 1
                if p == l:
                    snr_acc[s]["correct"] += 1
        x, y = [], []
        for s in sorted(snr_acc.keys()):
            t = snr_acc[s]["total"]
            if t > 0:
                x.append(s)
                y.append(snr_acc[s]["correct"] / t * 100)

        color_idx = list(exp_results.keys()).index(exp_name)
        ax.plot(x, y, color=COLORS[color_idx % len(COLORS)],
                marker=MARKERS[color_idx % len(MARKERS)],
                markersize=7, linewidth=2.0, label=exp_name)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Total Accuracy (%)")
    ax.set_title("Ablation Study — SNR Comparison")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    fn = "ablation_snr_comparison.png"
    fig.savefig(os.path.join(fig_dir, fn), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [图] {fn}")


# ================================================================
# 评估单个实验
# ================================================================
def evaluate_one(exp_name, args, device):
    """评估单个实验并输出所有图表。"""
    fig_dir = os.path.join(args.fig_dir, exp_name)
    os.makedirs(fig_dir, exist_ok=True)

    # 加载模型
    model, mode = load_model_by_exp(exp_name, args.save_dir, device)

    # 加载数据
    use_gaf = '_gaf' in exp_name
    dataset = FSODataset(args.eval_data, use_gaf=use_gaf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 推理
    results = evaluate_model(model, mode, loader, device)
    all_preds, all_labels, all_snrs, all_turbs, \
        distorted_imgs, repaired_imgs, ideal_imgs, vis_snrs, vis_labels, vis_preds = results

    total = len(all_labels)
    acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(total, 1)
    print(f"[结果] {exp_name}: 总样本 {total}, 准确率 {acc:.4f} ({acc*100:.2f}%)")

    # 绘图 (分类器/级联才出混淆矩阵等; 纯 U-Net 只出修复可视化)
    if mode in ("cnn", "vit_e2e", "unet_resnet18_e2e", "unet_swin_e2e"):
        plot_waterfall(all_preds, all_labels, all_snrs, exp_name, fig_dir)
        plot_confusion_matrix(all_preds, all_labels, exp_name, fig_dir)
        plot_turbulence_confusion(all_preds, all_labels, all_turbs, exp_name, fig_dir)
        plot_turbulence_snr_curves(all_preds, all_labels, all_snrs, all_turbs, exp_name, fig_dir)

    if mode in ("unet", "vit_e2e", "unet_resnet18_e2e", "unet_swin_e2e"):
        plot_repair_visualization(distorted_imgs, repaired_imgs, ideal_imgs,
                                  vis_labels, vis_preds,
                                  vis_snrs,
                                  exp_name, fig_dir)

    print(f"[完成] {exp_name} 评估完毕，图表保存至 {fig_dir}/")
    return results


# ================================================================
# 主入口
# ================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("  FSO 消融实验评估")
    print("=" * 60)
    print(f"[数据] {args.eval_data}")
    print(f"[权重] {args.save_dir}")
    print(f"[图表] {args.fig_dir}")
    print()

    if args.exp_name:
        # 评估单个实验
        evaluate_one(args.exp_name, args, device)

    elif args.compare_all:
        # 对比全部实验
        exp_names = ["unet_vit_e2e", "unet_swin_e2e", "unet_swin_gaf", "cnn_ideal", "cnn_gaf"]
        exp_results = {}
        for exp_name in exp_names:
            print(f"\n{'='*50}")
            print(f"  评估: {exp_name}")
            print(f"{'='*50}")
            try:
                results = evaluate_one(exp_name, args, device)
                exp_results[exp_name] = results
            except FileNotFoundError as e:
                print(f"  [跳过] {e}")
            except Exception as e:
                print(f"  [错误] {exp_name}: {e}")

        # 绘制对比图
        if len(exp_results) >= 2:
            print(f"\n[绘图] 实验对比图...")
            plot_comparison(exp_results, args.fig_dir)

        print(f"\n{'='*60}")
        print(f"  对比评估完成! 图表保存至 {args.fig_dir}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
