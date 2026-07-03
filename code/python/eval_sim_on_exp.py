"""
eval_sim_on_exp.py — 仿真预训练模型在实验数据上评估
=====================================================
加载仿真数据预训练的模型，在实验数据上测试分类准确率。

用法:
  python eval_sim_on_exp.py                          # 评估全部模型
  python eval_sim_on_exp.py --model cnn_ideal        # 评估单个模型
  python eval_sim_on_exp.py --model unet_resnet18_e2e
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E, UNet_Swin_E2E
from utils import set_seed, get_device
from train_exp_data import ExpDataset

# 预训练模型清单
AVAILABLE_MODELS = {
    "cnn_ideal":          {"file": "cnn_ideal_best.pth",          "arch": "cnn",              "gaf": False},
    "cnn_gaf":            {"file": "cnn_gaf_best.pth",            "arch": "cnn",              "gaf": True},
    "unet_denoiser":      {"file": "unet_denoiser_best.pth",      "arch": "unet",             "gaf": False},
    "unet_vit_e2e":       {"file": "unet_vit_e2e_best.pth",       "arch": "unet_vit",         "gaf": False},
    "unet_resnet18_e2e":  {"file": "unet_resnet18_e2e_best.pth",  "arch": "unet_resnet18",    "gaf": False},
    "unet_swin_e2e":      {"file": "unet_swin_e2e_best.pth",      "arch": "unet_swin",        "gaf": False},
    "unet_resnet18_gaf":  {"file": "unet_resnet18_gaf_best.pth",  "arch": "unet_resnet18",    "gaf": True},
    "unet_swin_gaf":      {"file": "unet_swin_gaf_best.pth",      "arch": "unet_swin",        "gaf": True},
}

CKPT_DIR = "./results/checkpoints"
EXP_DATA_DIR = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\dataset_cdm_exp_all"
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]


def parse_args():
    parser = argparse.ArgumentParser(description="仿真预训练模型 → 实验数据评估")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="指定单个模型 (默认评估全部)")
    parser.add_argument("--data_dir", type=str, default=EXP_DATA_DIR,
                        help="实验数据目录")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(name, info, device):
    """加载仿真预训练模型"""
    path = os.path.join(CKPT_DIR, info["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_channels = 2 if info["gaf"] else 1

    if info["arch"] == "cnn":
        model = ConstellationCNN(num_classes=5, in_channels=in_channels).to(device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.eval()
        return model, "cnn"

    elif info["arch"] == "unet":
        model = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.eval()
        return model, "unet"

    elif info["arch"] == "unet_vit":
        unet = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        vit = ConstellationViT(img_size=64, patch_size=16, in_channels=1, num_classes=5).to(device)
        unet.load_state_dict(ckpt["unet_state_dict"])
        vit.load_state_dict(ckpt["vit_state_dict"])
        unet.eval(); vit.eval()
        return (unet, vit), "unet_vit"

    elif info["arch"] == "unet_resnet18":
        model = UNet_ResNet18_E2E(num_classes=5, in_channels=in_channels).to(device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        model.resnet.load_state_dict(ckpt["resnet_state_dict"])
        model.eval()
        return (model.unet, model.resnet), "unet_resnet18"

    elif info["arch"] == "unet_swin":
        model = UNet_Swin_E2E(num_classes=5, in_channels=in_channels).to(device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        model.swin.load_state_dict(ckpt["swin_state_dict"])
        model.eval()
        return (model.unet, model.swin), "unet_swin"


@torch.no_grad()
def evaluate(model_or_tuple, mode, loader, device):
    """
    统一推理。返回 (preds, labels, mse_values)
    """
    all_preds, all_labels, all_mses = [], [], []

    for distorted, ideal, label in loader:
        distorted = distorted.to(device)
        ideal = ideal.to(device)
        label = label.to(device)
        B = distorted.size(0)

        if mode == "cnn":
            logits = model_or_tuple(distorted)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().tolist())
            # CNN 无修复图，skip MSE
            for _ in range(B):
                all_mses.append(float("nan"))

        elif mode == "unet":
            repaired = model_or_tuple(distorted)
            mse = nn.functional.mse_loss(repaired, ideal, reduction="none").mean(dim=(1,2,3))
            all_mses.extend(mse.cpu().tolist())
            # 纯去噪器无分类输出
            all_preds.extend([-1] * B)
            all_labels.extend(label.cpu().tolist())

        elif mode in ("unet_vit", "unet_resnet18", "unet_swin"):
            unet, cls = model_or_tuple
            repaired = unet(distorted)
            logits = cls(repaired)
            mse = nn.functional.mse_loss(repaired, ideal, reduction="none").mean(dim=(1,2,3))
            all_mses.extend(mse.cpu().tolist())
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().tolist())

    return all_preds, all_labels, all_mses


def print_results(name, preds, labels, mses, is_classifier):
    """打印单模型评估结果"""
    total = len(labels)
    print(f"\n{'='*60}")
    print(f"  模型: {name}")
    print(f"{'='*60}")

    if is_classifier:
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / max(total, 1)
        print(f"{'类别':<12} {'准确率':<12} {'样本数':<10}")
        print("-" * 34)
        for c in range(5):
            ct = sum(1 for l in labels if l == c)
            cc = sum(1 for p, l in zip(preds, labels) if p == l == c)
            ca = cc / max(ct, 1) * 100
            print(f"{CLASS_NAMES[c]:<12} {ca:<12.2f} {ct:<10}")
        print("-" * 34)
        print(f"{'总准确率':<12} {acc*100:<12.2f} {total:<10}")
    else:
        print(f"  (纯去噪器，无分类输出)")

    # MSE 统计
    valid_mses = [m for m in mses if not np.isnan(m)]
    if valid_mses:
        print(f"\n  MSE: avg={np.mean(valid_mses):.6f}, min={np.min(valid_mses):.6f}, max={np.max(valid_mses):.6f}")

    return acc if is_classifier else None


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("  仿真预训练模型 → 实验数据评估")
    print("=" * 60)
    print(f"[实验数据] {args.data_dir}")

    # 加载实验数据
    dataset = ExpDataset(args.data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[数据] 实验样本数: {len(dataset)}")

    models_to_eval = [args.model] if args.model else list(AVAILABLE_MODELS.keys())

    results = {}
    for name in models_to_eval:
        info = AVAILABLE_MODELS[name]
        try:
            print(f"\n[加载] {name} ({info['file']})...")
            model, mode = load_model(name, info, device)
            preds, labels, mses = evaluate(model, mode, loader, device)
            is_cls = mode != "unet"
            acc = print_results(name, preds, labels, mses, is_cls)
            if is_cls:
                results[name] = acc
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
        except Exception as e:
            print(f"  [错误] {e}")
            import traceback
            traceback.print_exc()

    # 汇总对比
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"  汇总对比")
        print(f"{'='*60}")
        print(f"{'模型':<25} {'准确率':<12}")
        print("-" * 37)
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            print(f"{name:<25} {acc*100:<12.2f}%")
        print("-" * 37)
        print(f"\n[实验数据] 最佳: {max(results, key=results.get)} ({results[max(results, key=results.get)]*100:.2f}%)")


if __name__ == "__main__":
    main()