"""
train_stage3_e2e.py — 端到端联合微调 (U-Net + ViT)
=====================================================
训练目标:
  1. 冻结 U-Net 的前几层编码器, 只微调解码器
  2. ViT 从头训练
  3. 联合损失: 0.5 * MSE(repaired, ideal) + 0.5 * CE(logits, labels)

输出:
  - e2e_best.pth    (联合验证损失最低的权重)
  - e2e_final.pth   (最后一个 epoch 的权重)
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")

sys.path.insert(0, SCRIPT_DIR)
from dataset import get_dataloaders, LABEL_MAPPING, NUM_CLASSES
from models import ConstellationUNet, ConstellationViT

# ==================== 配置 ====================
CONFIG = {
    "data_dir": DATA_DIR,
    "batch_size": 64,
    "split_ratio": 0.8,
    "num_workers": 0,
    "seed": 42,

    "num_epochs": 30,
    "lr_unet": 1e-5,        # U-Net 微调学习率 (小)
    "lr_vit":  1e-4,        # ViT 学习率 (大, 从头训练)
    "weight_decay": 1e-4,
    "mse_weight": 0.5,      # MSE 损失权重
    "ce_weight":  0.5,      # 交叉熵损失权重

    "pretrained_unet": os.path.join(CKPT_DIR, "gnn_denoiser_best.pth"),
    "save_dir": CKPT_DIR,
}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[设备] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("[设备] CPU")
    return d


# ==================== 训练函数 ====================
def train_one_epoch(unet, vit, loader, criterion_mse, criterion_ce, optimizer, device, scaler=None):
    unet.train()
    vit.train()
    total_loss, total_mse, total_ce, total_correct, total_samples = 0.0, 0.0, 0.0, 0, 0
    use_amp = scaler is not None

    for distorted, ideal, labels, _ in loader:
        distorted = distorted.to(device)
        ideal = ideal.to(device)
        labels_g = labels.to(device)
        B = distorted.size(0)

        with torch.amp.autocast(device.type, enabled=use_amp):
            repaired = unet(distorted)
            logits = vit(repaired)
            loss_mse = criterion_mse(repaired, ideal)
            loss_ce = criterion_ce(logits, labels_g)
            loss = CONFIG["mse_weight"] * loss_mse + CONFIG["ce_weight"] * loss_ce

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * B
        total_mse += loss_mse.item() * B
        total_ce += loss_ce.item() * B
        total_correct += (logits.argmax(dim=1) == labels_g).sum().item()
        total_samples += B

    n = max(total_samples, 1)
    return total_loss / n, total_mse / n, total_ce / n, total_correct / n


@torch.no_grad()
def evaluate(unet, vit, loader, criterion_mse, criterion_ce, device):
    unet.eval()
    vit.eval()
    total_loss, total_mse, total_ce, total_correct, total_samples = 0.0, 0.0, 0.0, 0, 0

    for distorted, ideal, labels, _ in loader:
        distorted = distorted.to(device)
        ideal = ideal.to(device)
        labels_g = labels.to(device)
        B = distorted.size(0)

        repaired = unet(distorted)
        logits = vit(repaired)
        loss_mse = criterion_mse(repaired, ideal)
        loss_ce = criterion_ce(logits, labels_g)
        loss = CONFIG["mse_weight"] * loss_mse + CONFIG["ce_weight"] * loss_ce

        total_loss += loss.item() * B
        total_mse += loss_mse.item() * B
        total_ce += loss_ce.item() * B
        total_correct += (logits.argmax(dim=1) == labels_g).sum().item()
        total_samples += B

    n = max(total_samples, 1)
    return total_loss / n, total_mse / n, total_ce / n, total_correct / n


# ==================== 主函数 ====================
def main():
    set_seed(CONFIG["seed"])
    device = get_device()

    print("\n" + "=" * 60)
    print("  Stage 3: End-to-End Joint Fine-Tuning (U-Net + ViT)")
    print("=" * 60)
    print(f"[数据] {CONFIG['data_dir']}")

    train_loader, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        split_ratio=CONFIG["split_ratio"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    # ---------- 构建模型 ----------
    unet = ConstellationUNet(in_channels=1, out_channels=1, bilinear=True).to(device)
    vit = ConstellationViT(img_size=64, patch_size=16, in_channels=1, num_classes=NUM_CLASSES).to(device)

    pu = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    pv = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"[模型] U-Net: {pu:,} 参数  |  ViT: {pv:,} 参数")

    # ---------- 加载 U-Net 预训练权重 ----------
    pretrained = CONFIG["pretrained_unet"]
    if os.path.isfile(pretrained):
        ckpt = torch.load(pretrained, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            unet.load_state_dict(ckpt["model_state_dict"])
        else:
            unet.load_state_dict(ckpt)
        print(f"[迁移] U-Net 已加载: {pretrained}")
    else:
        print(f"[警告] 未找到 U-Net 预训练权重 ({pretrained}), 继续...")

    # ---------- 优化器 (分层学习率) ----------
    optimizer = optim.AdamW([
        {"params": unet.parameters(), "lr": CONFIG["lr_unet"]},
        {"params": vit.parameters(),  "lr": CONFIG["lr_vit"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    # AMP
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")) if device.type == "cuda" else None

    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    # ---------- 训练循环 ----------
    best_loss = float("inf")
    best_epoch = 0

    print(f"\n{'Epoch':<6} {'Tr Loss':<10} {'Tr MSE':<10} {'Tr CE':<10} {'Tr Acc':<10} "
          f"{'Test Loss':<10} {'Test Acc':<10} {'LR':<10}")
    print("-" * 80)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss, train_mse, train_ce, train_acc = train_one_epoch(
            unet, vit, train_loader, criterion_mse, criterion_ce, optimizer, device, scaler
        )

        test_loss, test_mse, test_ce, test_acc = evaluate(
            unet, vit, test_loader, criterion_mse, criterion_ce, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"{epoch:<6} {train_loss:<10.4f} {train_mse:<10.6f} {train_ce:<10.4f} {train_acc:<10.4f} "
              f"{test_loss:<10.4f} {test_acc:<10.4f} {current_lr:<10.2e}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "unet_state_dict": unet.state_dict(),
                "vit_state_dict": vit.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": CONFIG,
            }, os.path.join(CONFIG["save_dir"], "e2e_best.pth"))
            print(f"  ==> [保存] 新最佳模型! (Epoch {epoch}, Loss={best_loss:.6f})")

    print("\n" + "=" * 60)
    print(f"  训练完成! 最佳 Loss={best_loss:.6f} (Epoch {best_epoch})")
    print(f"  模型保存至: {os.path.join(CONFIG['save_dir'], 'e2e_best.pth')}")
    print("=" * 60)

    torch.save({
        "epoch": epoch,
        "unet_state_dict": unet.state_dict(),
        "vit_state_dict": vit.state_dict(),
        "best_loss": best_loss,
        "config": CONFIG,
    }, os.path.join(CONFIG["save_dir"], "e2e_final.pth"))


if __name__ == "__main__":
    main()
