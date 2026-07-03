"""
train_stage1.py — 阶段一：纯净分类器预训练
========================================================
训练目标:
  让 ConstellationCNN 学会在「完美无湍流」的 Ideal_CDM 上识别调制格式。

数据流:
  Ideal_CDM (1,64,64) ──► ConstellationCNN ──► Logits (5类) ──► CrossEntropyLoss

输出:
  - cnn_ideal_best.pth    (测试集准确率最高的模型权重)
  - cnn_ideal_final.pth   (最后一个 epoch 的模型权重, 用于中断恢复)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入我们的自定义模块
from dataset import get_dataloaders, MODULATION_NAMES
from models import ConstellationCNN


# ================================================================
# 训练配置 (超参数) — 可根据需要调整
# ================================================================
CONFIG = {
    "data_dir": "./data",
    "batch_size": 128,
    "split_ratio": 0.8,
    "num_workers": 4,
    "seed": 42,

    "num_epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "distorted_mix_ratio": 0.3,  # 30%% Distorted CDM 混入训练 (防止域偏移崩塌)

    "lr_patience": 8,
    "lr_factor": 0.5,

    "early_stop_patience": 10,

    "save_dir": "./results/cnn_unet",
}


def set_seed(seed: int):
    """固定随机种子, 保证实验可复现"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # GPU 加速: 启用 cuDNN 自动调谐（牺牲微小可复现性，换取 3-5x 加速）
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """自动选择可用设备: CUDA > MPS (macOS) > CPU"""
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


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> tuple:
    """
    执行一个 epoch 的训练 (支持 AMP 混合精度)

    参数:
      model:     分类器模型
      loader:    训练集 DataLoader
      criterion: 损失函数 (CrossEntropyLoss)
      optimizer: 优化器
      device:    计算设备
      scaler:    torch.cuda.amp.GradScaler (None = 禁用 AMP)

    返回:
      (平均损失, 准确率)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for batch_idx, (distorted, ideal, labels, _) in enumerate(loader):
        # 数据增强: 30% 概率混入 Distorted_CDM, 强迫 Baseline 适应湍流散斑
        # 防止纯 Ideal_CDM 训练导致高 SNR 下因域偏移而崩塌
        mix_ratio = 0.3
        batch_size = ideal.size(0)
        mask = torch.rand(batch_size, 1, 1, 1, device=ideal.device) < mix_ratio
        # blended = mask * distorted + (1-mask) * ideal  (逐样本混合)
        blended = torch.where(mask, distorted, ideal)

        # stage1 训练使用 blended 输入
        blended = blended.to(device)
        labels = labels.to(device)

        # ---- 前向传播 (AMP autocast) ----
        # ideal (B,1,64,64) → logits (B,5)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(blended)

        # ---- 计算损失 ----
        loss = criterion(logits, labels)

        # ---- 反向传播与参数更新 ----
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ---- 统计 ----
        total_loss += loss.item() * blended.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    在验证/测试集上评估模型

    返回:
      (平均损失, 准确率)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for distorted, ideal, labels, _ in loader:
        ideal = ideal.to(device)
        labels = labels.to(device)

        logits = model(ideal)
        loss = criterion(logits, labels)

        total_loss += loss.item() * ideal.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # ---- 初始化 ----
    set_seed(CONFIG["seed"])
    device = get_device()

    # 创建保存目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ---- 加载数据 ----
    print("\n" + "=" * 60)
    print("  阶段一: 纯净分类器预训练")
    print("=" * 60)
    print(f"[数据] 数据目录: {CONFIG['data_dir']}")

    train_loader, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        split_ratio=CONFIG["split_ratio"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    # ---- 构建模型 ----
    model = ConstellationCNN(
        num_classes=5,
        dropout_rate=CONFIG["dropout_rate"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] ConstellationCNN 可训练参数: {total_params:,}")

    # ---- 损失函数 & 优化器 & 学习率调度器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
    )

    # ---- 训练循环 ----
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    print("\n" + "-" * 60)
    print(f"  {'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} {'Test Loss':<12} {'Test Acc':<10} {'LR':<10}")
    print("-" * 60)

    # ---- 混合精度 (GPU 加速 1.5-2x) ----
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda')) if device.type == 'cuda' else None

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # 测试集评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 学习率调度 (基于测试集准确率)
        scheduler.step(test_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印日志
        print(
            f"  {epoch:<6} {train_loss:<12.4f} {train_acc:<10.4f} "
            f"{test_loss:<12.4f} {test_acc:<10.4f} {current_lr:<10.2e}"
        )

        # ---- 保存最佳模型 ----
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": CONFIG,
                },
                os.path.join(CONFIG["save_dir"], "cnn_ideal_best.pth"),
            )
            print(f"  ==> [保存] 新最佳模型! (Epoch {epoch}, Test Acc = {best_acc:.4f})")
        else:
            early_stop_counter += 1

        # ---- 早停检查 ----
        if early_stop_counter >= CONFIG["early_stop_patience"]:
            print(f"\n[早停] 验证集准确率连续 {CONFIG['early_stop_patience']} 轮未提升, 停止训练。")
            break

    # ---- 训练结束 ----
    print("\n" + "=" * 60)
    print(f"  训练完成!")
    print(f"  最佳测试准确率: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"  最佳模型已保存至: {os.path.join(CONFIG['save_dir'], 'cnn_ideal_best.pth')}")
    print("=" * 60)

    # 保存最后一个 epoch 的模型 (用于中断恢复)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": CONFIG,
        },
        os.path.join(CONFIG["save_dir"], "cnn_ideal_final.pth"),
    )


if __name__ == "__main__":
    # 支持命令行指定数据目录: python train_stage1.py ./my_data
    if len(sys.argv) > 1:
        CONFIG["data_dir"] = sys.argv[1]
    main()
