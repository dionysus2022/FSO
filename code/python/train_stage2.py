"""
train_stage2.py — 阶段二：湍流畸变修复训练
========================================================
训练目标:
  训练 ConstellationUNet (GNN 去噪器) 学习将「带有大气湍流畸变」的星座密度图
  修复还原为「纯净理想」的星座密度图。

数据流:
  Distorted_CDM (1,64,64) ──► ConstellationUNet ──► Repaired_CDM (1,64,64)
                                  │
                                  └── 与 Ideal_CDM 计算 MSELoss
输出:
  - gnn_denoiser_best.pth   (测试集重构损失最低的模型权重)
  - gnn_denoiser_final.pth  (最后一个 epoch 的模型权重, 用于中断恢复)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入我们的自定义模块
from dataset import get_dataloaders
from models import ConstellationUNet


# ================================================================
# 训练配置 (超参数) — 可根据需要调整
# ================================================================
CONFIG = {
    "data_dir": "./data",
    "batch_size": 128,
    "split_ratio": 0.8,
    "num_workers": 4,
    "seed": 42,

    "num_epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-5,

    "lr_patience": 6,
    "lr_factor": 0.5,

    "early_stop_patience": 10,

    "save_dir": "./results/cnn_unet",

    # 迁移学习: 从旧 checkpoint 加载预训练权重作为起点
    "pretrained": "./checkpoints/gnn_denoiser_best.pth",
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
) -> float:
    """
    执行一个 epoch 的图像重建训练 (支持 AMP 混合精度)

    参数:
      model:     U-Net 去噪器
      loader:    训练集 DataLoader
      criterion: 损失函数 (MSELoss)
      optimizer: 优化器
      device:    计算设备
      scaler:    torch.cuda.amp.GradScaler (None = 禁用 AMP)

    返回:
      平均重构损失 (MSE)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None

    for batch_idx, (distorted, ideal, labels, snr_values) in enumerate(loader):
        
        # 阶段二: 使用 distorted (畸变图) 作为输入, ideal (纯净图) 作为目标
        # distorted: (B, 1, 64, 64)  — 待修复的畸变星座密度图
        # ideal:     (B, 1, 64, 64)  — 修复目标 (纯净星座密度图)
        distorted = distorted.to(device)
        ideal = ideal.to(device)
        # labels 在阶段二中不使用

        # ---- 前向传播 + 损失计算 (AMP autocast) ----
        # distorted (B,1,64,64) → repaired (B,1,64,64)  值域 [0, 1]
        # 注意：损失计算也在 autocast 内进行，保持精度一致避免 backward dtype 不匹配
        with torch.amp.autocast(device.type, enabled=use_amp):
            repaired = model(distorted)
            # ---- 计算重构损失 (MSE) ----
            # 衡量 repaired 与 ideal 之间逐像素的差异
            loss = criterion(repaired, ideal)

        # ---- SNR 感知权重: 低 SNR 抑制高阶调制先验坍塌 ----
        # 逐样本计算权重: BPSK/QPSK 始终 1.0, 高阶调制低 SNR 时降权
        device_w = distorted.device  # 使用数据张量所在设备 (GPU)
        snr_weights = torch.ones(len(labels), device=device_w)
        for i in range(len(labels)):
            s = snr_values[i].item()
            lbl = labels[i].item()
            if lbl >= 2:  # 16QAM(2)/64QAM(3)/256QAM(4): 密集星座, 低 SNR 易坍塌
                if   s <= -6: snr_weights[i] = 0.10
                elif s <= -4: snr_weights[i] = 0.15
                elif s <= -2: snr_weights[i] = 0.25
                elif s <=  0: snr_weights[i] = 0.35
                elif s <=  2: snr_weights[i] = 0.50
                elif s <=  4: snr_weights[i] = 0.70
                elif s <=  6: snr_weights[i] = 0.85
            # BPSK/QPSK: w=1.0 不变
        # 权重重塑为 (B,1,1,1) 与 repaired/ideal 逐像素相乘
        w = snr_weights.view(-1, 1, 1, 1)
        # 加权 MSE: 对加权后的逐像素误差取均值
        diff = (repaired - ideal) ** 2
        loss = (diff * w).mean()

        # ---- 反向传播与参数更新 ----
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ---- 统计 (加权 loss 用于反向传播, 原始 MSE 用于日志对比) ----
        total_loss += loss.item() * distorted.size(0)
        total_samples += distorted.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    在验证/测试集上评估重构损失

    返回:
      平均重构损失 (MSE)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for distorted, ideal, labels, _ in loader:
        distorted = distorted.to(device)
        ideal = ideal.to(device)

        repaired = model(distorted)
        loss = criterion(repaired, ideal)

        total_loss += loss.item() * distorted.size(0)
        total_samples += distorted.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    # ---- 初始化 ----
    set_seed(CONFIG["seed"])
    device = get_device()

    # 创建保存目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ---- 加载数据 ----
    print("\n" + "=" * 60)
    print("  阶段二: 湍流畸变修复训练 (U-Net 去噪器)")
    print("=" * 60)
    print(f"[数据] 数据目录: {CONFIG['data_dir']}")

    train_loader, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        split_ratio=CONFIG["split_ratio"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    print("\n[数据探针] 验证 BPSK / QPSK 输入张量（非零值检测）...")
    bpsk_ok = False
    qpsk_ok = False
    for distorted, ideal, labels, _ in train_loader:
        labels_np = labels.numpy()
        for i in range(distorted.size(0)):
            lbl = labels_np[i]
            d_tensor = distorted[i]
            d_cpu = d_tensor.cpu().numpy().squeeze()
            nonzero = int(np.count_nonzero(d_cpu))
            max_val = float(d_cpu.max())
            min_val = float(d_cpu.min())
            mod_names = {0: "BPSK", 1: "QPSK", 2: "16QAM", 3: "64QAM", 4: "256QAM"}
            if lbl == 0 and not bpsk_ok:
                print(f"  [BPSK]  sample: min={min_val:.4f}, max={max_val:.4f}, nonzero={nonzero}/4096")
                bpsk_ok = nonzero > 0
            elif lbl == 1 and not qpsk_ok:
                print(f"  [QPSK]  sample: min={min_val:.4f}, max={max_val:.4f}, nonzero={nonzero}/4096")
                qpsk_ok = nonzero > 0
        if bpsk_ok and qpsk_ok:
            break
    if not bpsk_ok:
        print("  [警告] BPSK Distorted_CDM 全为零！网络将无法区分 BPSK/QPSK！")
    if not qpsk_ok:
        print("  [警告] QPSK Distorted_CDM 全为零！")
    if bpsk_ok and qpsk_ok:
        print("  [OK] 所有调制格式 Distorted_CDM 均含非零值，数据探针通过。")
    print("")

    # ---- 构建模型 ----
    model = ConstellationUNet(
        in_channels=1,
        out_channels=1,
        bilinear=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] ConstellationUNet 可训练参数: {total_params:,}")

    # ---- 加载预训练权重 (迁移学习起点, 如果存在) ----
    pretrained = CONFIG.get("pretrained", "")
    if pretrained and os.path.isfile(pretrained):
        print(f"[迁移] 从预训练权重加载: {pretrained}")
        try:
            ckpt = torch.load(pretrained, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  [OK] 已加载 Epoch {ckpt.get('epoch','?')} 权重, 旧 best_loss={ckpt.get('best_loss','?'):.6f}")
            else:
                model.load_state_dict(ckpt)
                print("  [OK] 已加载原始权重 (无epoch信息)")
        except Exception as e:
            print(f"  [警告] 预训练权重加载失败: {e}，将从零开始。")
    elif pretrained:
        print(f"[提示] 预训练权重不存在 ({pretrained})，将从零开始训练。")
    else:
        print("[模型] 从零开始训练 (无预训练权重配置)")

    # ---- 损失函数 & 优化器 & 学习率调度器 ----
    # MSELoss: 逐像素均方误差, 驱动网络将畸变图逐点还原为纯净图
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
    )

    # ---- 训练循环 ----
    best_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    print("\n" + "-" * 60)
    print(f"  {'Epoch':<6} {'Train Loss':<14} {'Test Loss':<14} {'LR':<10}")
    print("-" * 60)

    # ---- 混合精度 (GPU 加速 1.5-2x) ----
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda')) if device.type == 'cuda' else None

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # 训练一个 epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # 测试集评估
        test_loss = evaluate(
            model, test_loader, criterion, device
        )

        # 学习率调度 (基于测试集损失)
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印日志
        print(
            f"  {epoch:<6} {train_loss:<14.6f} {test_loss:<14.6f} {current_lr:<10.2e}"
        )

        # ---- 保存最佳模型 ----
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": CONFIG,
                },
                os.path.join(CONFIG["save_dir"], "gnn_denoiser_best.pth"),
            )
            print(f"  ==> [保存] 新最佳模型! (Epoch {epoch}, Test Loss = {best_loss:.6f})")
        else:
            early_stop_counter += 1

        # ---- 早停检查 ----
        if early_stop_counter >= CONFIG["early_stop_patience"]:
            print(f"\n[早停] 测试集损失连续 {CONFIG['early_stop_patience']} 轮未降低, 停止训练。")
            break

    # ---- 训练结束 ----
    print("\n" + "=" * 60)
    print(f"  训练完成!")
    print(f"  最佳测试重构损失 (MSE): {best_loss:.6f} (Epoch {best_epoch})")
    print(f"  最佳模型已保存至: {os.path.join(CONFIG['save_dir'], 'gnn_denoiser_best.pth')}")
    print("=" * 60)

    # 保存最后一个 epoch 的模型 (用于中断恢复)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": CONFIG,
        },
        os.path.join(CONFIG["save_dir"], "gnn_denoiser_final.pth"),
    )


if __name__ == "__main__":
    # 支持命令行指定数据目录: python train_stage2.py ./my_data
    if len(sys.argv) > 1:
        CONFIG["data_dir"] = sys.argv[1]
    main()
