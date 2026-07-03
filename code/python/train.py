"""
train.py — FSO 统一训练入口，适配消融实验
=============================================
四种实验模式 (model_arch):
  cnn              — 纯 CNN 分类器训练 (ConstellationCNN, CrossEntropyLoss)
  unet             — 纯 U-Net 去噪器训练 (ConstellationUNet, MSELoss)
  vit              — U-Net + ViT 端到端联合微调 (ConstellationUNet + ConstellationViT, HybridLoss)
  unet_resnet18    — U-Net + 改造 ResNet-18 端到端 (ConstellationUNet + ResNet18-adapt, HybridLoss)

用法示例:
  python train.py --exp_name cnn_ideal --model_arch cnn --epochs 30
  python train.py --exp_name cnn_gaf --model_arch cnn --use_gaf --epochs 30
  python train.py --exp_name unet_denoiser --model_arch unet --epochs 20
  python train.py --exp_name unet_vit_e2e --model_arch vit --epochs 30 --pretrained_unet ./results/checkpoints/unet_denoiser_best.pth
  python train.py --exp_name unet_swin_e2e --model_arch unet_swin --epochs 30 --pretrained_unet ./results/checkpoints/unet_denoiser_best.pth
"""
import logging
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 项目内部导入
from dataset import get_dataloaders
from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E, UNet_Swin_E2E
from utils import set_seed, get_device, setup_logger


# ================================================================
# 参数解析
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSO 消融实验统一训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 实验标识 ----
    parser.add_argument("--exp_name", type=str, required=True,
                        help="实验名称 (用于保存权重: {exp_name}_best.pth)")
    parser.add_argument("--model_arch", type=str, required=True,
                        choices=["cnn", "unet", "vit", "unet_resnet18", "unet_swin"],
                        help="模型架构: cnn=分类器, unet=去噪器, vit=端到端联合, unet_resnet18=UNet+ResNet18, unet_swin=UNet+Swin-T")
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["ce", "mse", "hybrid"],
                        help="损失类型 (默认根据 model_arch 自动选择: cnn→ce, unet→mse, vit→hybrid)")

    # ---- 数据配置 ----
    parser.add_argument("--train_data", type=str, default="./data",
                        help="训练数据目录")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批大小")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="训练/测试集划分比例")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数 (0=主进程)")

    # ---- 训练超参数 ----
    parser.add_argument("--epochs", type=int, default=30,
                        help="总训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减 (AdamW)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--use_gaf", action="store_true",
                        help="启用 GAF 双通道模式 (CDM + GAF 时序编码, in_channels=2)")

    # ---- 模型特定参数 ----
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="CNN Dropout 概率")
    parser.add_argument("--pretrained_unet", type=str, default="",
                        help="vit 模式下预训练 U-Net 权重路径 (可选)")
    parser.add_argument("--pretrained_vit", type=str, default="",
                        help="vit 模式下预训练 ViT 权重路径 (可选)")

    # ---- 保存配置 ----
    parser.add_argument("--save_dir", type=str, default="./results/checkpoints",
                        help="模型权重保存目录")
    parser.add_argument("--log_dir", type=str, default="./results/logs",
                        help="日志保存目录")

    # ---- 学习率调度 ----
    parser.add_argument("--lr_patience", type=int, default=8,
                        help="ReduceLROnPlateau 等待轮数 (cnn/unet)")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="学习率衰减因子")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="早停等待轮数 (0=禁用)")

    # ---- unet_resnet18 模式特有 ----
    parser.add_argument("--lr_resnet", type=float, default=1e-4,
                        help="unet_resnet18 模式下 ResNet-18 学习率")

    # ---- vit 模式特有 ----
    parser.add_argument("--lr_unet", type=float, default=1e-5,
                        help="vit 模式下 U-Net 微调学习率")
    parser.add_argument("--lr_vit", type=float, default=1e-4,
                        help="vit 模式下 ViT 学习率")
    parser.add_argument("--lr_swin", type=float, default=1e-4,
                        help="unet_swin 模式下 Swin-T 学习率")
    parser.add_argument("--mse_weight", type=float, default=0.5,
                        help="vit 模式下 MSE 损失权重")
    parser.add_argument("--ce_weight", type=float, default=0.5,
                        help="vit 模式下 CE 损失权重")

    # ---- SNR 感知加权 (unet 模式, 沿用原 train_stage2 设计) ----
    parser.add_argument("--snr_weighted", type=bool, default=True,
                        help="unet 模式下是否启用 SNR 感知加权")

    return parser.parse_args()


# ================================================================
# 辅助函数: 学习率调度器自动选择
# ================================================================
def _build_scheduler(optimizer, args, mode="min"):
    """根据架构选择合适的学习率调度器。"""
    if args.model_arch == "vit":
        # vit 模式使用余弦退火
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        return ReduceLROnPlateau(
            optimizer, mode=mode,
            factor=args.lr_factor, patience=args.lr_patience,
        )


# ================================================================
# 训练函数: CNN 分类器 (对应原 train_stage1)
# ================================================================
def train_cnn(
    model, train_loader, test_loader,
    criterion, optimizer, scheduler, device, args, logger,
):
    """
    训练 CNN 调制格式分类器。

    数据流: 
      - CDM模式: Ideal_CDM (带 30% Distorted_CDM 混合增强) → Logits → CE Loss
      - GAF模式: Distorted_CDM+GAF (双通道) → Logits → CE Loss
    """
    use_gaf = args.use_gaf
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")) if device.type == "cuda" else None

    logger.info(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} "
                f"{'Test Loss':<12} {'Test Acc':<10} {'LR':<10}")
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for distorted, ideal, labels, _ in train_loader:
            batch_size = ideal.size(0)
            if use_gaf:
                # GAF 模式: 直接使用双通道 (CDM+GAF) 输入
                inputs = distorted.to(device)
            else:
                # CDM 模式: 30% 概率混入 Distorted_CDM
                mask = torch.rand(batch_size, 1, 1, 1, device=ideal.device) < 0.3
                inputs = torch.where(mask.to(device), distorted.to(device), ideal.to(device))
            labels = labels.to(device)

            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                logits = model(inputs)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- 评估 ----
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for distorted, ideal, labels, _ in test_loader:
                inputs = distorted.to(device) if use_gaf else ideal.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)

                test_loss += loss.item() * inputs.size(0)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += inputs.size(0)

        test_loss /= max(test_total, 1)
        test_acc = test_correct / max(test_total, 1)

        # ---- 学习率调度 ----
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_acc)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<10.4f} "
                    f"{test_loss:<12.4f} {test_acc:<10.4f} {current_lr:<10.2e}")

        # ---- 保存最佳模型 ----
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            early_stop_counter = 0
            _save_checkpoint(model, optimizer, epoch, args, best_acc=best_acc)
            logger.info(f"  ==> [保存] 新最佳! Epoch {epoch}, Acc={best_acc:.4f}")
        else:
            early_stop_counter += 1

        # ---- 早停 ----
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            logger.info(f"[早停] {args.early_stop_patience} 轮未提升, 停止训练")
            break

    _save_final_checkpoint(model, optimizer, epoch, args, best_acc=best_acc)
    return best_acc


# ================================================================
# 训练函数: U-Net 去噪器 (对应原 train_stage2)
# ================================================================
def train_unet(
    model, train_loader, test_loader,
    criterion, optimizer, scheduler, device, args, logger,
):
    """
    训练 U-Net 湍流畸变修复网络。

    数据流: Distorted_CDM → Repaired_CDM → MSE(Ideal_CDM)
    支持 SNR 感知逐样本加权 (防止低 SNR 高阶调制坍塌)
    """
    best_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")) if device.type == "cuda" else None

    logger.info(f"{'Epoch':<6} {'Train Loss':<14} {'Test Loss':<14} {'LR':<10}")
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss, total_samples = 0.0, 0

        for distorted, ideal, labels, snr_values in train_loader:
            distorted = distorted.to(device)
            ideal = ideal.to(device)
            labels = labels.to(device)
            B = distorted.size(0)

            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                repaired = model(distorted)
                # 基础 MSE
                loss = criterion(repaired, ideal)

            # ---- SNR 感知加权 ----
            if args.snr_weighted:
                snr_weights = torch.ones(B, device=device)
                for i in range(B):
                    s = snr_values[i].item()
                    lbl = labels[i].item()
                    if lbl >= 2:  # 16QAM(2)/64QAM(3)/256QAM(4)
                        if   s <= -6: snr_weights[i] = 0.10
                        elif s <= -4: snr_weights[i] = 0.15
                        elif s <= -2: snr_weights[i] = 0.25
                        elif s <=  0: snr_weights[i] = 0.35
                        elif s <=  2: snr_weights[i] = 0.50
                        elif s <=  4: snr_weights[i] = 0.70
                        elif s <=  6: snr_weights[i] = 0.85
                w = snr_weights.view(-1, 1, 1, 1)
                diff = (repaired - ideal) ** 2
                loss = (diff * w).mean()

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        train_loss = total_loss / max(total_samples, 1)

        # ---- 评估 ----
        model.eval()
        test_loss, test_samples = 0.0, 0
        with torch.no_grad():
            for distorted, ideal, labels, _ in test_loader:
                distorted = distorted.to(device)
                ideal = ideal.to(device)
                repaired = model(distorted)
                loss = criterion(repaired, ideal)
                test_loss += loss.item() * distorted.size(0)
                test_samples += distorted.size(0)

        test_loss /= max(test_samples, 1)

        # ---- 学习率调度 ----
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"{epoch:<6} {train_loss:<14.6f} {test_loss:<14.6f} {current_lr:<10.2e}")

        # ---- 保存最佳模型 ----
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            early_stop_counter = 0
            _save_checkpoint(model, optimizer, epoch, args, best_loss=best_loss)
            logger.info(f"  ==> [保存] 新最佳! Epoch {epoch}, Loss={best_loss:.6f}")
        else:
            early_stop_counter += 1

        # ---- 早停 ----
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            logger.info(f"[早停] {args.early_stop_patience} 轮未降低, 停止训练")
            break

    _save_final_checkpoint(model, optimizer, epoch, args, best_loss=best_loss)
    return best_loss


# ================================================================
# 训练函数: U-Net + ViT 端到端联合微调 (对应原 train_stage3_e2e)
# ================================================================
def train_e2e(
    unet, vit, train_loader, test_loader,
    criterion_mse, criterion_ce, optimizer, scheduler, device, args, logger,
    cls_key="vit_state_dict",
):
    """
    端到端联合微调 U-Net + ViT。

    数据流: Distorted_CDM → U-Net → Repaired → ViT → Logits
    损失:   mse_weight * MSE(repaired, ideal) + ce_weight * CE(logits, labels)
    """
    best_loss = float("inf")
    best_epoch = 0
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")) if device.type == "cuda" else None

    logger.info(f"{'Epoch':<6} {'Tr Loss':<10} {'Tr MSE':<10} {'Tr CE':<10} "
                f"{'Tr Acc':<10} {'Te Loss':<10} {'Te Acc':<10} {'LR':<10}")
    logger.info("-" * 90)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        unet.train()
        vit.train()
        tr_loss, tr_mse, tr_ce, tr_correct, tr_total = 0.0, 0.0, 0.0, 0, 0

        for distorted, ideal, labels, _ in train_loader:
            distorted = distorted.to(device)
            ideal = ideal.to(device)
            labels = labels.to(device)
            B = distorted.size(0)

            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                repaired = unet(distorted)
                logits = vit(repaired)
                loss_mse = criterion_mse(repaired, ideal)
                loss_ce = criterion_ce(logits, labels)
                loss = args.mse_weight * loss_mse + args.ce_weight * loss_ce

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            tr_loss += loss.item() * B
            tr_mse += loss_mse.item() * B
            tr_ce += loss_ce.item() * B
            tr_correct += (logits.argmax(dim=1) == labels).sum().item()
            tr_total += B

        tr_loss /= max(tr_total, 1)
        tr_mse /= max(tr_total, 1)
        tr_ce /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # ---- 评估 ----
        unet.eval()
        vit.eval()
        te_loss, te_mse, te_ce, te_correct, te_total = 0.0, 0.0, 0.0, 0, 0
        with torch.no_grad():
            for distorted, ideal, labels, _ in test_loader:
                distorted = distorted.to(device)
                ideal = ideal.to(device)
                labels = labels.to(device)
                B = distorted.size(0)

                repaired = unet(distorted)
                logits = vit(repaired)
                loss_mse = criterion_mse(repaired, ideal)
                loss_ce = criterion_ce(logits, labels)
                loss = args.mse_weight * loss_mse + args.ce_weight * loss_ce

                te_loss += loss.item() * B
                te_mse += loss_mse.item() * B
                te_ce += loss_ce.item() * B
                te_correct += (logits.argmax(dim=1) == labels).sum().item()
                te_total += B

        te_loss /= max(te_total, 1)
        te_mse /= max(te_total, 1)
        te_ce /= max(te_total, 1)
        te_acc = te_correct / max(te_total, 1)

        # ---- 学习率调度 ----
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(te_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"{epoch:<6} {tr_loss:<10.4f} {tr_mse:<10.6f} {tr_ce:<10.4f} "
                    f"{tr_acc:<10.4f} {te_loss:<10.4f} {te_acc:<10.4f} {current_lr:<10.2e}")

        # ---- 保存最佳模型 ----
        if te_loss < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            _save_checkpoint_e2e(unet, vit, optimizer, epoch, args, best_loss=best_loss, cls_key=cls_key)
            logger.info(f"  ==> [保存] 新最佳! Epoch {epoch}, Loss={best_loss:.6f}")

    _save_final_checkpoint_e2e(unet, vit, optimizer, epoch, args, best_loss=best_loss, cls_key=cls_key)
    return best_loss


# ================================================================
# Checkpoint 保存函数
# ================================================================
def _save_checkpoint(model, optimizer, epoch, args, **metrics):
    """保存单模型 checkpoint。"""
    path = os.path.join(args.save_dir, f"{args.exp_name}_best.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)


def _save_final_checkpoint(model, optimizer, epoch, args, **metrics):
    """保存最终 checkpoint (用于中断恢复)。"""
    path = os.path.join(args.save_dir, f"{args.exp_name}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)
    logger = logging.getLogger("train")
    logger.info(f"[保存] 最终模型: {path}")


def _save_checkpoint_e2e(unet, cls_model, optimizer, epoch, args, cls_key="vit_state_dict", **metrics):
    """保存 e2e (UNet+分类器) checkpoint。cls_key 指定分类器权重在 ckpt 中的 key 名。"""
    path = os.path.join(args.save_dir, f"{args.exp_name}_best.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "unet_state_dict": unet.state_dict(),
        cls_key: cls_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)


def _save_final_checkpoint_e2e(unet, cls_model, optimizer, epoch, args, cls_key="vit_state_dict", **metrics):
    """保存 e2e 最终 checkpoint。"""
    path = os.path.join(args.save_dir, f"{args.exp_name}_final.pth")
    torch.save({
        "epoch": epoch,
        "unet_state_dict": unet.state_dict(),
        cls_key: cls_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)
    logger = logging.getLogger("train")
    logger.info(f"[保存] 最终模型: {path}")


# ================================================================
# 主入口
# ================================================================
def main():
    args = parse_args()

    # 自动选择 loss_type
    if args.loss_type is None:
        AUTO_LOSS = {"cnn": "ce", "unet": "mse", "vit": "hybrid", "unet_swin": "hybrid",
                       "unet_resnet18": "hybrid", "unet_swin": "hybrid"}
        args.loss_type = AUTO_LOSS[args.model_arch]

    # ---- 初始化 ----
    set_seed(args.seed)
    device = get_device()

    logger = setup_logger(
        "train",
        log_file=os.path.join(args.log_dir, f"{args.exp_name}.log"),
    )
    logger.info("=" * 60)
    logger.info(f"  实验: {args.exp_name}  |  架构: {args.model_arch}  |  损失: {args.loss_type}")
    logger.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info("-" * 60)

    # ---- 数据加载 ----
    logger.info(f"[数据] 目录: {args.train_data}")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.train_data,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        use_gaf=args.use_gaf,
    )
    logger.info(f"  训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"  测试集: {len(test_loader.dataset)} 样本")
    logger.info(f"  输入通道数: {'2 (CDM+GAF)' if args.use_gaf else '1 (CDM only)'}")

    # ============ CNN 分类器训练 ============
    in_channels = 2 if args.use_gaf else 1
    if args.model_arch == "cnn":
        model = ConstellationCNN(num_classes=5, dropout_rate=args.dropout_rate,
                                 in_channels=in_channels).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[模型] ConstellationCNN({'GAF' if args.use_gaf else 'CDM'}) — {params:,} 参数")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = _build_scheduler(optimizer, args, mode="max")

        best_metric = train_cnn(model, train_loader, test_loader, criterion, optimizer,
                                scheduler, device, args, logger)
        logger.info(f"[完成] 最佳测试准确率: {best_metric:.4f}")

    # ============ U-Net 去噪器训练 ============
    elif args.model_arch == "unet":
        model = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[模型] ConstellationUNet({'GAF' if args.use_gaf else 'CDM'}) — {params:,} 参数")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = _build_scheduler(optimizer, args, mode="min")

        best_metric = train_unet(model, train_loader, test_loader, criterion, optimizer,
                                 scheduler, device, args, logger)
        logger.info(f"[完成] 最佳测试损失 (MSE): {best_metric:.6f}")

    # ============ U-Net + ViT 端到端 ============
    elif args.model_arch == "vit":
        unet = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        vit = ConstellationViT(img_size=64, patch_size=16, in_channels=1, num_classes=5).to(device)

        pu = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        pv = sum(p.numel() for p in vit.parameters() if p.requires_grad)
        logger.info(f"[模型] UNet: {pu:,} 参数  |  ViT: {pv:,} 参数")

        # 加载预训练 U-Net 权重
        if args.pretrained_unet and os.path.isfile(args.pretrained_unet):
            ckpt = torch.load(args.pretrained_unet, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                unet.load_state_dict(ckpt["model_state_dict"])
            else:
                unet.load_state_dict(ckpt)
            logger.info(f"[迁移] U-Net: {args.pretrained_unet}")
        elif args.pretrained_unet:
            logger.warning(f"[警告] 预训练 U-Net 未找到: {args.pretrained_unet}")

        if args.pretrained_vit and os.path.isfile(args.pretrained_vit):
            ckpt = torch.load(args.pretrained_vit, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            if hasattr(vit, "load_state_dict"):
                vit.load_state_dict(state)
            logger.info(f"[迁移] ViT: {args.pretrained_vit}")

        # 分层学习率
        optimizer = optim.AdamW([
            {"params": unet.parameters(), "lr": args.lr_unet},
            {"params": vit.parameters(), "lr": args.lr_vit},
        ], weight_decay=args.weight_decay)

        scheduler = _build_scheduler(optimizer, args, mode="min")
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()

        best_metric = train_e2e(unet, vit, train_loader, test_loader,
                                criterion_mse, criterion_ce, optimizer,
                                scheduler, device, args, logger)
        logger.info(f"[完成] 最佳联合损失: {best_metric:.6f}")

    # ============ U-Net + ResNet-18 端到端 ============
    elif args.model_arch == "unet_resnet18":
        model = UNet_ResNet18_E2E(num_classes=5, in_channels=in_channels).to(device)

        pu = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        pr = sum(p.numel() for p in model.resnet.parameters() if p.requires_grad)
        logger.info(f"[模型] UNet: {pu:,} 参数  |  ResNet-18 (adapt): {pr:,} 参数")

        # 加载预训练 U-Net 权重
        if args.pretrained_unet and os.path.isfile(args.pretrained_unet):
            ckpt = torch.load(args.pretrained_unet, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.unet.load_state_dict(ckpt["model_state_dict"])
            else:
                model.unet.load_state_dict(ckpt)
            logger.info(f"[迁移] U-Net: {args.pretrained_unet}")
        elif args.pretrained_unet:
            logger.warning(f"[警告] 预训练 U-Net 未找到: {args.pretrained_unet}")

        # 分层学习率
        optimizer = optim.AdamW([
            {"params": model.unet.parameters(), "lr": args.lr_unet},
            {"params": model.resnet.parameters(), "lr": args.lr_resnet},
        ], weight_decay=args.weight_decay)

        scheduler = _build_scheduler(optimizer, args, mode="min")
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()

        best_metric = train_e2e(model.unet, model.resnet, train_loader, test_loader,
                                criterion_mse, criterion_ce, optimizer,
                                scheduler, device, args, logger,
                                cls_key="resnet_state_dict")
        logger.info(f"[完成] 最佳联合损失: {best_metric:.6f}")

    # ============ U-Net + Swin-T 端到端 ============
    elif args.model_arch == "unet_swin":
        model = UNet_Swin_E2E(num_classes=5, in_channels=in_channels).to(device)

        pu = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        ps = sum(p.numel() for p in model.swin.parameters() if p.requires_grad)
        logger.info(f"[模型] UNet: {pu:,} 参数  |  Swin-T (adapt): {ps:,} 参数")

        # 加载预训练 U-Net 权重
        if args.pretrained_unet and os.path.isfile(args.pretrained_unet):
            ckpt = torch.load(args.pretrained_unet, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.unet.load_state_dict(ckpt["model_state_dict"])
            else:
                model.unet.load_state_dict(ckpt)
            logger.info(f"[迁移] U-Net: {args.pretrained_unet}")
        elif args.pretrained_unet:
            logger.warning(f"[警告] 预训练 U-Net 未找到: {args.pretrained_unet}")

        # 分层学习率: UNet 小步微调, Swin-T 从头训练
        optimizer = optim.AdamW([
            {"params": model.unet.parameters(), "lr": args.lr_unet},
            {"params": model.swin.parameters(), "lr": args.lr_swin},
        ], weight_decay=args.weight_decay)

        scheduler = _build_scheduler(optimizer, args, mode="min")
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()

        best_metric = train_e2e(model.unet, model.swin, train_loader, test_loader,
                                criterion_mse, criterion_ce, optimizer,
                                scheduler, device, args, logger,
                                cls_key="swin_state_dict")
        logger.info(f"[完成] 最佳联合损失: {best_metric:.6f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
