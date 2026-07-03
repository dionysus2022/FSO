"""
train_exp_data.py — 实验数据调制格式识别训练脚本
==================================================
基于 FSO 项目的模型架构，适配实验数据命名格式
数据文件: exp_frame_001_mod_1.mat → exp_frame_100_mod_8.mat
模型: ConstellationCNN / ConstellationUNet + ConstellationViT

用法:
  python train_exp_data.py
  python train_exp_data.py --model_arch cnn --epochs 50
  python train_exp_data.py --model_arch vit --epochs 30
"""

import os
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import scipy.io as sio

# 项目内部导入
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E
from utils import set_seed, get_device, setup_logger


# ================================================================
# 参数解析
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="实验数据调制格式识别训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str,
                        default=r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\dataset_cdm_exp_all",
                        help="实验数据目录 (exp_frame_XXX_mod_Y.mat)")
    parser.add_argument("--model_arch", type=str, default="cnn",
                        choices=["cnn", "vit", "unet_resnet18"],
                        help="模型架构")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split_ratio", type=float, default=0.7, help="训练/测试比例")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称（默认自动生成）")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="仿真预训练模型路径 (如 ./results/checkpoints/unet_resnet18_e2e_best.pth)")

    return parser.parse_args()


# ================================================================
# 实验数据集
# ================================================================
class ExpDataset(Dataset):
    """实验数据数据集：加载 exp_frame_XXX_mod_Y.mat 文件"""

    # 文件名正则: exp_frame_001_mod_1.mat
    FILENAME_PATTERN = re.compile(r'exp_frame_(\d+)_mod_(\d+)\.mat$')

    # 标签映射: {1:BPSK, 2:QPSK, 4:16QAM, 6:64QAM, 8:256QAM} → {0,1,2,3,4}
    LABEL_MAPPING = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
    CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]

    def __init__(self, data_dir: str, augment: bool = False):
        all_files = glob.glob(os.path.join(data_dir, "*.mat"))
        self.files = []
        self.augment = augment

        for f in all_files:
            basename = os.path.basename(f)
            m = self.FILENAME_PATTERN.search(basename)
            if m:
                mod_bits = int(m.group(2))
                if mod_bits in self.LABEL_MAPPING:
                    self.files.append(f)

        self.files.sort()
        print(f"[数据集] 总样本数: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _augment_cdm(cdm_np: np.ndarray) -> np.ndarray:
        """数据扩增: 随机相位旋转 (±5°) + 微弱高斯白噪声注入"""
        from scipy.ndimage import rotate

        angle = np.random.uniform(-5.0, 5.0)
        cdm_aug = rotate(cdm_np, angle, axes=(-2, -1), reshape=False,
                         order=1, mode='constant', cval=0.0)

        noise = np.random.randn(*cdm_aug.shape).astype(np.float32) * 0.005
        cdm_aug = cdm_aug.astype(np.float32) + noise
        cdm_aug = np.clip(cdm_aug, 0.0, None)
        return cdm_aug

    def __getitem__(self, idx):
        mat_path = self.files[idx]
        mat_data = sio.loadmat(mat_path)

        # 加载 CDM 数据 (64×64)
        distorted = mat_data["Distorted_CDM"].astype(np.float32)
        ideal = mat_data["Ideal_CDM"].astype(np.float32)

        # 提取标签
        label_bits = int(np.squeeze(mat_data["Label_Bits"]))
        label = self.LABEL_MAPPING[label_bits]

        # ---- 数据扩增 (仅在训练时开启) ----
        if self.augment:
            distorted = self._augment_cdm(distorted)
            ideal = self._augment_cdm(ideal)

        # 转为 Tensor，增加通道维度: (64,64) → (1,64,64)
        distorted = torch.from_numpy(distorted).unsqueeze(0)
        ideal = torch.from_numpy(ideal).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return distorted, ideal, label


# ================================================================
# 训练函数: CNN 分类器
# ================================================================
def train_cnn(model, train_loader, test_loader, criterion, optimizer,
              scheduler, device, args, logger):
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 10

    logger.info(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} "
                f"{'Test Loss':<12} {'Test Acc':<10} {'LR':<10}")
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for distorted, ideal, labels in train_loader:
            # 30% 概率混入 Distorted_CDM 做数据增强
            batch_size = ideal.size(0)
            mask = torch.rand(batch_size, 1, 1, 1) < 0.3
            inputs = torch.where(mask, distorted, ideal).to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- 测试 ----
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for distorted, ideal, labels in test_loader:
                inputs = ideal.to(device)
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

        if early_stop_counter >= early_stop_patience:
            logger.info(f"[早停] {early_stop_patience} 轮未提升")
            break

    return best_acc


# ================================================================
# 训练函数: U-Net 去噪器
# ================================================================
def train_unet(model, train_loader, test_loader, criterion, optimizer,
               scheduler, device, args, logger):
    best_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 10

    logger.info(f"{'Epoch':<6} {'Train Loss':<14} {'Test Loss':<14} {'LR':<10}")
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss, total_samples = 0.0, 0

        for distorted, ideal, labels in train_loader:
            distorted = distorted.to(device)
            ideal = ideal.to(device)
            B = distorted.size(0)

            repaired = model(distorted)
            loss = criterion(repaired, ideal)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        train_loss = total_loss / max(total_samples, 1)

        # ---- 测试 ----
        model.eval()
        test_loss, test_samples = 0.0, 0
        with torch.no_grad():
            for distorted, ideal, labels in test_loader:
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

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            early_stop_counter = 0
            _save_checkpoint(model, optimizer, epoch, args, best_loss=best_loss)
            logger.info(f"  ==> [保存] 新最佳! Epoch {epoch}, Loss={best_loss:.6f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            logger.info(f"[早停] {early_stop_patience} 轮未降低")
            break

    return best_loss


# ================================================================
# Checkpoint 保存
# ================================================================
def _save_checkpoint(model, optimizer, epoch, args, **metrics):
    path = os.path.join("./results/checkpoints", f"{args.exp_name}_best.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)


# ================================================================
# 评估
# ================================================================
@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    for distorted, ideal, labels in test_loader:
        inputs = ideal.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(total, 1)

    # 打印每类准确率
    print(f"\n{'='*50}")
    print(f"  分类报告")
    print(f"{'='*50}")
    print(f"{'类别':<12} {'准确率':<12} {'样本数':<10}")
    print("-" * 34)
    for c in range(5):
        class_total = sum(1 for l in all_labels if l == c)
        class_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l == c)
        class_acc = class_correct / max(class_total, 1) * 100
        print(f"{class_names[c]:<12} {class_acc:<12.2f} {class_total:<10}")
    print("-" * 34)
    print(f"{'总准确率':<12} {acc*100:<12.2f} {total:<10}")
    print(f"{'='*50}\n")

    return acc, all_preds, all_labels


# ================================================================
# 加载预训练权重
# ================================================================
def load_pretrained_weights(ckpt_path, model_arch, device, logger):
    """加载仿真预训练权重，返回 (模型组件, 加载信息)"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(f"[预训练] 加载: {ckpt_path}")
    logger.info(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")

    if model_arch == "cnn":
        sd = ckpt.get("model_state_dict", ckpt)
        return sd, ckpt

    elif model_arch == "vit":
        unet_sd = ckpt.get("unet_state_dict")
        vit_sd = ckpt.get("vit_state_dict")
        if unet_sd is None or vit_sd is None:
            raise KeyError("Checkpoint 缺少 unet_state_dict / vit_state_dict")
        return (unet_sd, vit_sd), ckpt

    elif model_arch == "unet_resnet18":
        unet_sd = ckpt.get("unet_state_dict")
        resnet_sd = ckpt.get("resnet_state_dict")
        if unet_sd is None or resnet_sd is None:
            raise KeyError("Checkpoint 缺少 unet_state_dict / resnet_state_dict")
        return (unet_sd, resnet_sd), ckpt

    return None, ckpt


# ================================================================
# 主入口
# ================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # 实验名称
    if args.exp_name is None:
        suffix = "_ft" if args.pretrained_ckpt else ""
        args.exp_name = f"exp_{args.model_arch}{suffix}"

    # 日志
    logger = setup_logger(
        "train_exp",
        log_file=os.path.join("./results/logs", f"{args.exp_name}.log"),
    )
    logger.info("=" * 60)
    logger.info(f"实验数据调制格式识别训练")
    logger.info(f"模型: {args.model_arch}  |  数据: {args.data_dir}")
    if args.pretrained_ckpt:
        logger.info(f"预训练: {args.pretrained_ckpt}")
    logger.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info("-" * 60)

    # ---- 1. 加载数据 ----
    logger.info(f"[数据] 加载中...")
    import random
    full_dataset = ExpDataset(args.data_dir, augment=False)
    n_total = len(full_dataset)
    n_train = int(n_total * args.split_ratio)

    indices = list(range(n_total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_dataset = ExpDataset(args.data_dir, augment=True)
    test_dataset = ExpDataset(args.data_dir, augment=False)
    train_ds = Subset(train_dataset, train_idx)
    test_ds = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"  训练集: {n_train} 样本")
    logger.info(f"  测试集: {n_total - n_train} 样本")

    # ---- 2. 构建模型 ----
    in_channels = 1

    if args.model_arch == "cnn":
        model = ConstellationCNN(num_classes=5, dropout_rate=0.3, in_channels=in_channels).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[模型] ConstellationCNN — {params:,} 参数")

        # 加载预训练权重
        if args.pretrained_ckpt:
            sd, pt_ckpt = load_pretrained_weights(args.pretrained_ckpt, "cnn", device, logger)
            model.load_state_dict(sd)
            logger.info("  [预训练] CNN 权重已加载，进行微调")

        criterion = nn.CrossEntropyLoss()
        # 微调时用更小的学习率
        ft_lr = args.lr * 0.1 if args.pretrained_ckpt else args.lr
        optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        best_acc = train_cnn(model, train_loader, test_loader, criterion, optimizer,
                             scheduler, device, args, logger)
        logger.info(f"[完成] 最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

        # 评估
        _ = evaluate_model(model, test_loader, device, ExpDataset.CLASS_NAMES)

    elif args.model_arch == "unet":
        model = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[模型] ConstellationUNet — {params:,} 参数")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

        best_loss = train_unet(model, train_loader, test_loader, criterion, optimizer,
                               scheduler, device, args, logger)
        logger.info(f"[完成] 最佳 MSE: {best_loss:.6f}")

    elif args.model_arch == "vit":
        # U-Net + ViT 端到端
        unet = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True).to(device)
        vit = ConstellationViT(img_size=64, patch_size=16, in_channels=1, num_classes=5).to(device)
        pu = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        pv = sum(p.numel() for p in vit.parameters() if p.requires_grad)
        logger.info(f"[模型] UNet: {pu:,} 参数  | ViT: {pv:,} 参数")

        # 加载预训练权重
        if args.pretrained_ckpt:
            (unet_sd, vit_sd), pt_ckpt = load_pretrained_weights(args.pretrained_ckpt, "vit", device, logger)
            unet.load_state_dict(unet_sd)
            vit.load_state_dict(vit_sd)
            logger.info("  [预训练] UNet + ViT 权重已加载，进行微调")

        unet_lr = 1e-5 if not args.pretrained_ckpt else 1e-6
        vit_lr = 1e-4 if not args.pretrained_ckpt else 1e-5
        optimizer = optim.AdamW([
            {"params": unet.parameters(), "lr": unet_lr},
            {"params": vit.parameters(), "lr": vit_lr},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()

        best_acc = 0.0
        mse_weight, ce_weight = 0.5, 0.5
        logger.info(f"{'Epoch':<6} {'Tr Acc':<10} {'Te Loss':<10} {'Te Acc':<10} {'LR':<10}")
        logger.info("-" * 50)

        for epoch in range(1, args.epochs + 1):
            unet.train()
            vit.train()
            tr_correct, tr_total = 0, 0
            for distorted, ideal, labels in train_loader:
                distorted, ideal, labels = distorted.to(device), ideal.to(device), labels.to(device)
                repaired = unet(distorted)
                logits = vit(repaired)
                loss = mse_weight * criterion_mse(repaired, ideal) + ce_weight * criterion_ce(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                tr_correct += (logits.argmax(dim=1) == labels).sum().item()
                tr_total += labels.size(0)

            unet.eval()
            vit.eval()
            te_loss, te_correct, te_total = 0.0, 0, 0
            with torch.no_grad():
                for distorted, ideal, labels in test_loader:
                    distorted, ideal, labels = distorted.to(device), ideal.to(device), labels.to(device)
                    repaired = unet(distorted)
                    logits = vit(repaired)
                    loss = mse_weight * criterion_mse(repaired, ideal) + ce_weight * criterion_ce(logits, labels)
                    te_loss += loss.item() * labels.size(0)
                    te_correct += (logits.argmax(dim=1) == labels).sum().item()
                    te_total += labels.size(0)

            tr_acc = tr_correct / max(tr_total, 1)
            te_loss /= max(te_total, 1)
            te_acc = te_correct / max(te_total, 1)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"{epoch:<6} {tr_acc:<10.4f} {te_loss:<10.4f} {te_acc:<10.4f} {current_lr:<10.2e}")

            if te_acc > best_acc:
                best_acc = te_acc
                _save_checkpoint_vit(unet, vit, optimizer, epoch, args, best_acc=best_acc)

        logger.info(f"[完成] 最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

    elif args.model_arch == "unet_resnet18":
        model = UNet_ResNet18_E2E(num_classes=5, in_channels=in_channels).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[模型] UNet_ResNet18_E2E — {params:,} 参数")

        # 加载预训练权重
        if args.pretrained_ckpt:
            (unet_sd, resnet_sd), pt_ckpt = load_pretrained_weights(args.pretrained_ckpt, "unet_resnet18", device, logger)
            model.unet.load_state_dict(unet_sd)
            model.resnet.load_state_dict(resnet_sd)
            logger.info("  [预训练] UNet + ResNet-18 权重已加载，进行微调")

        ft_lr = args.lr * 0.1 if args.pretrained_ckpt else args.lr
        optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        logger.info(f"{'Epoch':<6} {'Tr Acc':<10} {'Te Acc':<10} {'LR':<10}")
        logger.info("-" * 40)
        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_correct, tr_total = 0, 0
            for distorted, ideal, labels in train_loader:
                distorted, labels = distorted.to(device), labels.to(device)
                logits = model(distorted)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                tr_correct += (logits.argmax(dim=1) == labels).sum().item()
                tr_total += labels.size(0)

            model.eval()
            te_correct, te_total = 0, 0
            with torch.no_grad():
                for distorted, ideal, labels in test_loader:
                    distorted, labels = distorted.to(device), labels.to(device)
                    logits = model(distorted)
                    te_correct += (logits.argmax(dim=1) == labels).sum().item()
                    te_total += labels.size(0)

            tr_acc = tr_correct / max(tr_total, 1)
            te_acc = te_correct / max(te_total, 1)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"{epoch:<6} {tr_acc:<10.4f} {te_acc:<10.4f} {current_lr:<10.2e}")

            if te_acc > best_acc:
                best_acc = te_acc
                _save_checkpoint(model, optimizer, epoch, args, best_acc=best_acc)

        logger.info(f"[完成] 最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

    logger.info("=" * 60)


def _save_checkpoint_vit(unet, vit, optimizer, epoch, args, **metrics):
    path = os.path.join("./results/checkpoints", f"{args.exp_name}_best.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "unet_state_dict": unet.state_dict(),
        "vit_state_dict": vit.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **metrics,
    }, path)


if __name__ == "__main__":
    main()
