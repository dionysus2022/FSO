#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mrf_aug_da.py
实验2+3：在 RAP (M3) 基础上添加信号增强和域自适应 (MMD)。
复用 train_mrf_lowmem.py 的模型、特征、缓存基础设施。

新增：
  - AugmentedDataset: 训练时对 CDM 和 stats 施加轻量增强
    * CDM: 随机 90° 旋转 + 小高斯噪声 + 幅度缩放
    * Stats: 小高斯噪声
  - MMD loss: 对 A/B 来源的特征做最大均值差异对齐
  - source-aware DataLoader: 同时返回 dataset_source 标签
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# 复用主训练脚本的基础设施
sys.path.insert(0, r"D:\Project_code\FSO_research\code\python")
from train_mrf_lowmem import (
    load_manifest, group_split_by_rx_file, build_cache_for_turbulence,
    normalize_stats_by_train, MultiRepNet, MOD_NAMES, save_confusion_matrix,
    simple_confusion_matrix, class_weights_from_y,
)
from sklearn.metrics import f1_score, balanced_accuracy_score


# =========================
# 增强数据集
# =========================
class AugmentedDataset(Dataset):
    """训练时对 CDM 和 stats 施加轻量增强。"""
    def __init__(self, X_cdm, X_iq, X_stats, y, indices, sources,
                 augment=False, noise_std=0.01, scale_range=(0.9, 1.1)):
        self.X_cdm = X_cdm
        self.X_iq = X_iq
        self.X_stats = X_stats
        self.y = y
        self.indices = indices.astype(np.int64)
        self.sources = sources  # 0=A, 1=B
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        cdm = self.X_cdm[i].copy()
        stats = self.X_stats[i].copy()
        src = int(self.sources[i])

        if self.augment:
            # CDM: 随机 90° 旋转（QAM 对称性）
            k = np.random.randint(0, 4)
            cdm = np.rot90(cdm, k=k, axes=(1, 2)).copy()
            # CDM: 幅度缩放
            scale = np.random.uniform(*self.scale_range)
            cdm = cdm * scale
            # CDM: 小高斯噪声
            cdm = cdm + np.random.randn(*cdm.shape).astype(np.float32) * self.noise_std
            cdm = np.clip(cdm, 0, None)
            # Stats: 小高斯噪声
            stats = stats + np.random.randn(*stats.shape).astype(np.float32) * self.noise_std * 0.5

        return (
            torch.from_numpy(cdm),
            torch.from_numpy(self.X_iq[i]),
            torch.from_numpy(stats),
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor(src, dtype=torch.long),
        )


# =========================
# MMD 损失
# =========================
def gaussian_kernel(x, y, sigma_list=[1, 5, 10]):
    """多核高斯 MMD"""
    n = x.size(0)
    m = y.size(0)
    dist = torch.cdist(x, y) ** 2  # [n, m]
    loss = 0
    for s in sigma_list:
        loss += torch.exp(-dist / (2 * s ** 2)).mean()
    return loss


def mmd_loss(feat_a, feat_b):
    """MMD^2 = E[k(a,a)] + E[k(b,b)] - 2*E[k(a,b)]"""
    if feat_a.size(0) == 0 or feat_b.size(0) == 0:
        return torch.tensor(0.0, device=feat_a.device)
    kaa = gaussian_kernel(feat_a, feat_a)
    kbb = gaussian_kernel(feat_b, feat_b)
    kab = gaussian_kernel(feat_a, feat_b)
    return kaa + kbb - 2 * kab


def coral_loss(feat_a, feat_b):
    """CORAL: 对齐协方差矩阵"""
    if feat_a.size(0) < 2 or feat_b.size(0) < 2:
        return torch.tensor(0.0, device=feat_a.device)
    d = feat_a.size(1)
    fa = feat_a - feat_a.mean(0, keepdim=True)
    fb = feat_b - feat_b.mean(0, keepdim=True)
    ca = (fa.T @ fa) / (feat_a.size(0) - 1)
    cb = (fb.T @ fb) / (feat_b.size(0) - 1)
    return ((ca - cb) ** 2).mean()


# =========================
# 提取特征（分类器前的融合特征）
# =========================
def extract_features(model, cdm, iq, stats):
    feats = []
    if model.use_cdm:
        feats.append(model.cdm_branch(cdm))
    if model.use_iq:
        feats.append(model.iq_branch(iq))
    if model.use_stats:
        feats.append(model.stats_branch(stats))
    return torch.cat(feats, dim=1)


# =========================
# 训练单湍流
# =========================
def train_one_turbulence_da(df, turb_name, args, device):
    out_dir = Path(args.out) / f"{turb_name}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}\n[{time.strftime('%H:%M:%S')}] Turbulence={turb_name} | model={args.model} | aug={args.augment} | da={args.da_method}\n{'='*80}")

    df_turb = df[df["turb_name"].astype(str) == turb_name].copy()
    split_df = group_split_by_rx_file(df_turb, seed=args.seed)
    split_df.to_csv(out_dir / "split.csv", index=False, encoding="utf-8-sig")

    cache_path = out_dir / f"cache_{turb_name}_{args.model}.npz"
    data = build_cache_for_turbulence(split_df, cache_path, args.model, rebuild_cache=args.rebuild_cache)

    X_cdm = data["X_cdm"].astype(np.float32)
    X_iq = data["X_iq"].astype(np.float32)
    X_stats = data["X_stats"].astype(np.float32)
    y = data["y"].astype(np.int64)
    split = data["split"].astype(str)

    X_stats, stats_mean, stats_std = normalize_stats_by_train(X_stats, split)

    # 从 split_df 获取 dataset_source
    source_arr = np.zeros(len(split_df), dtype=np.int64)
    for i, row in enumerate(split_df.itertuples()):
        src = getattr(row, 'dataset_source', 'A')
        source_arr[i] = 0 if str(src) == 'A' else 1

    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]

    train_ds = AugmentedDataset(X_cdm, X_iq, X_stats, y, idx_train, source_arr,
                                 augment=args.augment)
    val_ds = AugmentedDataset(X_cdm, X_iq, X_stats, y, idx_val, source_arr, augment=False)
    test_ds = AugmentedDataset(X_cdm, X_iq, X_stats, y, idx_test, source_arr, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    iq_in_ch = int(X_iq.shape[1])
    stats_dim = int(X_stats.shape[1])
    model = MultiRepNet(args.model, num_classes=6, iq_in_ch=iq_in_ch, stats_dim=stats_dim, dropout=args.dropout).to(device)

    weights = class_weights_from_y(y, idx_train, 6).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val = -1; best_epoch = -1; bad = 0; history = []
    ckpt = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0; n_sum = 0
        for cdm, iq, stats, yy, src in train_loader:
            cdm, iq, stats, yy, src = cdm.to(device), iq.to(device), stats.to(device), yy.to(device), src.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(cdm, iq, stats)
            loss_cls = criterion(logits, yy)

            # 域自适应损失
            loss_da = torch.tensor(0.0, device=device)
            if args.da_method != "none":
                feat = extract_features(model, cdm, iq, stats)
                mask_a = (src == 0)
                mask_b = (src == 1)
                if mask_a.sum() > 1 and mask_b.sum() > 1:
                    feat_a = feat[mask_a]
                    feat_b = feat[mask_b]
                    if args.da_method == "mmd":
                        loss_da = mmd_loss(feat_a, feat_b)
                    elif args.da_method == "coral":
                        loss_da = coral_loss(feat_a, feat_b)

            loss = loss_cls + args.da_weight * loss_da
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bs = yy.size(0); loss_sum += float(loss.detach()) * bs; n_sum += bs

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            yv, pv = [], []
            for cdm, iq, stats, yy, src in val_loader:
                cdm, iq, stats = cdm.to(device), iq.to(device), stats.to(device)
                pred = model(cdm, iq, stats).argmax(1)
                yv.append(yy.numpy()); pv.append(pred.cpu().numpy())
            yv = np.concatenate(yv); pv = np.concatenate(pv)
            val_acc = float((yv == pv).mean())

            ytr, ptr = [], []
            for cdm, iq, stats, yy, src in train_loader:
                cdm, iq, stats = cdm.to(device), iq.to(device), stats.to(device)
                pred = model(cdm, iq, stats).argmax(1)
                ytr.append(yy.numpy()); ptr.append(pred.cpu().numpy())
            ytr = np.concatenate(ytr); ptr = np.concatenate(ptr)
            train_acc = float((ytr == ptr).mean())

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": loss_sum/max(1,n_sum), "train_acc": train_acc, "val_acc": val_acc, "lr": lr_now})
        print(f"  Epoch {epoch:03d}/{args.epochs} | loss={loss_sum/max(1,n_sum):.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e}")

        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch; bad = 0
            torch.save({"model_state": model.state_dict(), "model_mode": args.model,
                        "stats_mean": stats_mean, "stats_std": stats_std,
                        "iq_in_ch": iq_in_ch, "stats_dim": stats_dim,
                        "best_val_acc": best_val, "best_epoch": best_epoch}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  Early stopping: best_epoch={best_epoch}, best_val_acc={best_val:.4f}")
                break

    # 加载最佳模型测试
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model_state"])
    model.eval()

    all_preds = {"train": [], "val": [], "test": []}
    for name, loader, indices in [("train", train_loader, idx_train), ("val", val_loader, idx_val), ("test", test_loader, idx_test)]:
        ys, ps, probs = [], [], []
        with torch.no_grad():
            for cdm, iq, stats, yy, src in loader:
                cdm, iq, stats = cdm.to(device), iq.to(device), stats.to(device)
                logits = model(cdm, iq, stats)
                prob = F.softmax(logits, 1)
                ys.append(yy.numpy()); ps.append(logits.argmax(1).cpu().numpy()); probs.append(prob.cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps); probs = np.concatenate(probs)
        acc = float((ys == ps).mean())
        f1 = f1_score(ys, ps, average="macro", zero_division=0)
        bal = balanced_accuracy_score(ys, ps)
        cm = simple_confusion_matrix(ys, ps)
        print(f"\n  [{name}] acc={acc:.4f} | bal_acc={bal:.4f} | f1={f1:.4f} | n={len(ys)}")

        # 保存预测
        mod_labels = [MOD_NAMES[y] for y in ys]
        pred_labels = [MOD_NAMES[p] for p in ps]
        df_pred = pd.DataFrame({"y_true": ys, "y_pred": ps, "true_name": mod_labels, "pred_name": pred_labels})
        for j, m in enumerate(MOD_NAMES):
            df_pred[f"prob_{m}"] = probs[:, j]
        df_pred.to_csv(out_dir / f"{turb_name}_{args.model}_{name}_predictions.csv", index=False, encoding="utf-8-sig")

        if name == "test":
            save_confusion_matrix(cm, MOD_NAMES, out_dir / "confusion_matrix.png",
                                  f"{turb_name}_{args.model} test CM")
            return {"turb_name": turb_name, "test_accuracy": acc, "test_f1_macro": f1,
                    "test_balanced_accuracy": bal, "out_dir": str(out_dir)}

    return {}


# =========================
# 主函数
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="cdm_stats_rap")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--turbulence", nargs="+", default=["weak", "strong"])
    # 增强
    ap.add_argument("--augment", action="store_true", help="启用 CDM/stats 增强")
    # 域自适应
    ap.add_argument("--da-method", default="none", choices=["none", "mmd", "coral"])
    ap.add_argument("--da-weight", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_manifest(Path(args.data_root))

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*70}\n# Seed = {seed}\n{'#'*70}")
        args.seed = seed
        out_seed = Path(args.out) / f"seed_{seed}"
        args.out = str(out_seed)
        out_seed.mkdir(parents=True, exist_ok=True)

        for turb in args.turbulence:
            res = train_one_turbulence_da(df, turb, args, device)
            res["seed"] = seed
            all_results.append(res)

        # 恢复 args.out
        args.out = str(Path(args.out).parent)

    # 汇总
    summary_path = Path(args.out).parent / f"summary_{args.model}_aug{int(args.augment)}_da{args.da_method}.csv"
    df_sum = pd.DataFrame(all_results)
    df_sum.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to {summary_path}")

    for turb in args.turbulence:
        sub = df_sum[df_sum["turb_name"] == turb]
        if len(sub) > 0:
            print(f"\n{turb}: acc = {sub['test_accuracy'].mean()*100:.2f} ± {sub['test_accuracy'].std()*100:.2f}%")

if __name__ == "__main__":
    main()
