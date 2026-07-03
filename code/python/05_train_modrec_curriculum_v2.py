#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_train_modrec_curriculum_v2.py

Stable training script for modulation recognition on synthetic GNN-channel data.

Why this v2 exists:
- The first training script may collapse to uniform predictions on the full dataset.
- This version adds:
  1) SNR curriculum: train high-SNR first, then gradually include low-SNR samples.
  2) max_steps_per_epoch: prevents one epoch from doing hundreds of destructive updates.
  3) LR warmup + cosine decay.
  4) gradient clipping.
  5) optional label smoothing.
  6) balanced class sampling inside the active SNR range.
  7) logging of logit_std and grad_norm to catch collapse early.

Expected shard format from 04_build_modrec_dataset_from_generated_channels.py:
    modrec_synth_shard_XXXX.npz with fields:
        Y_iq          : float32, shape [N, 2, K, T]
        mod_label     : int64/int32, shape [N]
        turb_label    : int64/int32, shape [N]
        target_snr_db : float32, shape [N]
        split_label   : optional int, 0=train, 1=val

Recommended first run:
    python 05_train_modrec_curriculum_v2.py ^
      --data_dir "D:/.../modrec_dataset_gnn_5_20dB" ^
      --out_dir  "D:/.../modrec_cnn_curriculum_v2" ^
      --model cnn ^
      --epochs 60 ^
      --batch_size 256 ^
      --optimizer sgd ^
      --lr 1e-2 ^
      --min_lr 1e-4 ^
      --warmup_epochs 3 ^
      --curriculum high2low ^
      --curriculum_snr_start 16 ^
      --curriculum_snr_end 5 ^
      --curriculum_epochs 25 ^
      --max_steps_per_epoch 64 ^
      --label_smoothing 0.03 ^
      --grad_clip 1.0 ^
      --normalize sample_rms ^
      --num_workers 0
"""

import argparse
import csv
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


MOD_NAMES_DEFAULT = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
TURB_NAMES_DEFAULT = {0: "weak", 1: "strong"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class ShardInfo:
    path: str
    n: int
    mod_label: np.ndarray
    turb_label: np.ndarray
    target_snr_db: np.ndarray
    split_label: Optional[np.ndarray]


class NPZShardCache:
    """Multi-shard cache with persistent mmap handles for Y_iq.

    Uses memory-mapped ``_yiq.npy`` (if available) for zero-copy access to the
    large I/Q array, falling back to loading the full NPZ otherwise.
    """

    def __init__(self, max_items: int = 3):
        self.max_items = max(1, int(max_items))
        self._mmaps: Dict[str, np.ndarray] = {}
        self._mmap_order: List[str] = []

    def get(self, path: str) -> np.ndarray:
        if path in self._mmaps:
            self._mmap_order.remove(path)
            self._mmap_order.append(path)
            return self._mmaps[path]
        # Try mmap-backed .npy first
        npy_path = path.replace(".npz", "_yiq.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path, mmap_mode="r")
        else:
            arr = np.load(path, allow_pickle=True)["Y_iq"]
        self._mmaps[path] = arr
        self._mmap_order.append(path)
        self._evict()
        return arr

    def _evict(self) -> None:
        while len(self._mmap_order) > self.max_items:
            old = self._mmap_order.pop(0)
            old_arr = self._mmaps.pop(old, None)
            if old_arr is not None:
                del old_arr


class ModRecShardDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        max_samples: int = 0,
        cache_shards: int = 3,
        normalize: str = "none",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split.lower()
        self.normalize = normalize.lower()
        if self.normalize not in ["none", "sample_rms"]:
            raise ValueError("--normalize must be 'none' or 'sample_rms'")

        shard_paths = sorted(glob.glob(os.path.join(data_dir, "modrec_synth_shard_*.npz")))
        if not shard_paths:
            shard_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found in: {data_dir}")

        self.shards: List[ShardInfo] = []
        self.samples: List[Tuple[int, int]] = []
        self.cache = NPZShardCache(cache_shards)
        rng = np.random.default_rng(seed)

        for shard_id, path in enumerate(shard_paths):
            with np.load(path, allow_pickle=True) as z:
                if "Y_iq" not in z.files or "mod_label" not in z.files:
                    continue
                n = int(z["Y_iq"].shape[0])
                mod_label = np.asarray(z["mod_label"]).astype(np.int64)
                turb_label = np.asarray(z["turb_label"]).astype(np.int64) if "turb_label" in z.files else np.zeros(n, dtype=np.int64)
                target_snr_db = np.asarray(z["target_snr_db"]).astype(np.float32) if "target_snr_db" in z.files else np.zeros(n, dtype=np.float32)
                split_label = np.asarray(z["split_label"]).astype(np.int64) if "split_label" in z.files else None

            self.shards.append(ShardInfo(path, n, mod_label, turb_label, target_snr_db, split_label))

            if split_label is not None:
                if self.split == "train":
                    local_indices = np.where(split_label == 0)[0]
                elif self.split in ["val", "valid", "validation"]:
                    local_indices = np.where(split_label == 1)[0]
                else:
                    local_indices = np.arange(n)
            else:
                perm = rng.permutation(n)
                n_val = max(1, int(round(n * val_ratio)))
                val_set = set(perm[:n_val].tolist())
                if self.split == "train":
                    local_indices = np.array([i for i in range(n) if i not in val_set], dtype=np.int64)
                elif self.split in ["val", "valid", "validation"]:
                    local_indices = np.array(sorted(val_set), dtype=np.int64)
                else:
                    local_indices = np.arange(n)

            for idx in local_indices.tolist():
                self.samples.append((shard_id, int(idx)))

        if not self.samples:
            raise RuntimeError(f"No samples for split='{split}'. Check split_label in shards.")

        if max_samples and max_samples > 0 and max_samples < len(self.samples):
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        # Fast metadata arrays for filtering/sampling.
        labels, turbs, snrs = [], [], []
        for shard_id, local_idx in self.samples:
            sh = self.shards[shard_id]
            labels.append(int(sh.mod_label[local_idx]))
            turbs.append(int(sh.turb_label[local_idx]))
            snrs.append(float(sh.target_snr_db[local_idx]))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.turbs = np.asarray(turbs, dtype=np.int64)
        self.snrs = np.asarray(snrs, dtype=np.float32)

        first = self.cache.get(self.shards[self.samples[0][0]].path)
        self.input_shape = tuple(first.shape[1:])
        self.num_classes = int(max([self.labels.max()] + [0])) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        shard_id, local_idx = self.samples[index]
        sh = self.shards[shard_id]
        x = np.asarray(self.cache.get(sh.path)[local_idx], dtype=np.float32)
        if self.normalize == "sample_rms":
            rms = np.sqrt(np.mean(x ** 2) + 1e-12)
            x = x / rms
        # copy avoids non-writable memmap/npz warning and silent surprises.
        x = np.array(x, dtype=np.float32, copy=True)
        y = int(sh.mod_label[local_idx])
        turb = int(sh.turb_label[local_idx])
        snr = float(sh.target_snr_db[local_idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(turb, dtype=torch.long), torch.tensor(snr, dtype=torch.float32)


def active_snr_min_for_epoch(args, epoch: int) -> float:
    if args.curriculum == "none":
        return float(args.train_snr_min)
    # high2low: start with high-SNR samples, gradually include lower SNR.
    e = min(max(epoch - 1, 0), max(args.curriculum_epochs - 1, 1))
    alpha = e / max(args.curriculum_epochs - 1, 1)
    return float(args.curriculum_snr_start * (1.0 - alpha) + args.curriculum_snr_end * alpha)


def make_active_indices(dataset: ModRecShardDataset, snr_min: float, snr_max: float, max_active_samples: int, seed: int, epoch: int) -> np.ndarray:
    mask = np.isfinite(dataset.snrs) & (dataset.snrs >= snr_min) & (dataset.snrs <= snr_max)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError(f"No active samples for SNR range [{snr_min}, {snr_max}].")
    if max_active_samples and max_active_samples > 0 and idx.size > max_active_samples:
        rng = np.random.default_rng(seed + 1009 * epoch)
        idx = rng.choice(idx, size=max_active_samples, replace=False)
    return idx.astype(np.int64)


def make_balanced_sampler(labels: np.ndarray, active_indices: np.ndarray, num_samples: int) -> WeightedRandomSampler:
    active_labels = labels[active_indices]
    counts = np.bincount(active_labels, minlength=int(labels.max()) + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[active_labels]
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=int(num_samples), replacement=True)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, groups: int = 8):
        super().__init__()
        if p is None:
            p = k // 2
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.act(x + y)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class CNNBaselineModRec(nn.Module):
    def __init__(self, num_classes: int = 6, width: int = 32, dropout: float = 0.10):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 3
        self.net = nn.Sequential(
            ConvGNAct(2, c1, 3, 1),
            ResidualBlock(c1, dropout=dropout * 0.5),
            ConvGNAct(c1, c2, 3, 2),
            ResidualBlock(c2, dropout=dropout * 0.5),
            ConvGNAct(c2, c3, 3, 2),
            ResidualBlock(c3, dropout=dropout * 0.5),
            SEBlock(c3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNAttentionModRec(nn.Module):
    def __init__(self, num_classes: int = 6, width: int = 32, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 3
        if c3 % heads != 0:
            raise ValueError(f"embed_dim={c3} must be divisible by heads={heads}")
        self.stem = nn.Sequential(
            ConvGNAct(2, c1, 3, 1),
            ResidualBlock(c1, dropout=dropout * 0.5),
            ConvGNAct(c1, c2, 3, 2),
            ResidualBlock(c2, dropout=dropout * 0.5),
            ConvGNAct(c2, c3, 3, 2),
            ResidualBlock(c3, dropout=dropout * 0.5),
            SEBlock(c3),
        )
        self.norm1 = nn.LayerNorm(c3)
        self.attn = nn.MultiheadAttention(c3, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(c3)
        self.ffn = nn.Sequential(
            nn.Linear(c3, c3 * 2), nn.SiLU(inplace=True), nn.Dropout(dropout), nn.Linear(c3 * 2, c3)
        )
        self.cls = nn.Sequential(nn.LayerNorm(c3), nn.Dropout(dropout), nn.Linear(c3, num_classes))

    def forward(self, x):
        feat = self.stem(x)
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        z = self.norm1(tokens)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        pooled = tokens.mean(dim=1)
        return self.cls(pooled)


def make_model(model_name: str, num_classes: int, width: int, heads: int, dropout: float) -> nn.Module:
    name = model_name.lower()
    if name in ["cnn", "baseline"]:
        return CNNBaselineModRec(num_classes=num_classes, width=width, dropout=dropout)
    if name in ["cnn_attention", "attention", "attn"]:
        return CNNAttentionModRec(num_classes=num_classes, width=width, heads=heads, dropout=dropout)
    raise ValueError("--model must be cnn or cnn_attention")


def parse_mod_names(s: str, num_classes: int) -> List[str]:
    if s.strip():
        names = [x.strip() for x in s.split(",") if x.strip()]
    else:
        names = MOD_NAMES_DEFAULT.copy()
    while len(names) < num_classes:
        names.append(f"class{len(names)}")
    return names


def compute_lr(args, global_step: int, total_steps: int) -> float:
    warmup_steps = max(0, int(args.warmup_epochs * args.max_steps_per_epoch))
    if warmup_steps > 0 and global_step < warmup_steps:
        return args.lr * float(global_step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return args.lr
    t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * t))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def grad_norm_l2(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.pow(2).sum().item())
    return math.sqrt(max(total, 0.0))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, label_smoothing: float = 0.0):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, total, correct = 0.0, 0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    group = {}
    logit_std_sum, logit_std_n = 0.0, 0

    for x, y, turb, snr in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        b = int(y.numel())
        total_loss += float(loss.item()) * b
        total += b
        correct += int((pred == y).sum().item())
        logit_std_sum += float(logits.detach().std().item()) * b
        logit_std_n += b

        y_np, p_np = y.cpu().numpy(), pred.cpu().numpy()
        turb_np, snr_np = turb.cpu().numpy(), snr.cpu().numpy()
        for yy, pp, tt, ss in zip(y_np, p_np, turb_np, snr_np):
            cm[int(yy), int(pp)] += 1
            key = (int(round(float(ss))), int(tt), int(yy))
            if key not in group:
                group[key] = [0, 0]
            group[key][0] += int(yy == pp)
            group[key][1] += 1

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "cm": cm,
        "group": group,
        "total": total,
        "logit_std": logit_std_sum / max(1, logit_std_n),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, args, global_step: int, total_steps: int):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_loss, total, correct = 0.0, 0, 0
    grad_sum, grad_n = 0.0, 0
    logit_std_sum, logit_std_n = 0.0, 0
    steps_done = 0
    amp_enabled = bool(args.amp and device.type == "cuda")

    for step_in_epoch, (x, y, _, _) in enumerate(loader):
        if args.max_steps_per_epoch > 0 and step_in_epoch >= args.max_steps_per_epoch:
            break
        lr_now = compute_lr(args, global_step, total_steps)
        set_optimizer_lr(optimizer, lr_now)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = grad_norm_l2(model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            gn = grad_norm_l2(model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        pred = torch.argmax(logits.detach(), dim=1)
        b = int(y.numel())
        total_loss += float(loss.item()) * b
        total += b
        correct += int((pred == y).sum().item())
        grad_sum += float(gn)
        grad_n += 1
        logit_std_sum += float(logits.detach().std().item()) * b
        logit_std_n += b
        steps_done += 1
        global_step += 1

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "grad_norm": grad_sum / max(1, grad_n),
        "logit_std": logit_std_sum / max(1, logit_std_n),
        "steps": steps_done,
        "global_step": global_step,
        "lr": optimizer.param_groups[0]["lr"],
    }


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_confusion_matrix(path: str, cm: np.ndarray, mod_names: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(mod_names[: cm.shape[0]]))
        for i in range(cm.shape[0]):
            w.writerow([mod_names[i] if i < len(mod_names) else str(i)] + cm[i].tolist())


def save_group_accuracy(path: str, group: Dict, mod_names: Sequence[str]) -> None:
    rows = []
    for (snr, turb, mod), (c, n) in sorted(group.items()):
        rows.append({
            "target_snr_db": snr,
            "turb_label": turb,
            "turb_name": TURB_NAMES_DEFAULT.get(turb, str(turb)),
            "mod_label": mod,
            "mod_name": mod_names[mod] if mod < len(mod_names) else str(mod),
            "correct": int(c),
            "total": int(n),
            "acc": float(c) / max(1, int(n)),
        })
    save_csv(path, rows)


def save_class_accuracy(path: str, cm: np.ndarray, mod_names: Sequence[str]) -> None:
    rows = []
    for i in range(cm.shape[0]):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        rows.append({
            "mod_label": i,
            "mod_name": mod_names[i] if i < len(mod_names) else str(i),
            "correct": correct,
            "total": total,
            "acc": correct / max(1, total),
        })
    save_csv(path, rows)


def summarize_label_counts(labels: np.ndarray, indices: np.ndarray, num_classes: int) -> Dict[str, int]:
    counts = np.bincount(labels[indices], minlength=num_classes)
    return {str(i): int(counts[i]) for i in range(num_classes)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="cnn", choices=["cnn", "cnn_attention"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps_per_epoch", type=int, default=64, help="0 means use all batches")
    p.add_argument("--balanced_sampler", action="store_true", default=True)
    p.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false")
    p.add_argument("--curriculum", default="high2low", choices=["none", "high2low"])
    p.add_argument("--curriculum_snr_start", type=float, default=16.0)
    p.add_argument("--curriculum_snr_end", type=float, default=5.0)
    p.add_argument("--curriculum_epochs", type=int, default=25)
    p.add_argument("--train_snr_min", type=float, default=5.0)
    p.add_argument("--train_snr_max", type=float, default=20.0)
    p.add_argument("--max_active_samples", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--normalize", default="sample_rms", choices=["none", "sample_rms"])
    p.add_argument("--cache_shards", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mod_names", default=",".join(MOD_NAMES_DEFAULT))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = ModRecShardDataset(
        args.data_dir, split="train", val_ratio=args.val_ratio, seed=args.seed,
        max_samples=args.max_train_samples, cache_shards=args.cache_shards, normalize=args.normalize,
    )
    val_set = ModRecShardDataset(
        args.data_dir, split="val", val_ratio=args.val_ratio, seed=args.seed,
        max_samples=args.max_val_samples, cache_shards=args.cache_shards, normalize=args.normalize,
    )
    num_classes = max(train_set.num_classes, val_set.num_classes)
    mod_names = parse_mod_names(args.mod_names, num_classes)

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model = make_model(args.model, num_classes, args.width, args.heads, args.dropout).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    total_steps = max(1, args.epochs * max(1, args.max_steps_per_epoch))
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "input_shape": train_set.input_shape,
        "num_classes": num_classes,
        "num_train_samples_total": len(train_set),
        "num_val_samples": len(val_set),
        "mod_names": mod_names,
    })
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("========== ModRec Curriculum v2 ==========")
    print(f"data_dir       : {args.data_dir}")
    print(f"out_dir        : {args.out_dir}")
    print(f"device         : {device}")
    print(f"model          : {args.model}")
    print(f"optimizer      : {args.optimizer}")
    print(f"input_shape    : {train_set.input_shape}")
    print(f"classes        : {num_classes} {mod_names[:num_classes]}")
    print(f"train total    : {len(train_set)}")
    print(f"val total      : {len(val_set)}")
    print("==========================================")

    log_rows: List[Dict] = []
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        snr_min = active_snr_min_for_epoch(args, epoch)
        active_idx = make_active_indices(
            train_set, snr_min=snr_min, snr_max=args.train_snr_max,
            max_active_samples=args.max_active_samples, seed=args.seed, epoch=epoch,
        )
        active_subset = Subset(train_set, active_idx.tolist())

        if args.max_steps_per_epoch and args.max_steps_per_epoch > 0:
            num_samples_epoch = args.max_steps_per_epoch * args.batch_size
        else:
            num_samples_epoch = len(active_subset)

        if args.balanced_sampler:
            sampler = make_balanced_sampler(train_set.labels, active_idx, num_samples_epoch)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            active_subset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, args, global_step, total_steps)
        global_step = int(tr["global_step"])
        val = evaluate(model, val_loader, device, num_classes, label_smoothing=0.0)

        counts = summarize_label_counts(train_set.labels, active_idx, num_classes)
        row = {
            "epoch": epoch,
            "lr": tr["lr"],
            "active_snr_min": snr_min,
            "active_snr_max": args.train_snr_max,
            "active_samples": int(active_idx.size),
            "steps": int(tr["steps"]),
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_logit_std": tr["logit_std"],
            "train_grad_norm": tr["grad_norm"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_logit_std": val["logit_std"],
            "label_counts_json": json.dumps(counts, ensure_ascii=False),
        }
        log_rows.append(row)
        save_csv(os.path.join(args.out_dir, "train_log.csv"), log_rows)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"snr>={snr_min:.2f} | n={active_idx.size} | steps={tr['steps']} | "
            f"lr={tr['lr']:.3e} | "
            f"train loss={tr['loss']:.4f}, acc={tr['acc']:.4f}, logit_std={tr['logit_std']:.4f}, grad={tr['grad_norm']:.4f} | "
            f"val loss={val['loss']:.4f}, acc={val['acc']:.4f}, logit_std={val['logit_std']:.4f}"
        )

        is_best = val["acc"] > best_acc
        if is_best:
            best_acc = float(val["acc"])
            best_epoch = int(epoch)
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "num_classes": num_classes,
                "width": args.width,
                "heads": args.heads,
                "dropout": args.dropout,
                "input_shape": train_set.input_shape,
                "mod_names": mod_names,
                "config": config,
                "best_epoch": best_epoch,
                "best_val_acc": best_acc,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))
            save_confusion_matrix(os.path.join(args.out_dir, "val_confusion_matrix.csv"), val["cm"], mod_names)
            save_class_accuracy(os.path.join(args.out_dir, "class_accuracy.csv"), val["cm"], mod_names)
            save_group_accuracy(os.path.join(args.out_dir, "val_group_accuracy_by_snr_turb.csv"), val["group"], mod_names)
        else:
            bad_epochs += 1

        torch.save({
            "model_state": model.state_dict(),
            "model_name": args.model,
            "num_classes": num_classes,
            "width": args.width,
            "heads": args.heads,
            "dropout": args.dropout,
            "input_shape": train_set.input_shape,
            "mod_names": mod_names,
            "config": config,
            "epoch": epoch,
            "val_acc": float(val["acc"]),
        }, os.path.join(args.out_dir, "last_model.pt"))

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val acc={best_acc:.4f}")
            break

    print(f"Finished. Best epoch={best_epoch}, best val acc={best_acc:.4f}")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
