import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import glob
import csv
import time
import argparse
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader


# ==================== Dataset ====================

class NPZShardCache:
    """Multi-shard cache with persistent mmap handles for Y_iq.

    Keeps up to *max_items* of memory-mapped Y_iq handles open concurrently.
    The OS virtual-memory subsystem handles page caching transparently, so
    random-access reads are fast (~2 µs) without keeping the full array in RAM.
    """

    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._mmaps: Dict[str, np.ndarray] = {}
        self._mmap_order: List[str] = []
        self.meta_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.meta_order: List[str] = []

    # ---- Y_iq (mmap handles) ----

    def get_mmap(self, path: str) -> np.ndarray:
        arr = self._mmaps.get(path)
        if arr is not None:
            # LRU promotion
            self._mmap_order.remove(path)
            self._mmap_order.append(path)
            return arr
        npy_path = path.replace(".npz", "_yiq.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path, mmap_mode="r")
        else:
            arr = np.load(path, allow_pickle=True)["Y_iq"]
        self._mmaps[path] = arr
        self._mmap_order.append(path)
        self._evict_mmap()
        return arr

    def _evict_mmap(self) -> None:
        while len(self._mmap_order) > self.max_items:
            old = self._mmap_order.pop(0)
            old_arr = self._mmaps.pop(old, None)
            if old_arr is not None:
                del old_arr

    # ---- metadata (small, loaded into RAM) ----

    def get_meta(self, path: str) -> Dict[str, np.ndarray]:
        meta = self.meta_cache.get(path)
        if meta is not None:
            self.meta_order.remove(path)
            self.meta_order.append(path)
            return meta
        with np.load(path, allow_pickle=True) as z:
            meta = {k: z[k] for k in z.files if k != "Y_iq"}
        self.meta_cache[path] = meta
        self.meta_order.append(path)
        while len(self.meta_order) > self.max_items:
            old = self.meta_order.pop(0)
            self.meta_cache.pop(old, None)
        return meta

    # ---- convenience ----

    def load_yiq(self, path: str, idx: int) -> np.ndarray:
        arr = self.get_mmap(path)
        return np.asarray(arr[idx], dtype=np.float32)


class _ShardInfo:
    __slots__ = ("path", "mod_label", "turb_label", "target_snr_db")
    def __init__(self, path: str, mod_label: np.ndarray, turb_label: np.ndarray, target_snr_db: np.ndarray):
        self.path = path
        self.mod_label = mod_label
        self.turb_label = turb_label
        self.target_snr_db = target_snr_db


class ModRecShardDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        normalize: str = "sample_rms",
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache: Optional[NPZShardCache] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.cache = cache or NPZShardCache()
        self.samples: List[Tuple[int, int]] = []   # (shard_id, local_idx)
        self.shards: List[_ShardInfo] = []
        self.num_classes: int = 0
        self.input_shape: Optional[Tuple[int, ...]] = None

        shard_paths = sorted(glob.glob(os.path.join(data_dir, "modrec_synth_shard_*.npz")))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {data_dir}")

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

            if split_label is not None:
                if split == "train":
                    mask = split_label == 0
                elif split == "val":
                    mask = split_label == 1
                elif split == "test":
                    mask = split_label == 2
                else:
                    mask = np.ones(n, dtype=bool)
            else:
                mask = np.ones(n, dtype=bool) if split == "train" else np.zeros(n, dtype=bool)

            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # ensure all labels are valid
            unique_labels = sorted(set(mod_label[indices]))
            self.num_classes = max(self.num_classes, len(unique_labels))

            # multi-indexed samples
            for local_idx in indices:
                self.samples.append((shard_id, int(local_idx)))
            self.shards.append(_ShardInfo(
                path=path,
                mod_label=mod_label,
                turb_label=turb_label,
                target_snr_db=target_snr_db,
            ))

        # infer input shape
        first_shard_path = self.shards[self.samples[0][0]].path
        first_mmap = self.cache.get_mmap(first_shard_path)
        self.input_shape = tuple(first_mmap.shape[1:])

    def set_normalization_gains(self, gains: Optional[Dict[str, float]] = None):
        self._norm_gains = gains

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        shard_id, local_idx = self.samples[index]
        shard = self.shards[shard_id]
        x = self.cache.load_yiq(shard.path, local_idx)
        if self.normalize == "sample_rms":
            rms = np.sqrt(np.mean(x ** 2) + 1e-12)
            x = x / rms
        y = int(shard.mod_label[local_idx])
        turb = int(shard.turb_label[local_idx])
        snr = shard.target_snr_db[local_idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long), torch.tensor(turb, dtype=torch.long), torch.tensor(snr, dtype=torch.float32)

# ==================== CNN + Attention Model ====================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        return x


class CNN_Attention(nn.Module):
    def __init__(self, num_classes, width=48, heads=4, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(width, width * 2, blocks=2, stride=2, reduction=16)
        self.layer2 = self._make_layer(width * 2, width * 4, blocks=3, stride=2, reduction=16)
        self.layer3 = self._make_layer(width * 4, width * 8, blocks=3, stride=2, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = width * 8
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.attention = MultiHeadAttention(self.embed_dim, heads, dropout)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_c, out_c, blocks, stride, reduction):
        layers = [ResidualBlock(in_c, out_c, stride, reduction)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, 1, reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1).unsqueeze(1)  # (B, 1, E)
        x = x + self.pos_embed
        x = self.attention(x)
        x = self.norm(x).squeeze(1)
        x = self.classifier(x)
        return x


def make_model(model_name: str, num_classes: int, width: int = 48, heads: int = 4, dropout: float = 0.0):
    if model_name == "cnn_attention":
        return CNN_Attention(num_classes, width, heads, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ==================== Training helpers ====================

def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def save_csv_log(filepath: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_confusion_matrix(filepath: str, cm: np.ndarray, labels: List[str]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + list(row))


def save_class_accuracy(filepath: str, cm: np.ndarray, labels: List[str]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy", "total"])
        for i, label in enumerate(labels):
            total = cm[i].sum()
            acc = cm[i, i] / total if total > 0 else 0.0
            w.writerow([label, f"{acc:.4f}", int(total)])


def save_group_accuracy(filepath: str, groups: List[dict], labels: List[str]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "turbulence", "class", "correct", "total", "accuracy"])
        for g in groups:
            w.writerow([g["snr"], g["turbulence"], labels[g["mod"]], g["correct"], g["total"], f"{g['correct']/max(g['total'],1):.4f}"])


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    groups = []
    for x, y, turb, snr in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for i in range(y.size(0)):
            cm[int(y[i]), int(pred[i])] += 1
            groups.append({"snr": float(snr[i]), "turbulence": int(turb[i]), "mod": int(y[i]),
                           "correct": int(pred[i] == y[i]), "total": 1})
    model.train()
    return {"loss": float(np.mean(losses)), "acc": correct / max(total, 1), "cm": cm, "group": groups}


def load_checkpoint(args, model, optimizer, scaler):
    last_path = os.path.join(args.out_dir, "last_model.pt")
    best_path = os.path.join(args.out_dir, "best_model.pt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        scaler.load_state_dict(ckpt.get("scaler_state", scaler.state_dict()))
        return ckpt.get("epoch", 0)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="cnn_attention")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_shards", type=int, default=8,
                        help="Number of shard mmap handles to keep open (each ~123 MiB).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = NPZShardCache(max_items=args.cache_shards)

    train_set = ModRecShardDataset(
        args.data_dir, split="train", normalize="sample_rms", max_samples=None, seed=args.seed, cache=cache,
    )
    val_set = ModRecShardDataset(
        args.data_dir, split="val", normalize="sample_rms", max_samples=None, seed=args.seed, cache=cache,
    )

    num_classes = max(train_set.num_classes, val_set.num_classes)
    mod_names = ['QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'][:num_classes]

    os.makedirs(args.out_dir, exist_ok=True)

    amp = bool(args.amp and device.type == "cuda")
    model = make_model(args.model, num_classes, args.width, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    config = vars(args).copy()
    config.update({
        "device": str(device),
    })
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("========== ModRec training ==========", flush=True)
    print(f"data_dir       : {args.data_dir}", flush=True)
    print(f"out_dir        : {args.out_dir}", flush=True)
    print(f"device         : {device}", flush=True)
    print(f"model          : {args.model}", flush=True)
    print(f"input_shape    : {train_set.input_shape}", flush=True)
    print(f"classes        : {num_classes} {mod_names[:num_classes]}", flush=True)
    print(f"train / val    : {len(train_set)} / {len(val_set)}", flush=True)
    print("=====================================", flush=True)

    best_acc = -1.0
    best_epoch = 0
    bad_epochs = 0
    start_epoch = load_checkpoint(args, model, optimizer, scaler)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    loss_fn = nn.CrossEntropyLoss()
    log_rows = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, tr_count = 0.0, 0.0, 0
        t0 = time.time()
        for batch_idx, (x, y, _, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
            n = y.size(0)
            tr_loss += loss.item() * n
            tr_acc += accuracy(logits, y) * n
            tr_count += n

        scheduler.step()

        tr_loss /= max(tr_count, 1)
        tr_acc /= max(tr_count, 1)
        val = evaluate(model, val_loader, device, num_classes)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "lr": lr_now,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss": round(val["loss"], 6), "val_acc": round(val["acc"], 6),
        }
        log_rows.append(row)
        save_csv_log(os.path.join(args.out_dir, "train_log.csv"), log_rows)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.3e} | "
            f"train loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"val loss={val['loss']:.4f}, acc={val['acc']:.4f} | "
            f"{dt:.1f}s",
            flush=True,
        )

        is_best = val["acc"] > best_acc
        if is_best:
            best_acc = float(val["acc"])
            best_epoch = epoch
            bad_epochs = 0
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
                "best_epoch": best_epoch,
                "best_val_acc": best_acc,
            }, os.path.join(args.out_dir, "best_model.pt"))
            save_confusion_matrix(os.path.join(args.out_dir, "val_confusion_matrix.csv"), val["cm"], mod_names)
            save_class_accuracy(os.path.join(args.out_dir, "class_accuracy.csv"), val["cm"], mod_names)
            save_group_accuracy(os.path.join(args.out_dir, "val_group_accuracy_by_snr_turb.csv"), val["group"], mod_names)
        else:
            bad_epochs += 1

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
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