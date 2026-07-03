# -*- coding: utf-8 -*-
"""
02_train_gnn_channel_vae.py
------------------------------------------------------------
作用：
    使用 01_build_gnn_channel_dataset_from_rx1.m 生成的 gnn_channel_dataset.mat，
    训练一个不依赖 PyTorch Geometric 的轻量级 GNN-CVAE 信道建模网络。

核心思想：
    每一帧 = 一张图。
    每个子载波 = 一个节点。
    节点之间按相邻子载波连接，形成固定图结构。
    GNN 编码真实实验 residual channel / SNR 曲线，CVAE 解码生成新的子载波信道曲线。

输入：
    gnn_channel_dataset.mat

输出：
    gnn_channel_vae.pt
    gnn_norm_stats.npz
    generated_channels_preview.npz
    train_log.csv

说明：
    1) 当前 MATLAB 批处理保存的 rx_sc 已经过 LTS 均衡，因此这里学习的是 residual/effective channel，
       不是完全原始的物理湍流信道。
    2) 该模型适合用于“实验驱动的信道增强”：先生成 H_gen(k)，再在后续脚本中叠加不同 SNR 噪声。
    3) 本脚本只需要 torch、numpy、scipy，不需要 torch_geometric。

运行示例：
    python 02_train_gnn_channel_vae.py \
        --mat "D:/Project_code/OptDSP_FSO_2/practical waterfiling-exp/OptDSP_lite-master/2_Data_Results/rx1_batch_450frames_results/2026.06.26/gnn_channel_dataset/gnn_channel_dataset.mat" \
        --out_dir "D:/Project_code/OptDSP_FSO_2/practical waterfiling-exp/OptDSP_lite-master/2_Data_Results/rx1_batch_450frames_results/2026.06.26/gnn_channel_model" \
        --epochs 300 \
        --batch_size 64
------------------------------------------------------------
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# -----------------------------
# Config
# -----------------------------

@dataclass
class TrainConfig:
    mat: str
    out_dir: str
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 96
    latent_dim: int = 24
    beta_kl: float = 1e-3
    snr_loss_weight: float = 0.30
    val_ratio: float = 0.20
    seed: int = 2026
    edge_hop: int = 2
    preview_per_turb: int = 128
    device: str = "auto"
    use_mod_condition: bool = False
    num_workers: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# -----------------------------
# Data loading and normalization
# -----------------------------

def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).squeeze()


def load_mat_dataset(mat_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Cannot find mat file: {mat_path}")

    d = sio.loadmat(mat_path)

    required = ["H_re", "H_im", "snr_sc_db", "turb_label", "mod_label", "snr_frame_db", "subcarrier_index_norm"]
    for k in required:
        if k not in d:
            raise KeyError(f"Missing variable in mat file: {k}")

    H_re = np.asarray(d["H_re"], dtype=np.float32)
    H_im = np.asarray(d["H_im"], dtype=np.float32)
    snr_sc_db = np.asarray(d["snr_sc_db"], dtype=np.float32)

    turb_label = _as_1d(d["turb_label"]).astype(np.int64)
    mod_label = _as_1d(d["mod_label"]).astype(np.int64)
    snr_frame_db = _as_1d(d["snr_frame_db"]).astype(np.float32)
    k_norm = _as_1d(d["subcarrier_index_norm"]).astype(np.float32)

    # 清理 NaN/Inf 样本
    finite = (
        np.all(np.isfinite(H_re), axis=1)
        & np.all(np.isfinite(H_im), axis=1)
        & np.all(np.isfinite(snr_sc_db), axis=1)
        & np.isfinite(snr_frame_db)
        & np.isfinite(turb_label)
        & np.isfinite(mod_label)
    )

    H_re = H_re[finite]
    H_im = H_im[finite]
    snr_sc_db = snr_sc_db[finite]
    turb_label = turb_label[finite]
    mod_label = mod_label[finite]
    snr_frame_db = snr_frame_db[finite]

    if H_re.shape[0] < 20:
        raise RuntimeError(f"Too few valid samples after cleaning: {H_re.shape[0]}")

    return {
        "H_re": H_re,
        "H_im": H_im,
        "snr_sc_db": snr_sc_db,
        "turb_label": turb_label,
        "mod_label": mod_label,
        "snr_frame_db": snr_frame_db,
        "k_norm": k_norm,
    }


def make_norm_stats(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # 对 H 的实部、虚部分别标准化。
    # 对 SNR 曲线也标准化，但生成 5-20 dB 数据时，最终平均 SNR 仍应由噪声功率控制。
    eps = 1e-6
    stats = {}
    for name in ["H_re", "H_im", "snr_sc_db"]:
        x = data[name]
        stats[f"{name}_mean"] = np.array([np.mean(x)], dtype=np.float32)
        stats[f"{name}_std"] = np.array([np.std(x) + eps], dtype=np.float32)
    return stats


def normalize_data(data: Dict[str, np.ndarray], stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(data)
    for name in ["H_re", "H_im", "snr_sc_db"]:
        out[name + "_n"] = (data[name] - stats[f"{name}_mean"][0]) / stats[f"{name}_std"][0]
    return out


class ChannelGraphDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], use_mod_condition: bool = False):
        self.H_re = data["H_re_n"].astype(np.float32)
        self.H_im = data["H_im_n"].astype(np.float32)
        self.snr = data["snr_sc_db_n"].astype(np.float32)
        self.turb = data["turb_label"].astype(np.int64)
        self.mod = data["mod_label"].astype(np.int64)
        self.k_norm = data["k_norm"].astype(np.float32)
        self.use_mod_condition = use_mod_condition

        self.num_turb = int(np.max(self.turb)) + 1
        self.num_mod = int(np.max(self.mod)) + 1
        self.N, self.K = self.H_re.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # x_target: [K, 3] = H_re, H_im, snr_sc_profile
        x = np.stack([self.H_re[idx], self.H_im[idx], self.snr[idx]], axis=-1)
        k = self.k_norm[:, None]

        turb = self.turb[idx]
        mod = self.mod[idx]

        turb_onehot = np.zeros((self.K, self.num_turb), dtype=np.float32)
        turb_onehot[:, turb] = 1.0

        if self.use_mod_condition:
            mod_onehot = np.zeros((self.K, self.num_mod), dtype=np.float32)
            mod_onehot[:, mod] = 1.0
            cond = np.concatenate([k, turb_onehot, mod_onehot], axis=-1)
        else:
            cond = np.concatenate([k, turb_onehot], axis=-1)

        # encoder 输入：真实节点特征 + 条件
        enc_in = np.concatenate([x, cond], axis=-1)

        return {
            "enc_in": torch.from_numpy(enc_in),
            "target": torch.from_numpy(x),
            "cond": torch.from_numpy(cond),
            "turb": torch.tensor(turb, dtype=torch.long),
            "mod": torch.tensor(mod, dtype=torch.long),
        }


# -----------------------------
# Fixed subcarrier graph
# -----------------------------

def build_subcarrier_adjacency(K: int, hop: int = 2) -> torch.Tensor:
    """固定图：每个子载波连接左右 hop 个邻居，加自环，做 GCN 归一化。"""
    A = torch.zeros(K, K, dtype=torch.float32)
    for i in range(K):
        for d in range(-hop, hop + 1):
            j = i + d
            if 0 <= j < K:
                A[i, j] = 1.0

    # 已包含自环；做 D^{-1/2} A D^{-1/2}
    deg = torch.sum(A, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


# -----------------------------
# Model
# -----------------------------

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, K, F], A: [K, K]
        x = torch.einsum("ij,bjf->bif", A, x)
        x = self.lin(x)
        x = self.norm(x)
        return F.gelu(x)


class GNNChannelCVAE(nn.Module):
    def __init__(self, enc_in_dim: int, cond_dim: int, hidden_dim: int = 96, latent_dim: int = 24, out_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.enc_g1 = GraphConv(enc_in_dim, hidden_dim)
        self.enc_g2 = GraphConv(hidden_dim, hidden_dim)
        self.enc_g3 = GraphConv(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        dec_in_dim = cond_dim + latent_dim
        self.dec_g1 = GraphConv(dec_in_dim, hidden_dim)
        self.dec_g2 = GraphConv(hidden_dim, hidden_dim)
        self.dec_g3 = GraphConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def encode(self, enc_in: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_g1(enc_in, A)
        h = self.enc_g2(h, A) + h
        h = self.enc_g3(h, A) + h
        # 图级池化：平均所有子载波节点
        g = h.mean(dim=1)
        return self.mu(g), self.logvar(g)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, K, _ = cond.shape
        z_node = z[:, None, :].expand(B, K, z.shape[-1])
        x = torch.cat([cond, z_node], dim=-1)
        h = self.dec_g1(x, A)
        h = self.dec_g2(h, A) + h
        h = self.dec_g3(h, A) + h
        y = self.out(h)
        return y

    def forward(self, enc_in: torch.Tensor, cond: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(enc_in, A)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, A)
        return recon, mu, logvar


# -----------------------------
# Loss and train utilities
# -----------------------------

def loss_fn(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float, snr_w: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    # target/recon: [B, K, 3]
    loss_h = F.mse_loss(recon[:, :, 0:2], target[:, :, 0:2])
    loss_snr = F.mse_loss(recon[:, :, 2], target[:, :, 2])
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_h + snr_w * loss_snr + beta_kl * kld
    return loss, {"loss_h": float(loss_h.item()), "loss_snr": float(loss_snr.item()), "kld": float(kld.item())}


def run_epoch(model, loader, A, optimizer, device, cfg: TrainConfig, train: bool) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    total_h = 0.0
    total_snr = 0.0
    total_kld = 0.0
    n = 0

    for batch in loader:
        enc_in = batch["enc_in"].to(device)
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(enc_in, cond, A)
            loss, parts = loss_fn(recon, target, mu, logvar, cfg.beta_kl, cfg.snr_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                recon, mu, logvar = model(enc_in, cond, A)
                loss, parts = loss_fn(recon, target, mu, logvar, cfg.beta_kl, cfg.snr_loss_weight)

        bs = enc_in.shape[0]
        total += float(loss.item()) * bs
        total_h += parts["loss_h"] * bs
        total_snr += parts["loss_snr"] * bs
        total_kld += parts["kld"] * bs
        n += bs

    return {
        "loss": total / max(n, 1),
        "loss_h": total_h / max(n, 1),
        "loss_snr": total_snr / max(n, 1),
        "kld": total_kld / max(n, 1),
    }


def save_log_row(csv_path: str, row: Dict[str, float], header: bool = False) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(row.keys())
    mode = "w" if header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# Generation preview
# -----------------------------

def build_condition_batch(K: int, k_norm: np.ndarray, num_turb: int, num_mod: int, turb_id: int, n: int, use_mod_condition: bool, mod_id: int = 0) -> torch.Tensor:
    k = k_norm[:, None].astype(np.float32)
    turb_onehot = np.zeros((K, num_turb), dtype=np.float32)
    turb_onehot[:, turb_id] = 1.0
    if use_mod_condition:
        mod_onehot = np.zeros((K, num_mod), dtype=np.float32)
        mod_onehot[:, mod_id] = 1.0
        cond_one = np.concatenate([k, turb_onehot, mod_onehot], axis=-1)
    else:
        cond_one = np.concatenate([k, turb_onehot], axis=-1)
    cond = np.repeat(cond_one[None, :, :], n, axis=0)
    return torch.from_numpy(cond)


def denormalize_generated(y: np.ndarray, stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H_re = y[:, :, 0] * stats["H_re_std"][0] + stats["H_re_mean"][0]
    H_im = y[:, :, 1] * stats["H_im_std"][0] + stats["H_im_mean"][0]
    snr_sc = y[:, :, 2] * stats["snr_sc_db_std"][0] + stats["snr_sc_db_mean"][0]
    H = H_re + 1j * H_im
    return H.astype(np.complex64), H_re.astype(np.float32), snr_sc.astype(np.float32)


def generate_preview(model, A, dataset: ChannelGraphDataset, raw_data: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], cfg: TrainConfig, device: torch.device) -> str:
    model.eval()
    K = dataset.K
    z_dim = cfg.latent_dim
    out = {}
    labels = []

    with torch.no_grad():
        for turb_id in range(dataset.num_turb):
            cond = build_condition_batch(
                K=K,
                k_norm=raw_data["k_norm"],
                num_turb=dataset.num_turb,
                num_mod=dataset.num_mod,
                turb_id=turb_id,
                n=cfg.preview_per_turb,
                use_mod_condition=cfg.use_mod_condition,
                mod_id=0,
            ).to(device)
            z = torch.randn(cfg.preview_per_turb, z_dim, device=device)
            y = model.decode(z, cond, A).cpu().numpy()
            H, H_re, snr_sc = denormalize_generated(y, stats)
            out[f"H_gen_turb_{turb_id}"] = H
            out[f"H_re_gen_turb_{turb_id}"] = H_re
            out[f"snr_sc_gen_turb_{turb_id}"] = snr_sc
            labels.append(turb_id)

    # 也保存真实数据的统计曲线，方便你画论文图。
    out["real_H_abs_mean_by_sc"] = np.mean(np.abs(raw_data["H_re"] + 1j * raw_data["H_im"]), axis=0).astype(np.float32)
    out["real_H_abs_std_by_sc"] = np.std(np.abs(raw_data["H_re"] + 1j * raw_data["H_im"]), axis=0).astype(np.float32)
    out["real_snr_mean_by_sc"] = np.mean(raw_data["snr_sc_db"], axis=0).astype(np.float32)
    out["real_snr_std_by_sc"] = np.std(raw_data["snr_sc_db"], axis=0).astype(np.float32)
    out["k_norm"] = raw_data["k_norm"].astype(np.float32)

    path = os.path.join(cfg.out_dir, "generated_channels_preview.npz")
    np.savez_compressed(path, **out)
    return path


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, required=True, help="Path to gnn_channel_dataset.mat")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=24)
    parser.add_argument("--beta_kl", type=float, default=1e-3)
    parser.add_argument("--snr_loss_weight", type=float, default=0.30)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--edge_hop", type=int, default=2)
    parser.add_argument("--preview_per_turb", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_mod_condition", action="store_true", help="Condition CVAE on modulation label as well. Default: turbulence only.")
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    print("=" * 70)
    print("GNN-CVAE channel modeling")
    print("=" * 70)
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print(f"Device: {device}")

    raw_data = load_mat_dataset(cfg.mat)
    stats = make_norm_stats(raw_data)
    data = normalize_data(raw_data, stats)

    np.savez_compressed(os.path.join(cfg.out_dir, "gnn_norm_stats.npz"), **stats)

    dataset = ChannelGraphDataset(data, use_mod_condition=cfg.use_mod_condition)
    print(f"Samples: {len(dataset)}, Subcarriers: {dataset.K}, Turb labels: {dataset.num_turb}, Mod labels: {dataset.num_mod}")

    K = dataset.K
    A = build_subcarrier_adjacency(K, hop=cfg.edge_hop).to(device)

    val_len = max(1, int(len(dataset) * cfg.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    # 从一个样本推断输入维度
    sample = dataset[0]
    enc_in_dim = sample["enc_in"].shape[-1]
    cond_dim = sample["cond"].shape[-1]

    model = GNNChannelCVAE(
        enc_in_dim=enc_in_dim,
        cond_dim=cond_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        out_dim=3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

    log_csv = os.path.join(cfg.out_dir, "train_log.csv")
    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "gnn_channel_vae.pt")

    save_log_row(log_csv, {
        "epoch": 0,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": np.nan,
        "train_loss_h": np.nan,
        "train_loss_snr": np.nan,
        "train_kld": np.nan,
        "val_loss": np.nan,
        "val_loss_h": np.nan,
        "val_loss_snr": np.nan,
        "val_kld": np.nan,
    }, header=True)

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, A, optimizer, device, cfg, train=True)
        va = run_epoch(model, val_loader, A, optimizer, device, cfg, train=False)
        scheduler.step(va["loss"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "train_loss_h": tr["loss_h"],
            "train_loss_snr": tr["loss_snr"],
            "train_kld": tr["kld"],
            "val_loss": va["loss"],
            "val_loss_h": va["loss_h"],
            "val_loss_snr": va["loss_snr"],
            "val_kld": va["kld"],
        }
        save_log_row(log_csv, row, header=False)

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "K": K,
                "enc_in_dim": enc_in_dim,
                "cond_dim": cond_dim,
                "num_turb": dataset.num_turb,
                "num_mod": dataset.num_mod,
                "A": A.detach().cpu(),
                "norm_stats": stats,
            }, best_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:04d} | "
                f"train={tr['loss']:.5f} h={tr['loss_h']:.5f} snr={tr['loss_snr']:.5f} kld={tr['kld']:.5f} | "
                f"val={va['loss']:.5f} h={va['loss_h']:.5f} snr={va['loss_snr']:.5f} kld={va['kld']:.5f} | "
                f"best={best_val:.5f}"
            )

    # 载入最佳模型并生成 preview
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    preview_path = generate_preview(model, A, dataset, raw_data, stats, cfg, device)

    # 保存配置
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Training finished.")
    print(f"Best model : {best_path}")
    print(f"Norm stats : {os.path.join(cfg.out_dir, 'gnn_norm_stats.npz')}")
    print(f"Preview    : {preview_path}")
    print(f"Train log  : {log_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
