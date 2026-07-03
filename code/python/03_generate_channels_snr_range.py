# -*- coding: utf-8 -*-
"""
03_generate_channels_snr_range.py
------------------------------------------------------------
作用：
    使用 02_train_gnn_channel_vae.py 训练好的 GNN-CVAE 模型，
    为弱/强湍流生成目标平均 SNR = 5~20 dB 的信道样本索引。

注意：
    本脚本生成的是 H_gen(k) 和 target_snr_db 标签，
    还没有直接生成最终调制识别输入 Y(k,n)。

    后续生成识别数据时应使用：
        Y(k,n) = H_gen(k) * X(k,n) + W(k,n)
    其中 W 的功率由 target_snr_db 控制。

输出：
    generated_channels_snr_5_20.npz
    包含：
        H_gen              complex64 [N, K]
        H_re_gen           float32   [N, K]
        H_im_gen           float32   [N, K]
        snr_shape_gen_db   float32   [N, K]   GNN生成的子载波SNR形状，仅作参考
        target_snr_db      float32   [N]
        turb_label         int64     [N]
        k_norm             float32   [K]

运行示例：
    python 03_generate_channels_snr_range.py \
        --ckpt "D:/.../gnn_channel_model/gnn_channel_vae.pt" \
        --out_dir "D:/.../gnn_channel_model/generated" \
        --n_per_snr 500 \
        --snr_min 5 --snr_max 20 --snr_step 1
------------------------------------------------------------
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
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

    def decode(self, z: torch.Tensor, cond: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, K, _ = cond.shape
        z_node = z[:, None, :].expand(B, K, z.shape[-1])
        x = torch.cat([cond, z_node], dim=-1)
        h = self.dec_g1(x, A)
        h = self.dec_g2(h, A) + h
        h = self.dec_g3(h, A) + h
        return self.out(h)


def select_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_condition_batch(K: int, k_norm: np.ndarray, num_turb: int, num_mod: int, turb_id: int, n: int,
                          use_mod_condition: bool, mod_id: int = 0) -> torch.Tensor:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to gnn_channel_vae.pt")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_per_snr", type=int, default=500)
    parser.add_argument("--snr_min", type=float, default=5.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--snr_step", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_gen", type=int, default=256)
    parser.add_argument("--use_mod_condition", action="store_true", help="Must match the setting used in training.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)

    K = int(ckpt["K"])
    A = ckpt["A"].to(device)
    num_turb = int(ckpt["num_turb"])
    num_mod = int(ckpt["num_mod"])
    stats = ckpt["norm_stats"]
    train_cfg = ckpt.get("cfg", {})

    latent_dim = int(train_cfg.get("latent_dim", 24))
    hidden_dim = int(train_cfg.get("hidden_dim", 96))
    enc_in_dim = int(ckpt["enc_in_dim"])
    cond_dim = int(ckpt["cond_dim"])

    model = GNNChannelCVAE(
        enc_in_dim=enc_in_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        out_dim=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # k_norm 是固定的 -1~1。训练时也是这个形式。
    k_norm = np.linspace(-1.0, 1.0, K).astype(np.float32)

    snr_values = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step, dtype=np.float32)

    H_all = []
    H_re_all = []
    H_im_all = []
    snr_shape_all = []
    target_snr_all = []
    turb_all = []

    with torch.no_grad():
        for turb_id in range(num_turb):
            for snr_db in snr_values:
                remain = args.n_per_snr
                while remain > 0:
                    b = min(args.batch_gen, remain)
                    cond = build_condition_batch(
                        K=K,
                        k_norm=k_norm,
                        num_turb=num_turb,
                        num_mod=num_mod,
                        turb_id=turb_id,
                        n=b,
                        use_mod_condition=args.use_mod_condition,
                        mod_id=0,
                    ).to(device)
                    z = torch.randn(b, latent_dim, device=device)
                    y = model.decode(z, cond, A).cpu().numpy()
                    H, H_re, snr_shape = denormalize_generated(y, stats)

                    H_all.append(H)
                    H_re_all.append(H_re)
                    H_im_all.append(np.imag(H).astype(np.float32))
                    snr_shape_all.append(snr_shape)
                    target_snr_all.append(np.full((b,), snr_db, dtype=np.float32))
                    turb_all.append(np.full((b,), turb_id, dtype=np.int64))

                    remain -= b

                print(f"Generated turb={turb_id}, target_snr={snr_db:.1f} dB, n={args.n_per_snr}")

    H_gen = np.concatenate(H_all, axis=0).astype(np.complex64)
    H_re_gen = np.concatenate(H_re_all, axis=0).astype(np.float32)
    H_im_gen = np.concatenate(H_im_all, axis=0).astype(np.float32)
    snr_shape_gen_db = np.concatenate(snr_shape_all, axis=0).astype(np.float32)
    target_snr_db = np.concatenate(target_snr_all, axis=0).astype(np.float32)
    turb_label = np.concatenate(turb_all, axis=0).astype(np.int64)

    out_file = os.path.join(args.out_dir, "generated_channels_snr_5_20.npz")
    np.savez_compressed(
        out_file,
        H_gen=H_gen,
        H_re_gen=H_re_gen,
        H_im_gen=H_im_gen,
        snr_shape_gen_db=snr_shape_gen_db,
        target_snr_db=target_snr_db,
        turb_label=turb_label,
        k_norm=k_norm,
    )

    with open(os.path.join(args.out_dir, "generate_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Generation finished.")
    print(f"Output: {out_file}")
    print(f"Total generated channels: {H_gen.shape[0]}")
    print(f"Subcarriers: {H_gen.shape[1]}")
    print("Next step for recognition dataset:")
    print("    Y(k,n) = H_gen(k) * X(k,n) + W(k,n)")
    print("    W is controlled by target_snr_db, not by snr_shape_gen_db.")
    print("=" * 70)


if __name__ == "__main__":
    main()
