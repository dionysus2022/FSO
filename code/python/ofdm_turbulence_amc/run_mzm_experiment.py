"""MZM optical OFDM experiment: constellation figure + per-alpha MS-CNN training."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\Project_code\FSO_research\code\python\ofdm_turbulence_amc")

from simulation import CONSTELLATIONS, MODULATION_ORDERS, MODULATIONS
from models import MultiScaleAMCNet
from mzm_additions import MZMConfig, generate_optical_ofdm_with_metadata, MZMOFDMDataset

ROOT = Path(
    r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
    r"\OptDSP_lite-master\2_Data_Results"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mzm_mscnn_experiment")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--n-subcarriers", type=int, default=64)
    parser.add_argument("--cp-length", type=int, default=8)
    parser.add_argument("--snr", type=float, default=15.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95])
    parser.add_argument("--train-examples", type=int, default=500)
    parser.add_argument("--val-examples", type=int, default=100)
    parser.add_argument("--test-examples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def plot_constellation_compression(output_dir: Path, args):
    print("=" * 60, flush=True)
    print("[Fig 1] Generating constellation compression (256QAM, alpha=0.85)...", flush=True)
    print("=" * 60, flush=True)

    cfg = MZMConfig(
        sequence_length=8192, n_subcarriers=args.n_subcarriers,
        cp_length=args.cp_length, alpha=0.85, v_pi=5.0,
    )
    mod_idx = MODULATIONS.index("256QAM")
    seq, _ = generate_optical_ofdm_with_metadata(
        mod_idx, 30.0, "none", 0, args.seed, cfg,
        CONSTELLATIONS[MODULATION_ORDERS[mod_idx]],
    )

    n_fft, cp = cfg.n_subcarriers, cfg.cp_length
    sl = n_fft + cp
    x = seq.astype(np.float64)
    best_offset = best_metric = 0
    best_corr = 0.0j
    for offset in range(sl):
        starts = np.arange(offset, x.size - sl + 1, sl, dtype=np.int64)
        if not starts.size: continue
        ci = (starts[:, None] + np.arange(cp)).reshape(-1)
        ti = ci + n_fft
        corr = np.vdot(x[ci], x[ti])
        den = np.sqrt(np.vdot(x[ci], x[ci]).real * np.vdot(x[ti], x[ti]).real + 1e-12)
        m = float(np.abs(corr) / den)
        if m > best_metric: best_metric = m; best_offset = offset; best_corr = corr

    aligned = x[best_offset:]
    n_complete = (aligned.size - n_fft) // sl
    syms = np.zeros((n_complete, n_fft), dtype=np.float64)
    for i in range(n_complete):
        st = i * sl
        cp_part = aligned[st:st+cp]
        tail = aligned[st+cp:st+sl]
        merged = 0.5 * (tail + np.concatenate([cp_part, tail[cp:]]))
        syms[i] = merged
    freq = np.fft.fft(syms, axis=1, norm="ortho")
    n_data = n_fft // 2 - 1
    pts = freq[:, 1:n_data+1].reshape(-1)
    pts /= np.sqrt(np.mean(np.abs(pts)**2) + 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    ref = CONSTELLATIONS[MODULATION_ORDERS[mod_idx]]
    ax1.scatter(ref.real, ref.imag, s=8, c="#1f77b4", alpha=0.8)
    ax1.set_title("Theoretical 256QAM", fontsize=12)
    ax1.set_xlabel("In-Phase"); ax1.set_ylabel("Quadrature")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

    ax2.scatter(pts.real, pts.imag, s=3, c="#d62728", alpha=0.4)
    ax2.set_title(f"After MZM (alpha=0.85), CP Merge + FFT\nN={n_fft}, CP={cp}, 30 dB", fontsize=11)
    ax2.set_xlabel("In-Phase"); ax2.set_ylabel("Quadrature")
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.annotate("Outer ring\ncompression",
                 xy=(float(np.percentile(pts.real, 95)), float(np.percentile(pts.imag, 95))),
                 xytext=(float(np.percentile(pts.real, 60)), float(np.percentile(pts.imag, 60))),
                 fontsize=10, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.suptitle("MZM Sinusoidal Transfer → Outer Constellation Compression (256QAM)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "constellation_mzm_compression.png", dpi=220)
    plt.close(fig)
    print(f"  Saved constellation figure", flush=True)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    for seq, lab, _, _ in loader:
        seq, lab = seq.to(device), lab.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        logits = model(seq)
        loss = criterion(logits, lab)
        if training: loss.backward(); optimizer.step()
        loss_sum += loss.item() * lab.size(0)
        correct += (logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for seq, lab, _, _ in loader:
        logits = model(seq.to(device))
        all_p.extend(logits.argmax(1).cpu().numpy().tolist())
        all_l.extend(lab.numpy().tolist())
    return np.array(all_p), np.array(all_l)


def train_model(train_loader, val_loader, device, args, tag=""):
    model = MultiScaleAMCNet(num_classes=len(MODULATIONS), in_channels=1, use_se=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_p, va_l = evaluate(model, val_loader, device)
        va_acc = accuracy_score(va_l, va_p)
        scheduler.step()
        print(f"  [{tag}] E{epoch:2d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} va_acc={va_acc:.4f}", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc; best_state = model.state_dict(); patience = 0
        else:
            patience += 1
            if patience >= 8:
                print(f"  [{tag}] Early stop at epoch {epoch}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_p, test_l = evaluate(model, val_loader, device)  # re-eval on val
    val_final = accuracy_score(test_l, test_p)
    return model, val_final


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    # ---- Figure 1: Constellation compression ----
    plot_constellation_compression(output_dir, args)

    # ---- Per-alpha training (moderate turbulence only) ----
    print("=" * 60, flush=True)
    print("[Training] Per-alpha models on moderate turbulence", flush=True)
    print("=" * 60, flush=True)

    cfg_template = MZMConfig(
        sequence_length=args.sequence_length, n_subcarriers=args.n_subcarriers,
        cp_length=args.cp_length, alpha=0.5, v_pi=5.0,
    )

    results = {}  # alpha_str -> {"accuracy": ..., "errors": ...}
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for alpha in args.alphas:
        alpha_str = f"{alpha:.2f}"
        tag = f"alpha_{alpha_str}"
        print(f"\n--- {tag} ---", flush=True)

        seed_off = int(alpha * 1000)
        train_set = MZMOFDMDataset(
            MODULATIONS, MODULATION_ORDERS, CONSTELLATIONS,
            alphas=[alpha], turbulence_levels=["moderate"], snrs=[args.snr],
            examples_per_condition=args.train_examples,
            seed=args.seed + 1000 + seed_off,
            config_template=cfg_template, cache=True,
        )
        val_set = MZMOFDMDataset(
            MODULATIONS, MODULATION_ORDERS, CONSTELLATIONS,
            alphas=[alpha], turbulence_levels=["moderate"], snrs=[args.snr],
            examples_per_condition=args.val_examples,
            seed=args.seed + 2000 + seed_off,
            config_template=cfg_template, cache=True,
        )
        test_set = MZMOFDMDataset(
            MODULATIONS, MODULATION_ORDERS, CONSTELLATIONS,
            alphas=[alpha], turbulence_levels=["moderate"], snrs=[args.snr],
            examples_per_condition=args.test_examples,
            seed=args.seed + 3000 + seed_off,
            config_template=cfg_template, cache=True,
        )

        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, args.batch_size, num_workers=0)

        model, val_acc = train_model(train_loader, val_loader, device, args, tag=tag)

        # Test
        te_p, te_l = evaluate(model, test_loader, device)
        te_acc = accuracy_score(te_l, te_p)
        te_errs = int(np.sum(te_p != te_l))
        total = len(te_l)
        results[alpha_str] = {"accuracy": float(te_acc), "errors": te_errs, "total": total}
        print(f"  >> {tag}: test_acc={te_acc:.5f} errors={te_errs}/{total}", flush=True)

        # Save model
        torch.save(model.state_dict(), model_dir / f"model_{tag}.pt")

    # ---- Figure 2: Accuracy vs alpha ----
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas_sorted = sorted(results.keys(), key=float)
    acc_vals = [results[a]["accuracy"] for a in alphas_sorted]
    err_vals = [results[a]["errors"] for a in alphas_sorted]
    ax.plot([float(a) for a in alphas_sorted], [100 * v for v in acc_vals],
            marker="o", linewidth=2, color="#ff7f0e", markersize=8)
    # Add error counts as text annotations
    for a, acc, err in zip(alphas_sorted, acc_vals, err_vals):
        ax.annotate(f"{err} err", (float(a), 100 * acc),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha="center")
    ax.set_xlabel("MZM Drive Swing Ratio alpha")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title(
        f"MS-CNN Accuracy vs MZM Nonlinearity\n"
        f"(Moderate Turbulence, SNR={args.snr} dB, N={args.n_subcarriers}, CP={args.cp_length})",
        fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_alpha.png", dpi=220)
    plt.close(fig)
    print(f"\nSaved accuracy_vs_alpha.png", flush=True)

    # ---- Save metrics ----
    metrics = {
        "task": "MZM optical OFDM MS-CNN experiment (per-alpha)",
        "modulations": list(MODULATIONS),
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "test_accuracy_vs_alpha_moderate": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"\nDone! Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
