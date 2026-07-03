#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_build_awgn_baseline_modrec_dataset.py

Purpose
-------
Generate a pure AWGN modulation-recognition baseline dataset that is compatible
with 05_train_modrec_curriculum_v2.py and 07_audit_synthetic_modrec_dataset.py.

Why this script matters
-----------------------
It isolates the data-generation pipeline:
    X(k,n) + W(k,n)
without GNN channel H_gen. If this AWGN baseline cannot classify QAM orders,
then the constellation/label/dataset code is wrong. If AWGN works well but the
GNN-channel dataset performs poorly, the problem is in the GNN channel/equalized
view/domain gap.

Output NPZ keys per shard
-------------------------
Y_iq              float32 [N, 2, n_sc, ofdm_symbols]
mod_label         int64   [N]
mod_name          unicode [N]
turb_label        int64   [N]     dummy 0/1, balanced if requested
 target_snr_db    float32 [N]
actual_snr_db     float32 [N]
channel_index     int64   [N]     -1 for AWGN baseline
split_label       int64   [N]     0=train, 1=val

Example
-------
python 08_build_awgn_baseline_modrec_dataset.py ^
  --out_dir "D:/.../modrec_dataset_awgn_baseline_5_20dB" ^
  --mods QPSK,16QAM,32QAM,64QAM,128QAM,256QAM ^
  --n_per_mod_snr 1000 ^
  --snr_min 5 --snr_max 20 --snr_step 1 ^
  --n_sc 64 --ofdm_symbols 32 ^
  --shard_samples 4096 ^
  --constellation_norm unit_avg_power ^
  --split_ratio 0.9 ^
  --seed 20260628
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


MODS_DEFAULT = "QPSK,16QAM,32QAM,64QAM,128QAM,256QAM"


def parse_mods(s: str) -> List[str]:
    mods = [x.strip() for x in s.split(",") if x.strip()]
    if not mods:
        raise ValueError("empty --mods")
    return mods


def square_qam_constellation(M: int) -> np.ndarray:
    """Square QAM constellation for M=16,64,256, unnormalized complex64."""
    m = int(round(math.sqrt(M)))
    if m * m != M:
        raise ValueError(f"M={M} is not square QAM")
    levels = np.arange(-(m - 1), m, 2, dtype=np.float64)
    I, Q = np.meshgrid(levels, levels)
    c = I.reshape(-1) + 1j * Q.reshape(-1)
    return c.astype(np.complex128)


def cross_qam_constellation(M: int) -> np.ndarray:
    """Meshgrid-corner-removed cross-QAM for 32QAM and 128QAM.

    32QAM: 6x6 grid, remove 4 outer corners.
    128QAM: 12x12 grid, remove four 2x2 corner blocks.
    """
    if M == 32:
        levels = np.arange(-5, 6, 2, dtype=np.float64)  # -5,-3,-1,1,3,5
        I, Q = np.meshgrid(levels, levels)
        keep = ~((np.abs(I) == 5) & (np.abs(Q) == 5))
        c = I[keep].reshape(-1) + 1j * Q[keep].reshape(-1)
    elif M == 128:
        levels = np.arange(-11, 12, 2, dtype=np.float64)  # 12 levels
        I, Q = np.meshgrid(levels, levels)
        # remove 2x2 blocks at four corners: |I|>=9 and |Q|>=9 -> 16 points
        keep = ~((np.abs(I) >= 9) & (np.abs(Q) >= 9))
        c = I[keep].reshape(-1) + 1j * Q[keep].reshape(-1)
    else:
        raise ValueError("cross_qam_constellation supports only 32 and 128")
    if c.size != M:
        raise RuntimeError(f"Cross QAM {M} generated {c.size} points, expected {M}")
    return c.astype(np.complex128)


def constellation(mod: str, norm: str) -> np.ndarray:
    mod = mod.upper()
    if mod == "QPSK":
        c = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128)
    elif mod == "16QAM":
        c = square_qam_constellation(16)
    elif mod == "32QAM":
        c = cross_qam_constellation(32)
    elif mod == "64QAM":
        c = square_qam_constellation(64)
    elif mod == "128QAM":
        c = cross_qam_constellation(128)
    elif mod == "256QAM":
        c = square_qam_constellation(256)
    else:
        raise ValueError(f"Unsupported modulation: {mod}")

    if norm == "unit_avg_power":
        c = c / np.sqrt(np.mean(np.abs(c) ** 2) + 1e-12)
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown constellation_norm: {norm}")
    return c.astype(np.complex64)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    sig_power = float(np.mean(np.abs(x) ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    w = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    y = x + w.astype(np.complex64)
    measured_noise = y - x
    actual_snr = 10.0 * np.log10(
        (np.mean(np.abs(x) ** 2) + 1e-12) / (np.mean(np.abs(measured_noise) ** 2) + 1e-12)
    )
    return y.astype(np.complex64), float(actual_snr)


def sample_symbols(c: np.ndarray, n_sc: int, n_sym: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(c), size=(n_sc, n_sym), endpoint=False)
    return c[idx].astype(np.complex64)


def save_shard(out_dir: Path, shard_id: int, rows: List[dict]) -> Path:
    y = np.stack([r["Y_iq"] for r in rows], axis=0).astype(np.float32)
    mod_label = np.asarray([r["mod_label"] for r in rows], dtype=np.int64)
    mod_name = np.asarray([r["mod_name"] for r in rows])
    turb_label = np.asarray([r["turb_label"] for r in rows], dtype=np.int64)
    target_snr_db = np.asarray([r["target_snr_db"] for r in rows], dtype=np.float32)
    actual_snr_db = np.asarray([r["actual_snr_db"] for r in rows], dtype=np.float32)
    channel_index = np.asarray([r["channel_index"] for r in rows], dtype=np.int64)
    split_label = np.asarray([r["split_label"] for r in rows], dtype=np.int64)

    p = out_dir / f"modrec_awgn_shard_{shard_id:04d}.npz"
    np.savez_compressed(
        p,
        Y_iq=y,
        mod_label=mod_label,
        mod_name=mod_name,
        turb_label=turb_label,
        target_snr_db=target_snr_db,
        actual_snr_db=actual_snr_db,
        channel_index=channel_index,
        split_label=split_label,
    )
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mods", default=MODS_DEFAULT)
    ap.add_argument("--n_per_mod_snr", type=int, default=1000)
    ap.add_argument("--snr_min", type=int, default=5)
    ap.add_argument("--snr_max", type=int, default=20)
    ap.add_argument("--snr_step", type=int, default=1)
    ap.add_argument("--snr_jitter_db", type=float, default=0.0)
    ap.add_argument("--n_sc", type=int, default=64)
    ap.add_argument("--ofdm_symbols", type=int, default=32)
    ap.add_argument("--shard_samples", type=int, default=4096)
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--constellation_norm", choices=["unit_avg_power", "none"], default="unit_avg_power")
    ap.add_argument("--balanced_dummy_turb", action="store_true", help="Alternate turb_label 0/1 so downstream grouping code can still run.")
    ap.add_argument("--seed", type=int, default=20260628)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mods = parse_mods(args.mods)
    consts: Dict[str, np.ndarray] = {m: constellation(m, args.constellation_norm) for m in mods}
    snrs = list(range(args.snr_min, args.snr_max + 1, args.snr_step))

    shard_rows: List[dict] = []
    manifest_rows: List[dict] = []
    shard_id = 0
    sample_id = 0

    for snr_db_int in snrs:
        for mod_label, mod in enumerate(mods):
            c = consts[mod]
            for local_i in range(args.n_per_mod_snr):
                snr_db = float(snr_db_int)
                if args.snr_jitter_db > 0:
                    snr_db += float(rng.uniform(-args.snr_jitter_db, args.snr_jitter_db))

                x = sample_symbols(c, args.n_sc, args.ofdm_symbols, rng)
                y_complex, actual_snr = add_awgn(x, snr_db, rng)

                y_iq = np.stack([y_complex.real, y_complex.imag], axis=0).astype(np.float32)
                split_label = 0 if rng.random() < args.split_ratio else 1
                turb_label = int(sample_id % 2) if args.balanced_dummy_turb else 0

                row = dict(
                    Y_iq=y_iq,
                    mod_label=mod_label,
                    mod_name=mod,
                    turb_label=turb_label,
                    target_snr_db=snr_db,
                    actual_snr_db=actual_snr,
                    channel_index=-1,
                    split_label=split_label,
                )
                shard_rows.append(row)
                manifest_rows.append(
                    dict(
                        global_index=sample_id,
                        shard_id=shard_id,
                        mod_label=mod_label,
                        mod_name=mod,
                        turb_label=turb_label,
                        target_snr_db=snr_db,
                        actual_snr_db=actual_snr,
                        channel_index=-1,
                        split_label=split_label,
                    )
                )
                sample_id += 1

                if len(shard_rows) >= args.shard_samples:
                    p = save_shard(out_dir, shard_id, shard_rows)
                    print(f"Saved {p} with {len(shard_rows)} samples")
                    shard_id += 1
                    shard_rows = []

    if shard_rows:
        p = save_shard(out_dir, shard_id, shard_rows)
        print(f"Saved {p} with {len(shard_rows)} samples")

    # manifest CSV
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_index", "shard_id", "mod_label", "mod_name", "turb_label",
                "target_snr_db", "actual_snr_db", "channel_index", "split_label"
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # dataset info
    info = dict(
        dataset_type="pure_awgn_baseline",
        equation="Y(k,n)=X(k,n)+W(k,n)",
        mods=mods,
        n_per_mod_snr=args.n_per_mod_snr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_step=args.snr_step,
        snr_jitter_db=args.snr_jitter_db,
        n_sc=args.n_sc,
        ofdm_symbols=args.ofdm_symbols,
        constellation_norm=args.constellation_norm,
        split_ratio=args.split_ratio,
        total_samples=sample_id,
        seed=args.seed,
        note="Use this as a sanity baseline. If AWGN fails, check constellation/labels. If AWGN succeeds but GNN dataset fails, check GNN channel/equalization/domain gap.",
    )
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # constellation dump for visual/manual comparison
    const_dir = out_dir / "constellations"
    const_dir.mkdir(exist_ok=True)
    for mod, c in consts.items():
        np.savetxt(const_dir / f"{mod}_constellation.csv", np.column_stack([c.real, c.imag]), delimiter=",", header="I,Q", comments="")

    print("Done.")
    print(f"Total samples: {sample_id}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
