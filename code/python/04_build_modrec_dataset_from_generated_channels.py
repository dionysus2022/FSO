# -*- coding: utf-8 -*-
"""
04_build_modrec_dataset_from_generated_channels.py
------------------------------------------------------------
作用：
    将 03_generate_channels_snr_range.py 输出的 GNN 生成信道 H_gen(k)
    转换为调制识别训练数据：

        Y(k,n) = H_gen(k) * X(k,n) + W(k,n)

    其中：
        k : 子载波索引
        n : 一个样本内的 OFDM 符号/时间片索引
        X : 随机调制符号，平均功率归一化为 1
        W : 按 target_snr_db 控制功率的复高斯噪声

输入：
    generated_channels_snr_5_20.npz
        H_gen          complex64 [Nch, K]
        target_snr_db  float32   [Nch]
        turb_label     int64     [Nch]

输出：
    out_dir/
        modrec_synth_shard_0000.npz
        modrec_synth_shard_0001.npz
        ...
        manifest.csv
        dataset_info.json

每个 shard 内包含：
    Y_iq            float32/float16 [Ns, 2, K, T]  默认 channel_first
                    Y_iq[:,0,:,:] = real(Y)
                    Y_iq[:,1,:,:] = imag(Y)
    mod_label       int64   [Ns]
    mod_name        str     [Ns]
    turb_label      int64   [Ns]
    target_snr_db   float32 [Ns]
    actual_snr_db   float32 [Ns]
    channel_index   int64   [Ns]
    split_label     int64   [Ns]   0=train, 1=val

运行示例 Windows PowerShell / CMD：
    python 04_build_modrec_dataset_from_generated_channels.py ^
      --channel_npz "D:/Project_code/OptDSP_FSO_2/practical waterfiling-exp/OptDSP_lite-master/2_Data_Results/rx1_batch_450frames_results/2026.06.26/gnn_channel_model/generated/generated_channels_snr_5_20.npz" ^
      --out_dir "D:/Project_code/OptDSP_FSO_2/practical waterfiling-exp/OptDSP_lite-master/2_Data_Results/rx1_batch_450frames_results/2026.06.26/modrec_dataset_gnn_5_20dB" ^
      --mods QPSK,16QAM,32QAM,64QAM,128QAM,256QAM ^
      --ofdm_symbols 32 ^
      --repeat_per_channel 1 ^
      --shard_samples 4096 ^
      --sample_norm rms

注意：
    1) 如果你的真实 TX 使用了自定义 32QAM/128QAM 星座，请把 qam_constellation() 换成你的真实星座映射。
    2) 真实测试数据也必须做同样的 sample_norm 预处理，否则训练/测试分布不一致。
------------------------------------------------------------
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_mods(mods: str) -> List[str]:
    return [m.strip().upper() for m in mods.split(",") if m.strip()]


def qam_constellation(mod_name: str) -> np.ndarray:
    """Return complex constellation points normalized to unit average power.

    Square QAM (16/64/256): standard rectangular grid.
    Cross QAM (32/128): standard cross constellation (matches MATLAB qammod /
    QAM_loadConstellation with class='cross'). Verified against TX's
    32QAM_cross.mat and 128QAM_cross.mat.
    """
    name = mod_name.upper()
    if name == "QPSK":
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64)
    else:
        if not name.endswith("QAM"):
            raise ValueError(f"Unsupported modulation: {mod_name}")
        M = int(name.replace("QAM", ""))

        root = int(round(math.sqrt(M)))
        if root * root == M:
            # square QAM: 16/64/256...
            i_levels = np.arange(-(root - 1), root, 2, dtype=np.float32)
            q_levels = np.arange(-(root - 1), root, 2, dtype=np.float32)
            I, Q = np.meshgrid(i_levels, q_levels, indexing="xy")
            pts = (I.reshape(-1) + 1j * Q.reshape(-1)).astype(np.complex64)
        elif M == 32:
            # cross-32QAM: 6×6 grid minus 4 corners
            levels = np.arange(-5, 6, 2, dtype=np.float32)
            I, Q = np.meshgrid(levels, levels, indexing="xy")
            pts = (I.reshape(-1) + 1j * Q.reshape(-1)).astype(np.complex64)
            mask = ~((np.abs(I.reshape(-1)) == 5) & (np.abs(Q.reshape(-1)) == 5))
            pts = pts[mask]
        elif M == 128:
            # cross-128QAM: 12×12 grid minus 4×4 corner block
            levels = np.arange(-11, 12, 2, dtype=np.float32)
            I, Q = np.meshgrid(levels, levels, indexing="xy")
            pts = (I.reshape(-1) + 1j * Q.reshape(-1)).astype(np.complex64)
            mask = ~((np.abs(I.reshape(-1)) >= 9) & (np.abs(Q.reshape(-1)) >= 9))
            pts = pts[mask]
        else:
            raise ValueError(
                f"Unsupported non-square QAM M={M}. Please define its constellation manually."
            )

    pts = pts.astype(np.complex64)
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2)).astype(np.float32)
    return pts.astype(np.complex64)


def generate_symbols(rng: np.random.Generator, constellation: np.ndarray, K: int, T: int) -> np.ndarray:
    idx = rng.integers(0, constellation.size, size=(K, T), endpoint=False)
    return constellation[idx].astype(np.complex64)


def add_awgn_by_target_snr(
    rng: np.random.Generator,
    y0: np.ndarray,
    target_snr_db: float,
) -> Tuple[np.ndarray, float]:
    """Add complex AWGN according to average SNR of y0."""
    p_signal = float(np.mean(np.abs(y0) ** 2))
    if p_signal <= 0 or not np.isfinite(p_signal):
        p_signal = 1e-12
    p_noise = p_signal / (10.0 ** (float(target_snr_db) / 10.0))
    noise = np.sqrt(p_noise / 2.0) * (
        rng.standard_normal(y0.shape).astype(np.float32)
        + 1j * rng.standard_normal(y0.shape).astype(np.float32)
    )
    y = y0 + noise.astype(np.complex64)
    actual_snr_db = 10.0 * np.log10(p_signal / max(float(np.mean(np.abs(noise) ** 2)), 1e-20))
    return y.astype(np.complex64), float(actual_snr_db)


def normalize_sample(y: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "none":
        return y.astype(np.complex64)
    if mode == "rms":
        rms = np.sqrt(np.mean(np.abs(y) ** 2)).astype(np.float32)
        if float(rms) < 1e-12 or not np.isfinite(float(rms)):
            return y.astype(np.complex64)
        return (y / rms).astype(np.complex64)
    if mode == "zero_mean_rms":
        y2 = y - np.mean(y)
        rms = np.sqrt(np.mean(np.abs(y2) ** 2)).astype(np.float32)
        if float(rms) < 1e-12 or not np.isfinite(float(rms)):
            return y2.astype(np.complex64)
        return (y2 / rms).astype(np.complex64)
    raise ValueError(f"Unknown sample_norm mode: {mode}")


def complex_to_iq(y: np.ndarray, layout: str, dtype: str) -> np.ndarray:
    """Convert complex [K,T] to real IQ tensor."""
    if layout == "channel_first":
        out = np.stack([np.real(y), np.imag(y)], axis=0)  # [2,K,T]
    elif layout == "channel_last":
        out = np.stack([np.real(y), np.imag(y)], axis=-1)  # [K,T,2]
    else:
        raise ValueError(f"Unknown layout: {layout}")
    if dtype == "float16":
        return out.astype(np.float16)
    if dtype == "float32":
        return out.astype(np.float32)
    raise ValueError(f"Unknown dtype: {dtype}")


def save_shard(
    out_dir: str,
    shard_id: int,
    buffer: Dict[str, list],
    layout: str,
    dtype: str,
    manifest_rows: List[Dict[str, str]],
    global_start: int,
) -> int:
    n = len(buffer["Y_iq"])
    if n == 0:
        return global_start

    shard_name = f"modrec_synth_shard_{shard_id:04d}.npz"
    shard_path = os.path.join(out_dir, shard_name)

    Y_iq = np.stack(buffer["Y_iq"], axis=0)
    mod_label = np.asarray(buffer["mod_label"], dtype=np.int64)
    mod_name = np.asarray(buffer["mod_name"], dtype=object)
    turb_label = np.asarray(buffer["turb_label"], dtype=np.int64)
    target_snr_db = np.asarray(buffer["target_snr_db"], dtype=np.float32)
    actual_snr_db = np.asarray(buffer["actual_snr_db"], dtype=np.float32)
    channel_index = np.asarray(buffer["channel_index"], dtype=np.int64)
    split_label = np.asarray(buffer["split_label"], dtype=np.int64)

    np.savez_compressed(
        shard_path,
        Y_iq=Y_iq,
        mod_label=mod_label,
        mod_name=mod_name,
        turb_label=turb_label,
        target_snr_db=target_snr_db,
        actual_snr_db=actual_snr_db,
        channel_index=channel_index,
        split_label=split_label,
        layout=np.asarray(layout),
        dtype=np.asarray(dtype),
    )

    manifest_rows.append(
        {
            "shard": shard_name,
            "num_samples": str(n),
            "global_start": str(global_start),
            "global_end_exclusive": str(global_start + n),
            "shape": str(tuple(Y_iq.shape)),
        }
    )
    print(f"Saved {shard_name}: {Y_iq.shape}")

    for key in buffer:
        buffer[key].clear()
    return global_start + n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_npz", type=str, required=True, help="generated_channels_snr_5_20.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mods", type=str, default="QPSK,16QAM,32QAM,64QAM,128QAM,256QAM")
    parser.add_argument("--ofdm_symbols", type=int, default=32, help="T in Y(k,n), number of OFDM symbols/time slices per sample")
    parser.add_argument("--repeat_per_channel", type=int, default=1, help="Number of random symbol blocks generated for each H_gen and each modulation")
    parser.add_argument("--max_channels", type=int, default=0, help="0 means use all generated channels; otherwise use first max_channels after shuffle")
    parser.add_argument("--shard_samples", type=int, default=4096)
    parser.add_argument("--sample_norm", type=str, default="rms", choices=["none", "rms", "zero_mean_rms"])
    parser.add_argument("--layout", type=str, default="channel_first", choices=["channel_first", "channel_last"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Synthetic validation ratio. Final test should use real collected data.")
    parser.add_argument("--snr_jitter_db", type=float, default=0.0, help="Uniform jitter added to target SNR, e.g. 0.25 means ±0.25 dB")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.channel_npz, allow_pickle=True)
    required = ["H_gen", "target_snr_db", "turb_label"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {args.channel_npz}. Existing keys: {list(data.keys())}")

    H_gen = data["H_gen"].astype(np.complex64)  # [Nch,K]
    target_snr_db_all = data["target_snr_db"].astype(np.float32)
    turb_label_all = data["turb_label"].astype(np.int64)

    if H_gen.ndim != 2:
        raise ValueError(f"H_gen must be [Nch,K], got {H_gen.shape}")
    if target_snr_db_all.shape[0] != H_gen.shape[0] or turb_label_all.shape[0] != H_gen.shape[0]:
        raise ValueError("H_gen, target_snr_db, turb_label length mismatch.")

    n_channels, K = H_gen.shape
    channel_indices = np.arange(n_channels, dtype=np.int64)
    rng.shuffle(channel_indices)
    if args.max_channels and args.max_channels > 0:
        channel_indices = channel_indices[: min(args.max_channels, n_channels)]

    mods = parse_mods(args.mods)
    constellations = {m: qam_constellation(m) for m in mods}
    mod_to_label = {m: i for i, m in enumerate(mods)}

    total_samples_expected = len(channel_indices) * len(mods) * args.repeat_per_channel
    print("=" * 72)
    print("Build modulation-recognition dataset from GNN-generated channels")
    print(f"Input channels       : {n_channels}")
    print(f"Used channels        : {len(channel_indices)}")
    print(f"Subcarriers K        : {K}")
    print(f"OFDM symbols T       : {args.ofdm_symbols}")
    print(f"Modulations          : {mods}")
    print(f"Repeat/channel/mod   : {args.repeat_per_channel}")
    print(f"Expected samples     : {total_samples_expected}")
    print(f"Sample normalization : {args.sample_norm}")
    print(f"Output layout        : {args.layout}")
    print("=" * 72)

    buffer = {
        "Y_iq": [],
        "mod_label": [],
        "mod_name": [],
        "turb_label": [],
        "target_snr_db": [],
        "actual_snr_db": [],
        "channel_index": [],
        "split_label": [],
    }
    manifest_rows: List[Dict[str, str]] = []
    shard_id = 0
    global_count = 0

    for count_ch, ch_idx in enumerate(channel_indices, start=1):
        H = H_gen[ch_idx].reshape(K, 1).astype(np.complex64)
        base_snr = float(target_snr_db_all[ch_idx])
        turb = int(turb_label_all[ch_idx])

        for mod_name in mods:
            const = constellations[mod_name]
            mod_label = mod_to_label[mod_name]

            for _ in range(args.repeat_per_channel):
                X = generate_symbols(rng, const, K=K, T=args.ofdm_symbols)
                Y0 = H * X

                if args.snr_jitter_db > 0:
                    snr_db = base_snr + float(rng.uniform(-args.snr_jitter_db, args.snr_jitter_db))
                else:
                    snr_db = base_snr

                Y, actual_snr = add_awgn_by_target_snr(rng, Y0, snr_db)
                Y = normalize_sample(Y, args.sample_norm)
                Y_iq = complex_to_iq(Y, layout=args.layout, dtype=args.dtype)

                # Synthetic train/val split. Real second acquisition should be the final test set.
                split_label = 1 if rng.random() < args.val_ratio else 0

                buffer["Y_iq"].append(Y_iq)
                buffer["mod_label"].append(mod_label)
                buffer["mod_name"].append(mod_name)
                buffer["turb_label"].append(turb)
                buffer["target_snr_db"].append(snr_db)
                buffer["actual_snr_db"].append(actual_snr)
                buffer["channel_index"].append(int(ch_idx))
                buffer["split_label"].append(split_label)

                if len(buffer["Y_iq"]) >= args.shard_samples:
                    global_count = save_shard(
                        args.out_dir,
                        shard_id,
                        buffer,
                        layout=args.layout,
                        dtype=args.dtype,
                        manifest_rows=manifest_rows,
                        global_start=global_count,
                    )
                    shard_id += 1

        if count_ch % 500 == 0 or count_ch == len(channel_indices):
            print(f"Processed channels: {count_ch}/{len(channel_indices)}")

    if len(buffer["Y_iq"]) > 0:
        global_count = save_shard(
            args.out_dir,
            shard_id,
            buffer,
            layout=args.layout,
            dtype=args.dtype,
            manifest_rows=manifest_rows,
            global_start=global_count,
        )

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shard", "num_samples", "global_start", "global_end_exclusive", "shape"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    info = {
        "channel_npz": args.channel_npz,
        "num_input_channels": int(n_channels),
        "num_used_channels": int(len(channel_indices)),
        "num_output_samples": int(global_count),
        "subcarriers_K": int(K),
        "ofdm_symbols_T": int(args.ofdm_symbols),
        "mods": mods,
        "mod_to_label": mod_to_label,
        "repeat_per_channel": int(args.repeat_per_channel),
        "sample_norm": args.sample_norm,
        "layout": args.layout,
        "dtype": args.dtype,
        "val_ratio": float(args.val_ratio),
        "snr_jitter_db": float(args.snr_jitter_db),
        "seed": int(args.seed),
        "note": "split_label: 0=train, 1=synthetic val. Final test should use the second real acquisition data.",
    }
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Finished.")
    print(f"Output samples : {global_count}")
    print(f"Output dir     : {args.out_dir}")
    print(f"Manifest       : {manifest_path}")
    print("Next step: train the modulation classifier on split_label=0, tune on split_label=1, and test on the second real acquisition.")
    print("=" * 72)


if __name__ == "__main__":
    main()
