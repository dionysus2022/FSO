# -*- coding: utf-8 -*-
"""
04_build_modrec_dataset_v2_preserve_physics.py
------------------------------------------------------------
Purpose
    Build a modulation-recognition dataset from GNN-generated channel responses.

    Physical forward model:
        Y_raw(k,n) = H_gen(k) * X(k,n) + W(k,n)

    Optional receiver view:
        raw       : input is Y_raw
        equalized : input is Y_eq = Y_raw / H_gen, matching a receiver after channel equalization

Why this v2 exists
    The previous version commonly used per-sample RMS normalization. In the current
    experiment this can hide or distort useful constellation-distribution cues and can
    make QAM classes collapse in feature space. This script therefore defaults to:
        --receiver_view equalized
        --sample_norm none
    and lets the training script use --normalize none.

Input
    generated_channels_snr_5_20.npz from 03_generate_channels_snr_range.py
        H_gen          complex [Nch, K]
        target_snr_db  float   [Nch]
        turb_label     int     [Nch]

Output shard format, compatible with 05_train_modrec_curriculum_v2.py
    modrec_synth_shard_XXXX.npz
        Y_iq             float32 [N, 2, K, T]
        mod_label        int64   [N]
        mod_name         object  [N]
        turb_label       int64   [N]
        target_snr_db    float32 [N]
        actual_snr_db    float32 [N]
        channel_index    int64   [N]
        split_label      int64   [N]   0=train, 1=synthetic val
        receiver_view    object  [N]
        constellation_norm object scalar
        sample_norm        object scalar

Recommended command
    python 04_build_modrec_dataset_v2_preserve_physics.py ^
      --channel_npz "D:/.../generated_channels_snr_5_20.npz" ^
      --out_dir "D:/.../modrec_dataset_gnn_5_20dB_eq_nonorm" ^
      --mods QPSK,16QAM,32QAM,64QAM,128QAM,256QAM ^
      --receiver_view equalized ^
      --sample_norm none ^
      --constellation_norm unit_avg_power ^
      --ofdm_symbols 32 ^
      --repeat_per_channel 1 ^
      --shard_samples 4096
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
    return [m.strip().upper() for m in mods.split(',') if m.strip()]


def cross_qam_constellation(M: int) -> np.ndarray:
    """Meshgrid-corner-removal cross QAM.

    32QAM:  6 x 6 grid with four corner points removed -> 32.
    128QAM: 12 x 12 grid with four 2 x 2 corner blocks removed -> 128.
    """
    if M == 32:
        levels = np.arange(-5, 6, 2, dtype=np.float32)  # -5,-3,-1,1,3,5
        I, Q = np.meshgrid(levels, levels, indexing='xy')
        mask = ~((np.abs(I) == 5) & (np.abs(Q) == 5))
        pts = I[mask] + 1j * Q[mask]
    elif M == 128:
        levels = np.arange(-11, 12, 2, dtype=np.float32)  # 12 levels
        I, Q = np.meshgrid(levels, levels, indexing='xy')
        # remove four 2x2 corner blocks: |I| in {9,11} and |Q| in {9,11}
        mask = ~((np.abs(I) >= 9) & (np.abs(Q) >= 9))
        pts = I[mask] + 1j * Q[mask]
    else:
        raise ValueError(f'cross_qam_constellation supports only 32/128, got M={M}')
    pts = np.asarray(pts, dtype=np.complex64).reshape(-1)
    if pts.size != M:
        raise RuntimeError(f'Cross-QAM size mismatch: got {pts.size}, expected {M}')
    return pts


def square_qam_constellation(M: int) -> np.ndarray:
    root = int(round(math.sqrt(M)))
    if root * root != M:
        raise ValueError(f'M={M} is not square QAM')
    levels = np.arange(-(root - 1), root, 2, dtype=np.float32)
    I, Q = np.meshgrid(levels, levels, indexing='xy')
    return (I.reshape(-1) + 1j * Q.reshape(-1)).astype(np.complex64)


def qam_constellation(mod_name: str, constellation_norm: str = 'unit_avg_power') -> np.ndarray:
    name = mod_name.upper()
    if name == 'QPSK':
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64)
    else:
        if not name.endswith('QAM'):
            raise ValueError(f'Unsupported modulation: {mod_name}')
        M = int(name.replace('QAM', ''))
        if M in (32, 128):
            pts = cross_qam_constellation(M)
        else:
            pts = square_qam_constellation(M)

    constellation_norm = constellation_norm.lower()
    if constellation_norm in ('unit', 'unit_avg', 'unit_avg_power'):
        p = float(np.mean(np.abs(pts) ** 2))
        pts = pts / np.sqrt(max(p, 1e-12))
    elif constellation_norm in ('none', 'raw'):
        pass
    else:
        raise ValueError(f'Unknown constellation_norm: {constellation_norm}')
    return pts.astype(np.complex64)


def generate_symbols(rng: np.random.Generator, constellation: np.ndarray, K: int, T: int) -> np.ndarray:
    idx = rng.integers(0, constellation.size, size=(K, T), endpoint=False)
    return constellation[idx].astype(np.complex64)


def add_awgn_by_target_snr(rng: np.random.Generator, y0: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float]:
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


def equalize_by_channel(y_raw: np.ndarray, H: np.ndarray, channel_floor: float) -> np.ndarray:
    """Equalize y_raw [K,T] by H [K,1] with a magnitude floor for numerical safety."""
    Hc = H.astype(np.complex64).copy()
    mag = np.abs(Hc)
    phase = np.exp(1j * np.angle(Hc)).astype(np.complex64)
    floor = float(channel_floor)
    if floor > 0:
        Hc = np.where(mag < floor, floor * phase, Hc).astype(np.complex64)
    else:
        Hc = np.where(mag < 1e-12, 1e-12 * phase, Hc).astype(np.complex64)
    return (y_raw / Hc).astype(np.complex64)


def normalize_sample(y: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == 'none':
        return y.astype(np.complex64)
    if mode in ('rms', 'sample_rms'):
        rms = np.sqrt(np.mean(np.abs(y) ** 2)).astype(np.float32)
        if float(rms) < 1e-12 or not np.isfinite(float(rms)):
            return y.astype(np.complex64)
        return (y / rms).astype(np.complex64)
    if mode == 'zero_mean_rms':
        y2 = y - np.mean(y)
        rms = np.sqrt(np.mean(np.abs(y2) ** 2)).astype(np.float32)
        if float(rms) < 1e-12 or not np.isfinite(float(rms)):
            return y2.astype(np.complex64)
        return (y2 / rms).astype(np.complex64)
    raise ValueError(f'Unknown sample_norm mode: {mode}')


def complex_to_iq(y: np.ndarray, layout: str, dtype: str) -> np.ndarray:
    if layout == 'channel_first':
        out = np.stack([np.real(y), np.imag(y)], axis=0)  # [2,K,T]
    elif layout == 'channel_last':
        out = np.stack([np.real(y), np.imag(y)], axis=-1)  # [K,T,2]
    else:
        raise ValueError(f'Unknown layout: {layout}')
    if dtype == 'float16':
        return out.astype(np.float16)
    if dtype == 'float32':
        return out.astype(np.float32)
    raise ValueError(f'Unknown dtype: {dtype}')


def save_shard(out_dir: str, shard_id: int, buffer: Dict[str, list], manifest_rows: List[Dict[str, str]], global_start: int, layout: str, dtype: str, constellation_norm: str, sample_norm: str) -> int:
    n = len(buffer['Y_iq'])
    if n == 0:
        return global_start
    shard_name = f'modrec_synth_shard_{shard_id:04d}.npz'
    shard_path = os.path.join(out_dir, shard_name)
    Y_iq = np.stack(buffer['Y_iq'], axis=0)
    np.savez_compressed(
        shard_path,
        Y_iq=Y_iq,
        mod_label=np.asarray(buffer['mod_label'], dtype=np.int64),
        mod_name=np.asarray(buffer['mod_name'], dtype=object),
        turb_label=np.asarray(buffer['turb_label'], dtype=np.int64),
        target_snr_db=np.asarray(buffer['target_snr_db'], dtype=np.float32),
        actual_snr_db=np.asarray(buffer['actual_snr_db'], dtype=np.float32),
        channel_index=np.asarray(buffer['channel_index'], dtype=np.int64),
        split_label=np.asarray(buffer['split_label'], dtype=np.int64),
        receiver_view=np.asarray(buffer['receiver_view'], dtype=object),
        layout=np.asarray(layout),
        dtype=np.asarray(dtype),
        constellation_norm=np.asarray(constellation_norm),
        sample_norm=np.asarray(sample_norm),
    )
    manifest_rows.append({
        'shard': shard_name,
        'num_samples': str(n),
        'global_start': str(global_start),
        'global_end_exclusive': str(global_start + n),
        'shape': str(tuple(Y_iq.shape)),
    })
    print(f'Saved {shard_name}: {Y_iq.shape}')
    for key in buffer:
        buffer[key].clear()
    return global_start + n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--channel_npz', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mods', default='QPSK,16QAM,32QAM,64QAM,128QAM,256QAM')
    p.add_argument('--ofdm_symbols', type=int, default=32)
    p.add_argument('--repeat_per_channel', type=int, default=1)
    p.add_argument('--max_channels', type=int, default=0)
    p.add_argument('--shard_samples', type=int, default=4096)
    p.add_argument('--receiver_view', default='equalized', choices=['raw', 'equalized'])
    p.add_argument('--sample_norm', default='none', choices=['none', 'sample_rms', 'rms', 'zero_mean_rms'])
    p.add_argument('--constellation_norm', default='unit_avg_power', choices=['unit_avg_power', 'unit_avg', 'unit', 'none', 'raw'])
    p.add_argument('--channel_floor', type=float, default=1e-3, help='Magnitude floor used only in equalized view.')
    p.add_argument('--layout', default='channel_first', choices=['channel_first', 'channel_last'])
    p.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
    p.add_argument('--val_ratio', type=float, default=0.10)
    p.add_argument('--snr_jitter_db', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.channel_npz, allow_pickle=True)
    for key in ['H_gen', 'target_snr_db', 'turb_label']:
        if key not in data.files:
            raise KeyError(f'Missing {key} in {args.channel_npz}. Existing keys: {data.files}')
    H_gen = data['H_gen'].astype(np.complex64)
    target_snr_db_all = data['target_snr_db'].astype(np.float32)
    turb_label_all = data['turb_label'].astype(np.int64)
    if H_gen.ndim != 2:
        raise ValueError(f'H_gen must be [Nch,K], got {H_gen.shape}')
    n_channels, K = H_gen.shape

    channel_indices = np.arange(n_channels, dtype=np.int64)
    rng.shuffle(channel_indices)
    if args.max_channels and args.max_channels > 0:
        channel_indices = channel_indices[: min(args.max_channels, n_channels)]

    mods = parse_mods(args.mods)
    constellations = {m: qam_constellation(m, args.constellation_norm) for m in mods}
    mod_to_label = {m: i for i, m in enumerate(mods)}

    constellation_info = {}
    for m, pts in constellations.items():
        constellation_info[m] = {
            'M': int(pts.size),
            'avg_power': float(np.mean(np.abs(pts) ** 2)),
            'min_abs': float(np.min(np.abs(pts))),
            'max_abs': float(np.max(np.abs(pts))),
        }

    print('=' * 80)
    print('Build modulation-recognition dataset v2')
    print(f'Input channels       : {n_channels}')
    print(f'Used channels        : {len(channel_indices)}')
    print(f'Subcarriers K        : {K}')
    print(f'OFDM symbols T       : {args.ofdm_symbols}')
    print(f'Mods                 : {mods}')
    print(f'Receiver view        : {args.receiver_view}')
    print(f'Constellation norm   : {args.constellation_norm}')
    print(f'Sample norm          : {args.sample_norm}')
    print(f'Channel floor        : {args.channel_floor}')
    print(f'Expected samples     : {len(channel_indices) * len(mods) * args.repeat_per_channel}')
    print('=' * 80)

    buffer = {k: [] for k in ['Y_iq', 'mod_label', 'mod_name', 'turb_label', 'target_snr_db', 'actual_snr_db', 'channel_index', 'split_label', 'receiver_view']}
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
                snr_db = base_snr + float(rng.uniform(-args.snr_jitter_db, args.snr_jitter_db)) if args.snr_jitter_db > 0 else base_snr
                Y_raw, actual_snr = add_awgn_by_target_snr(rng, Y0, snr_db)
                if args.receiver_view == 'raw':
                    Y_in = Y_raw
                else:
                    Y_in = equalize_by_channel(Y_raw, H, args.channel_floor)
                Y_in = normalize_sample(Y_in, args.sample_norm)
                Y_iq = complex_to_iq(Y_in, layout=args.layout, dtype=args.dtype)
                split_label = 1 if rng.random() < args.val_ratio else 0

                buffer['Y_iq'].append(Y_iq)
                buffer['mod_label'].append(mod_label)
                buffer['mod_name'].append(mod_name)
                buffer['turb_label'].append(turb)
                buffer['target_snr_db'].append(snr_db)
                buffer['actual_snr_db'].append(actual_snr)
                buffer['channel_index'].append(int(ch_idx))
                buffer['split_label'].append(split_label)
                buffer['receiver_view'].append(args.receiver_view)

                if len(buffer['Y_iq']) >= args.shard_samples:
                    global_count = save_shard(args.out_dir, shard_id, buffer, manifest_rows, global_count, args.layout, args.dtype, args.constellation_norm, args.sample_norm)
                    shard_id += 1

        if count_ch % 500 == 0 or count_ch == len(channel_indices):
            print(f'Processed channels: {count_ch}/{len(channel_indices)}')

    if len(buffer['Y_iq']) > 0:
        global_count = save_shard(args.out_dir, shard_id, buffer, manifest_rows, global_count, args.layout, args.dtype, args.constellation_norm, args.sample_norm)

    manifest_path = os.path.join(args.out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['shard', 'num_samples', 'global_start', 'global_end_exclusive', 'shape'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    info = {
        'channel_npz': args.channel_npz,
        'num_input_channels': int(n_channels),
        'num_used_channels': int(len(channel_indices)),
        'num_output_samples': int(global_count),
        'subcarriers_K': int(K),
        'ofdm_symbols_T': int(args.ofdm_symbols),
        'mods': mods,
        'mod_to_label': mod_to_label,
        'repeat_per_channel': int(args.repeat_per_channel),
        'receiver_view': args.receiver_view,
        'sample_norm': args.sample_norm,
        'constellation_norm': args.constellation_norm,
        'channel_floor': float(args.channel_floor),
        'constellation_info': constellation_info,
        'layout': args.layout,
        'dtype': args.dtype,
        'val_ratio': float(args.val_ratio),
        'snr_jitter_db': float(args.snr_jitter_db),
        'seed': int(args.seed),
        'important_note': 'Recommended training option for this dataset: --normalize none. Do not re-apply sample_rms unless you intentionally test that ablation.',
    }
    with open(os.path.join(args.out_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print('=' * 80)
    print('Finished v2 dataset generation.')
    print(f'Output samples : {global_count}')
    print(f'Output dir     : {args.out_dir}')
    print(f'Manifest       : {manifest_path}')
    print('Next: audit this dataset, then train with 05_train_modrec_curriculum_v2.py --normalize none')
    print('=' * 80)


if __name__ == '__main__':
    main()
