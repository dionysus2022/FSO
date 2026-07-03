"""Extract Y_iq from each compressed .npz shard into uncompressed .npy files.

The .npy format supports memory-mapped loading (mmap_mode='r'), which avoids
decompressing the full ~123 MiB array into RAM when training.
Run once after dataset generation. Deletes .npy files on success to save space.
"""

import glob
import os
import sys

import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master"
    r"\2_Data_Results\rx1_batch_450frames_results"
    r"\2026.06.26\modrec_dataset_gnn_5_20dB"
)

shard_paths = sorted(glob.glob(os.path.join(DATA_DIR, "modrec_synth_shard_*.npz")))
if not shard_paths:
    print(f"No shards found in: {DATA_DIR}")
    sys.exit(1)

print(f"Found {len(shard_paths)} shards in: {DATA_DIR}")

for sp in shard_paths:
    base = sp.replace(".npz", "")
    npy_path = base + "_yiq.npy"
    if os.path.exists(npy_path):
        fs = os.path.getsize(npy_path) / 1024 / 1024
        print(f"  Skipping (exists): {os.path.basename(npy_path)} ({fs:.1f} MB)")
        continue
    print(f"  Extracting: {os.path.basename(sp)} -> {os.path.basename(npy_path)}")
    with np.load(sp, allow_pickle=True) as z:
        yiq = z["Y_iq"]
        np.save(npy_path, yiq)
    fsize = os.path.getsize(npy_path) / 1024 / 1024
    print(f"    Saved: {fsize:.1f} MB")

print("Done. All shards converted.")