"""Generate long, synchronization-free time-domain sequence datasets.

The script reads the raw Keysight waveform, resamples 80 GSa/s to 16 GSa/s,
removes DC, normalizes by waveform standard deviation, and takes deterministic
random crops. It does not perform frame synchronization, FFT, channel
estimation, equalization, or use transmitted reference symbols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly


CLASS_NAMES = ["2QAM", "4QAM", "16QAM", "64QAM", "256QAM"]
LABELS = {name: index for index, name in enumerate(CLASS_NAMES)}
SUBDIR_PATTERN = re.compile(r"sub(\d+)$", re.IGNORECASE)


def parse_int_set(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def parse_args() -> argparse.Namespace:
    project_data = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_data / "rx_data" / "2026.06.21",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_data / "dataset_time_domain_blind",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--crops-per-file", type=int, default=8)
    parser.add_argument("--source-rate", type=float, default=80e9)
    parser.add_argument("--target-rate", type=float, default=16e9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["npy", "mat"], default="npy")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing files in an existing output dataset.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Optional frame selection, e.g. 1-60 or 1,2,61,62.",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional debug limit after frame filtering.",
    )
    return parser.parse_args()


def read_keysight_bin(path: Path) -> np.ndarray:
    """Read the binary layout used by the existing MATLAB receiver scripts."""
    with path.open("rb") as file:
        cookie = file.read(2)
        version = file.read(2)
        if len(cookie) != 2 or len(version) != 2:
            raise ValueError(f"{path}: incomplete header")

        integers = struct.unpack("<7i", file.read(28))
        num_points = integers[5]
        file.read(4)  # x display range
        file.read(8 * 3)  # x display origin, increment, origin
        file.read(4 * 2)  # x/y units
        file.read(16 + 16 + 24 + 16)
        file.read(8)  # time tag
        file.read(4)  # segment index
        file.read(4)  # data header size
        file.read(2)  # buffer type
        bytes_per_point = struct.unpack("<h", file.read(2))[0]
        file.read(4)  # buffer size

        dtype_map = {1: "<i1", 2: "<i2", 4: "<f4", 8: "<f8"}
        if bytes_per_point not in dtype_map:
            raise ValueError(f"{path}: unsupported bytes_per_point={bytes_per_point}")
        signal = np.fromfile(file, dtype=dtype_map[bytes_per_point], count=num_points)

    if signal.size != num_points:
        raise ValueError(f"{path}: expected {num_points} points, read {signal.size}")
    return signal.astype(np.float32, copy=False)


def preprocess(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    ratio = source_rate / target_rate
    rounded = round(ratio)
    if not np.isclose(ratio, rounded):
        raise ValueError("This implementation requires an integer downsampling ratio.")
    processed = resample_poly(signal, up=1, down=int(rounded)).astype(
        np.float32, copy=False
    )
    processed -= np.mean(processed, dtype=np.float64)
    scale = float(np.std(processed, dtype=np.float64))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Waveform has invalid standard deviation.")
    processed /= scale
    if not np.all(np.isfinite(processed)):
        raise ValueError("Waveform contains non-finite values after preprocessing.")
    return processed


def discover_files(input_root: Path, selected_frames: set[int] | None) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}
    for modulation in CLASS_NAMES:
        class_root = input_root / modulation
        files = sorted(
            class_root.glob("sub*/*.bin"),
            key=lambda path: (
                int(SUBDIR_PATTERN.fullmatch(path.parent.name).group(1)),
                int(path.stem),
            ),
        )
        if selected_frames is not None:
            files = [path for path in files if int(path.stem) in selected_frames]
        discovered[modulation] = files
    return discovered


def sample_seed(base_seed: int, source_file: Path, length: int) -> int:
    payload = f"{base_seed}|{source_file.as_posix()}|{length}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def save_sample(
    output_path: Path,
    file_format: str,
    signal_seq: np.ndarray,
    metadata: dict[str, object],
) -> None:
    if file_format == "npy":
        np.save(output_path, signal_seq.astype(np.float32, copy=False))
        return
    savemat(
        output_path,
        {
            "signal_seq": signal_seq.astype(np.float32, copy=False),
            "label": np.asarray([[metadata["label"]]], dtype=np.int16),
            "frame_id": np.asarray([[metadata["frame_id"]]], dtype=np.int16),
            "modulation": metadata["modulation"],
            "source_file": metadata["source_file"],
            "crop_start": np.asarray([[metadata["crop_start"]]], dtype=np.int64),
            "source_rate": np.asarray([[metadata["source_rate"]]], dtype=np.float64),
            "target_rate": np.asarray([[metadata["target_rate"]]], dtype=np.float64),
        },
        do_compression=True,
    )


def main() -> None:
    args = parse_args()
    if args.crops_per_file < 1:
        raise ValueError("--crops-per-file must be positive")
    if any(length < 128 for length in args.lengths):
        raise ValueError("All sequence lengths must be at least 128.")
    existing_outputs = [
        args.output_root / "dataset_summary.json",
        *[
            args.output_root / f"M_{length}" / "manifest.csv"
            for length in sorted(set(args.lengths))
        ],
    ]
    if not args.allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            "Output dataset already contains generated files. Choose a new "
            "--output-root or pass --allow-overwrite explicitly."
        )

    selected_frames = parse_int_set(args.frames)
    discovered = discover_files(args.input_root, selected_frames)
    for modulation, files in discovered.items():
        if args.max_files_per_class is not None:
            discovered[modulation] = files[: args.max_files_per_class]
        if not discovered[modulation]:
            raise RuntimeError(f"No input files found for {modulation}")

    manifests: dict[int, tuple[Path, object, csv.DictWriter]] = {}
    fields = [
        "sample_path",
        "label",
        "frame_id",
        "modulation",
        "source_file",
        "source_subfolder",
        "crop_index",
        "crop_start",
        "sequence_length",
        "source_rate",
        "target_rate",
        "waveform_mean_before_normalization",
        "waveform_std_before_normalization",
        "crop_mean",
        "crop_std",
    ]
    for length in sorted(set(args.lengths)):
        length_root = args.output_root / f"M_{length}"
        sample_root = length_root / "samples"
        sample_root.mkdir(parents=True, exist_ok=True)
        handle = (length_root / "manifest.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        manifests[length] = (sample_root, handle, writer)

    counts = {length: {name: 0 for name in CLASS_NAMES} for length in manifests}
    failures: list[dict[str, str]] = []
    try:
        for modulation in CLASS_NAMES:
            for source_file in discovered[modulation]:
                frame_id = int(source_file.stem)
                try:
                    raw = read_keysight_bin(source_file)
                    raw_mean = float(np.mean(raw, dtype=np.float64))
                    raw_std = float(np.std(raw, dtype=np.float64))
                    signal = preprocess(raw, args.source_rate, args.target_rate)
                    relative_source = source_file.relative_to(args.input_root).as_posix()

                    for length, (sample_root, _, writer) in manifests.items():
                        if signal.size < length:
                            raise ValueError(
                                f"resampled length {signal.size} is shorter than M={length}"
                            )
                        rng = np.random.default_rng(
                            sample_seed(args.seed, source_file, length)
                        )
                        starts = rng.integers(
                            0,
                            signal.size - length + 1,
                            size=args.crops_per_file,
                            endpoint=False,
                        )
                        class_dir = sample_root / modulation
                        class_dir.mkdir(parents=True, exist_ok=True)
                        extension = ".npy" if args.format == "npy" else ".mat"

                        for crop_index, start_value in enumerate(starts):
                            start = int(start_value)
                            crop = signal[start : start + length].copy()
                            sample_name = (
                                f"frame_{frame_id:04d}_{source_file.parent.name}"
                                f"_crop_{crop_index:03d}{extension}"
                            )
                            sample_path = class_dir / sample_name
                            metadata = {
                                "label": LABELS[modulation],
                                "frame_id": frame_id,
                                "modulation": modulation,
                                "source_file": relative_source,
                                "crop_start": start,
                                "source_rate": args.source_rate,
                                "target_rate": args.target_rate,
                            }
                            save_sample(
                                sample_path, args.format, crop, metadata
                            )
                            writer.writerow(
                                {
                                    "sample_path": sample_path.relative_to(
                                        args.output_root / f"M_{length}"
                                    ).as_posix(),
                                    **metadata,
                                    "source_subfolder": source_file.parent.name,
                                    "crop_index": crop_index,
                                    "sequence_length": length,
                                    "waveform_mean_before_normalization": raw_mean,
                                    "waveform_std_before_normalization": raw_std,
                                    "crop_mean": float(np.mean(crop)),
                                    "crop_std": float(np.std(crop)),
                                }
                            )
                            counts[length][modulation] += 1
                except Exception as error:  # keep a complete failure audit
                    failures.append(
                        {"source_file": str(source_file), "error": repr(error)}
                    )
                    print(f"[FAILED] {source_file}: {error}", flush=True)
    finally:
        for _, handle, _ in manifests.values():
            handle.close()

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "source_rate": args.source_rate,
        "target_rate": args.target_rate,
        "lengths": sorted(manifests),
        "crops_per_file": args.crops_per_file,
        "seed": args.seed,
        "format": args.format,
        "counts": counts,
        "failures": failures,
        "processing": [
            "read raw Keysight waveform",
            "resample to target rate",
            "remove DC",
            "normalize by full-waveform standard deviation",
            "randomly crop M points",
        ],
        "explicitly_not_used": [
            "packet synchronization",
            "known preamble",
            "FFT",
            "CFO/STO compensation",
            "channel estimation",
            "equalization",
            "transmitted reference symbols",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"{len(failures)} source files failed; see dataset_summary.json")


if __name__ == "__main__":
    main()
