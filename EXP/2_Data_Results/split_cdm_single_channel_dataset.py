"""Create train/val single-channel CDM files from the existing 5-subband dataset."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_cdm_5subbands_blind"
TARGET = ROOT / "dataset_cdm_5subbands_single" / "dataset"
CLASS_NAMES = ["2QAM", "4QAM", "16QAM", "64QAM", "256QAM"]
LABEL_BITS = {"2QAM": 1, "4QAM": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}


def main() -> None:
    train_count = 0
    val_count = 0

    for modulation in CLASS_NAMES:
        for frame_id in range(1, 76):
            split = "train" if frame_id <= 60 else "val"
            output_dir = TARGET / split / modulation
            output_dir.mkdir(parents=True, exist_ok=True)

            for band_id in range(1, 6):
                source_file = (
                    SOURCE
                    / modulation
                    / f"band{band_id:02d}"
                    / f"sig_{frame_id:04d}.mat"
                )
                if not source_file.is_file():
                    raise FileNotFoundError(source_file)

                source = loadmat(source_file)
                cdm = source["Distorted_CDM"].astype(np.float32, copy=False)
                if cdm.shape != (64, 64):
                    raise ValueError(f"{source_file}: unexpected shape {cdm.shape}")

                output_file = output_dir / f"frame_{frame_id:04d}_band{band_id:02d}.mat"
                savemat(
                    output_file,
                    {
                        "Distorted_CDM": cdm,
                        "Label_Bits": np.array([[LABEL_BITS[modulation]]], dtype=np.uint8),
                        "modulation": modulation,
                        "Frame_ID": np.array([[frame_id]], dtype=np.uint8),
                        "Split": split,
                        "Subband_ID": np.array([[band_id]], dtype=np.uint8),
                        "Subband_Size": source["Subband_Size"],
                        "Subcarrier_Idx": source["Subcarrier_Idx"],
                        "Source_Subfolder": source["Source_Subfolder"],
                        "Source_File": source["Source_File"],
                    },
                    do_compression=True,
                )

                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Total samples: {train_count + val_count}")
    print(f"Output: {TARGET}")


if __name__ == "__main__":
    main()
