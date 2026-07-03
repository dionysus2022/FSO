"""Train a CUDA CNN with 5-subcarrier single-channel CDM samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["2QAM", "4QAM", "16QAM", "64QAM", "256QAM"]
FILE_PATTERN = re.compile(r"frame_(\d+)_band(\d+)\.mat$")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=base / "dataset_cdm_sc5_single" / "dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "training_cdm_sc5_single_channel_cuda",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class SingleChannelCDMDataset(Dataset):
    def __init__(self, split_dir: Path, augment: bool):
        self.augment = augment
        self.records: list[tuple[Path, int, int, int]] = []

        for label, modulation in enumerate(CLASS_NAMES):
            class_dir = split_dir / modulation
            for path in sorted(class_dir.glob("frame_*_band*.mat")):
                match = FILE_PATTERN.fullmatch(path.name)
                if match:
                    self.records.append(
                        (path, label, int(match.group(1)), int(match.group(2)))
                    )

        if not self.records:
            raise RuntimeError(f"No CDM files found under {split_dir}")

        features = []
        for path, _, _, _ in self.records:
            cdm = sio.loadmat(path, variable_names=["Distorted_CDM"])[
                "Distorted_CDM"
            ].astype(np.float32, copy=False)
            if cdm.shape != (64, 64):
                raise ValueError(f"{path}: expected (64, 64), got {cdm.shape}")
            features.append(cdm)
        self.features = np.stack(features, axis=0)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        _, label, frame_id, band_id = self.records[index]
        cdm = self.features[index]

        if self.augment:
            cdm = self._augment(cdm)

        feature = torch.from_numpy(np.ascontiguousarray(cdm[None, :, :]))
        return feature, torch.tensor(label, dtype=torch.long), frame_id, band_id

    @staticmethod
    def _augment(cdm: np.ndarray) -> np.ndarray:
        # A 90-degree rotation represents an unknown common carrier phase.
        cdm = np.rot90(cdm, k=np.random.randint(0, 4)).copy()
        noise = np.random.normal(0.0, 0.004, cdm.shape).astype(np.float32)
        gain = np.float32(np.random.uniform(0.95, 1.05))
        return np.clip(cdm * gain + noise, 0.0, 1.0)


class SingleChannelCDMCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels, _, _ in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(features)
            loss = criterion(logits, labels)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, list]:
    model.eval()
    result = {
        "labels": [],
        "predictions": [],
        "frame_ids": [],
        "band_ids": [],
        "probabilities": [],
    }

    for features, labels, frame_ids, band_ids in loader:
        probabilities = model(features.to(device, non_blocking=True)).softmax(1).cpu()
        result["labels"].extend(labels.tolist())
        result["predictions"].extend(probabilities.argmax(1).tolist())
        result["frame_ids"].extend(frame_ids.tolist())
        result["band_ids"].extend(band_ids.tolist())
        result["probabilities"].extend(probabilities.tolist())

    return result


def fuse_frame_predictions(sample_result: dict[str, list]) -> dict[str, list]:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for label, frame_id, probabilities in zip(
        sample_result["labels"],
        sample_result["frame_ids"],
        sample_result["probabilities"],
    ):
        grouped[(label, frame_id)].append(np.asarray(probabilities, dtype=np.float32))

    result = {"labels": [], "predictions": [], "frame_ids": [], "probabilities": []}
    for (label, frame_id), probabilities in sorted(grouped.items()):
        if len(probabilities) != 24:
            raise RuntimeError(
                f"Class {CLASS_NAMES[label]} frame {frame_id} has "
                f"{len(probabilities)} subbands instead of 24."
            )
        mean_probability = np.mean(probabilities, axis=0)
        result["labels"].append(label)
        result["predictions"].append(int(mean_probability.argmax()))
        result["frame_ids"].append(frame_id)
        result["probabilities"].append(mean_probability.tolist())
    return result


def save_confusion_matrix(
    output_path: Path, labels: list[int], predictions: list[int], title: str
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_training_curves(output_dir: Path, history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Subband accuracy", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def write_predictions(
    path: Path, result: dict[str, list], include_band: bool
) -> None:
    fields = ["frame_id"]
    if include_band:
        fields.append("band_id")
    fields.extend(["true_class", "predicted_class"])
    fields.extend(f"prob_{name}" for name in CLASS_NAMES)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        count = len(result["labels"])
        for index in range(count):
            row = {
                "frame_id": result["frame_ids"][index],
                "true_class": CLASS_NAMES[result["labels"][index]],
                "predicted_class": CLASS_NAMES[result["predictions"][index]],
            }
            if include_band:
                row["band_id"] = result["band_ids"][index]
            row.update(
                {
                    f"prob_{name}": probability
                    for name, probability in zip(
                        CLASS_NAMES, result["probabilities"][index]
                    )
                }
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but PyTorch cannot access a CUDA GPU.")
    device = torch.device("cuda")

    train_dataset = SingleChannelCDMDataset(args.data_root / "train", augment=True)
    val_dataset = SingleChannelCDMDataset(args.data_root / "val", augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SingleChannelCDMCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Training subband samples: {len(train_dataset)}")
    print(f"Validation subband samples: {len(val_dataset)}")
    print("Input shape: (1, 64, 64)")

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | lr {learning_rate:.2e}",
            flush=True,
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "input_shape": [1, 64, 64],
                    "best_subband_val_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    sample_result = predict(model, val_loader, device)
    frame_result = fuse_frame_predictions(sample_result)

    sample_accuracy = float(
        np.mean(
            np.asarray(sample_result["labels"])
            == np.asarray(sample_result["predictions"])
        )
    )
    frame_accuracy = float(
        np.mean(
            np.asarray(frame_result["labels"])
            == np.asarray(frame_result["predictions"])
        )
    )
    sample_report = classification_report(
        sample_result["labels"],
        sample_result["predictions"],
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    frame_report = classification_report(
        frame_result["labels"],
        frame_result["predictions"],
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    write_predictions(
        args.output_dir / "subband_predictions.csv", sample_result, include_band=True
    )
    write_predictions(
        args.output_dir / "frame_fused_predictions.csv", frame_result, include_band=False
    )
    save_training_curves(args.output_dir, history)
    save_confusion_matrix(
        args.output_dir / "subband_confusion_matrix.png",
        sample_result["labels"],
        sample_result["predictions"],
        "Validation: independent subband samples",
    )
    save_confusion_matrix(
        args.output_dir / "frame_fused_confusion_matrix.png",
        frame_result["labels"],
        frame_result["predictions"],
        "Validation: 24-group probability fusion",
    )

    metrics = {
        "device": torch.cuda.get_device_name(0),
        "best_epoch": best_epoch,
        "best_subband_validation_accuracy": best_accuracy,
        "final_best_checkpoint_subband_accuracy": sample_accuracy,
        "frame_fused_validation_accuracy": frame_accuracy,
        "training_subband_samples": len(train_dataset),
        "validation_subband_samples": len(val_dataset),
        "validation_frames": len(frame_result["labels"]),
        "subband_classification_report": sample_report,
        "frame_classification_report": frame_report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Best subband validation accuracy: {sample_accuracy:.4f}")
    print(f"24-group fused frame accuracy: {frame_accuracy:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
