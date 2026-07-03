"""Train and evaluate raw-waveform 1-D modulation classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .models import build_model
except ImportError:
    from models import build_model


CLASS_NAMES = ["2QAM", "4QAM", "16QAM", "64QAM", "256QAM"]


def parse_int_set(spec: str) -> set[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            result.update(range(start, end + 1))
        elif part:
            result.add(int(part))
    if not result:
        raise ValueError(f"Empty frame specification: {spec}")
    return result


def parse_args() -> argparse.Namespace:
    base = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=base / "dataset_time_domain_blind" / "M_8192"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "training_time_domain_blind" / "M_8192_resnet_se",
    )
    parser.add_argument(
        "--model", choices=["cnn1d", "resnet1d", "resnet_se"], default="resnet_se"
    )
    parser.add_argument("--train-frames", default="1-60")
    parser.add_argument("--val-frames", default="61-75")
    parser.add_argument("--test-frames", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--gain-jitter", type=float, default=0.05)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing files in an existing result directory.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        frame_ids: set[int],
        augment: bool,
        noise_std: float,
        gain_jitter: float,
    ):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.noise_std = noise_std
        self.gain_jitter = gain_jitter
        self.records: list[dict[str, str]] = []
        with (dataset_dir / "manifest.csv").open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if int(row["frame_id"]) in frame_ids:
                    self.records.append(row)
        if not self.records:
            raise RuntimeError(f"No samples selected from {dataset_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.dataset_dir / record["sample_path"]
        if path.suffix == ".npy":
            sequence = np.load(path).astype(np.float32, copy=False)
        else:
            sequence = (
                loadmat(path, variable_names=["signal_seq"])["signal_seq"]
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
        if self.augment:
            gain = np.float32(np.random.uniform(1 - self.gain_jitter, 1 + self.gain_jitter))
            noise = np.random.normal(0, self.noise_std, sequence.shape).astype(np.float32)
            sequence = sequence * gain + noise
        return (
            torch.from_numpy(np.ascontiguousarray(sequence[None, :])),
            torch.tensor(int(record["label"]), dtype=torch.long),
            int(record["frame_id"]),
            record["source_file"],
            int(record["crop_index"]),
        )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for sequences, labels, _, _, _ in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(sequences)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    result = {
        "labels": [],
        "predictions": [],
        "frame_ids": [],
        "source_files": [],
        "crop_indices": [],
        "probabilities": [],
    }
    for sequences, labels, frame_ids, source_files, crop_indices in loader:
        probabilities = model(sequences.to(device, non_blocking=True)).softmax(1).cpu()
        result["labels"].extend(labels.tolist())
        result["predictions"].extend(probabilities.argmax(1).tolist())
        result["frame_ids"].extend(frame_ids.tolist())
        result["source_files"].extend(source_files)
        result["crop_indices"].extend(crop_indices.tolist())
        result["probabilities"].extend(probabilities.tolist())
    return result


def fuse_source_files(sample_result):
    grouped: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    frames: dict[tuple[int, str], int] = {}
    for label, frame_id, source_file, probabilities in zip(
        sample_result["labels"],
        sample_result["frame_ids"],
        sample_result["source_files"],
        sample_result["probabilities"],
    ):
        key = (label, source_file)
        grouped[key].append(np.asarray(probabilities, dtype=np.float32))
        frames[key] = frame_id
    fused = {
        "labels": [],
        "predictions": [],
        "frame_ids": [],
        "source_files": [],
        "probabilities": [],
    }
    for (label, source_file), probabilities in sorted(grouped.items()):
        mean_probability = np.mean(probabilities, axis=0)
        fused["labels"].append(label)
        fused["predictions"].append(int(mean_probability.argmax()))
        fused["frame_ids"].append(frames[(label, source_file)])
        fused["source_files"].append(source_file)
        fused["probabilities"].append(mean_probability.tolist())
    return fused


def accuracy(result) -> float:
    return float(
        np.mean(np.asarray(result["labels"]) == np.asarray(result["predictions"]))
    )


def save_confusion(path: Path, result, title: str) -> None:
    matrix = confusion_matrix(result["labels"], result["predictions"], labels=range(5))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path.with_name("training_curves.png"), dpi=180)
    plt.close(fig)


def write_predictions(path: Path, result, sample_level: bool) -> None:
    fields = ["frame_id", "source_file"]
    if sample_level:
        fields.append("crop_index")
    fields += ["true_class", "predicted_class"]
    fields += [f"prob_{name}" for name in CLASS_NAMES]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate(result["labels"]):
            row = {
                "frame_id": result["frame_ids"][index],
                "source_file": result["source_files"][index],
                "true_class": CLASS_NAMES[label],
                "predicted_class": CLASS_NAMES[result["predictions"][index]],
            }
            if sample_level:
                row["crop_index"] = result["crop_indices"][index]
            row.update(
                {
                    f"prob_{name}": value
                    for name, value in zip(CLASS_NAMES, result["probabilities"][index])
                }
            )
            writer.writerow(row)


def evaluate_split(name: str, model, loader, device, output_dir: Path):
    sample_result = predict(model, loader, device)
    frame_result = fuse_source_files(sample_result)
    write_predictions(output_dir / f"{name}_crop_predictions.csv", sample_result, True)
    write_predictions(output_dir / f"{name}_frame_predictions.csv", frame_result, False)
    save_confusion(
        output_dir / f"{name}_crop_confusion_matrix.png",
        sample_result,
        f"{name}: random waveform crops",
    )
    save_confusion(
        output_dir / f"{name}_frame_confusion_matrix.png",
        frame_result,
        f"{name}: source-file probability fusion",
    )
    return {
        "crop_accuracy": accuracy(sample_result),
        "frame_accuracy": accuracy(frame_result),
        "crop_count": len(sample_result["labels"]),
        "frame_count": len(frame_result["labels"]),
        "crop_classification_report": classification_report(
            sample_result["labels"],
            sample_result["predictions"],
            labels=range(5),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "frame_classification_report": classification_report(
            frame_result["labels"],
            frame_result["predictions"],
            labels=range(5),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    protected_outputs = [
        args.output_dir / "best_model.pt",
        args.output_dir / "metrics.json",
        args.output_dir / "history.csv",
    ]
    if not args.allow_overwrite and any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            "Result directory already contains an experiment. Choose a new "
            "--output-dir or pass --allow-overwrite explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_frames = parse_int_set(args.train_frames)
    val_frames = parse_int_set(args.val_frames)
    test_frames = parse_int_set(args.test_frames) if args.test_frames else None
    if train_frames & val_frames:
        raise ValueError("Training and validation frame IDs overlap.")
    if test_frames and ((train_frames & test_frames) or (val_frames & test_frames)):
        raise ValueError("Test frame IDs overlap with training or validation.")

    train_dataset = SequenceDataset(
        args.dataset_dir,
        train_frames,
        augment=True,
        noise_std=args.noise_std,
        gain_jitter=args.gain_jitter,
    )
    val_dataset = SequenceDataset(
        args.dataset_dir, val_frames, augment=False, noise_std=0, gain_jitter=0
    )
    test_dataset = (
        SequenceDataset(
            args.dataset_dir, test_frames, augment=False, noise_std=0, gain_jitter=0
        )
        if test_frames
        else None
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True, workers=args.num_workers
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, shuffle=False, workers=args.num_workers
    )
    test_loader = (
        make_loader(test_dataset, args.batch_size, False, args.num_workers)
        if test_dataset
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

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
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f}/{train_acc:.4f} "
            f"| val={val_loss:.4f}/{val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "best_val_crop_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "class_names": CLASS_NAMES,
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
            break

    save_history(args.output_dir / "history.csv", history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "sequence_length": int(train_dataset.records[0]["sequence_length"]),
        "train_frames": sorted(train_frames),
        "validation_frames": sorted(val_frames),
        "test_frames": sorted(test_frames) if test_frames else None,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_online_validation_crop_accuracy": best_accuracy,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset) if test_dataset else 0,
        "validation": evaluate_split(
            "validation", model, val_loader, device, args.output_dir
        ),
        "blind_preprocessing": {
            "uses_fft": False,
            "uses_equalization": False,
            "uses_known_training_sequence": False,
            "uses_transmitted_reference": False,
        },
    }
    if test_loader:
        metrics["test"] = evaluate_split(
            "test", model, test_loader, device, args.output_dir
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
