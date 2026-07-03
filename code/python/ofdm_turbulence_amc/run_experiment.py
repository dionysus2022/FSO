"""Train AMC ResNet-SE and produce turbulence-specific SNR curves/matrices."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

try:
    from .models import AMCResNetSE, GAFCNN, IQAmplitudePhaseNet, MultiScaleAMCNet
    from .simulation import (
        GAMMA_GAMMA,
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        TURBULENCE_LEVELS,
        AmplitudePhaseDataset,
        OFDMTurbulenceDataset,
        GAFDataset,
        SimulationConfig,
    )
except ImportError:
    from models import AMCResNetSE, GAFCNN, IQAmplitudePhaseNet, MultiScaleAMCNet
    from simulation import (
        GAMMA_GAMMA,
        MODULATIONS,
        TRAINING_TURBULENCE_LEVELS,
        TURBULENCE_LEVELS,
        AmplitudePhaseDataset,
        OFDMTurbulenceDataset,
        GAFDataset,
        SimulationConfig,
    )


def parse_args() -> argparse.Namespace:
    base = Path(
        r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp"
        r"\OptDSP_lite-master\2_Data_Results"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "simulation_ofdm_turbulence_amc_resnet_se",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--snrs", type=float, nargs="+", default=list(range(-10, 21, 5)))
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=300)
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Deprecated alias. Use curriculum stage epochs instead.",
    )
    parser.add_argument("--curriculum-clean-epochs", type=int, default=5)
    parser.add_argument("--curriculum-weak-epochs", type=int, default=5)
    parser.add_argument("--curriculum-full-turbulence-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--confusion-snr", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-se", action="store_true")
    parser.add_argument(
        "--representation",
        choices=["raw_iq", "gaf", "gaf_gadf", "iq_amp_phase", "multiscale_iq"],
        default="raw_iq",
    )
    parser.add_argument("--gaf-size", type=int, default=128)
    parser.add_argument("--train-dynamic", action="store_true", default=True)
    parser.add_argument("--no-train-dynamic", dest="train_dynamic", action="store_false")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def wrap_representation(dataset, args):
    if args.representation == "gaf":
        return GAFDataset(dataset, args.gaf_size, mode="gasf")
    if args.representation == "gaf_gadf":
        return GAFDataset(dataset, args.gaf_size, mode="gasf_gadf")
    if args.representation == "iq_amp_phase":
        return AmplitudePhaseDataset(dataset)
    return dataset


def build_model(args):
    if args.representation == "gaf":
        return GAFCNN(len(MODULATIONS), in_channels=2)
    if args.representation == "gaf_gadf":
        return GAFCNN(len(MODULATIONS), in_channels=4)
    if args.representation == "iq_amp_phase":
        return IQAmplitudePhaseNet(len(MODULATIONS), use_se=not args.no_se)
    if args.representation == "multiscale_iq":
        return MultiScaleAMCNet(len(MODULATIONS), in_channels=2, use_se=not args.no_se)
    return AMCResNetSE(len(MODULATIONS), use_se=not args.no_se)


def build_curriculum(args):
    highest_snr = max(args.snrs)
    stages = []
    clean_epochs = args.curriculum_clean_epochs + max(args.pretrain_epochs, 0)
    if clean_epochs > 0:
        stages.append(
            {
                "name": "clean_high_snr",
                "epochs": clean_epochs,
                "snrs": [highest_snr],
                "turbulence_levels": ["none"],
            }
        )
    if args.curriculum_weak_epochs > 0:
        stages.append(
            {
                "name": "weak_high_snr",
                "epochs": args.curriculum_weak_epochs,
                "snrs": [highest_snr],
                "turbulence_levels": ["weak"],
            }
        )
    if args.curriculum_full_turbulence_epochs > 0:
        stages.append(
            {
                "name": "all_turbulence_high_snr",
                "epochs": args.curriculum_full_turbulence_epochs,
                "snrs": [highest_snr],
                "turbulence_levels": list(TRAINING_TURBULENCE_LEVELS),
            }
        )
    if args.epochs > 0:
        stages.append(
            {
                "name": "all_snr_finetune",
                "epochs": args.epochs,
                "snrs": args.snrs,
                "turbulence_levels": list(TRAINING_TURBULENCE_LEVELS),
            }
        )
    return stages


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for sequences, labels, _, _ in loader:
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
def evaluate(model, loader, device):
    model.eval()
    records = []
    for sequences, labels, snrs, turbulence in loader:
        probabilities = model(sequences.to(device, non_blocking=True)).softmax(1).cpu()
        predictions = probabilities.argmax(1)
        for index in range(labels.size(0)):
            records.append(
                {
                    "label": int(labels[index]),
                    "prediction": int(predictions[index]),
                    "snr": float(snrs[index]),
                    "turbulence": turbulence[index],
                    "probabilities": probabilities[index].tolist(),
                }
            )
    return records


def save_predictions(path: Path, records) -> None:
    fields = ["turbulence", "snr", "true_class", "predicted_class"] + [
        f"prob_{name}" for name in MODULATIONS
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "turbulence": record["turbulence"],
                "snr": record["snr"],
                "true_class": MODULATIONS[record["label"]],
                "predicted_class": MODULATIONS[record["prediction"]],
            }
            row.update(
                {
                    f"prob_{name}": probability
                    for name, probability in zip(
                        MODULATIONS, record["probabilities"]
                    )
                }
            )
            writer.writerow(row)


def summarize(records):
    rows = []
    for turbulence in TURBULENCE_LEVELS:
        for snr in sorted({record["snr"] for record in records}):
            selected = [
                record
                for record in records
                if record["turbulence"] == turbulence and record["snr"] == snr
            ]
            rows.append(
                {
                    "turbulence": turbulence,
                    "snr": snr,
                    "accuracy": float(
                        np.mean(
                            [
                                record["label"] == record["prediction"]
                                for record in selected
                            ]
                        )
                    ),
                    "samples": len(selected),
                }
            )
    return rows


def plot_snr_curves(output_dir: Path, rows) -> None:
    colors = {"weak": "#2ca02c", "moderate": "#ff7f0e", "strong": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for turbulence in TURBULENCE_LEVELS:
        selected = [row for row in rows if row["turbulence"] == turbulence]
        ax.plot(
            [row["snr"] for row in selected],
            [100 * row["accuracy"] for row in selected],
            marker="o",
            linewidth=2,
            color=colors[turbulence],
            label=turbulence.capitalize(),
        )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Classification accuracy (%)",
        title="Six-class OFDM AMC under atmospheric turbulence",
        ylim=(0, 102),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_snr_all_turbulence.png", dpi=220)
    plt.close(fig)

    for turbulence in TURBULENCE_LEVELS:
        selected = [row for row in rows if row["turbulence"] == turbulence]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            [row["snr"] for row in selected],
            [100 * row["accuracy"] for row in selected],
            marker="o",
            linewidth=2,
            color=colors[turbulence],
        )
        alpha, beta = GAMMA_GAMMA[turbulence]
        ax.set(
            xlabel="SNR (dB)",
            ylabel="Classification accuracy (%)",
            title=(
                f"{turbulence.capitalize()} turbulence "
                f"(Gamma-Gamma alpha={alpha}, beta={beta})"
            ),
            ylim=(0, 102),
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"accuracy_vs_snr_{turbulence}.png", dpi=220)
        plt.close(fig)


def plot_confusions(output_dir: Path, records, target_snr: float) -> None:
    available_snrs = sorted({float(record["snr"]) for record in records})
    if not available_snrs:
        return
    resolved_snr = min(available_snrs, key=lambda value: abs(value - target_snr))
    for turbulence in TURBULENCE_LEVELS:
        selected = [
            record
            for record in records
            if record["turbulence"] == turbulence
            and np.isclose(record["snr"], resolved_snr)
        ]
        if not selected:
            continue
        matrix = confusion_matrix(
            [record["label"] for record in selected],
            [record["prediction"] for record in selected],
            labels=range(len(MODULATIONS)),
            normalize="true",
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(
            matrix, display_labels=MODULATIONS
        ).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(
            f"{turbulence.capitalize()} turbulence, SNR={resolved_snr:g} dB"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"confusion_matrix_{turbulence}_snr_{resolved_snr:g}dB.png",
            dpi=220,
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    protected = args.output_dir / "metrics.json"
    if protected.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"{protected} already exists. Use a new output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    config = SimulationConfig(
        sequence_length=args.sequence_length,
        n_subcarriers=args.n_subcarriers,
        cp_length=args.cp_length,
    )

    train_dataset = OFDMTurbulenceDataset(
        args.snrs,
        list(TRAINING_TURBULENCE_LEVELS),
        args.train_examples,
        args.seed + 1000,
        config,
        cache=False,
        dynamic_per_epoch=args.train_dynamic,
    )
    val_dataset = OFDMTurbulenceDataset(
        args.snrs,
        list(TURBULENCE_LEVELS),
        args.val_examples,
        args.seed + 2000,
        config,
        cache=True,
    )
    test_dataset = OFDMTurbulenceDataset(
        args.snrs,
        list(TURBULENCE_LEVELS),
        args.test_examples,
        args.seed + 3000,
        config,
        cache=True,
    )
    train_dataset = wrap_representation(train_dataset, args)
    val_dataset = wrap_representation(val_dataset, args)
    test_dataset = wrap_representation(test_dataset, args)
    train_loader = make_loader(train_dataset, args, True)
    val_loader = make_loader(val_dataset, args, False)
    test_loader = make_loader(test_dataset, args, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = args.output_dir / "best_model.pt"
    history = []
    best_acc = -1.0
    best_epoch = 0
    stale = 0

    global_epoch = 0
    curriculum = build_curriculum(args)
    for stage in curriculum:
        stage_dataset = OFDMTurbulenceDataset(
            stage["snrs"],
            stage["turbulence_levels"],
            args.train_examples,
            args.seed + 500 + global_epoch * 17,
            config,
            cache=False,
            dynamic_per_epoch=args.train_dynamic,
        )
        stage_dataset = wrap_representation(stage_dataset, args)
        stage_loader = make_loader(stage_dataset, args, True)
        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            if hasattr(stage_loader.dataset, "base") and hasattr(stage_loader.dataset.base, "set_epoch"):
                stage_loader.dataset.base.set_epoch(global_epoch)
            elif hasattr(stage_loader.dataset, "set_epoch"):
                stage_loader.dataset.set_epoch(global_epoch)
            train_loss, train_acc = run_epoch(
                model, stage_loader, criterion, device, optimizer, scaler
            )
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_acc)
            history.append(
                {
                    "stage": stage["name"],
                    "epoch": global_epoch,
                    "stage_epoch": stage_epoch,
                    "stage_snrs": ",".join(str(value) for value in stage["snrs"]),
                    "stage_turbulence": ",".join(stage["turbulence_levels"]),
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"{stage['name']} {stage_epoch:03d} | "
                f"train={train_loss:.4f}/{train_acc:.4f} "
                f"| val={val_loss:.4f}/{val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = global_epoch
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_validation_accuracy": best_acc,
                        "best_epoch": best_epoch,
                        "modulations": MODULATIONS,
                        "simulation_config": config.__dict__,
                        "gamma_gamma": GAMMA_GAMMA,
                        "args": vars(args),
                        "curriculum": curriculum,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            if stale >= args.patience:
                break
        if stale >= args.patience:
            break

    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    records = evaluate(model, test_loader, device)
    save_predictions(args.output_dir / "test_predictions.csv", records)
    rows = summarize(records)
    with (args.output_dir / "accuracy_vs_snr.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_snr_curves(args.output_dir, rows)
    plot_confusions(args.output_dir, records, args.confusion_snr)

    metrics = {
        "task": "six-class time-domain OFDM AMC under Gamma-Gamma turbulence",
        "modulations": MODULATIONS,
        "snrs_db": args.snrs,
        "turbulence_levels": TURBULENCE_LEVELS,
        "gamma_gamma_parameters": GAMMA_GAMMA,
        "sequence_length": args.sequence_length,
        "representation": args.representation,
        "gaf_size": args.gaf_size if "gaf" in args.representation else None,
        "n_subcarriers": args.n_subcarriers,
        "cp_length": args.cp_length,
        "curriculum": curriculum,
        "train_examples_per_condition": args.train_examples,
        "val_examples_per_condition": args.val_examples,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_acc,
        "test_examples_per_condition": args.test_examples,
        "accuracy_vs_snr": rows,
        "evaluation_unit": "independent randomly generated time-domain IQ sequence",
        "frame_level_accuracy_used": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
