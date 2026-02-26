#!/usr/bin/env python3
"""Main training script for Dhwani FER model."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.emotion_model import build_emotion_model, print_model_summary
from utils.dataloader import get_dataloaders
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.trainer import train

# Logging configuration
LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w"),
    ],
)
logger = logging.getLogger("dhwani.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dhwani FER Model Training Pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, default="Dataset",
        help="Root directory of the dataset (must contain train/ and val/).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Initial learning rate (default: 0.0003).",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay for Adam (default: 1e-4).",
    )
    parser.add_argument(
        "--early_stopping", type=int, default=5,
        help="Early-stopping patience (0 = disabled).",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Max gradient norm for clipping (default: 1.0).",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", default=True,
        help="Freeze backbone layers (default: True).",
    )
    parser.add_argument(
        "--no_freeze_backbone", dest="freeze_backbone", action="store_false",
        help="Train all layers without freezing.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="outputs/checkpoints",
        help="Directory for saving model checkpoints.",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="outputs/plots",
        help="Directory for saving training plots.",
    )
    return parser.parse_args()


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
) -> None:
    """Plot training and validation loss curves."""
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs_range, val_losses, "s-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Loss curve saved → %s", save_path)


def plot_accuracy_curve(
    val_accuracies: list[float],
    save_path: str,
) -> None:
    epochs_range = range(1, len(val_accuracies) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, val_accuracies, "D-", color="green", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Accuracy curve saved → %s", save_path)


def main() -> None:
    args = parse_args()

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    # Device detection: CUDA > MPS (Apple Metal) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS (Metal) backend detected.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected — training on CPU.")
    logger.info("Using device: %s", device)

    logger.info("Loading dataset from: %s", args.data_dir)
    train_loader, val_loader, train_ds, val_ds = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Dynamically detect emotion classes from dataset folder structure
    class_names = train_ds.classes
    num_classes = len(class_names)
    logger.info("Detected %d emotion classes: %s", num_classes, class_names)

    model = build_emotion_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)
    print_model_summary(model)

    # Loss function with label smoothing for regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("Mixed-precision training (AMP) enabled.")

    history, final_labels, final_preds = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=use_amp,
        early_stopping_patience=args.early_stopping,
        max_grad_norm=args.grad_clip,
    )

    logger.info("Computing final evaluation metrics …")
    metrics = compute_metrics(
        final_labels, final_preds, class_names=list(class_names)
    )
    logger.info(
        "Final  —  Acc: %.4f  |  Prec: %.4f  |  Rec: %.4f  |  F1: %.4f",
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )

    plot_loss_curves(
        history.train_loss,
        history.val_loss,
        save_path=os.path.join(args.plots_dir, "loss_curve.png"),
    )
    plot_accuracy_curve(
        history.val_accuracy,
        save_path=os.path.join(args.plots_dir, "accuracy_curve.png"),
    )
    plot_confusion_matrix(
        final_labels,
        final_preds,
        class_names=list(class_names),
        save_path=os.path.join(args.plots_dir, "confusion_matrix.png"),
    )

    logger.info("All outputs saved under: outputs/")
    logger.info("Training pipeline finished successfully. ✓")


if __name__ == "__main__":
    main()
