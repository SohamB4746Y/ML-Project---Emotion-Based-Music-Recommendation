import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Store training metrics across epochs."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


class EarlyStopping:
    """Stop training if metric doesn't improve after patience epochs."""
    def __init__(
        self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss, "max" for accuracy
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(loader, desc="  Train", leave=False, ncols=100)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use mixed precision if scaler provided (CUDA only)
        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Validate model. Returns (avg_loss, accuracy, true_labels, predictions)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    progress = tqdm(loader, desc="  Val  ", leave=False, ncols=100)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_labels, all_preds


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    path: str,
) -> None:
    """Save model checkpoint with training state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    epochs: int = 10,
    checkpoint_dir: str = "outputs/checkpoints",
    use_amp: bool = False,
    early_stopping_patience: int = 5,
    max_grad_norm: float = 1.0,
) -> Tuple[TrainingHistory, List[int], List[int]]:
    history = TrainingHistory()
    early_stop = (
        EarlyStopping(patience=early_stopping_patience, mode="max")
        if early_stopping_patience > 0
        else None
    )
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None
    best_val_acc = 0.0

    logger.info("=" * 60)
    logger.info("TRAINING STARTED  |  Epochs: %d  |  Device: %s", epochs, device)
    logger.info("=" * 60)

    final_labels: List[int] = []
    final_preds: List[int] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, max_grad_norm=max_grad_norm,
        )

        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)
        history.learning_rates.append(current_lr)

        final_labels = val_labels
        final_preds = val_preds

        logger.info(
            "Epoch [%02d/%02d]  "
            "Train Loss: %.4f  |  Val Loss: %.4f  |  Val Acc: %.4f  |  "
            "LR: %.6f  |  Time: %.1fs",
            epoch, epochs, train_loss, val_loss, val_acc, current_lr, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                path=f"{checkpoint_dir}/best_model.pth",
            )

        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            path=f"{checkpoint_dir}/last_model.pth",
        )

        if early_stop is not None and early_stop(val_acc):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE  |  Best Val Acc: %.4f", best_val_acc)
    logger.info("=" * 60)

    return history, final_labels, final_preds
