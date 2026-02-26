import logging
import os
import platform
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Training augmentation: resize, random crop, flip, rotation, color jitter."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Validation transforms: resize, center crop, no augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _optimal_workers() -> int:
    """Determine optimal DataLoader workers based on CPU count and OS."""
    cpu_count = os.cpu_count() or 1
    # MacOS performs better with fewer workers
    if platform.system() == "Darwin":
        return min(2, cpu_count)
    return min(4, cpu_count)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    # Validate dataset structure
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Load datasets using ImageFolder (auto-detects classes from folder names)
    train_dataset = datasets.ImageFolder(
        root=str(train_dir), transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir), transform=get_val_transforms()
    )

    logger.info("Class-to-index mapping: %s", train_dataset.class_to_idx)
    logger.info("Training samples  : %d", len(train_dataset))
    logger.info("Validation samples: %d", len(val_dataset))

    if num_workers is None:
        num_workers = _optimal_workers()
    logger.info("DataLoader num_workers: %d", num_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset
