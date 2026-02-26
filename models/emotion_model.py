import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def build_emotion_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 2,
) -> nn.Module:
    # Load EfficientNet-B0 with ImageNet weights for transfer learning
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    logger.info(
        "Loaded EfficientNet-B0 (pretrained=%s)", pretrained
    )

    if freeze_backbone:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze last N feature blocks for fine-tuning
        feature_blocks = list(model.features.children())
        total_blocks = len(feature_blocks)
        unfreeze_from = max(0, total_blocks - unfreeze_last_n_blocks)

        for block in feature_blocks[unfreeze_from:]:
            for param in block.parameters():
                param.requires_grad = True

        logger.info(
            "Backbone frozen. Unfroze last %d / %d feature blocks.",
            min(unfreeze_last_n_blocks, total_blocks),
            total_blocks,
        )

    # Replace classifier head: 1280 → 256 → num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    logger.info(
        "Classifier head replaced: %d → 256 → %d", in_features, num_classes
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and frozen parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def print_model_summary(model: nn.Module) -> None:
    stats = count_parameters(model)
    logger.info("=" * 55)
    logger.info("MODEL SUMMARY")
    logger.info("-" * 55)
    logger.info("Total parameters      : {:>12,}".format(stats["total"]))
    logger.info("Trainable parameters  : {:>12,}".format(stats["trainable"]))
    logger.info("Frozen parameters     : {:>12,}".format(stats["frozen"]))
    logger.info("=" * 55)
