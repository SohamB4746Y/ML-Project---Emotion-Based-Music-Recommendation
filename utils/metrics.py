import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute classification metrics: accuracy, precision, recall, F1."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    logger.info("\n%s", report)

    return metrics


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    save_path: str = "outputs/plots/confusion_matrix.png",
) -> None:
    """Generate and save confusion matrix plots (counts + normalized)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalised = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Counts)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    _annotate_matrix(axes[0], cm, fmt="d")
    _set_axis_labels(axes[0], class_names)

    im1 = axes[1].imshow(cm_normalised, interpolation="nearest", cmap=plt.cm.Blues,
                          vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalised)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    _annotate_matrix(axes[1], cm_normalised, fmt=".2f")
    _set_axis_labels(axes[1], class_names)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def _annotate_matrix(ax, matrix: np.ndarray, fmt: str = "d") -> None:
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            colour = "white" if value > thresh else "black"
            ax.text(
                j, i, format(value, fmt),
                ha="center", va="center", color=colour, fontsize=11,
            )


def _set_axis_labels(ax, class_names: List[str]) -> None:
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
