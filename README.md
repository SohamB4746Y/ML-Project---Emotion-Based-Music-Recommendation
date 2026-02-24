# Dhwani – Mood & Emotion-Based Song Recommender
## Partial Implementation: Facial Emotion Recognition (FER) Model Training

> **Scope Notice**
> This repository contains **only** the model-training pipeline for the Facial
> Emotion Recognition component of the larger *Dhwani* project. It does **not**
> include the Spotify API integration, Streamlit UI, real-time webcam inference,
> music recommendation engine, lyrics classification, or backend APIs.

---

## Overview

| Item | Detail |
|---|---|
| **Model** | EfficientNet-B0 (ImageNet-pretrained, fine-tuned) |
| **Task** | 4-class emotion classification |
| **Classes** | `calm`, `energetic`, `happy`, `sad` |
| **Framework** | PyTorch + Torchvision |
| **Metrics** | Accuracy, Precision, Recall, F1-score, Confusion Matrix |

---

## Project Structure

```
Dhwani_Emotion_Training/
│
├── Dataset/                    # Place your images here
│   ├── train/
│   │   ├── calm/
│   │   ├── energetic/
│   │   ├── happy/
│   │   └── sad/
│   └── val/
│       ├── calm/
│       ├── energetic/
│       ├── happy/
│       └── sad/
│
├── models/
│   └── emotion_model.py        # EfficientNet-B0 architecture
│
├── utils/
│   ├── dataloader.py           # Transforms & DataLoaders
│   ├── metrics.py              # Evaluation metrics & confusion matrix
│   └── trainer.py              # Training loop, validation, early stopping
│
├── outputs/
│   ├── checkpoints/            # Saved model weights
│   └── plots/                  # Loss, accuracy, confusion matrix plots
│
├── train.py                    # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place facial-emotion images into the `Dataset/` directory following the
structure shown above. Each sub-folder name is treated as the class label
(alphabetical order: calm → 0, energetic → 1, happy → 2, sad → 3).

---

## Training

```bash
# Default configuration (10 epochs, batch_size=32, lr=0.0003)
python train.py

# Custom training
python train.py --epochs 20 --batch_size 64 --lr 0.0001

# Disable early stopping
python train.py --early_stopping 0

# Train all layers (no backbone freezing)
python train.py --no_freeze_backbone
```

### Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `Dataset` | Root dataset directory |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.0003` | Initial learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--early_stopping` | `5` | Patience (0 = disabled) |
| `--grad_clip` | `1.0` | Max gradient norm |
| `--freeze_backbone` | `True` | Freeze EfficientNet backbone |
| `--checkpoint_dir` | `outputs/checkpoints` | Checkpoint save path |
| `--plots_dir` | `outputs/plots` | Plot save path |

---

## Expected Outputs

After training completes you will find:

| File | Description |
|---|---|
| `outputs/checkpoints/best_model.pth` | Weights of the best-performing epoch |
| `outputs/checkpoints/last_model.pth` | Weights of the final epoch |
| `outputs/plots/loss_curve.png` | Training & validation loss over epochs |
| `outputs/plots/accuracy_curve.png` | Validation accuracy over epochs |
| `outputs/plots/confusion_matrix.png` | Per-class confusion matrix |
| `training.log` | Full training log |

Console output per epoch:

```
Epoch [01/10]  Train Loss: 1.2345  |  Val Loss: 0.9876  |  Val Acc: 0.6500  |  LR: 0.000300  |  Time: 23.4s
```

---

## What to Show During Academic Review

1. **Code Architecture** – modular design with clear separation of concerns.
2. **Model Choice** – EfficientNet-B0 with transfer learning justification.
3. **Training Curves** – `loss_curve.png` and `accuracy_curve.png`.
4. **Classification Report** – precision, recall, F1 per class (printed in logs).
5. **Confusion Matrix** – `confusion_matrix.png`.
6. **Checkpoint** – load `best_model.pth` and demonstrate inference if needed.
7. **Techniques Used** – data augmentation, gradient clipping, LR scheduling,
   early stopping, mixed-precision training (CUDA), label smoothing.

---

## Technical Highlights

- **Transfer Learning** – ImageNet-pretrained EfficientNet-B0 with selective
  layer unfreezing for efficient domain adaptation.
- **Data Augmentation** – RandomCrop, HorizontalFlip, Rotation, ColorJitter,
  RandomGrayscale.
- **Regularisation** – Dropout, weight decay, label smoothing (0.1).
- **Mixed Precision** – automatic `torch.cuda.amp` when running on CUDA.
- **Gradient Clipping** – prevents exploding gradients (`max_norm=1.0`).
- **Learning Rate Scheduling** – `ReduceLROnPlateau` halves LR after 2
  epochs of stagnation.
- **Early Stopping** – avoids overfitting by monitoring validation loss.

---

## License

This code is part of an academic project and is provided for educational
purposes only.
