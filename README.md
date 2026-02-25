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
| **Model** | EfficientNet-B0 (ImageNet-pretrained, transfer learning) |
| **Task** | Multi-class emotion classification |
| **Classes** | Dynamic detection (7 FER classes: angry, disgust, fear, happy, neutral, sad, surprise) |
| **Framework** | PyTorch 2.8.0+ + Torchvision |
| **Python** | 3.9+ |
| **Metrics** | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| **Best Model** | Selected by validation accuracy (highest) |

---

## Project Structure

```
Dhwani_Emotion_Training/
│
├── processed_data/              # Raw dataset source (organize by emotion class)
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── Dataset/                     # Created after running split_dataset.py
│   ├── train/                   # 80% of images per class
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── val/                     # 20% of images per class
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── models/
│   ├── __init__.py
│   └── emotion_model.py         # EfficientNet-B0 architecture (1280→256→num_classes)
│
├── utils/
│   ├── __init__.py
│   ├── dataloader.py            # Data transforms & DataLoaders (with ImageNet normalization)
│   ├── metrics.py               # Evaluation metrics & confusion matrix plots
│   └── trainer.py               # Training loop, validation, early stopping, checkpointing
│
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   │   ├── best_model.pth       # Best model (highest validation accuracy)
│   │   └── last_model.pth       # Final epoch model
│   └── plots/                   # Training visualizations
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       └── confusion_matrix.png
│
├── train.py                     # Main entry point with CLI arguments
├── split_dataset.py             # Dataset splitting script (80/20 deterministic split)
├── requirements.txt             # Python dependencies
├── training.log                 # Training logs generated during run
├── README.md                    # This file
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

Organize your facial emotion images into `processed_data/` with one folder per emotion:

```
processed_data/
├── angry/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

### 4. Split the dataset

Run the dataset splitting script to create `Dataset/train/` and `Dataset/val/` with an 80/20 deterministic split:

```bash
python split_dataset.py
```

Output:
```
====================================================================
  Source directory : /path/to/processed_data
  Output directory : /path/to/Dataset
  Train ratio      : 80%
  Val ratio        : 20%
  Random seed      : 42
  Classes detected : 7 -> ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
====================================================================

  angry       |  total:  1000  |  train:   800  |  val:   200
  disgust     |  total:   900  |  train:   720  |  val:   180
  ...
====================================================================
  Total train images : 5600
  Total val images   : 1400
  Detected classes   : 7 -> ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
====================================================================

[SUCCESS] Dataset split completed successfully. Ready for training.
```

---

## Training

### Quick Start

```bash
# Default configuration (10 epochs, batch_size=32, lr=0.0003)
python train.py
```

### Custom Training

```bash
python train.py --epochs 20 --batch_size 64 --lr 2e-4 --early_stopping 7
```

### Disable Features

```bash
# Train all layers without freezing backbone
python train.py --no_freeze_backbone

# Disable early stopping
python train.py --early_stopping 0
```

### Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `Dataset` | Root dataset directory (must contain `train/` and `val/`) |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `0.0003` | Initial learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation (Adam) |
| `--early_stopping` | `5` | Early stopping patience (0 = disabled) |
| `--grad_clip` | `1.0` | Max gradient norm for clipping |
| `--freeze_backbone` | `True` | Freeze EfficientNet backbone layers |
| `--no_freeze_backbone` | — | Train all layers without freezing |
| `--checkpoint_dir` | `outputs/checkpoints` | Directory for model checkpoints |
| `--plots_dir` | `outputs/plots` | Directory for training plots |

---

## Training Configuration

### Model Architecture
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Custom Head**: 1280 → 256 (ReLU) → num_classes (Softmax)
- **Unfrozen Layers**: Last 2 feature blocks + classifier head (transfer learning)

### Hyperparameters
- **Optimizer**: Adam (lr=3e-4, weight_decay=1e-4)
- **Loss Function**: CrossEntropyLoss (label_smoothing=0.1)
- **Scheduler**: ReduceLROnPlateau (mode="min", factor=0.5, patience=2)
- **Gradient Clipping**: max_norm=1.0
- **Mixed Precision**: Auto-enabled on CUDA (AMP)
- **Best Model**: Selected by **validation accuracy** (highest)
- **Early Stopping**: Monitors validation accuracy with patience=5

### Data Augmentation (Train Only)
- Resize → 256px
- RandomCrop → 224×224px
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±15°)
- ColorJitter (brightness, contrast, saturation ±0.2, hue ±0.1)
- RandomGrayscale (p=0.05)
- ImageNet normalization

### Validation Transforms
- Resize → 256px
- CenterCrop → 224×224px
- ImageNet normalization (no augmentation)

---

## Expected Outputs

After training completes:

| File | Description |
|---|---|
| `outputs/checkpoints/best_model.pth` | Best model by validation accuracy |
| `outputs/checkpoints/last_model.pth` | Final epoch model |
| `outputs/plots/loss_curve.png` | Training & validation loss over epochs |
| `outputs/plots/accuracy_curve.png` | Validation accuracy over epochs |
| `outputs/plots/confusion_matrix.png` | Per-class confusion matrix (counts + normalized) |
| `training.log` | Full training logs with timestamps |

### Console Output Example

```
Epoch [01/10]  Train Loss: 1.2345  |  Val Loss: 0.9876  |  Val Acc: 0.6500  |  LR: 0.000300  |  Time: 23.4s
Epoch [02/10]  Train Loss: 0.9876  |  Val Loss: 0.8765  |  Val Acc: 0.7200  |  LR: 0.000300  |  Time: 22.8s
...
Detected 7 emotion classes: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

---

## Features

✅ **Dynamic Class Detection** – Automatically detects emotion classes from dataset structure  
✅ **Modular Design** – Separate concerns: model, data, training, metrics  
✅ **EfficientNet Transfer Learning** – Leverages ImageNet pretraining for efficiency  
✅ **Comprehensive Metrics** – Accuracy, Precision, Recall, F1, Confusion Matrix  
✅ **Early Stopping** – Prevents overfitting with configurable patience  
✅ **Checkpoint Management** – Best and last model snapshots  
✅ **Mixed Precision Training** – AMP on CUDA for faster training and lower memory  
✅ **Gradient Clipping** – Prevents exploding gradients  
✅ **Learning Rate Scheduling** – Adaptive LR reduction on validation loss plateau  
✅ **Device Flexibility** – Auto-detection: CUDA > Apple Metal (MPS) > CPU  
✅ **Deterministic Splitting** – 80/20 reproducible split with seed=42  
✅ **Comprehensive Logging** – Timestamped training history  
✅ **Production-Grade Code** – Clean code, type hints, Python 3.9+ compatible  

---

## What's NOT Included

❌ Spotify API integration  
❌ Song recommendation logic  
❌ Real-time inference / webcam  
❌ Web UI / Streamlit dashboard  
❌ Deployment configurations  
❌ Ensemble methods  

This is a **training-focused implementation** for emotion model development only.

---

## Technical Requirements

- **Python**: 3.9 or higher
- **PyTorch**: 2.0.0 or higher (tested with 2.8.0)
- **Torchvision**: 0.15.0 or higher
- **scikit-learn**: 1.3.0 or higher
- **matplotlib**: 3.7.0 or higher
- **tqdm**: 4.65.0 or higher
- **numpy**: 1.24.0 or higher
- **Pillow**: 9.5.0 or higher

All dependencies listed in `requirements.txt`.

---

## Workflow Summary

1. **Setup**: Create venv, install requirements
2. **Prepare Data**: Organize images in `processed_data/`
3. **Split Dataset**: Run `python split_dataset.py` → creates `Dataset/train/` and `Dataset/val/`
4. **Train Model**: Run `python train.py` → trains and saves best model
5. **Review Results**: Check logs, confusion matrix, and loss curves in `outputs/`

---

## License

This code is part of an academic project and is provided for educational purposes only.
