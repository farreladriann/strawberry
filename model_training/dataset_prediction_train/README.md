# Dataset Prediction and Training

This directory contains scripts for training YOLO models and making predictions on strawberry disease datasets.

## Files Overview

### Training Scripts

#### `feb14train.py`
Advanced training script optimized for A100 GPU:
- Configured for 3 classes:
  1. Strawberry leaf (2,095 annotations)
  2. Strawberry fruit (4,459 annotations)
  3. Strawberry flower (271 annotations)
- Features:
  - Optimized hyperparameters for A100 40GB GPU
  - Advanced training configuration
  - Memory bandwidth utilization up to 1,555GB/s
- Key functions:
  - `setup_training_environment()`: Configures CUDA and training environment
  - `train_strawberry_leaves_model()`: Main training loop with optimized parameters
  - `validate_model()`: Model validation on test split

#### `train.py`
Basic training script for initial experimentation:
- Simpler configuration for YOLOv8n
- Suitable for CPU/smaller GPU setups
- Parameters:
  - 20 epochs
  - 280px image size
  - 32 batch size
  - Early stopping patience: 50

### Prediction Scripts

#### `predict_inastek_dataset.py`
Script for running predictions on INASTEK dataset:
- Supports two detection methods:
  1. Part-first: Detects plant parts then diseases
  2. Direct: Detects diseases directly
- Features:
  - Multiple model support
  - Automated output organization
  - JSON prediction output
- Model paths:
  - Leaf blight model
  - Best strawberry disease model

## Usage

### Training

```python
# For A100 GPU optimized training
python feb14train.py

# For basic training
python train.py
```

### Prediction

```python
# Run predictions on INASTEK dataset
python predict_inastek_dataset.py
```

## Model Paths

```python
partModelPaths = {
    "part": "../WebApp/models/part.pt",
    "part2": "../WebApp/models/part2.pt"
}

diseaseModelPaths = {
    "leafblight": "../WebApp/models/leafblight.pt",
    "best_strawberry_disease_model": "../WebApp/models/best_strawberry_disease_model.pt"
}
```

## Requirements
- Python 3.x
- PyTorch
- Ultralytics YOLO
- OpenCV
- CUDA capable GPU (for GPU training)