# Model Training

This directory contains model training pipelines for strawberry disease detection.

## Directory Structure

```
model_training/
├── data_integration_cleaning/    # Dataset preparation & combining
├── data_augmentation/           # Image augmentation tools
└── dataset_prediction_train/    # Training & prediction scripts
```

## Components

### Data Integration (`data_integration_cleaning/`)
- Combines multiple YOLO datasets
- Handles class mapping and normalization
- Analyzes dataset distributions

### Data Augmentation (`data_augmentation/`) 
- HSV color adjustments
- Lighting simulation (indoor farming)
- Advanced augmentation suite

### Training & Prediction (`dataset_prediction_train/`)
- YOLOv11 training configurations
- A100 GPU optimized settings
- Prediction pipeline for INASTEK dataset

## Requirements

See `requirements.txt` in root directory.