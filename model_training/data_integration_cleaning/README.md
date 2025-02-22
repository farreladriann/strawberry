# Data Integration and Cleaning

This directory contains scripts for data integration, cleaning and analysis of the strawberry disease dataset.

## Files Overview

### `object_detection.ipynb`
Jupyter notebook for processing object detection dataset:
- Analyzes YOLO format label files
- Combines multiple datasets while handling class mapping
- Processes data.yaml files for class names and paths
- Key functions:
  - `analyze_dataset_labels_for_classification()`: Analyzes label distribution
  - `combine_dataset()`: Merges two YOLO datasets
  - `dictionary_yaml_old_to_new()`: Maps class IDs between datasets
  - `normalize_name()`: Standardizes class names

### `klasifikasi.ipynb` 
Jupyter notebook for classification dataset analysis:
- Analyzes image classification dataset structure
- Generates statistics and visualizations
- Main functionality:
  - Dataset folder structure analysis (train/valid/test)
  - Class distribution calculations
  - Visualization of data distribution
  - Percentage calculations per split

## Dataset File Structure

### Object Detection Dataset
```
object_detection/
├── train/
│   ├── images/             # Training images (.jpg, .png)
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/            # YOLO format annotations (.txt)
│       ├── img1.txt      # Format: <class> <x> <y> <width> <height>
│       └── img2.txt
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml             # Dataset configuration file
    # Contains:
    # - path: dataset root
    # - train: train images path
    # - val: validation images path
    # - test: test images path
    # - nc: number of classes
    # - names: list of class names
```

### Classification Dataset
```
classification/
├── train/
│   ├── healthy/          # One folder per class
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── anthracnose/
│   └── leaf_spot/
├── valid/
│   ├── healthy/
│   ├── anthracnose/
│   └── leaf_spot/
└── test/
    ├── healthy/
    ├── anthracnose/
    └── leaf_spot/
```

## Usage

1. For object detection dataset:
```python
analyze_dataset_labels_for_classification('path/to/dataset', 'data.yaml')
```

2. For classification dataset:
```python
analyze_dataset('path/to/classification/dataset')
```

## Requirements
- Python 3.x
- pandas
- matplotlib
- PyYAML