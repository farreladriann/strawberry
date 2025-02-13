# This is my roboflow dataset for object detection

# there are 3 clasess to detect
# 1. strawberry leaf (5,614 annotation)
# 2. strawberry fruit (6580 annotation)
# 3. strawberry flower (370 annotation)

# All the image already annotate.

# I want to train this roboflow dataset with yolo11x model, but I still confuse to configurate the hyperparameter of the training using ultralytics YOLO.

# GPU: NVIDIA GeForce RTX 3060
# VRAM: 12.62 GB

import torch
from ultralytics import YOLO
import os

def setup_training_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Periksa instalasi GPU kamu.")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = True

def train_strawberry_leaves_model():
    setup_training_environment()

    # Optimized training configuration for Strawberry-Leaves2 dataset
    config = {
        'data': '/data/strawberry-2/data.yaml',
        'epochs': 150,
        'imgsz': 640,
        'batch': 32,
        'device': 0,
        'optimizer': 'SGD',
        'lr0': 1e-3,
        'lrf': 0.1,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'patience': 30,
        'cache': False,
        'pretrained': True,
        'project': 'strawberry_tuned',
        'name': 'strawberry_tuned',
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.3,
        'shear': 3.0,
        'perspective': 0.0001,
        'flipud': 0.3,
        'fliplr': 0.3,
        'workers': 8,
        'plots': True,
        'hsv_h': 0.015,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'save_period': 30,
    }

    model = YOLO('yolo11m.pt')  # Use YOLOv8n for faster training with limited data
    print("\n=== Mulai Training Model ===")
    results = model.train(**config)

    # Save the trained model
    model.save('strawberry_tuned.pt')

def validate_model():
    # Load the trained modelso what is the best configuration
    model = YOLO('strawberry_tuned.pt')
    print("\n=== Mulai Validasi Model ===")

    # Perform validation on the test split
    metrics = model.val(data='/data/strawberry-2/data.yaml', split='test')

    # Display validation metrics
    print(f"Validasi: mAP50-95: {metrics.box.map:.2f} | mAP50: {metrics.box.map50:.2f}")

if __name__ == "__main__":
    try:
        # Display GPU information
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.2f} GB")

        # Run training and validation
        train_strawberry_leaves_model()
        validate_model()

    except Exception as e:
        print(f"Error: {str(e)}")