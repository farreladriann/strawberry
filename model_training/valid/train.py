# This is my roboflow dataset for object detection

# there are 3 clasess to detect
# 1. strawberry leaf (2095 annotation)
# 2. strawberry fruit (4459 annotation)
# 3. strawberry flower (271 annotation)

# All the image already annotate.

# I want to train this roboflow dataset with yolo11x model, but I still confuse to configurate the hyperparameter of the training using ultralytics YOLO.

# I use Nvidia A100 GPU, so you don't fear of the resource, and I need high accuration.

# Specification	A100 40GB PCIe
# GPU Memory	40GB HBM2
# GPU Memory Bandwidth	1,555GB/s

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
        'data': '/content/strawberry-1/data.yaml',  # Path to your dataset YAML file
        'epochs': 150,  # Number of training epochs
        'imgsz': 640,  # Image size
        'batch': 32,  # Batch size
        'device': 0,  # GPU device (0 for default GPU)
        'optimizer': 'AdamW',  # Optimizer (AdamW is generally preferred)
        'lr0': 1e-4,  # Initial learning rate
        'lrf': 0.05,  # Final OneCycleLR learning rate (lr0 * lrf)
        'weight_decay': 0.0005,  # Weight decay for regularization
        'patience': 30,  # Early stopping patience
        'cache': True, # Cache images for faster training
        'pretrained': True, # Use pre-trained weights
        'project': 'strawberry_part_detection',  # Save to project/name
        'name': 'strawberry_part_detection',  # Save to project/name
        'close_mosaic': 10, # Force close mosaic augmentation
        'degrees': 10.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 2.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.5,  # Image flip up-down (probability)
        'fliplr': 0.5,  # Image flip left-right (probability)
        'mosaic': 1.0, # Mosaic augmentation (probability)
        'mixup': 0.2, # MixUp augmentation (probability)
        'copy_paste': 0.2, # Copy-Paste augmentation (probability)
        'workers': 16,  # Number of data loading workers
        'plots': True,  # Visualize training results
        'pretrained': True,  # Use pre-trained weights
        'save_period': 30,  # Save model every 10 epochs
    }

    # Initialize the model (YOLOv8x is a large variant; consider YOLOv8n for smaller datasets)
    model = YOLO('yolov8x.pt')  # Use YOLOv8n for faster training with limited data
    print("\n=== Mulai Training Model ===")
    results = model.train(**config)

    # Save the trained model
    model.save('strawberry_part_detection.pt')

def validate_model():
    # Load the trained modelso what is the best configuration
    model = YOLO('strawberry_part_detection.pt')
    print("\n=== Mulai Validasi Model ===")

    # Perform validation on the test split
    metrics = model.val(data='/content/Strawberry-Leaves2-2/data.yaml', split='test')

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