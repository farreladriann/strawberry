import torch
from ultralytics import YOLO

def train_model():
    # Initialize YOLO model
    model = YOLO('yolov8n.yaml')  # Load YOLOv8 nano model configuration
    
    # Training configuration
    config = {
        'data': 'data.yaml',           # Dataset configuration
        'epochs': 20,                 # Number of epochs
        'imgsz': 280,                 # Image size
        'batch': 32,                  # Batch size
        'device': 'cpu',  # GPU/CPU
        'workers': 8,                 # Number of workers
        'patience': 50,               # Early stopping patience
        'project': 'strawberry_disease',  # Project name
        'name': 'experiment12'         # Run name
    }
    
    # Train the model
    results = model.train(**config)
    
    # Save the trained model
    model.save('best_model2.pt')
    
    return results

if __name__ == "__main__":
    train_model()
