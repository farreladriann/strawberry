from roboflow import Roboflow
import torch
from ultralytics import YOLO
import os

# Setup Roboflow dataset
rf = Roboflow(api_key="pZl3oXtPnrV76f4g5qtS")
project = rf.workspace("farrelganteng").project("strawberry-skcti-ew21i")
version = project.version(2)
dataset = version.download("yolov11")

def setup_training_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Periksa instalasi GPU kamu.")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = True

def train_strawberry_leaves_model():
    setup_training_environment()

    # Konfigurasi hyperparameter (disarankan sebagai starting point)
    config = {
        'data': '/data/strawberry-2/data.yaml',  # Pastikan path data.yaml sudah benar
        'epochs': 200,              # Tambah epoch untuk pelatihan yang lebih mendalam
        'imgsz': 640,
        'batch': 16,                # Batch size disesuaikan dengan kapasitas VRAM
        'device': 0,
        'optimizer': 'AdamW',       # Optimizer yang sering memberikan kestabilan saat fine-tuning
        'lr0': 5e-4,                # Learning rate awal yang lebih rendah untuk fine-tuning
        'lrf': 0.1,
        'weight_decay': 0.0005,
        'momentum': 0.937,          # Momentum yang umum dipakai (meski lebih relevan untuk SGD)
        'patience': 50,             # Kesabaran (patience) yang lebih tinggi untuk menghindari early stopping terlalu dini
        'cache': True,              # Aktifkan caching untuk mempercepat akses data
        'pretrained': True,
        'project': 'strawberry_tuned_best',
        'name': 'strawberry_tuned_best',
        # Augmentasi Geometris:
        'degrees': 10.0,            # Rotasi yang lebih kecil untuk menjaga kealamian objek
        'translate': 0.1,           # Translasi yang dikurangi
        'scale': 0.5,               # Variasi skala lebih besar agar model lebih robust terhadap perubahan ukuran
        'shear': 2.0,               # Pengurangan intensitas shear
        'perspective': 0.0001,      # Augmentasi perspektif minimal
        'flipud': 0.0,              # Tidak membalik vertikal (karena objek tidak mungkin terbalik secara natural)
        'fliplr': 0.5,              # Pembalikan horizontal dengan probabilitas lebih tinggi
        'workers': 8,
        'plots': True,
        # Augmentasi Warna (HSV):
        'hsv_h': 0.2,             # Intensitas perubahan hue yang ringan
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'save_period': 10,          # Simpan model secara berkala untuk monitoring
    }

    # Pastikan kamu menggunakan model yang sesuai (misalnya yolo11m.pt atau yolo11x.pt)
    model = YOLO('yolo11m.pt')
    print("\n=== Mulai Training Model ===")
    results = model.train(**config)
    
    # Simpan model terlatih
    model.save('strawberry_tuned_best.pt')

def validate_model():
    model = YOLO('strawberry_tuned_best.pt')
    print("\n=== Mulai Validasi Model ===")
    metrics = model.val(data='/data/strawberry-2/data.yaml', split='test')
    print(f"Validasi: mAP50-95: {metrics.box.map:.2f} | mAP50: {metrics.box.map50:.2f}")

if __name__ == "__main__":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.2f} GB")
        
        train_strawberry_leaves_model()
        validate_model()
    except Exception as e:
        print(f"Error: {str(e)}")
