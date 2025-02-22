from ultralytics import YOLO
import cv2
import os

def predict_disease(image_path, final_result_path):
    # Load model pertama (untuk deteksi ROI)
    model1 = YOLO('strawberry_tuned.pt')
    # Load model kedua (untuk deteksi pada crop)
    model2 = YOLO('best_model.pt')
    
    # Atur confidence threshold untuk masing-masing model
    conf_thresh1 = 0.5  # untuk model1
    conf_thresh2 = 0.5  # untuk model2
    
    # Baca gambar asli
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Gagal memuat gambar: {image_path}")
        return
    
    # Lakukan inferensi dengan model1
    results1 = model1(original_image)
    
    # Proses setiap hasil deteksi dari model1
    for result in results1:
        boxes = result.boxes  # Bounding boxes dari model1
        
        # Filter kotak berdasarkan confidence threshold model1
        filtered_boxes = [box for box in boxes if box.conf[0] >= conf_thresh1]
        
        # Untuk setiap ROI dari model1, crop dan lakukan deteksi dengan model2
        for box in filtered_boxes:
            # Ekstrak koordinat ROI (asumsi format xyxy)
            coords = box.xyxy[0]  # Tensor dengan [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords.cpu().numpy())
            
            # Pastikan koordinat berada di dalam batas gambar asli
            h, w, _ = original_image.shape
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Crop ROI secara in-memory (tanpa menyimpan ke disk)
            crop_img = original_image[y1:y2, x1:x2].copy()
            
            # Lakukan inferensi pada crop dengan model2
            results2 = model2(crop_img)
            
            # Proses hasil deteksi dari model2 dan anotasi crop (jika ada deteksi)
            for res2 in results2:
                boxes2 = res2.boxes
                filtered_boxes2 = [b for b in boxes2 if b.conf[0] >= conf_thresh2]
                if filtered_boxes2:
                    # Gunakan method plot() untuk mendapatkan crop dengan anotasi dari model2
                    annotated_crop = res2.plot()
                    
                    # Resize crop yang sudah dianotasi agar sesuai dengan ukuran ROI di gambar asli
                    annotated_crop_resized = cv2.resize(annotated_crop, (x2 - x1, y2 - y1))
                    
                    # Gantikan ROI pada gambar asli dengan crop yang sudah dianotasi
                    original_image[y1:y2, x1:x2] = annotated_crop_resized
                    
                    # (Opsional) Cetak hasil prediksi dari model2 untuk setiap crop
                    for b in filtered_boxes2:
                        conf = b.conf[0]
                        class_id = int(b.cls[0])
                        print(f"Deteksi pada ROI: {model2.names[class_id]}, Confidence: {conf:.2f}")
    
    # Simpan gambar akhir yang sudah dianotasi ke disk
    cv2.imwrite(final_result_path, original_image)
    print(f"Hasil akhir disimpan di: {final_result_path}")

if __name__ == "__main__":
    # Path gambar input dan output
    input_image_path = "./stest/Angular Leaf Spot/1573030.jpg"
    ext = os.path.splitext(input_image_path)[1]
    basename = os.path.basename(input_image_path)
    final_result_path = f"./final_{basename}"  # Hanya gambar akhir yang disimpan
    
    predict_disease(input_image_path, final_result_path)
