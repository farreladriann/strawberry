
import sys
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

modelPaths = {
    "strawberry_tuned": os.path.join(os.getcwd() ,"..", "WebApp", "models", "strawberry_tuned.pt"),
    "strawberry_part_detection": os.path.join(os.getcwd(), "..", "WebApp", "models", "strawberry_part_detection.pt"),
    "leafblight": os.path.join(os.getcwd(), "..", "WebApp", "models", "leafblight.pt"),
    "best_strawberry_disease_model": os.path.join(os.getcwd(), "..", "WebApp", "models", "best_strawberry_disease_model.pt"),
}

partModelPaths = {
    "strawberry_part_detection": os.path.join(os.getcwd(), "..", "WebApp", "models", "strawberry_part_detection.pt"),
    "strawberry_tuned": os.path.join(os.getcwd(), "..", "WebApp", "models", "strawberry_tuned.pt"),
}

diseaseModelPaths = {
    "leafblight": os.path.join(os.getcwd(), "..", "WebApp", "models", "leafblight.pt"),
    "best_strawberry_disease_model": os.path.join(os.getcwd(), "..", "WebApp", "models", "best_strawberry_disease_model.pt"),
}

def predict_disease(image_path, detection_method, part_model_path, disease_model_path, output_folder):
    try:
        conf_thresh = 0.1
        
        if detection_method == "part-first":
            part_model = YOLO(part_model_path)
            disease_model = YOLO(disease_model_path)
        else:
            disease_model = YOLO(disease_model_path)
        
        if not os.path.exists(image_path):
            print(json.dumps({"error": "Input image not found"}))
            sys.stdout.flush()
            return
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(json.dumps({"error": "Failed to load image"}))
            sys.stdout.flush()
            return
            
        predictions = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        part_model_name = os.path.splitext(os.path.basename(part_model_path))[0]
        disease_model_name = os.path.splitext(os.path.basename(disease_model_path))[0]
        crop_folder = os.path.join(output_folder, f"crops_{part_model_name}_{disease_model_name}")
        os.makedirs(crop_folder, exist_ok=True)
        crop_index = 0
        
        if detection_method == "part-first":
            # First detect parts
            results1 = part_model(original_image)
            for result in results1:
                boxes = result.boxes
                filtered_boxes = [box for box in boxes if float(box.conf[0]) >= conf_thresh]
                
                for box in filtered_boxes:
                    coords = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords.cpu().numpy())
                    h, w = original_image.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, w), min(y2, h)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    crop_img = original_image[y1:y2, x1:x2].copy()
                    
                    # Then detect diseases in the cropped part
                    results2 = disease_model(crop_img)
                    secondary_detections = []
                    
                    has_detections = False
                    annotated_crop = None
                    
                    for res2 in results2:
                        boxes2 = res2.boxes
                        filtered_boxes2 = [b for b in boxes2 if float(b.conf[0]) >= conf_thresh]
                        
                        if filtered_boxes2:
                            has_detections = True
                            # Create annotated crop only if we have detections
                            annotated_crop = res2.plot(font_size=10, line_width=4)
                            
                            for b in filtered_boxes2:
                                disease_name = disease_model.names[int(b.cls[0])]
                                confidence = float(b.conf[0])
                                secondary_detections.append({
                                    "disease": disease_name,
                                    "confidence": confidence
                                })
                    
                    # If no disease detected, use original crop for visualization
                    if not has_detections:
                        annotated_crop = crop_img
                    
                    # Always resize the annotated crop (or original if no detections)
                    annotated_crop_resized = cv2.resize(annotated_crop, (x2 - x1, y2 - y1))
                    original_image[y1:y2, x1:x2] = annotated_crop_resized
                    
                    crop_index += 1
                    crop_filename = f"crop_{crop_index}.jpg"
                    crop_filepath = os.path.join(crop_folder, crop_filename)
                    cv2.imwrite(crop_filepath, annotated_crop_resized)
                    rel_path = os.path.relpath(crop_filepath, os.path.join(os.getcwd(), "public"))
                    crop_url = "/" + rel_path.replace(os.sep, "/")
                    
                    predictions.append({
                        "primary_detection": {
                            "disease": part_model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        },
                        "secondary_detections": secondary_detections,
                        "crop_url": crop_url
                    })
        else:
            # Direct disease detection (annotate full image at once)
            results = disease_model(original_image)
            
            # ↓ Ubah font_size dan line_width di sini ↓
            annotated_full = results[0].plot(font_size=10, line_width=4)
            original_image = annotated_full
            
            # Kumpulkan data bounding box tanpa menyimpan crop
            for result in results:
                boxes = result.boxes
                filtered_boxes = [box for box in boxes if float(box.conf[0]) >= conf_thresh]
                
                for box in filtered_boxes:
                    coords = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords.cpu().numpy())
                    predictions.append({
                        "primary_detection": {
                            "disease": disease_model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": [x1, y1, x2, y2]
                        },
                        "secondary_detections": [{
                            "disease": disease_model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0])
                        }],
                        "crop_url": ""
                    })
        
        # Save annotated image
        path_file = os.path.join(output_folder, f"{base_name}_{part_model_name}_{disease_model_name}_annotated.jpg")
        cv2.imwrite(path_file, original_image)
        print(json.dumps({"success": True, "predictions": predictions}))
        sys.stdout.flush()
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

# Get model paths based on selection
# part_model_path = r"${detectionMethod === 'part-first' ? modelPaths[partModel as keyof typeof modelPaths].replace(/\\/g, '/') : ''}"
# disease_model_path = r"${modelPaths[diseaseModel as keyof typeof modelPaths].replace(/\\/g, '/')}"

# predict_disease(r"${imagePath.replace(/\\/g, '/')}", "${detectionMethod}", part_model_path, disease_model_path)

# root_folder = os.path.join(os.path.dirname(image_path), "test_inastek")
# print(os.getcwd())

inastek_folder = os.path.join(os.getcwd(), "test_inastek")
for folder_diseases in os.listdir(inastek_folder):
    full_path = os.path.join(inastek_folder, folder_diseases)
    if not os.path.isdir(full_path):
        continue
    folder_output_disease = os.path.join(inastek_folder, f"{folder_diseases}_output")
    os.makedirs(folder_output_disease, exist_ok=True)
    # process the folder as needed
    for image in os.listdir(full_path):
        image_path = os.path.join(full_path, image)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_folder = os.path.join(folder_output_disease, image_name)

        os.makedirs(output_folder, exist_ok=True)

        # part-first
        for part_model in partModelPaths:
            for disease_model in diseaseModelPaths:
                part_model_path = partModelPaths[part_model]
                disease_model_path = diseaseModelPaths[disease_model]
                predict_disease(image_path, "part-first", part_model_path, disease_model_path, output_folder)

        # direct
        for disease_model in diseaseModelPaths:
            disease_model_path = diseaseModelPaths[disease_model]
            predict_disease(image_path, "direct", "", disease_model_path, output_folder)
