import type { NextApiRequest, NextApiResponse } from "next";
import formidable, { Fields, Files } from "formidable";
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { spawn } from "child_process";

export const config = {
  api: {
    bodyParser: false,
  },
};

interface SecondaryDetection {
  disease: string;
  confidence: number;
}

interface PredictionResult {
  primary_detection: {
    disease: string;
    confidence: number;
    bbox: number[];
  };
  secondary_detections: SecondaryDetection[];
  crop_url: string;
}

async function runPythonPrediction(
  imagePath: string,
  detectionMethod: string,
  partModel: string,
  diseaseModel: string
): Promise<PredictionResult[]> {
  const modelPaths = {
    strawberry_tuned: path.join(process.cwd(), "models", "strawberry_tuned.pt"),
    strawberry_part_detection: path.join(process.cwd(), "models", "strawberry_part_detection.pt"),
    best_model: path.join(process.cwd(), "models", "best_model.pt"),
    leafblight: path.join(process.cwd(), "models", "leafblight.pt"),
    best_strawberry_disease_model: path.join(process.cwd(), "models", "best_strawberry_disease_model.pt"),
  };

  const pythonCode = `
import sys
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

def predict_disease(image_path, detection_method, part_model_path, disease_model_path):
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
        crop_folder = os.path.join(os.path.dirname(image_path), "crops", base_name)
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
                    
                    for res2 in results2:
                        boxes2 = res2.boxes
                        filtered_boxes2 = [b for b in boxes2 if float(b.conf[0]) >= conf_thresh]
                        
                        if filtered_boxes2:
                            # ↓ Ubah font_size dan line_width di sini ↓
                            annotated_crop = res2.plot(font_size=10, line_width=4)
                            annotated_crop_resized = cv2.resize(annotated_crop, (x2 - x1, y2 - y1))
                            original_image[y1:y2, x1:x2] = annotated_crop_resized
                            
                            for b in filtered_boxes2:
                                disease_name = disease_model.names[int(b.cls[0])]
                                confidence = float(b.conf[0])
                                secondary_detections.append({
                                    "disease": disease_name,
                                    "confidence": confidence
                                })
                    
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
            annotated_full = results[0].plot(font_size=12, line_width=10)
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
        cv2.imwrite(image_path, original_image)
        print(json.dumps({"success": True, "predictions": predictions}))
        sys.stdout.flush()
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

# Get model paths based on selection
part_model_path = r"${detectionMethod === 'part-first' ? modelPaths[partModel as keyof typeof modelPaths].replace(/\\/g, '/') : ''}"
disease_model_path = r"${modelPaths[diseaseModel as keyof typeof modelPaths].replace(/\\/g, '/')}"

predict_disease(r"${imagePath.replace(/\\/g, '/')}", "${detectionMethod}", part_model_path, disease_model_path)
`;

  return new Promise<PredictionResult[]>((resolve, reject) => {
    const pythonProcess = spawn("python", ["-c", pythonCode]);

    let outputData = "";
    let errorData = "";

    pythonProcess.stdout.on("data", (data) => {
      outputData += data.toString();
      console.log("Python stdout:", data.toString());
    });

    pythonProcess.stderr.on("data", (data) => {
      errorData += data.toString();
      console.error("Python stderr:", data.toString());
    });

    pythonProcess.on("close", (code) => {
      console.log("Python process closed with code:", code);
      if (code !== 0) {
        return reject(new Error(`Python process exited with code ${code}\nError: ${errorData}`));
      }

      try {
        const lines = outputData.trim().split("\n");
        const lastLine = lines[lines.length - 1];
        const result = JSON.parse(lastLine);

        if (result.error) {
          return reject(new Error(result.error));
        } else if (result.success && Array.isArray(result.predictions)) {
          return resolve(result.predictions);
        } else {
          return reject(new Error("Invalid prediction results format"));
        }
      } catch (error) {
        console.error("JSON Parse Error:", error);
        console.error("Attempted to parse:", outputData);
        return reject(new Error("Failed to parse prediction results"));
      }
    });
  });
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const form = formidable({
      keepExtensions: true,
      maxFiles: 1,
      maxFileSize: 10 * 1024 * 1024,
      filter: (part) => part.mimetype?.includes("image/") || false,
    });

    const [fields, files] = await new Promise<[Fields, Files]>((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) return reject(err);
        resolve([fields, files]);
      });
    });

    const file = Array.isArray(files.image) ? files.image[0] : files.image;
    if (!file) {
      return res.status(400).json({ message: "No image uploaded" });
    }

    const detectionMethod = String(fields.detectionMethod);
    const partModel = String(fields.partModel);
    const diseaseModel = String(fields.diseaseModel);

    const uploadsDir = path.join(process.cwd(), "public", "uploads");
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    const safeFilename = `${Date.now()}-${path.basename(file.originalFilename || "image.jpg").replace(/[^a-zA-Z0-9.-]/g, "_")}`;
    const processedImagePath = path.join(uploadsDir, `processed-${safeFilename}`);

    await sharp(file.filepath)
      .sharpen()
      .jpeg({ quality: 100 })
      .toFile(processedImagePath);

    const predictions = await runPythonPrediction(
      processedImagePath,
      detectionMethod,
      partModel,
      diseaseModel
    );

    return res.status(200).json({
      message: "Image processed successfully",
      processedImageUrl: `/uploads/processed-${safeFilename}?t=${Date.now()}`,
      predictions,
    });
  } catch (error) {
    console.error("Error processing image:", error);
    return res.status(500).json({
      message: "Error processing image",
      error: String(error),
    });
  }
}