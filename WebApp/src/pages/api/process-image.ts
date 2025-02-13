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

async function runPythonPrediction(imagePath: string): Promise<PredictionResult[]> {
  const modelPath = path.join(process.cwd(), "models", "strawberry_tuned.pt");
  const secondModelPath = path.join(process.cwd(), "models", "best_strawberry_disease_model.pt");

  if (!fs.existsSync(modelPath) || !fs.existsSync(secondModelPath)) {
    throw new Error(`Model files not found at ${modelPath} or ${secondModelPath}`);
  }

  // Gunakan concatenation alih-alih f-string yang mengandung {...} agar tidak bentrok dengan TS
  const pythonCode = `
import sys
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

def predict_disease(image_path):
    try:
        conf_thresh1 = 0.1
        conf_thresh2 = 0.1

        model1 = YOLO(r'${modelPath.replace(/\\\\/g, "/")}')
        model2 = YOLO(r'${secondModelPath.replace(/\\\\/g, "/")}')

        if not os.path.exists(image_path):
            print(json.dumps({"error": "Input image not found at " + image_path}))
            sys.stdout.flush()
            return

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(json.dumps({"error": "Failed to load image: " + image_path}))
            sys.stdout.flush()
            return

        predictions = []

        # Subfolder unik untuk crop, berdasarkan nama file processed (tanpa ekstensi)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        crop_folder = os.path.join(os.path.dirname(image_path), "crops", base_name)
        os.makedirs(crop_folder, exist_ok=True)
        crop_index = 0

        results1 = model1(original_image)
        for result in results1:
            boxes = result.boxes
            filtered_boxes = [box for box in boxes if float(box.conf[0]) >= conf_thresh1]

            for box in filtered_boxes:
                coords = box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords.cpu().numpy())
                h, w = original_image.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                crop_img = original_image[y1:y2, x1:x2].copy()
                crop_to_save = crop_img

                results2 = model2(crop_img)
                secondary_detections = []
                for res2 in results2:
                    boxes2 = res2.boxes
                    filtered_boxes2 = [b for b in boxes2 if float(b.conf[0]) >= conf_thresh2]
                    if filtered_boxes2:
                        annotated_crop = res2.plot(font_size=8, line_width=4)
                        annotated_crop_resized = cv2.resize(annotated_crop, (x2 - x1, y2 - y1))
                        original_image[y1:y2, x1:x2] = annotated_crop_resized
                        crop_to_save = annotated_crop_resized
                        for b in filtered_boxes2:
                            disease_name = model2.names[int(b.cls[0])]
                            confidence = float(b.conf[0])
                            secondary_detections.append({
                                "disease": disease_name,
                                "confidence": confidence
                            })

                crop_index += 1
                crop_filename = "crop_" + str(crop_index) + ".jpg"
                crop_filepath = os.path.join(crop_folder, crop_filename)
                cv2.imwrite(crop_filepath, crop_to_save)
                rel_path = os.path.relpath(crop_filepath, os.path.join(os.getcwd(), "public"))
                crop_url = "/" + rel_path.replace(os.sep, "/")

                predictions.append({
                    "primary_detection": {
                        "disease": model1.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    },
                    "secondary_detections": secondary_detections,
                    "crop_url": crop_url
                })

        # Simpan gambar utama yang sudah dianotasi
        cv2.imwrite(image_path, original_image)
        print(json.dumps({"success": True, "predictions": predictions}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

predict_disease(r"${imagePath.replace(/\\\\/g, "/")}")
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
      console.log("Final output:", outputData);

      if (code !== 0) {
        return reject(
          new Error(`Python process exited with code ${code}\nError: ${errorData}`)
        );
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
    const modelsDir = path.join(process.cwd(), "models");
    const uploadsDir = path.join(process.cwd(), "public", "uploads");

    [modelsDir, uploadsDir].forEach((dir) => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    const form = formidable({
      keepExtensions: true,
      maxFiles: 1,
      maxFileSize: 10 * 1024 * 1024, // 10MB
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

    try {
      await fs.promises.access(file.filepath, fs.constants.R_OK);
    } catch (error) {
      return res.status(400).json({
        message: "Unable to access uploaded file",
        error: String(error),
      });
    }

    // Nama file aman
    const safeFilename = `${Date.now()}-${path
      .basename(file.originalFilename || "image.jpg")
      .replace(/[^a-zA-Z0-9.-]/g, "_")}`;
    const processedImagePath = path.join(uploadsDir, `processed-${safeFilename}`);

    // Proses gambar dengan sharp
    try {
      await sharp(file.filepath)
        .rotate()
        .sharpen()
        .jpeg({ quality: 80 })
        .toFile(processedImagePath);
    } catch (error) {
      return res.status(400).json({
        message: "Error processing image file",
        error: String(error),
      });
    }

    // Jalankan Python
    const predictions = await runPythonPrediction(processedImagePath);
    // Tambahkan query string agar browser tidak cache
    const processedImageUrl = `/uploads/processed-${safeFilename}?t=${Date.now()}`;

    return res.status(200).json({
      message: "Image processed successfully",
      processedImageUrl,
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
