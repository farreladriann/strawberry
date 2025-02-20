import type { NextApiRequest, NextApiResponse } from 'next';
import formidable, { Fields, Files } from 'formidable';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { spawn } from 'child_process';

export const config = {
  api: {
    bodyParser: false,
  },
};

interface PredictionResult {
  disease: string;
  confidence: number;
}

async function runPythonPrediction(imagePath: string): Promise<PredictionResult[]> {
  const modelPath = path.join(process.cwd(), 'models', 'best_model.pt');

  if (!fs.existsSync(modelPath)) {
    throw new Error(`Model file not found at ${modelPath}`);
  }

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      '-c',
      `
import sys
import json
from ultralytics import YOLO
import cv2
import os

def predict_disease(image_path):
    try:
        # Load the trained model
        model_path = r'${modelPath.replace(/\\/g, '/')}'
        if not os.path.exists(model_path):
            print(json.dumps({"error": f"Model file not found at {model_path}"}))
            return None
        
        model = YOLO(model_path)
        
        # Check if input image exists
        if not os.path.exists(image_path):
            print(json.dumps({"error": f"Input image not found at {image_path}"}))
            return None
        
        # Run inference
        results = model(image_path)
        predictions = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            # Get image with boxes drawn
            im_array = result.plot()
            
            # Save the output image back to the same file
            cv2.imwrite(image_path, im_array)
            
            # Collect predictions
            for box in boxes:
                if float(box.conf[0]) >= 0.5:  # Confidence threshold
                    predictions.append({
                        "disease": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })
        
        # Print the predictions as JSON to stdout
        print(json.dumps({"success": True, "predictions": predictions}))
        sys.stdout.flush()
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

# Run prediction
predict_disease(r"${imagePath.replace(/\\/g, '/')}")
      `,
    ]);

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log('Python stdout:', data.toString());  // Debug log
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error('Python stderr:', data.toString());  // Debug log
    });

    pythonProcess.on('close', (code) => {
      console.log('Python process closed with code:', code);  // Debug log
      console.log('Final output:', outputData);  // Debug log
      
      if (code !== 0) {
        return reject(new Error(`Python process exited with code ${code}\nError: ${errorData}`));
      }

      try {
        const lastLine = outputData.trim().split('\n').pop() || '';
        const result = JSON.parse(lastLine);
        
        if (result.error) {
          reject(new Error(result.error));
        } else if (result.success && Array.isArray(result.predictions)) {
          resolve(result.predictions);
        } else {
          reject(new Error('Invalid prediction results format'));
        }
      } catch (error) {
        console.error('JSON Parse Error:', error);  // Debug log
        console.error('Attempted to parse:', outputData);  // Debug log
        reject(new Error(`Failed to parse prediction results`));
      }
    });
  });
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Create required directories
    const modelsDir = path.join(process.cwd(), 'models');
    const uploadsDir = path.join(process.cwd(), 'public', 'uploads');
    
    [modelsDir, uploadsDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    const modelPath = path.join(modelsDir, 'best_model.pt');
    if (!fs.existsSync(modelPath)) {
      return res.status(500).json({ 
        message: 'Model file not found', 
        expectedPath: modelPath 
      });
    }

    const form = formidable({});
    const [, files]: [Fields, Files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });

    const file = files.image?.[0];
    if (!file) {
      return res.status(400).json({ message: 'No image uploaded' });
    }

    // Process the image using Sharp first
    const processedImagePath = path.join(uploadsDir, `processed-${file.originalFilename}`);
    await sharp(file.filepath)
      .sharpen()
      .toFile(processedImagePath);

    // // Run disease prediction on the processed image - it will update the same image
    const predictions = await runPythonPrediction(processedImagePath);

    // Return the results
    const processedImageUrl = `/uploads/processed-${file.originalFilename}`;

    res.status(200).json({
      message: 'Image processed successfully',
      processedImageUrl,
      predictions,
    });

  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ 
      message: 'Error processing image', 
      error: String(error)
    });
  }
}