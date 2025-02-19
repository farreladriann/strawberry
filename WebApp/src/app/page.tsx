"use client"

import { useState, useRef } from "react";
import Image from "next/image";
import { Button } from "src/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "src/components/ui/card";
import { Upload, Download, Image as ImageIcon } from "lucide-react";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "src/components/ui/select";
import { RadioGroup, RadioGroupItem } from "src/components/ui/radio-group";
import { Label } from "src/components/ui/label";

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

interface ProcessedImage {
  originalUrl: string;
  processedUrl: string;
}

const PART_DETECTION_MODELS = [
  { id: "strawberry_tuned", name: "Strawberry Tuned" },
  { id: "strawberry_part_detection", name: "Strawberry Part Detection" },
];

const DISEASE_DETECTION_MODELS = [
  { id: "best_model", name: "Best Model" },
  { id: "leafblight", name: "Leaf Blight Detector" },
  { id: "best_strawberry_disease_model", name: "Best Strawberry Disease" },
];

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<ProcessedImage | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // New state for detection options
  const [detectionMethod, setDetectionMethod] = useState<"part-first" | "direct">("direct");
  const [selectedPartModel, setSelectedPartModel] = useState(PART_DETECTION_MODELS[0].id);
  const [selectedDiseaseModel, setSelectedDiseaseModel] = useState(DISEASE_DETECTION_MODELS[0].id);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === "image/jpeg" || file.type === "image/jpg" || file.type === "image/png") {
        setSelectedImage(file);
        const imageUrl = URL.createObjectURL(file);
        setProcessedImage({
          originalUrl: imageUrl,
          processedUrl: ""
        });
        setPredictions([]);
      } else {
        alert("Please upload only JPEG, JPG, or PNG files.");
      }
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append("image", selectedImage);
      formData.append("detectionMethod", detectionMethod);
      formData.append("partModel", selectedPartModel);
      formData.append("diseaseModel", selectedDiseaseModel);
      
      const response = await fetch("/api/process-image-new", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) throw new Error("Image processing failed");
      const result = await response.json();
      
      setProcessedImage((prev) => ({
        originalUrl: prev?.originalUrl || "",
        processedUrl: result.processedImageUrl,
      }));
      setPredictions(result.predictions || []);
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadImage = () => {
    if (processedImage?.processedUrl) {
      const link = document.createElement("a");
      link.href = processedImage.processedUrl;
      link.download = "processed-image.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Helper: menghasilkan string label untuk secondary detection secara agregat
  const getAllSecondaryLabels = () => {
    const allSec = predictions.flatMap((pred) => pred.secondary_detections);
    if (allSec.length === 0) return "No disease in this cropped area";
    return allSec
      .map((sd) => `${sd.disease} (${(sd.confidence * 100).toFixed(0)}%)`)
      .join(", ");
  };

  // Untuk tiap crop: label primary (first detection) dan label secondary detection (jika ada)
  const getCropLabels = (pred: PredictionResult) => {
    const primary = `${pred.primary_detection.disease} (${(
      pred.primary_detection.confidence * 100
    ).toFixed(0)}%)`;
    const secondary =
      pred.secondary_detections.length > 0
        ? pred.secondary_detections
            .map((sd) => `${sd.disease} (${(sd.confidence * 100).toFixed(0)}%)`)
            .join(", ")
        : "No disease in this cropped area";
    return { primary, secondary };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <Card className="shadow-lg">
          <CardHeader className="bg-gradient-to-r from-green-400 to-blue-500 p-6 rounded-t-lg">
            <CardTitle className="text-center text-white text-3xl font-bold">
              Strawberry Disease Detection
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="space-y-8">
              {/* Detection Method Selection */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Detection Method</h3>
                <RadioGroup 
                  value={detectionMethod} 
                  onValueChange={(value) => setDetectionMethod(value as "part-first" | "direct")}
                  className="space-y-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="part-first" id="part-first" />
                    <Label htmlFor="part-first">Detect parts first, then diseases</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="direct" id="direct" />
                    <Label htmlFor="direct">Detect diseases directly</Label>
                  </div>
                </RadioGroup>
              </div>

              {/* Model Selection */}
              <div className="space-y-4">
                {detectionMethod === "part-first" && (
                  <div className="space-y-2">
                    <Label>Part Detection Model</Label>
                    <Select value={selectedPartModel} onValueChange={setSelectedPartModel}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select part detection model" />
                      </SelectTrigger>
                      <SelectContent>
                        {PART_DETECTION_MODELS.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                <div className="space-y-2">
                  <Label>Disease Detection Model</Label>
                  <Select value={selectedDiseaseModel} onValueChange={setSelectedDiseaseModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select disease detection model" />
                    </SelectTrigger>
                    <SelectContent>
                      {DISEASE_DETECTION_MODELS.map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Upload Section */}
              <div className="flex flex-col items-center space-y-4">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
                >
                  <Upload size={20} />
                  Upload Image
                </Button>
                <p className="text-sm text-gray-600">
                  Supported formats: JPEG, JPG, PNG
                </p>
              </div>

              {/* Rest of the existing JSX remains the same */}
              {/* Image Preview Section */}
              {processedImage && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Original Image */}
                  <div className="space-y-4">
                    <p className="font-medium text-center text-gray-700">
                      Original Image
                    </p>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-white">
                      <div className="relative w-full h-64">
                        <Image
                          src={processedImage.originalUrl}
                          alt="Original"
                          className="object-contain"
                          fill
                        />
                      </div>
                    </div>
                  </div>
                  {/* Processed Image & Overall Secondary Detection */}
                  <div className="space-y-4">
                    <p className="font-medium text-center text-gray-700">
                      Disease Detection Result
                    </p>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-white">
                      {processedImage.processedUrl ? (
                        <div className="relative w-full h-64">
                          <Image
                            src={processedImage.processedUrl}
                            alt="Processed"
                            className="object-contain"
                            fill
                          />
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-64">
                          <ImageIcon className="text-gray-400" size={48} />
                        </div>
                      )}
                    </div>
                    
                    {/* Tampilkan seluruh secondary detection untuk keseluruhan gambar */}
                    {processedImage?.processedUrl && (
                    <div className="mt-4">
                      <h3 className="text-center text-xl font-bold text-gray-700">
                        Diseases
                      </h3>
                      {predictions.length === 0 ? (
                        <p className="mt-2 text-center text-gray-700">
                          There are no diseases
                        </p>
                      ) : (
                        <p className="mt-2 text-center text-gray-700">
                          {getAllSecondaryLabels()}
                        </p>
                      )}
                    </div>
                    )}
                  </div>
                </div>
              )}

              {/* Detected Crops Section */}
              {detectionMethod === "part-first" && predictions.length > 0 && (
                <div className="space-y-4">
                  <p className="font-medium text-center text-gray-700">
                    Detected Crops
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {predictions.filter(pred => pred.crop_url).map((pred, index) => {
                      const labels = getCropLabels(pred);
                      return (
                        <div key={index} className="border rounded-lg p-2 bg-white">
                          <div className="relative w-full h-40">
                            <Image
                              src={pred.crop_url + `?t=${Date.now()}`}
                              alt={`Crop ${index + 1}`}
                              className="object-contain"
                              fill
                            />
                          </div>
                          <p className="mt-2 text-center text-sm font-medium text-gray-700">
                            {labels.primary}
                          </p>
                          <p className="text-center text-xs text-gray-600">
                            {labels.secondary}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              {selectedImage && (
                <div className="flex justify-center gap-4">
                  <Button
                    onClick={processImage}
                    disabled={isProcessing}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
                  >
                    {isProcessing ? "Processing..." : "Process Image"}
                  </Button>
                  {processedImage?.processedUrl && (
                    <Button
                      onClick={downloadImage}
                      className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
                    >
                      <Download size={20} />
                      Download
                    </Button>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}