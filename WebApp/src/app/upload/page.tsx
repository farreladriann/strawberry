"use client"

import { useState, useRef } from 'react';
import Image from 'next/image';
import { Button } from 'src/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from 'src/components/ui/card';
import { Upload, Download, Image as ImageIcon } from 'lucide-react';

interface ProcessedImage {
  originalUrl: string;
  processedUrl: string;
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<ProcessedImage | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === "image/jpeg" || file.type === "image/jpg" || file.type === "image/png") {
        setSelectedImage(file);
        const imageUrl = URL.createObjectURL(file);
        setProcessedImage({
          originalUrl: imageUrl,
          processedUrl: "" // Will be set after processing
        });
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
      formData.append('image', selectedImage);

      // Replace with your actual API endpoint
      const response = await fetch('/api/process-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Image processing failed');

      const result = await response.json();
      
      setProcessedImage(prev => ({
        originalUrl: prev?.originalUrl || '',
        processedUrl: result.processedImageUrl
      }));
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadImage = () => {
    if (processedImage?.processedUrl) {
      const link = document.createElement('a');
      link.href = processedImage.processedUrl;
      link.download = 'processed-image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
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

              {/* Image Preview Section */}
              {processedImage && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="font-medium text-center text-gray-700">Original Image</p>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-white">
                      <div className="relative w-full h-64">
                        <Image
                          src={processedImage.originalUrl}
                          alt="Original"
                          className="object-contain"
                          fill // Use fill to make the image responsive within the container
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <p className="font-medium text-center text-gray-700">Disease Detection Result</p>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-white">
                      {processedImage.processedUrl ? (
                        <div className="relative w-full h-64">
                          <Image
                            src={processedImage.processedUrl}
                            alt="Processed"
                            className="object-contain"
                            fill // Use fill to make the image responsive within the container
                          />
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-64">
                          <ImageIcon className="text-gray-400" size={48} />
                        </div>
                      )}
                    </div>
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
                    {isProcessing ? 'Processing...' : 'Process Image'}
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