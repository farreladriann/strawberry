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
      const response = await fetch('/api/process-image-direct', {
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
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-center">Image Processing App</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Upload Section */}
              <div className="flex flex-col items-center">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2"
                >
                  <Upload size={20} />
                  Upload Image
                </Button>
                <p className="mt-2 text-sm text-gray-500">
                  Supported formats: JPEG, JPG, PNG
                </p>
              </div>

              {/* Image Preview Section */}
              {processedImage && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <p className="font-medium text-center">Original Image</p>
                    <div className="border rounded-lg p-2 h-64 flex items-center justify-center">
                      <Image
                        src={processedImage.originalUrl}
                        alt="Original"
                        className="max-h-full max-w-full object-contain"
                        width={256}
                        height={256}
                        style={{ width: 'auto', height: 'auto' }}
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <p className="font-medium text-center">Processed Image</p>
                    <div className="border rounded-lg p-2 h-64 flex items-center justify-center">
                      {processedImage.processedUrl ? (
                        <Image
                          src={processedImage.processedUrl}
                          alt="Processed"
                          className="max-h-full max-w-full object-contain"
                          width={256}
                          height={256}
                          style={{ width: 'auto', height: 'auto' }}
                        />
                      ) : (
                        <div className="flex items-center justify-center">
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
                    className="flex items-center gap-2"
                  >
                    {isProcessing ? 'Processing...' : 'Process Image'}
                  </Button>
                  
                  {processedImage?.processedUrl && (
                    <Button
                      onClick={downloadImage}
                      className="flex items-center gap-2"
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