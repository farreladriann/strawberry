"use client"

import React, { useRef, useState } from 'react';
import { Camera, Download } from 'lucide-react';
import { Button } from 'src/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from 'src/components/ui/card';
import { Switch } from 'src/components/ui/switch';
import { Label } from 'src/components/ui/label';

const CameraPage = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [photo, setPhoto] = useState<string | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');

  const startCamera = async () => {
    // Hentikan stream yang ada jika masih aktif
    if (isStreamActive) {
      stopCamera();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreamActive(true);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreamActive(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      const context = canvas.getContext('2d');
      if (context) {
        // Flip horizontal jika menggunakan kamera depan
        if (facingMode === 'user') {
          context.scale(-1, 1);
          context.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
        } else {
          context.drawImage(videoRef.current, 0, 0);
        }
        const photoData = canvas.toDataURL('image/jpeg', 0.8);
        setPhoto(photoData);
      }
    }
  };

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    if (isStreamActive) {
      startCamera();
    }
  };

  const downloadPhoto = () => {
    if (photo) {
      const link = document.createElement('a');
      link.href = photo;
      link.download = `photo-${new Date().toISOString()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="w-6 h-6" />
            Kamera Web
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className={`w-full h-full object-cover ${facingMode === 'user' ? 'scale-x-[-1]' : ''}`}
            />
          </div>
          
          <div className="flex items-center justify-between px-2">
            <div className="flex items-center space-x-2">
              <Switch
                id="camera-switch"
                checked={facingMode === 'environment'}
                onCheckedChange={toggleCamera}
                disabled={!isStreamActive}
              />
              <Label htmlFor="camera-switch">
                {facingMode === 'user' ? 'Kamera Depan' : 'Kamera Belakang'}
              </Label>
            </div>
          </div>

          <div className="flex gap-4 justify-center">
            {!isStreamActive ? (
              <Button onClick={startCamera}>Buka Kamera</Button>
            ) : (
              <>
                <Button onClick={capturePhoto}>Ambil Foto</Button>
                <Button variant="destructive" onClick={stopCamera}>
                  Tutup Kamera
                </Button>
              </>
            )}
          </div>

          {photo && (
            <div className="mt-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">Hasil Foto:</h3>
                <Button 
                  variant="outline" 
                  onClick={downloadPhoto}
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Unduh Foto
                </Button>
              </div>
              <img
                src={photo}
                alt="Captured"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CameraPage;