from ultralytics import YOLO
import cv2

def predict_disease(image_path, result_image_path):
    # Load the trained model
    model = YOLO('best_model.pt')
    
    # Confidence threshold
    confidence_threshold = 0.5
    
    # Run inference
    results = model(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        
        # Filter boxes by confidence
        filtered_boxes = [box for box in boxes if box.conf[0] >= confidence_threshold]
        
        # Get image with boxes drawn
        im_array = result.plot()
        
        # Save the output image
        cv2.imwrite(result_image_path, im_array)
        print(f"Processed {image_path} and saved to {result_image_path}")
        
        # Print predictions
        for box in filtered_boxes:
            confidence = box.conf[0]
            class_id = box.cls[0]
            print(f"Disease: {model.names[int(class_id)]}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # Path to the input image and output result image
    input_image_path = "./test/img/image1.png"  # Ganti dengan path gambar Anda
    output_image_path = "./test/result1/image1.png"  # Ganti dengan path output yang diinginkan
    
    # Predict disease on the single image
    predict_disease(input_image_path, output_image_path)


# from ultralytics import YOLO
# import torch
# import torch.nn as nn
# import cv2

# class CustomYOLO(nn.Module):
#     def __init__(self, model_path):
#         super(CustomYOLO, self).__init__()
#         # Load the YOLO model
#         self.yolo = YOLO(model_path)
#         self.yolo.eval()  # Set YOLO to evaluation mode
        
#         # Add custom convolutional layers
#         self.custom_layers = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )

#     def forward(self, image_path):
#         # Get YOLO results
#         results = self.yolo(image_path)
        
#         # Load image and preprocess
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (640, 640))  # Resize for simplicity
#         image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
#         # Disable gradient calculations
#         with torch.no_grad():
#             custom_output = self.custom_layers(image_tensor)
        
#         return results, custom_output


# def predict_disease_with_custom_model(image_path, result_image_path, model_path='best_model.pt'):
#     # Load the custom YOLO model
#     model = CustomYOLO(model_path)
#     model.eval()  # Ensure the model is in evaluation mode
    
#     # Confidence threshold
#     confidence_threshold = 0.5

#     # Run inference
#     results, custom_output = model(image_path)

#     # Process results
#     for result in results:
#         boxes = result.boxes  # Bounding boxes
#         filtered_boxes = [box for box in boxes if box.conf[0] >= confidence_threshold]
#         im_array = result.plot()
#         cv2.imwrite(result_image_path, im_array)
#         print(f"Processed {image_path} and saved to {result_image_path}")
#         for box in filtered_boxes:
#             confidence = box.conf[0]
#             class_id = box.cls[0]
#             print(f"Disease: {model.yolo.names[int(class_id)]}, Confidence: {confidence:.2f}")

#     # Optional: Print custom layer output shape
#     print(f"Custom layers output shape: {custom_output.shape}")


# if __name__ == "__main__":
#     input_image_path = "./test/img/image1.png"
#     output_image_path = "./test/result1/image1.png"
#     predict_disease_with_custom_model(input_image_path, output_image_path)
