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