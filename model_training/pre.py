from ultralytics import YOLO
import cv2
import os

def predict_disease():
    # Load the trained model
    model = YOLO('best_model.pt')
    
    # Paths to test images and result directory
    test_images_dir = "./test/img"
    result_images_dir = "./test/result1"
    
    # Confidence threshold
    confidence_threshold = 0.5
    
    # Process each image in the test images directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        
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
            result_image_path = os.path.join(result_images_dir, image_name)
            cv2.imwrite(result_image_path, im_array)
            print(f"Processed {image_name} and saved to {result_image_path}")
            
            # Print predictions
            for box in filtered_boxes:
                confidence = box.conf[0]
                class_id = box.cls[0]
                print(f"Disease: {model.names[int(class_id)]}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    predict_disease()