import cv2
import numpy as np
from skimage import color
from scipy.interpolate import LinearNDInterpolator

class LightingNormalizer:
    def __init__(self):
        """Initialize the lighting normalizer with different methods"""
        pass
        
    def white_balance(self, img):
        """
        Apply simple white balancing using the Gray World assumption
        """
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def color_transfer(self, source, target):
        """
        Transfer the color distribution from the source to the target image
        using Reinhard's method
        """
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)

        # Compute mean and std for each channel
        source_mean = np.mean(source, axis=(0, 1))
        source_std = np.std(source, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        target_std = np.std(target, axis=(0, 1))

        # Subtract the mean from the source image
        result = source - source_mean

        # Scale by relative standard deviations
        for i in range(3):
            if source_std[i] != 0:
                result[:, :, i] *= (target_std[i] / source_std[i])

        # Add in the target mean
        result += target_mean
        
        # Clip and convert back to BGR
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def illumination_normalization(self, img):
        """
        Normalize illumination using Retinex-based normalization
        """
        # Convert to float
        img_float = img.astype(float)
        
        # Apply log transform
        log_img = np.log1p(img_float)
        
        # Apply Gaussian filter to estimate illumination
        blur_size = max(3, int(min(img.shape[:2]) * 0.02))
        if blur_size % 2 == 0:
            blur_size += 1
        illumination = cv2.GaussianBlur(log_img, (blur_size, blur_size), 0)
        
        # Subtract illumination and apply exp transform
        reflectance = np.exp(log_img - illumination) - 1
        
        # Normalize to 0-255 range
        normalized = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def process_image(self, img, reference_img=None, method='illumination'):
        """
        Process an image using the specified method
        
        Args:
            img: Input image
            reference_img: Reference image for color transfer (optional)
            method: 'white_balance', 'color_transfer', or 'illumination'
        """
        if method == 'white_balance':
            return self.white_balance(img)
        elif method == 'color_transfer' and reference_img is not None:
            return self.color_transfer(img, reference_img)
        elif method == 'illumination':
            return self.illumination_normalization(img)
        else:
            raise ValueError("Invalid method or missing reference image for color transfer")

def normalize_dataset_lighting(input_folder, output_folder, reference_image_path=None, method='illumination'):
    """
    Process all images in a folder to normalize their lighting conditions
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save processed images
        reference_image_path: Path to reference image for color transfer (optional)
        method: Normalization method to use
    """
    import os
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize normalizer
    normalizer = LightingNormalizer()
    
    # Load reference image if provided
    reference_img = None
    if reference_image_path and os.path.exists(reference_image_path):
        reference_img = cv2.imread(reference_image_path)
    
    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # Process image
            try:
                processed_img = normalizer.process_image(img, reference_img, method)
                
                # Save processed image
                output_path = os.path.join(output_folder, f"normalized_{filename}")
                cv2.imwrite(output_path, processed_img)
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")