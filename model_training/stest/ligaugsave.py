import cv2
import numpy as np
import random
import os
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class AugmentationInfo:
    """Store information about applied augmentations"""
    colored_light: Dict[str, float] = None

class ImageAugmentor:
    def __init__(self, image_path: str):
        """Initialize with an image path."""
        self.original_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.base_filename = os.path.splitext(os.path.basename(image_path))[0]
        self.output_dir = "augmented_images"
        os.makedirs(self.output_dir, exist_ok=True)

    def add_colored_lighting(self, image: np.ndarray, 
                             color: Tuple[int, int, int], 
                             intensity: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add colored lighting effect."""
        color_layer = np.full_like(image, color)
        blend = cv2.addWeighted(image, 1, color_layer, intensity, 0)
        
        color_name = "red (630-660nm)" if color == (255, 0, 0) else "indoor purple"
        light_info = {
            'color': color_name,
            'intensity': round(intensity, 2)
        }
        return blend, light_info

    def generate_variations(self, num_variations: int = 8) -> List[Tuple[np.ndarray, AugmentationInfo]]:
        """Generate variations with specific colors and random intensities."""
        variations = []
        colors = [
            (255, 0, 0),     # Red for spectrum 630-660nm
            (127, 0, 255)    # Purple for indoor farming
        ]

        num_variations = min(num_variations, len(colors))

        for color in colors[:num_variations]:
            img = self.original.copy()
            info = AugmentationInfo()

            intensity = random.uniform(0.1, 0.3)
            img, light_info = self.add_colored_lighting(img, color, intensity)
            info.colored_light = light_info

            variations.append((img, info))

        return variations


    def save_variation(self, image: np.ndarray, info: AugmentationInfo, index: int):
        """Save a single augmented variation to the output directory."""
        filename = f"{self.base_filename}_aug_{index}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, image_bgr)

    def process_and_save(self, num_variations: int = 8):
        """Generate and save augmented variations."""
        variations = self.generate_variations(num_variations)
        for i, (img, info) in enumerate(variations):
            self.save_variation(img, info, i)
        print(f"Saved {num_variations} augmented images to '{self.output_dir}'")

# Example usage:
augmentor = ImageAugmentor("Leaf Spot/20250112_074254.jpg")  # Replace with your image path
augmentor.process_and_save(num_variations=2)