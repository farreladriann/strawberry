import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class AugmentationInfo:
    """Store information about applied augmentations"""
    colored_light: Dict[str, float] = None

class ImageAugmentor:
    def __init__(self, image_path: str):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

    def add_colored_lighting(self, image: np.ndarray, 
                             color: Tuple[int, int, int], 
                             intensity: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add colored lighting effect."""
        color_layer = np.full_like(image, color)
        blend = cv2.addWeighted(image, 1, color_layer, intensity, 0)
        
        # Adjust color name based on provided color
        color_name = "red (630-660nm)" if color == (255, 0, 0) else "indoor purple"
        light_info = {
            'color': color_name,
            'intensity': round(intensity, 2)
        }
        return blend, light_info

    def generate_variations(self, num_variations: int = 8) -> List[Tuple[np.ndarray, AugmentationInfo]]:
        """Generate variations using only colored lighting effects."""
        variations = []
        
        for _ in range(num_variations):
            img = self.original.copy()
            info = AugmentationInfo()
            
            # Apply subtle lighting effects only
            color = random.choice([
                (255, 0, 0),     # Red for spectrum 630-660nm
                (127, 0, 255)    # Purple for indoor farming
            ])
            img, light_info = self.add_colored_lighting(
                img, color, random.uniform(0.1, 0.3))
            info.colored_light = light_info
            
            variations.append((img, info))
        
        return variations

    def format_augmentation_title(self, info: AugmentationInfo) -> str:
        """Format augmentation information into a title string."""
        parts = []
        
        if info.colored_light is not None:
            parts.append(f"Light({info.colored_light['color']}:{info.colored_light['intensity']})")
            
        return "\n".join(parts)

    def display_variations(self, variations: List[Tuple[np.ndarray, AugmentationInfo]]):
        """Display original image and its variations in a grid with detailed titles."""
        n = len(variations) + 1
        grid_size = int(np.ceil(np.sqrt(n)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle('Original Image and Augmented Variations', fontsize=16)
        
        # Display original image
        axes[0, 0].imshow(self.original)
        axes[0, 0].set_title('Original', fontsize=8)
        axes[0, 0].axis('off')
        
        # Display variations
        for idx, (img, info) in enumerate(variations, 1):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].imshow(img)
            axes[row, col].set_title(self.format_augmentation_title(info), fontsize=8, pad=3)
            axes[row, col].axis('off')
        
        # Turn off any unused subplots
        for idx in range(len(variations) + 1, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

    def process_and_display(self, num_variations: int = 8):
        """Generate and display augmented variations."""
        variations = self.generate_variations(num_variations)
        self.display_variations(variations)

# Usage remains the same
augmentor = ImageAugmentor("Leaf Spot/20250112_074254.jpg")
augmentor.process_and_display(num_variations=20)
