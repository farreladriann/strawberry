import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class AugmentationInfo:
    """Store information about applied augmentations"""
    temperature: float = None
    jitter: Dict[str, float] = None
    channel_shift: Dict[str, float] = None
    colored_light: Dict[str, float] = None

class ImageAugmentor:
    def __init__(self, image_path: str):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

    def adjust_color_temperature(self, image: np.ndarray, 
                               temperature: float) -> Tuple[np.ndarray, float]:
        """Adjust color temperature (warm/cool)."""
        img = image.copy()
        if temperature > 0:  # Warmer
            img = img.astype(float)
            img[:,:,0] *= (1 + temperature * 0.1)  # More red
            img[:,:,2] *= (1 - temperature * 0.1)  # Less blue
        else:  # Cooler
            img = img.astype(float)
            img[:,:,0] *= (1 + temperature * 0.1)  # Less red
            img[:,:,2] *= (1 - temperature * 0.1)  # More blue
        return np.clip(img, 0, 255).astype(np.uint8), temperature

    def color_jitter(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply random color jittering."""
        # Generate random values
        hue_shift = random.uniform(-10, 10)
        sat_scale = random.uniform(0.5, 1.5)
        val_scale = random.uniform(0.7, 1.3)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply adjustments
        hsv[:,:,0] += hue_shift
        hsv[:,:,1] *= sat_scale
        hsv[:,:,2] *= val_scale
        
        # Ensure values are in valid range
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        
        jitter_info = {
            'hue': round(hue_shift, 2),
            'saturation': round(sat_scale, 2),
            'value': round(val_scale, 2)
        }
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), jitter_info

    def add_colored_lighting(self, image: np.ndarray, 
                           color: Tuple[int, int, int], 
                           intensity: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add colored lighting effect."""
        color_layer = np.full_like(image, color)
        blend = cv2.addWeighted(image, 1, color_layer, intensity, 0)
        
        color_name = "yellow" if color == (255, 255, 0) else "purple"
        light_info = {
            'color': color_name,
            'intensity': round(intensity, 2)
        }
        
        return blend, light_info

    def channel_shift(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply random channel shifting."""
        shifts = [random.uniform(-30, 30) for _ in range(3)]
        shifted = image.astype(np.float32)
        
        for i in range(3):
            shifted[:,:,i] += shifts[i]
            
        shift_info = {
            'R': round(shifts[0], 2),
            'G': round(shifts[1], 2),
            'B': round(shifts[2], 2)
        }
        
        return np.clip(shifted, 0, 255).astype(np.uint8), shift_info

    def generate_variations(self, num_variations: int = 8) -> List[Tuple[np.ndarray, AugmentationInfo]]:
        """Generate multiple variations of the image with augmentation info."""
        variations = []
        
        for _ in range(num_variations):
            img = self.original.copy()
            info = AugmentationInfo()
            
            # Random color temperature
            if random.random() > 0.5:
                temp = random.uniform(-1, 1)
                img, temp_value = self.adjust_color_temperature(img, temp)
                info.temperature = temp_value
            
            # Random color jitter
            if random.random() > 0.5:
                img, jitter_info = self.color_jitter(img)
                info.jitter = jitter_info
            
            # Random channel shift
            if random.random() > 0.5:
                img, shift_info = self.channel_shift(img)
                info.channel_shift = shift_info
            
            # Add colored lighting
            if random.random() > 0.5:
                color = random.choice([
                    (255, 255, 0),  # Yellow
                    (255, 0, 255)   # Purple
                ])
                img, light_info = self.add_colored_lighting(
                    img, color, random.uniform(0.1, 0.3))
                info.colored_light = light_info
            
            variations.append((img, info))
        
        return variations

    def format_augmentation_title(self, info: AugmentationInfo) -> str:
        """Format augmentation information into a title string."""
        parts = []
        
        if info.temperature is not None:
            temp_type = "Warm" if info.temperature > 0 else "Cool"
            parts.append(f"Temp({temp_type}:{info.temperature:.2f})")
            
        if info.jitter is not None:
            parts.append(f"Jitter(H:{info.jitter['hue']},S:{info.jitter['saturation']},V:{info.jitter['value']})")
            
        if info.channel_shift is not None:
            parts.append(f"Shift(R:{info.channel_shift['R']},G:{info.channel_shift['G']},B:{info.channel_shift['B']})")
            
        if info.colored_light is not None:
            parts.append(f"Light({info.colored_light['color']}:{info.colored_light['intensity']})")
            
        return "\n".join(parts)

    def display_variations(self, variations: List[Tuple[np.ndarray, AugmentationInfo]]):
        """Display original image and its variations in a grid with detailed titles."""
        n = len(variations) + 1
        grid_size = int(np.ceil(np.sqrt(n)))
        
        fig, axes = plt.subplots(grid_size, grid_size, 
                               figsize=(20, 20))
        fig.suptitle('Original Image and Augmented Variations', 
                    fontsize=16)
        
        # Display original
        axes[0, 0].imshow(self.original)
        axes[0, 0].set_title('Original', fontsize=8)
        axes[0, 0].axis('off')
        
        # Display variations
        for idx, (img, info) in enumerate(variations, 1):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].imshow(img)
            axes[row, col].set_title(
                self.format_augmentation_title(info), 
                fontsize=8, 
                pad=3
            )
            axes[row, col].axis('off')
        
        # Turn off empty subplots
        for idx in range(len(variations) + 1, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

    def process_and_display(self, num_variations: int = 8):
        """Generate and display variations."""
        variations = self.generate_variations(num_variations)
        self.display_variations(variations)