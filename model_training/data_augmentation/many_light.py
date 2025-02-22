import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass

# Define optimal grow light colors
GROW_LIGHT_COLORS = {
    'deep_red': {
        'rgb': (255, 0, 0),  # RGB for ~660nm
        'wavelength': '660nm',
        'purpose': 'Flowering, fruiting, stem growth',
        'intensity_range': (0.2, 0.4)
    },
    'red': {
        'rgb': (255, 20, 0),  # RGB for ~630nm
        'wavelength': '630nm',
        'purpose': 'Photosynthesis, flowering',
        'intensity_range': (0.15, 0.35)
    },
    'blue': {
        'rgb': (0, 0, 255),  # RGB for ~450nm
        'wavelength': '450nm',
        'purpose': 'Vegetative growth, compactness',
        'intensity_range': (0.1, 0.25)
    },
    'purple_mix': {
        'rgb': (127, 0, 255),  # Combined red and blue
        'wavelength': '430-660nm mix',
        'purpose': 'Full growth cycle',
        'intensity_range': (0.15, 0.30)
    },
    'warm_white': {
        'rgb': (255, 244, 229),  # Full spectrum
        'wavelength': '380-700nm',
        'purpose': 'General purpose, inspection',
        'intensity_range': (0.1, 0.2)
    }
}

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
                           intensity: float,
                           light_info: dict) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add colored lighting effect."""
        color_layer = np.full_like(image, color)
        blend = cv2.addWeighted(image, 1, color_layer, intensity, 0)
        
        # Create detailed light information
        light_params = {
            'color': f"{light_info['name']} ({light_info['wavelength']})",
            'intensity': round(intensity, 2),
            'purpose': light_info['purpose']
        }
        
        return blend, light_params

    def generate_variations(self, num_variations: int = 8) -> List[Tuple[np.ndarray, AugmentationInfo]]:
        """Generate variations using only colored lighting effects."""
        variations = []
        for _ in range(num_variations):
            img = self.original.copy()
            info = AugmentationInfo()
            
            # Select random light type
            light_type = random.choice(list(GROW_LIGHT_COLORS.keys()))
            light_params = GROW_LIGHT_COLORS[light_type]
            
            # Get intensity within recommended range
            intensity = random.uniform(*light_params['intensity_range'])
            
            # Add light name to params
            light_params['name'] = light_type.replace('_', ' ').title()
            
            img, light_info = self.add_colored_lighting(
                img,
                light_params['rgb'],
                intensity,
                light_params
            )
            
            info.colored_light = light_info
            variations.append((img, info))
            
        return variations

    def format_augmentation_title(self, info: AugmentationInfo) -> str:
        """Format augmentation information into a title string."""
        if info.colored_light is not None:
            return f"Light: {info.colored_light['color']}\nIntensity: {info.colored_light['intensity']}\nPurpose: {info.colored_light['purpose']}"
        return "No augmentations applied"

    def display_variations(self, variations: List[Tuple[np.ndarray, AugmentationInfo]]):
        """Display original image and its variations in a grid with detailed titles."""
        n = len(variations) + 1
        grid_size = int(np.ceil(np.sqrt(n)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle('Indoor Farming Lighting Variations', fontsize=16)
        
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

# Usage
augmentor = ImageAugmentor("Leaf Spot/20250112_074254.jpg")
augmentor.process_and_display(num_variations=20)