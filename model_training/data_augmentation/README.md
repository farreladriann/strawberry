# Data Augmentation

This directory contains various image augmentation implementations for strawberry disease detection.

## Files Overview

### Color and Lighting Augmentation

#### `hsv.py` and `hsv2.py`
- Basic HSV color space augmentation
- `hsv.py`: Conservative HSV adjustments (h_gain=0.015)
- `hsv2.py`: More aggressive HSV adjustments (h_gain=0.2)
- Functions:
  - `augment_hsv()`: Modifies hue, saturation, and value
  - Includes visualization capabilities

#### `lighting_augment.py`
- Basic lighting condition simulation
- Focuses on red (630-660nm) and purple light spectrums
- Suitable for indoor farming lighting conditions
- Features:
  - Basic colored lighting effects
  - Visualization tools

#### `many_light.py`
- Advanced lighting simulation for indoor farming
- Implements multiple grow light types:
  - Purple mix (430-660nm)
  - Warm white (380-700nm)
  - Different intensity ranges
- Includes detailed information about light purposes

#### `temp_light.py`
- Temperature-based lighting adjustments
- Simulates warm/cool lighting conditions
- Combines color temperature with lighting effects

#### `ligaugsave.py`
- Similar to lighting_augment.py but with save functionality
- Saves augmented images to disk
- Organized output directory structure

### Advanced Augmentation

#### `augment.py`
- Comprehensive augmentation suite
- Features:
  - Color temperature adjustment
  - Color jittering
  - Channel shifting
  - Colored lighting effects
  - Complex variation generation
  - Visualization tools

## Usage

Each script can be run independently. Most scripts include example usage at the bottom of the file. For example:

```python
# Basic HSV augmentation
python hsv.py

# Advanced lighting simulation
augmentor = ImageAugmentor("path/to/image.jpg")
augmentor.process_and_display(num_variations=20)
```

## Dependencies
- OpenCV (cv2)
- NumPy
- Matplotlib
- PyTorch (for training scripts)
- Ultralytics YOLO