import cv2
import numpy as np
import albumentations as A

def erode_mask(mask, **kwargs):
    """
    Morphological erosion to thin the roads in the ground truth mask.
    This penalizes the model if it predicts 6m wide roads (like DeepGlobe)
    and forces it to learn narrower 3-4m Indian roads.
    """
    kernel = np.ones((3, 3), np.uint8)
    # Mask is expected to be {0, 1}
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(mask.dtype)

def get_bengaluru_transforms():
    """
    Phase 21: Bengaluru Domain Adaptation Pipeline
    Simulates LISS-IV satellite imagery (5.8m resolution) over chaotic Indian roads
    starting from DeepGlobe 0.5m high-resolution Western data.
    """
    return A.Compose([
        # 1. Spatial Transforms (Intersection density variation)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # 2. Road Thinning (Task 2)
        # Erodes the Ground Truth mask heavily to simulate narrow 3-4m lanes
        A.Lambda(mask=erode_mask, p=0.7),
        
        # 3. Resolution Gap Bridging (Task 4)
        # LISS-IV is 5.8m/px, DeepGlobe is 0.5m/px. Scale = 0.5/5.8 = 0.086
        # Downscale by ~0.1x to simulate the severe pixelation, then upscale back
        A.Downscale(scale_min=0.08, scale_max=0.12, interpolation=cv2.INTER_NEAREST, p=0.8),
        
        # 4. Texture Mixing & Degradation (Task 2)
        # Add heavy noise and color shifts to break the pristine Western road texture
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ImageCompression(quality_range=(40, 80), p=0.5),
        
        # Elastic deformation for irregular, non-grid curves
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    ])
