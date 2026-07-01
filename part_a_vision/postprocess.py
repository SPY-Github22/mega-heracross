import numpy as np
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes, label

def apply_morphology(binary_mask: np.ndarray, uncertainty_map: np.ndarray = None, 
                     base_min_area: int = 20, uncertain_min_area: int = 100, 
                     uncertainty_threshold: float = 0.3) -> np.ndarray:
    """
    Phase 18: Morphological Post-processing Pipeline
    Cleans the raw neural network output to prepare it for Phase 20 (Skeletonization).
    
    Args:
        binary_mask: (H, W) boolean or uint8 mask (0 = bg, 1 = road).
        uncertainty_map: (H, W) float map of standard deviation from TTA.
        base_min_area: Minimum area (pixels) for confident road blobs.
        uncertain_min_area: Minimum area (pixels) for highly uncertain road blobs.
        uncertainty_threshold: Threshold above which a blob is considered "uncertain".
        
    Returns:
        Cleaned binary mask (H, W) uint8.
    """
    # Ensure boolean
    mask = binary_mask.astype(bool)
    
    # Task 1: Binary Closing (heal 3px breaks)
    # This connects road segments that were just barely broken by a few pixels of noise
    mask = closing(mask, disk(3))
    
    # Task 3: Smart Hole filling
    # Enclosed background pixels within a solid road are noise (small holes).
    # But massive enclosed backgrounds (city blocks bounded by ring roads) are NOT holes!
    # We invert the mask, find connected components, and only fill the small ones.
    bg_mask = ~mask
    labeled_bg, num_bg = label(bg_mask)
    for i in range(1, num_bg + 1):
        bg_pixels = (labeled_bg == i)
        if bg_pixels.sum() < 200: # only fill small holes
            mask[bg_pixels] = True
    
    # Task 2: Uncertainty-Guided Component Filtering
    # Label connected components
    labeled_mask, num_features = label(mask)
    
    cleaned_mask = np.zeros_like(mask)
    
    for i in range(1, num_features + 1):
        component_pixels = (labeled_mask == i)
        area = component_pixels.sum()
        
        # Determine dynamic deletion threshold
        threshold = base_min_area
        if uncertainty_map is not None:
            mean_uncertainty = uncertainty_map[component_pixels].mean()
            if mean_uncertainty > uncertainty_threshold:
                threshold = uncertain_min_area
                
        # Keep if it meets the dynamic threshold
        if area >= threshold:
            cleaned_mask[component_pixels] = True
            
    return cleaned_mask.astype(np.uint8)
