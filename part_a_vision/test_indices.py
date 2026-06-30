import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part_a_vision.dataset import RoadDataset

def test_spectral_indices():
    print("--- Phase 22: Testing Spectral Indices (NDVI & NDWI) ---")
    
    # Load dataset
    ds = RoadDataset(
        tile_size=512,
        num_tiles=10,
        split="train",
        augment=False,
    )
    
    # Get a sample
    # fused is (C, H, W)
    fused, mask = ds[0]
    fused = fused.numpy()
    
    # In Phase 22, fused has 12 channels:
    # 0: Green, 1: Red, 2: NIR, 3: SWIR
    # 4: NDVI, 5: NDWI
    # 6: VV, 7: VH
    # 8-11: Temporal Diff
    
    print(f"Fused tensor shape: {fused.shape}")
    assert fused.shape[0] == 12, f"Expected 12 channels, got {fused.shape[0]}"
    
    green = fused[0]
    red = fused[1]
    nir = fused[2]
    
    ndvi = fused[4]
    ndwi = fused[5]
    
    # We will plot the RGB proxy (using SWIR/NIR/Red since true Blue is missing in LISS-IV)
    # LISS-IV standard false color is NIR, Red, Green
    false_color = np.stack([nir, red, green], axis=-1)
    # Normalize for display
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min() + 1e-6)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 22 LISS-IV Band Engineering (Spectral Indices)", fontsize=16)
    
    axs[0].imshow(false_color)
    axs[0].set_title("LISS-IV False Color (NIR/R/G)")
    axs[0].axis('off')
    
    im1 = axs[1].imshow(ndvi, cmap='YlGn')
    axs[1].set_title("NDVI (Vegetation)")
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    im2 = axs[2].imshow(ndwi, cmap='Blues')
    axs[2].set_title("NDWI (Water)")
    axs[2].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), "test_indices.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_spectral_indices()
