import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part_a_vision.dataset import RoadDataset
from part_a_vision.bengaluru_transforms import get_bengaluru_transforms

def test_bengaluru_augmentations():
    print("--- Phase 21: Testing Bengaluru Augmentation Pipeline ---")
    
    # 1. Load the original standard dataset (DeepGlobe style)
    print("Loading pristine dataset...")
    clean_ds = RoadDataset(
        tile_size=512,
        num_tiles=10,
        split="train",
        augment=False, # pristine
    )
    
    # 2. Load the dataset with Bengaluru transforms
    print("Loading Bengaluru-adapted dataset...")
    bengaluru_ds = RoadDataset(
        tile_size=512,
        num_tiles=10,
        split="train",
        augment=True,
        custom_transform=get_bengaluru_transforms()
    )
    
    # Get a deterministic sample
    idx = 4 # Arbitrary tile
    
    # Clean
    # Remember: __getitem__ returns (fused_tensor, mask_tensor)
    # fused_tensor is (C, H, W)
    clean_fused, clean_mask = clean_ds[idx]
    clean_opt = clean_fused[:3].numpy().transpose(1, 2, 0) # RGB
    clean_mask = clean_mask.numpy()
    
    # Augmented
    aug_fused, aug_mask = bengaluru_ds[idx]
    aug_opt = aug_fused[:3].numpy().transpose(1, 2, 0)
    aug_mask = aug_mask.numpy()
    
    print(f"Clean road pixels: {clean_mask.sum()}")
    print(f"Augmented road pixels: {aug_mask.sum()}")
    if aug_mask.sum() < clean_mask.sum():
        print("SUCCESS: Road thinning (erosion) applied successfully.")
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Phase 21 Domain Adaptation: DeepGlobe (Top) vs Bengaluru LISS-IV (Bottom)", fontsize=14)
    
    axs[0, 0].imshow(clean_opt)
    axs[0, 0].set_title("Pristine Optical (0.5m)")
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(clean_mask, cmap='gray')
    axs[0, 1].set_title("Pristine Ground Truth (6m wide)")
    axs[0, 1].axis("off")
    
    axs[1, 0].imshow(aug_opt)
    axs[1, 0].set_title("Degraded Optical (simulated 5.8m, noisy)")
    axs[1, 0].axis("off")
    
    axs[1, 1].imshow(aug_mask, cmap='gray')
    axs[1, 1].set_title("Eroded Ground Truth (3-4m wide)")
    axs[1, 1].axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "bengaluru_aug_test.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_bengaluru_augmentations()
