import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from dataset import RoadDataset
from postprocess import apply_morphology

import sys

# Hack for import, since I fixed it in evaluate_tta but maybe not here
sys.path.append("d:\\BAH\\mega\\part_a_vision")
from evaluate_baseline import skeleton_iou

def inject_network_noise(gt_mask: np.ndarray, seed: int) -> tuple:
    """
    Simulates what a raw, untrained neural network output might look like.
    Adds salt & pepper noise, punches small holes, and creates topological breaks.
    Returns: (noisy_mask, mock_uncertainty_map)
    """
    rng = np.random.RandomState(seed)
    noisy = gt_mask.copy().astype(float)
    
    # 1. Topological breaks (gaps of ~2-3 pixels)
    h, w = noisy.shape
    for _ in range(50):
        y, x = rng.randint(5, h-5), rng.randint(5, w-5)
        noisy[y-1:y+2, x-1:x+2] = 0.0 # 3x3 gap
        
    # 2. Holes inside roads
    for _ in range(100):
        y, x = rng.randint(0, h), rng.randint(0, w)
        if noisy[y, x] == 1.0:
            noisy[y, x] = 0.0
            
    # 3. False positives (salt noise)
    noise_mask = rng.rand(h, w) < 0.01
    noisy[noise_mask] = 1.0
    
    # Create a mock uncertainty map where false positives are highly uncertain
    uncertainty = rng.uniform(0.0, 0.2, size=(h, w))
    uncertainty[noise_mask] = rng.uniform(0.4, 0.8, size=noise_mask.sum())
    
    return noisy.astype(np.uint8), uncertainty

def calc_iou(pred, gt):
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-8)

def main():
    print("Initializing Synthetic Network Output...")
    ds = RoadDataset(num_tiles=25, split="val", augment=False)
    
    all_raw_ious = []
    all_post_ious = []
    
    all_raw_ccs = []
    all_post_ccs = []
    
    for i in range(5):
        _, gt_tensor = ds[i]
        gt = gt_tensor.numpy()
        
        raw, uncert = inject_network_noise(gt, seed=42+i)
        
        post = apply_morphology(raw, uncertainty_map=uncert)
        
        # Metrics
        raw_iou = calc_iou(raw, gt)
        post_iou = calc_iou(post, gt)
        
        _, raw_cc = label(raw)
        _, post_cc = label(post)
        _, gt_cc = label(gt)
        
        all_raw_ious.append(raw_iou)
        all_post_ious.append(post_iou)
        all_raw_ccs.append(raw_cc)
        all_post_ccs.append(post_cc)
        
        if i == 0:
            # Save visual
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Ground Truth")
            plt.imshow(gt, cmap='gray')
            
            plt.subplot(1, 3, 2)
            plt.title(f"Raw Output (CC={raw_cc}, IoU={raw_iou:.3f})")
            plt.imshow(raw, cmap='gray')
            
            plt.subplot(1, 3, 3)
            plt.title(f"Post-Processed (CC={post_cc}, IoU={post_iou:.3f})")
            plt.imshow(post, cmap='gray')
            
            plt.tight_layout()
            plt.savefig("test_morphology.png")
            print("Saved visual to test_morphology.png")
            
    print("\n============================================================")
    print("MORPHOLOGY EVALUATION REPORT")
    print("============================================================")
    print(f"Mean Raw IoU       : {np.mean(all_raw_ious):.4f}")
    print(f"Mean Post IoU      : {np.mean(all_post_ious):.4f}")
    print(f"Mean Raw CC count  : {np.mean(all_raw_ccs):.1f}")
    print(f"Mean Post CC count : {np.mean(all_post_ccs):.1f}")
    print("============================================================")
    
if __name__ == "__main__":
    main()
