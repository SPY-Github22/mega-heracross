import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from part_a_vision.dataset import RoadDataset
from part_a_vision.model import SegformerB3Custom
from part_a_vision.postprocess import apply_morphology

def main():
    print("Generating PPT Visuals...")
    
    # 1. Load the dataset (90% cloud cover)
    ds = RoadDataset(
        tile_size=512, 
        num_tiles=1, 
        split="test", 
        augment=False, 
        cloud_level=0.9,
        force_occlusion=False,
        force_shadow=False
    )
    fused_tensor, gt_mask = ds[0]

    # Extract Optical (Clouds)
    green, red, nir = fused_tensor[0].numpy(), fused_tensor[1].numpy(), fused_tensor[2].numpy()
    optical_rgb = np.stack([red, green, nir], axis=-1)
    optical_rgb = (optical_rgb - optical_rgb.min()) / (optical_rgb.max() - optical_rgb.min() + 1e-6)

    # Extract SAR (Radar passing through clouds)
    # SAR VV and VH are channels 4 and 5 in the fused tensor (0-indexed)
    sar_vv = fused_tensor[4].numpy()
    sar_vh = fused_tensor[5].numpy()
    # Create a pseudo-color SAR image (VV, VH, VV/VH)
    sar_ratio = sar_vv / (sar_vh + 1e-6)
    sar_rgb = np.stack([sar_vv, sar_vh, sar_ratio], axis=-1)
    sar_rgb = (sar_rgb - sar_rgb.min()) / (sar_rgb.max() - sar_rgb.min() + 1e-6)

    # 2. Load the trained model for the POC Output
    device = torch.device("cpu")
    model = SegformerB3Custom(input_channels=12, num_classes=1).to(device)
    checkpoint_path = "outputs/best_checkpoint.pth"
    
    poc_mask = np.zeros((512, 512))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        with torch.no_grad():
            output = model(fused_tensor.unsqueeze(0))
            if isinstance(output, dict):
                output = output["out"]
            pred_prob = torch.sigmoid(output).numpy()[0, 0]
        poc_mask = apply_morphology((pred_prob > 0.5).astype(np.uint8))

    # 3. Plot a 4-panel PPT Slide
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(optical_rgb)
    axes[0].set_title("1. Optical (90% Clouds)", fontsize=14, fontweight='bold')
    axes[0].axis("off")
    
    axes[1].imshow(sar_rgb)
    axes[1].set_title("2. SAR (Sees through clouds)", fontsize=14, fontweight='bold')
    axes[1].axis("off")
    
    axes[2].imshow(gt_mask.numpy(), cmap='gray')
    axes[2].set_title("3. Target (Fully Trained)", fontsize=14, fontweight='bold')
    axes[2].axis("off")
    
    axes[3].imshow(poc_mask, cmap='magma')
    axes[3].set_title("4. Our POC (3 Min Training)", fontsize=14, fontweight='bold')
    axes[3].axis("off")
    
    plt.tight_layout()
    output_file = "ppt_visual_slide.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}! Drag and drop this into your PPT.")

if __name__ == "__main__":
    main()
