import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from part_a_vision.dataset import RoadDataset
from part_a_vision.model import SegformerB3Custom
from part_a_vision.postprocess import apply_morphology

def main():
    print("Loading data and model...")
    # 1. Load the dataset (force heavy cloud occlusion to show the cloud-piercing)
    ds = RoadDataset(
        tile_size=512, 
        num_tiles=1, 
        split="test", 
        augment=False, 
        cloud_level=0.9,  # 90% cloud cover
        force_occlusion=False,
        force_shadow=False
    )
    fused_tensor, gt_mask = ds[0]

    # Extract the RGB optical bands for visualization (first 3 channels: Green, Red, NIR)
    # We'll map them loosely to RGB for a generic display
    green = fused_tensor[0].numpy()
    red = fused_tensor[1].numpy()
    nir = fused_tensor[2].numpy()
    
    # Create a pseudo-RGB image
    rgb = np.stack([red, green, nir], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    # 2. Load the trained model
    device = torch.device("cpu")
    model = SegformerB3Custom(input_channels=12, num_classes=1).to(device)
    
    checkpoint_path = "outputs/best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Strip 'model.' from the state dict keys (since train_custom wrapped it)
        state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model.'):
                state_dict[k[6:]] = v
            else:
                state_dict[k] = v
                
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    else:
        print("Model checkpoint not found. Ensure it is at outputs/best_checkpoint.pth")
        return

    model.eval()
    
    # 3. Run Inference
    with torch.no_grad():
        batch = fused_tensor.unsqueeze(0)
        output = model(batch)
        if isinstance(output, dict):
            output = output["out"]
        pred_prob = torch.sigmoid(output).numpy()[0, 0]
        
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    final_mask = apply_morphology(pred_mask)

    # 4. Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title("Optical Imagery (90% Cloud Cover)")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask.numpy(), cmap='gray')
    axes[1].set_title("Ground Truth (Actual Roads)")
    axes[1].axis("off")
    
    axes[2].imshow(final_mask, cmap='magma')
    axes[2].set_title("AI Prediction (Cloud Pierced via SAR)")
    axes[2].axis("off")
    
    plt.tight_layout()
    print("Opening visualizer window...")
    plt.show()

if __name__ == "__main__":
    main()
