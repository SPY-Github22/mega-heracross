import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'part_a_vision'))
from dataset import RoadDataset
from model import SegformerB3Custom
from tta import tta_infer
from postprocess import apply_morphology

def run():
    print("Generating cloudy image vs prediction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset with cloud occlusion
    ds = RoadDataset(
        tile_size=512,
        num_tiles=1,
        split="test",
        augment=False,
        cloud_level=0.8,
        force_occlusion=True,
        force_shadow=True
    )
    
    fused_tensor, gt_mask = ds[0]
    fused_np = fused_tensor.numpy()
    gt_np = gt_mask.numpy()
    
    # 2. Model Setup
    model = SegformerB3Custom(input_channels=12, num_classes=1).to(device)
    checkpoint_path = os.path.join("outputs", "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            if 'model.segformer.encoder.patch_embeddings.0.proj.weight' in state_dict:
                old_proj_weight = state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight']
                if old_proj_weight.shape[1] == 10:
                    new_proj_weight = torch.zeros((64, 12, 7, 7), dtype=old_proj_weight.dtype, device=old_proj_weight.device)
                    new_proj_weight[:, :4] = old_proj_weight[:, :4]
                    new_proj_weight[:, 6:8] = old_proj_weight[:, 4:6]
                    new_proj_weight[:, 8:12] = old_proj_weight[:, 6:10]
                    state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight'] = new_proj_weight
            model.load_state_dict(state_dict)
            print(f"Loaded {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print("No checkpoint found, using random weights.")
    
    model.eval()
    
    # 3. Inference
    batch = fused_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        mean_prob, aleatoric_unc = tta_infer(model, batch)
        
    pred_prob = mean_prob.cpu().numpy()[0, 0]
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    
    aleatoric_unc_np = aleatoric_unc.cpu().numpy()[0, 0] if aleatoric_unc is not None else None
    final_mask = apply_morphology(pred_mask, uncertainty_map=aleatoric_unc_np)
    
    # 4. Plotting
    BG = '#0D1117'; CARD = '#161B22'; WHITE = '#E6EDF3'
    plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': CARD, 'text.color': WHITE})
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Optical (false color from fused channels)
    # fused_tensor has 6 channels (4 optical + 2 SAR) if cloud fusion fails/isn't there, or 12.
    # Assuming first 3 channels are RGB/Optical proxy
    if fused_np.shape[0] >= 3:
        fc = np.clip(np.stack([fused_np[2], fused_np[1], fused_np[0]], -1), 0, 1)
    else:
        fc = np.zeros((512, 512, 3))
    
    axes[0].imshow(fc)
    axes[0].set_title('Synthetic Cloudy Optical', color=WHITE)
    axes[0].axis('off')
    
    axes[1].imshow(gt_np, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Mask', color=WHITE)
    axes[1].axis('off')
    
    axes[2].imshow(final_mask, cmap='Greens', vmin=0, vmax=1)
    axes[2].set_title('Predicted Mask', color=WHITE)
    axes[2].axis('off')
    
    # Error Map
    diff = np.zeros((*gt_np.shape, 3), dtype=np.float32)
    diff[..., 0] = ((final_mask == 1) & (gt_np == 0)).astype(float)  # FP red
    diff[..., 1] = ((final_mask == 0) & (gt_np == 1)).astype(float)  # FN green
    diff[..., 2] = ((final_mask == 1) & (gt_np == 1)).astype(float)  # TP blue
    
    axes[3].imshow(diff)
    axes[3].set_title('Error Map (Blue=TP, Red=FP, Green=FN)', color=WHITE)
    axes[3].axis('off')
    
    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/cloudy_prediction_grid.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    print(f"Saved visualization to: {out_path}")

if __name__ == '__main__':
    run()
