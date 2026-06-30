import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part_a_vision.dataset import RoadDataset
from part_a_vision.model import SegformerB3Custom
from part_a_vision.tta import tta_infer
from part_a_vision.postprocess import apply_morphology
from part_a_vision.output_writer import write_road_mask, write_meta

def run_pipeline(mode="synthetic", occlusion="none", output_dir="outputs"):
    print("======================================================")
    print("  Part A: Vision & Occlusion Engine (Integration)     ")
    print(f"  Mode: {mode} | Occlusion: {occlusion}")
    print("======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset & Occlusion Setup
    cloud_level = 0.0
    force_occlusion = False
    force_shadow = False
    
    if occlusion == "cloud" or occlusion == "all":
        cloud_level = 0.8
    if occlusion == "canopy" or occlusion == "all":
        force_occlusion = True
    if occlusion == "shadow" or occlusion == "all":
        force_shadow = True
        
    ds = RoadDataset(
        tile_size=512,
        num_tiles=1,
        split="test",
        augment=False,
        cloud_level=cloud_level,
        force_occlusion=force_occlusion,
        force_shadow=force_shadow
    )
    
    fused_tensor, gt_mask = ds[0]
    
    # 2. Model Setup
    print("Loading 12-channel SegFormer model...")
    model = SegformerB3Custom(input_channels=12, num_classes=1).to(device)
    
    # We load the weights from best_checkpoint if available, else we just run initialized
    checkpoint_path = os.path.join("outputs", "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            # Adaptation for 12 channels (from 10)
            if 'model.segformer.encoder.patch_embeddings.0.proj.weight' in state_dict:
                old_proj_weight = state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight']
                if old_proj_weight.shape[1] == 10:
                    new_proj_weight = torch.zeros((64, 12, 7, 7), dtype=old_proj_weight.dtype, device=old_proj_weight.device)
                    new_proj_weight[:, :4] = old_proj_weight[:, :4]
                    new_proj_weight[:, 6:8] = old_proj_weight[:, 4:6]
                    new_proj_weight[:, 8:12] = old_proj_weight[:, 6:10]
                    state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight'] = new_proj_weight
                
            model.load_state_dict(state_dict)
            print("✓ Loaded pre-trained weights")
        except Exception as e:
            print(f"⚠ Failed to load weights ({e}). Running with random weights.")
    else:
        print("⚠ best_checkpoint.pth not found. Running with random weights for integration test.")
        
    model.eval()
    
    # 3. TTA Inference
    print("Running Test-Time Augmentation (TTA)...")
    batch = fused_tensor.unsqueeze(0).to(device)
    
    # For integration test we just simulate the uncertainty output to save compute
    # In real deployment we'd run MC Dropout. Here we just run TTA for standard prediction.
    with torch.no_grad():
        mean_prob, aleatoric_unc = tta_infer(model, batch)
        
    # Convert tensor to numpy
    pred_prob = mean_prob.cpu().numpy()[0, 0]
    
    # Base thresholding
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    
    # 4. Post-processing
    print("Applying Morphological Post-Processing...")
    aleatoric_unc_np = aleatoric_unc.cpu().numpy()[0, 0] if aleatoric_unc is not None else None
    final_mask = apply_morphology(pred_mask, uncertainty_map=aleatoric_unc_np)
    
    # Evaluate IoU vs Ground Truth
    gt_np = gt_mask.numpy()
    intersection = np.logical_and(final_mask == 1, gt_np == 1).sum()
    union = np.logical_or(final_mask == 1, gt_np == 1).sum()
    iou = intersection / (union + 1e-6)
    print(f"Internal IoU Metric: {iou:.4f}")
    
    # 5. Contract Output
    print("Writing Contract Outputs...")
    # Bbox for Koramangala test area
    bbox = [77.6101, 12.9177, 77.6401, 12.9377]
    mask_path = os.path.join(output_dir, "road_mask.npy")
    meta_path = os.path.join(output_dir, "meta.json")
    write_road_mask(final_mask, path=mask_path)
    write_meta(crs="EPSG:4326", bbox=bbox, resolution_m=5.8, source="LISS-IV", path=meta_path)
    print(f"✓ Part A complete. Output saved to {output_dir}/road_mask.npy")
    
    return iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="synthetic")
    parser.add_argument("--occlusion", type=str, default="none", choices=["none", "cloud", "canopy", "shadow", "all"])
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.occlusion)
