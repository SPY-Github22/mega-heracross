import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Use local modules
import sys
sys.path.append(os.path.dirname(__file__))

from dataset import RoadDataset
from evaluate_baseline import load_model, evaluate

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model")
    parser.add_argument("--backbone", type=str, default="segformer_b3")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_clouds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device, backbone=args.backbone)

    cloud_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    ious = []

    for level in cloud_levels:
        print(f"Evaluating at cloud level: {level * 100:.0f}%...")
        
        ds = RoadDataset(
            num_tiles=50,  # Evaluate on a subset for speed
            split="val",
            augment=False,
            cloud_level=level
        )
        
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        
        metrics = evaluate(
            model=model,
            dataloader=loader,
            device=device,
            output_dir=None,
            save_visuals=False
        )
        
        ious.append(metrics.get("iou", 0.0))
        print(f"  -> IoU: {ious[-1]:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot([c * 100 for c in cloud_levels], ious, marker='o', linewidth=2, color='blue', label='Dynamic SAR Fusion (Phase 13)')
    
    plt.title("Robustness to Cloud Cover")
    plt.xlabel("Cloud Cover (%)")
    plt.ylabel("Road Segmentation IoU")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, max(max(ious) + 0.1, 1.0))
    
    plot_path = os.path.join(args.output_dir, "cloud_robustness_curve.png")
    plt.savefig(plot_path)
    print(f"Saved robustness curve to {plot_path}")

if __name__ == "__main__":
    main()
