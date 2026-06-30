import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
    args = parser.parse_args()

    device = args.device
    print(f"Loading model from {args.checkpoint}...")
    
    try:
        model = load_model(args.checkpoint, device=device, backbone=args.backbone)
    except Exception as e:
        print(f"Could not load model, probably because it's untrained locally. Error: {e}")
        return

    print("\n--- Evaluating WITH Temporal Channels (has_temporal=True) ---")
    ds_temporal = RoadDataset(
        num_tiles=50,
        split="val",
        augment=False,
        force_shadow=True,
        has_temporal=True
    )
    loader_temporal = DataLoader(ds_temporal, batch_size=4, shuffle=False)
    metrics_temporal = evaluate(
        model=model,
        dataloader=loader_temporal,
        device=device,
        output_dir=None,
        save_visuals=False
    )
    iou_temporal = metrics_temporal.get("iou", 0.0)
    print(f"IoU (Temporal): {iou_temporal:.4f}")

    print("\n--- Evaluating WITHOUT Temporal Channels (has_temporal=False) ---")
    ds_no_temporal = RoadDataset(
        num_tiles=50,
        split="val",
        augment=False,
        force_shadow=True,
        has_temporal=False
    )
    loader_no_temporal = DataLoader(ds_no_temporal, batch_size=4, shuffle=False)
    metrics_no_temporal = evaluate(
        model=model,
        dataloader=loader_no_temporal,
        device=device,
        output_dir=None,
        save_visuals=False
    )
    iou_no_temporal = metrics_no_temporal.get("iou", 0.0)
    print(f"IoU (No Temporal): {iou_no_temporal:.4f}")

    print("\n============================================================")
    print("SHADOW ROBUSTNESS REPORT")
    print("============================================================")
    print(f"Shadow IoU without temporal : {iou_no_temporal:.4f}")
    print(f"Shadow IoU with temporal    : {iou_temporal:.4f}")
    delta = iou_temporal - iou_no_temporal
    print(f"Shadow IoU Delta            : {delta:+.4f}")
    if delta > 0.05:
        print("✅ Multi-temporal reasoning successfully mitigates shadow occlusion!")
    else:
        print("⚠️ Temporal channels did not provide significant improvement.")
    print("============================================================")

if __name__ == "__main__":
    main()
