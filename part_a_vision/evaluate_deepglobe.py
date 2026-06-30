import os
import torch
from torch.utils.data import DataLoader

from deepglobe_dataset import DeepGlobeDataset
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

    print("\n--- Evaluating DeepGlobe Dataset ---")
    val_dataset = DeepGlobeDataset(root_dir="data/deepglobe", split="val", augment=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=None,
        save_visuals=False
    )
    
    iou = metrics.get("iou", 0.0)
    print("\n============================================================")
    print("DEEPGLOBE VALIDATION REPORT")
    print("============================================================")
    print(f"DeepGlobe Val: IoU={iou:.4f}, F1={metrics.get('f1', 0.0):.4f}, Skeleton IoU={metrics.get('skeleton_iou', 0.0):.4f}")
    
    if iou > 0.60:
        print("✅ Target Met! IoU > 0.60 on DeepGlobe.")
    else:
        print("⚠️ Model did not hit 0.60 IoU. Ensure model was trained with Focal Loss and AdaIN.")
    print("============================================================")

if __name__ == "__main__":
    main()
