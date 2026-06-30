import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataset import RoadDataset
from evaluate_baseline import load_model, skeleton_iou
from tta import tta_infer

def evaluate_tta_vs_baseline(model, dataloader, device):
    model.eval()
    
    baseline_y_true = []
    baseline_y_pred = []
    
    tta_y_true = []
    tta_y_pred = []
    
    skel_ious_base = []
    skel_ious_tta = []
    
    start_base = time.time()
    
    # --- Baseline Evaluation ---
    print("Running Baseline Inference...")
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).float()
            
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            baseline_y_true.append(y.cpu().numpy().flatten())
            baseline_y_pred.append(preds.cpu().numpy().flatten())
            
            for i in range(preds.shape[0]):
                p_np = preds[i, 0].cpu().numpy()
                y_np = y[i].cpu().numpy()
                try:
                    s_iou = skeleton_iou(p_np, y_np)
                    if not np.isnan(s_iou):
                        skel_ious_base.append(s_iou)
                except Exception:
                    pass
                    
    end_base = time.time()
    base_time = end_base - start_base
    
    # --- TTA Evaluation ---
    print("Running TTA Inference (8x slower)...")
    start_tta = time.time()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).float()
            
            mean_prob, uncertainty = tta_infer(model, x)
            preds = (mean_prob > 0.5).float()
            
            tta_y_true.append(y.cpu().numpy().flatten())
            tta_y_pred.append(preds.cpu().numpy().flatten())
            
            for i in range(preds.shape[0]):
                p_np = preds[i, 0].cpu().numpy()
                y_np = y[i].cpu().numpy()
                try:
                    s_iou = skeleton_iou(p_np, y_np)
                    if not np.isnan(s_iou):
                        skel_ious_tta.append(s_iou)
                except Exception:
                    pass
                    
            # Save the last batch's first image for visualization
            last_x = x[0].cpu().numpy()
            last_y = y[0].cpu().numpy()
            last_prob = mean_prob[0, 0].cpu().numpy()
            last_uncert = uncertainty[0, 0].cpu().numpy()
                    
    end_tta = time.time()
    tta_time = end_tta - start_tta
    
    # --- Metrics Computation ---
    def calc_iou(y_true, y_pred):
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        iou = tp / (tp + fp + fn + 1e-8)
        return iou
        
    base_iou = calc_iou(baseline_y_true, baseline_y_pred)
    tta_iou = calc_iou(tta_y_true, tta_y_pred)
    
    base_skel = np.mean(skel_ious_base) if skel_ious_base else 0
    tta_skel = np.mean(skel_ious_tta) if skel_ious_tta else 0
    
    print("\n============================================================")
    print("TTA EVALUATION REPORT")
    print("============================================================")
    print(f"Baseline IoU   : {base_iou:.4f} (Time: {base_time:.2f}s)")
    print(f"TTA IoU        : {tta_iou:.4f} (Time: {tta_time:.2f}s)")
    print(f"IoU Delta      : {tta_iou - base_iou:+.4f}")
    print(f"Base Skel IoU  : {base_skel:.4f}")
    print(f"TTA Skel IoU   : {tta_skel:.4f}")
    print(f"Skel IoU Delta : {tta_skel - base_skel:+.4f}")
    print("============================================================")
    
    # Save visual
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Optical T")
    opt = last_x[:3].transpose(1, 2, 0)
    opt = (opt - opt.min()) / (opt.max() - opt.min() + 1e-8)
    plt.imshow(opt)
    
    plt.subplot(1, 3, 2)
    plt.title("TTA Prediction")
    plt.imshow(last_prob, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("TTA Uncertainty Map")
    plt.imshow(last_uncert, cmap='inferno')
    
    plt.tight_layout()
    plt.savefig("test_tta.png")
    print("Saved visual to test_tta.png")

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
        
    ds = RoadDataset(num_tiles=20, split="val", augment=False, has_temporal=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    
    evaluate_tta_vs_baseline(model, loader, device)

if __name__ == "__main__":
    main()
