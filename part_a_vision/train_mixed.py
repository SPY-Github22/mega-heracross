import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import time

from dataset import RoadDataset
from deepglobe_dataset import DeepGlobeDataset
from model import SegformerB3Custom
from loss import CombinedLoss

def adain(content_batch, style_batch):
    """
    Adaptive Instance Normalization (AdaIN)
    Aligns the mean and variance of the content_batch to match the style_batch.
    Applied only to the optical RGB channels (0-2) to transfer DeepGlobe style to Koramangala.
    """
    # content: Koramangala (B, 10, H, W)
    # style: DeepGlobe (B, 10, H, W)
    c_opt = content_batch[:, :3, :, :]
    s_opt = style_batch[:, :3, :, :]
    
    # Calculate mean and std over spatial dimensions (H, W)
    mu_c = c_opt.mean(dim=(2, 3), keepdim=True)
    std_c = c_opt.std(dim=(2, 3), keepdim=True) + 1e-8
    
    mu_s = s_opt.mean(dim=(2, 3), keepdim=True)
    std_s = s_opt.std(dim=(2, 3), keepdim=True) + 1e-8
    
    # Apply style transfer
    styled_opt = (c_opt - mu_c) / std_c * std_s + mu_s
    styled_opt = torch.clamp(styled_opt, 0.0, 1.0)
    
    # Reconstruct tensor
    styled_batch = content_batch.clone()
    styled_batch[:, :3, :, :] = styled_opt
    return styled_batch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Checkpoint (Phase 11)
    model = SegformerB3Custom(input_channels=10, num_classes=1).to(device)
    ckpt_path = "outputs/baseline/best_model.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded synthetic checkpoint.")
    else:
        print("No synthetic checkpoint found. Starting from ImageNet weights.")
        
    # Freeze encoder for first 10 epochs
    print("Freezing encoder for Phase 15 initial fine-tuning...")
    for name, param in model.named_parameters():
        if "decode_head" not in name:
            param.requires_grad = False
            
    # 2. Datasets
    dg_dataset = DeepGlobeDataset(root_dir="data/deepglobe", split="train", augment=True)
    kora_dataset = RoadDataset(num_tiles=100, split="train", augment=True, has_temporal=True)
    
    dg_loader = DataLoader(dg_dataset, batch_size=2, shuffle=True)
    kora_loader = DataLoader(kora_dataset, batch_size=2, shuffle=True)
    
    # 3. Loss & Optimizer (Focal Loss active!)
    criterion = CombinedLoss(dice_weight=0.4, bce_weight=0.3, boundary_weight=0.2, conn_weight=0.1, use_focal=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
    # 4. Mixed Training Loop (Mock)
    epochs = 2
    print("\nStarting Mixed Training (DeepGlobe + Styled Koramangala)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Zip loaders to get mixed batches
        kora_iter = iter(kora_loader)
        
        for i, (dg_inputs, dg_masks) in enumerate(dg_loader):
            try:
                kora_inputs, kora_masks = next(kora_iter)
            except StopIteration:
                kora_iter = iter(kora_loader)
                kora_inputs, kora_masks = next(kora_iter)
                
            dg_inputs, dg_masks = dg_inputs.to(device), dg_masks.to(device).float().unsqueeze(1)
            kora_inputs, kora_masks = kora_inputs.to(device), kora_masks.to(device).float().unsqueeze(1)
            
            # Domain Adaptation: Style Koramangala to look like DeepGlobe
            kora_styled = adain(kora_inputs, dg_inputs)
            
            # Concatenate batches
            inputs = torch.cat([dg_inputs, kora_styled], dim=0)
            masks = torch.cat([dg_masks, kora_masks], dim=0)
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs)
            loss, _ = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
                
            if i >= 20: # Short mock run
                break
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / 21:.4f}")
        
    print("Mixed training complete!")

if __name__ == "__main__":
    main()
