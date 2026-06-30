#!/usr/bin/env python3
"""
Phase 21: Bengaluru Domain Adaptation Fine-Tuning
=================================================
This script takes the Phase 16 model (trained on high-res, pristine data)
and fine-tunes it on the Bengaluru-specific augmentation pipeline.

Key adaptations:
1. Loads Phase 16 `best_checkpoint.pth`
2. Uses `get_bengaluru_transforms()` from `bengaluru_transforms.py`
3. Freezes the encoder to prevent catastrophic forgetting.
4. Uses a very low learning rate (1e-6) on the decoder.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoadDataset, get_class_weights
from model import SegformerB3Custom
from bengaluru_transforms import get_bengaluru_transforms

def train_bengaluru(epochs=10, batch_size=4, lr=1e-6, checkpoint_path="outputs/best_checkpoint.pth"):
    print("--- Phase 21: Bengaluru Domain Adaptation Fine-Tuning ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please run Phase 16 training first.")
        return
        
    print("Building datasets with Bengaluru Domain Adaptation...")
    train_ds = RoadDataset(
        tile_size=512,
        num_tiles=800,
        split="train",
        augment=True,
        custom_transform=get_bengaluru_transforms()
    )
    
    val_ds = RoadDataset(
        tile_size=512,
        num_tiles=200,
        split="val",
        augment=False # Validation stays clean or matches target domain
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print("Loading model and weights...")
    # Phase 22: Model now takes 12 channels (4 Opt + 2 Idx + 2 SAR + 4 Temp)
    model = SegformerB3Custom(
        input_channels=12,
        num_classes=1
    ).to(device)
    
    # Load Phase 16 weights (which was trained with 10 channels)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Surgical weight transfer for the first conv layer (patch_embeddings[0].proj)
    # Phase 16 shape: (64, 10, 7, 7)
    # Phase 22 shape: (64, 12, 7, 7)
    old_proj_weight = state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight']
    new_proj_weight = model.model.segformer.encoder.patch_embeddings[0].proj.weight.data
    
    # Copy the first 4 channels (Optical)
    new_proj_weight[:, :4] = old_proj_weight[:, :4]
    
    # The next 2 channels (4, 5) are the new NDVI/NDWI indices. 
    # They are already randomly initialized by the SegformerB3Custom constructor, we leave them.
    
    # Copy the SAR channels (which were at 4,5 and move to 6,7)
    new_proj_weight[:, 6:8] = old_proj_weight[:, 4:6]
    
    # Copy the Temporal channels (which were at 6-9 and move to 8-11)
    new_proj_weight[:, 8:12] = old_proj_weight[:, 6:10]
    
    # Update the state_dict with the newly surgically-spliced weights
    state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight'] = new_proj_weight
    
    model.load_state_dict(state_dict)
    print(f"Loaded and adapted weights from {checkpoint_path}")
    
    print("Freezing encoder to prevent catastrophic forgetting...")
    # Freeze the encoder base
    for param in model.model.segformer.encoder.parameters():
        param.requires_grad = False
        
    # Only decoder (and fusion neck) will be trained
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=get_class_weights().to(device))
    
    print(f"Starting Fine-Tuning for {epochs} epochs with LR={lr}...")
    
    best_loss = float('inf')
    os.makedirs("outputs", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for opt_sar, mask in pbar:
            opt_sar, mask = opt_sar.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(opt_sar)
            
            # mask is (B, H, W) int64, logits is (B, 1, H, W)
            mask_f = mask.unsqueeze(1).float()
            loss = criterion(logits, mask_f)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * opt_sar.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        # VAL
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for opt_sar, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                opt_sar, mask = opt_sar.to(device), mask.to(device)
                logits = model(opt_sar)
                mask_f = mask.unsqueeze(1).float()
                loss = criterion(logits, mask_f)
                val_loss += loss.item() * opt_sar.size(0)
                
        val_loss /= len(val_ds)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = "outputs/bengaluru_adapted_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }, save_path)
            print(f"  -> Saved improved model to {save_path}")

    print("--- Domain Adaptation Complete ---")

if __name__ == "__main__":
    train_bengaluru()
