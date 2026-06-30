import os
import sys
import json
import numpy as np

def verify_contract(output_dir="outputs"):
    print("=== Part B Contract Verification Simulator ===")
    
    mask_path = os.path.join(output_dir, "road_mask.npy")
    meta_path = os.path.join(output_dir, "meta.json")
    
    # 1. Check files exist
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing {mask_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
        
    print("✓ Files found")
    
    # 2. Check meta.json schema
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    required_keys = ["crs", "bbox", "resolution_m", "source"]
    for k in required_keys:
        if k not in meta:
            raise ValueError(f"meta.json missing required key: {k}")
            
    if meta["crs"] != "EPSG:4326":
        raise ValueError(f"Invalid CRS: {meta['crs']} (expected EPSG:4326)")
        
    if not isinstance(meta["bbox"], list) or len(meta["bbox"]) != 4:
        raise ValueError("Invalid bbox format")
        
    print("✓ meta.json schema validated")
    
    # 3. Check road_mask.npy properties
    mask = np.load(mask_path)
    
    if mask.dtype != np.uint8:
        raise ValueError(f"Invalid dtype: {mask.dtype} (expected uint8)")
        
    if mask.ndim != 2:
        raise ValueError(f"Invalid shape: {mask.shape} (expected 2D array)")
        
    unique_vals = np.unique(mask)
    for val in unique_vals:
        if val not in [0, 1]:
            raise ValueError(f"Invalid value in mask: {val} (expected only 0 or 1)")
            
    print(f"✓ road_mask.npy validated (shape: {mask.shape}, dtype: {mask.dtype})")
    print("\n✓ Part A -> Part B Contract: PASSED")
    return True

if __name__ == "__main__":
    verify_contract(os.path.join(os.path.dirname(__file__), "outputs"))
