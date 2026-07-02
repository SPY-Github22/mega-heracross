import os
import sys
import math
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import osmnx as ox

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'part_a_vision'))
from model import SegformerB3Custom
from tta import tta_infer
from postprocess import apply_morphology

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def run():
    print("Fetching REAL satellite imagery for testing...")
    
    # Koramangala coordinates
    lat, lon = 12.9277, 77.6251
    z = 16
    x, y = deg2num(lat, lon, z)
    
    # Bounding box for OSMnx
    north, west = num2deg(x, y, z)
    south, east = num2deg(x+1, y+1, z)
    
    # 1. Download ESRI World Imagery Tile
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 2. Prepare 12-channel tensor
    fused = np.zeros((12, 512, 512), dtype=np.float32)
    fused[0] = img_np[:,:,2] # R
    fused[1] = img_np[:,:,1] # G
    fused[2] = img_np[:,:,0] # B
    # Other channels remain 0
    fused_tensor = torch.from_numpy(fused)
    
    # 3. Get Real OSMnx Ground Truth Mask
    print("Fetching OSMnx roads...")
    try:
        try:
            # OSMnx 2.0+
            G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
        except TypeError:
            # OSMnx < 2.0
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        fig, ax = ox.plot_graph(G, show=False, close=False, edge_color='white', edge_linewidth=2, node_size=0, bgcolor='black', figsize=(5.12, 5.12), dpi=100)
        ax.set_position([0, 0, 1, 1])
        fig.canvas.draw()
        gt_mask_img = np.array(fig.canvas.renderer.buffer_rgba())
        gt_mask = (gt_mask_img[:,:,0] > 128).astype(np.uint8)
        plt.close(fig)
    except Exception as e:
        print(f"OSMnx failed: {e}")
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        
    # 4. Model Setup
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerB3Custom(input_channels=12, num_classes=1).to(device)
    checkpoint_path = os.path.join("outputs", "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        if 'model.segformer.encoder.patch_embeddings.0.proj.weight' in state_dict:
            old_proj = state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight']
            if old_proj.shape[1] == 10:
                new_proj = torch.zeros((64, 12, 7, 7), dtype=old_proj.dtype, device=old_proj.device)
                new_proj[:, :4] = old_proj[:, :4]
                new_proj[:, 6:8] = old_proj[:, 4:6]
                new_proj[:, 8:12] = old_proj[:, 6:10]
                state_dict['model.segformer.encoder.patch_embeddings.0.proj.weight'] = new_proj
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint.")
    
    model.eval()
    
    # 5. Inference
    print("Running inference...")
    batch = fused_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        mean_prob, _ = tta_infer(model, batch)
    pred_prob = mean_prob.cpu().numpy()[0, 0]
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    final_mask = apply_morphology(pred_mask)
    
    # 6. Plotting
    BG = '#0D1117'; CARD = '#161B22'; WHITE = '#E6EDF3'
    plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': CARD, 'text.color': WHITE})
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Real Optical (ESRI Satellite)', color=WHITE)
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Real OSM Ground Truth', color=WHITE)
    axes[1].axis('off')
    
    axes[2].imshow(final_mask, cmap='Greens', vmin=0, vmax=1)
    axes[2].set_title('Predicted Mask', color=WHITE)
    axes[2].axis('off')
    
    diff = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    diff[..., 0] = ((final_mask == 1) & (gt_mask == 0)).astype(float)
    diff[..., 1] = ((final_mask == 0) & (gt_mask == 1)).astype(float)
    diff[..., 2] = ((final_mask == 1) & (gt_mask == 1)).astype(float)
    axes[3].imshow(diff)
    axes[3].set_title('Error Map', color=WHITE)
    axes[3].axis('off')
    
    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/real_satellite_prediction.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    print(f"Saved real visualization to: {out_path}")

if __name__ == '__main__':
    run()
