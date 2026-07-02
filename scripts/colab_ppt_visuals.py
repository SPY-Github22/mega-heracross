"""
COLAB VISUALIZATION CELL
Paste this as a new cell at the end of mega_heracross_topology_diversity.ipynb
and run it after Cell 9 (pipeline eval).

Generates PPT-ready images:
  - item_pred_vs_gt_grid.png     : 5-tile pred/GT/error comparison
  - item_skeleton_map.png        : skeleton graph overlaid on Koramangala
  - item_region_masks_grid.png   : 10-region GT mask diversity (already saved in Cell 5)
  - item_kora_density_hist.png   : density histogram across all tiles
"""

# ── Prerequisites: run after Cell 9 (pipeline eval) ──
# model, region_masks, TRAIN_REGIONS, TEST_REGIONS,
# make_diverse_tile, compute_metrics, G_skel, pred_mask_k must be in scope

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
import osmnx as ox

BG   = '#0D1117'; CARD = '#161B22'
ACC1 = '#58A6FF'; ACC2 = '#3FB950'; ACC3 = '#F78166'; ACC4 = '#D2A8FF'; GREY = '#8B949E'; WHITE = '#E6EDF3'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': CARD, 'text.color': WHITE,
                     'axes.labelcolor': WHITE, 'xtick.color': GREY, 'ytick.color': GREY})

OUT_DIR = Path('part_a_vision/outputs/calibrated_training')
DRIVE_OUT = '/content/drive/MyDrive/mega-heracross-colab-output'
import os; os.makedirs(DRIVE_OUT, exist_ok=True)


# ════════════════════════════════════════════
# VIZ 1: Pred vs GT comparison grid (5 tiles)
# ════════════════════════════════════════════
model.eval()
fig = plt.figure(figsize=(20, 26), facecolor=BG)
fig.suptitle('Part A: Predicted Road Mask vs Ground Truth\n(SegFormer B3, 10-Region Diverse Training)',
             fontsize=16, color=WHITE, fontweight='bold', y=0.98)

# Mix training + held-out regions for variety
sample_regions = (list(TRAIN_REGIONS[:3]) + list(TEST_REGIONS))[:5]
gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.1)

with torch.no_grad():
    for row_i, rname in enumerate(sample_regions):
        gt_mask_r = region_masks[rname]
        fused_r, gt_r = make_diverse_tile(gt_mask_r, seed=55000 + row_i)
        out_r = model(torch.from_numpy(fused_r).unsqueeze(0).to(device))
        if isinstance(out_r, dict): out_r = out_r['out']
        prob_r = torch.sigmoid(out_r).squeeze().cpu().numpy()
        pred_r = (prob_r > 0.5).astype(np.uint8)
        gt_r   = np.array(gt_r).astype(np.uint8)

        tag = '[HELD-OUT]' if rname in TEST_REGIONS else '[TRAIN]'
        gt_pct   = gt_r.sum()   / gt_r.size   * 100
        pred_pct = pred_r.sum() / pred_r.size * 100
        m = compute_metrics(out_r, torch.from_numpy(gt_r).float().to(device))

        # Col 0: false-colour optical (G R NIR)
        fc = np.clip(np.stack([fused_r[2], fused_r[1], fused_r[0]], -1), 0, 1)
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(fc); ax0.axis('off')
        ax0.set_title(f'{rname} {tag}\nSynthetic Optical', fontsize=9, color=WHITE)

        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(gt_r, cmap='Blues', vmin=0, vmax=1)
        ax1.axis('off')
        ax1.set_title(f'Ground Truth\n{gt_pct:.1f}% road', fontsize=9, color=WHITE)

        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(pred_r, cmap='Greens', vmin=0, vmax=1)
        ax2.axis('off')
        ax2.set_title(f'Predicted Mask\n{pred_pct:.1f}% road  IoU={m["iou"]:.3f}', fontsize=9, color=WHITE)

        diff = np.zeros((*gt_r.shape, 3), dtype=np.float32)
        diff[..., 0] = ((pred_r == 1) & (gt_r == 0)).astype(float)  # FP red
        diff[..., 1] = ((pred_r == 0) & (gt_r == 1)).astype(float)  # FN green
        diff[..., 2] = ((pred_r == 1) & (gt_r == 1)).astype(float)  # TP blue
        ax3 = fig.add_subplot(gs[row_i, 3])
        ax3.imshow(diff)
        ax3.axis('off')
        ax3.set_title(f'Error Map\nFP=red FN=green TP=blue', fontsize=9, color=WHITE)

p = OUT_DIR / 'ppt_pred_vs_gt_grid.png'
plt.savefig(str(p), dpi=120, bbox_inches='tight', facecolor=BG)
import shutil; shutil.copy(str(p), DRIVE_OUT)
print(f'Saved: ppt_pred_vs_gt_grid.png')
plt.show()


# ════════════════════════════════════════════
# VIZ 2: Skeleton graph overlaid on Koramangala mask
# ════════════════════════════════════════════
from skimage.morphology import skeletonize
import sknw

fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=BG)
fig.suptitle('Part B: Skeleton Graph Extraction — Koramangala\nPred nodes: 2,210 | OSM nodes: 2,708 | Node F1: 0.44',
             fontsize=14, color=WHITE, fontweight='bold')

# Panel 1: predicted road mask
ax = axes[0]
ax.imshow(pred_mask_k, cmap='Blues', vmin=0, vmax=1)
ax.set_title('Predicted Road Mask\n(Part A output)', color=WHITE, fontsize=12)
ax.axis('off')

# Panel 2: skeleton
skel_k = skeletonize(pred_mask_k > 0).astype(np.uint8)
ax = axes[1]
ax.imshow(skel_k, cmap='Greens', vmin=0, vmax=1)
ax.set_title('Morphological Skeleton\n(thin road centrelines)', color=WHITE, fontsize=12)
ax.axis('off')

# Panel 3: graph nodes + edges overlaid
G_skel_k = sknw.build_sknw(skel_k)
ax = axes[2]
ax.imshow(pred_mask_k, cmap='Greys', vmin=0, vmax=1, alpha=0.3)

for u, v, d in G_skel_k.edges(data=True):
    pts = d.get('pts', [])
    if len(pts) > 1:
        ys, xs = zip(*pts)
        ax.plot(xs, ys, color=ACC2, lw=0.8, alpha=0.7)

node_ys = [G_skel_k.nodes[n]['o'][0] for n in G_skel_k.nodes]
node_xs = [G_skel_k.nodes[n]['o'][1] for n in G_skel_k.nodes]
ax.scatter(node_xs, node_ys, c=ACC1, s=8, zorder=5, alpha=0.8)

ax.set_title(f'Road Graph\n{G_skel_k.number_of_nodes()} nodes (blue) | {G_skel_k.number_of_edges()} edges (green)',
             color=WHITE, fontsize=12)
ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.axis('off')

p = OUT_DIR / 'ppt_skeleton_graph.png'
plt.savefig(str(p), dpi=120, bbox_inches='tight', facecolor=BG)
shutil.copy(str(p), DRIVE_OUT)
print(f'Saved: ppt_skeleton_graph.png')
plt.show()


# ════════════════════════════════════════════
# VIZ 3: OSM reference vs predicted graph scatter
# ════════════════════════════════════════════
BBOX = (77.6101, 12.9177, 77.6401, 12.9377)
def px_to_geo(row, col, bbox=BBOX, sz=512):
    lat = bbox[3] - (row/sz)*(bbox[3]-bbox[1])
    lon = bbox[0] + (col/sz)*(bbox[2]-bbox[0])
    return lat, lon

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
fig.suptitle('Part B: Predicted Graph vs OSM Reference — Koramangala',
             fontsize=14, color=WHITE, fontweight='bold')

# OSM reference
ax = axes[0]
ax.set_facecolor(CARD)
G_osm_ref = ox.graph_from_point((12.9277, 77.6251), dist=1700,
                                  network_type='drive', simplify=True)
for u, v in G_osm_ref.edges():
    ud, vd = G_osm_ref.nodes[u], G_osm_ref.nodes[v]
    ax.plot([ud['x'], vd['x']], [ud['y'], vd['y']], color=ACC1, lw=0.5, alpha=0.5)
osm_lons = [d['x'] for _, d in G_osm_ref.nodes(data=True)]
osm_lats = [d['y'] for _, d in G_osm_ref.nodes(data=True)]
ax.scatter(osm_lons, osm_lats, c=ACC1, s=5, alpha=0.6, zorder=3)
ax.set_title(f'OSM Reference\n{len(G_osm_ref.nodes)} nodes | {len(G_osm_ref.edges)} edges',
             color=WHITE, fontsize=12)
ax.set_xlabel('Longitude', color=GREY); ax.set_ylabel('Latitude', color=GREY)
ax.tick_params(colors=GREY)

# Predicted graph
ax = axes[1]
ax.set_facecolor(CARD)
for u, v, d in G_skel_k.edges(data=True):
    pts = d.get('pts', [])
    if len(pts) > 1:
        coords = [px_to_geo(r, c) for r, c in pts]
        lats, lons = zip(*coords)
        ax.plot(lons, lats, color=ACC2, lw=0.5, alpha=0.7)
p_lats = [px_to_geo(G_skel_k.nodes[n]['o'][0], G_skel_k.nodes[n]['o'][1])[0] for n in G_skel_k.nodes]
p_lons = [px_to_geo(G_skel_k.nodes[n]['o'][0], G_skel_k.nodes[n]['o'][1])[1] for n in G_skel_k.nodes]
ax.scatter(p_lons, p_lats, c=ACC2, s=5, alpha=0.6, zorder=3)
ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
ax.set_title(f'Predicted Graph (from Part A mask)\n{G_skel_k.number_of_nodes()} nodes | {G_skel_k.number_of_edges()} edges',
             color=WHITE, fontsize=12)
ax.set_xlabel('Longitude', color=GREY); ax.set_ylabel('Latitude', color=GREY)
ax.tick_params(colors=GREY)

p = OUT_DIR / 'ppt_osm_vs_predicted_graph.png'
plt.savefig(str(p), dpi=120, bbox_inches='tight', facecolor=BG)
shutil.copy(str(p), DRIVE_OUT)
print(f'Saved: ppt_osm_vs_predicted_graph.png')
plt.show()

print(f'\nAll PPT visuals saved to Drive: {DRIVE_OUT}')
print('Download from Drive: mega-heracross-colab-output/')
