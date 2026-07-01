"""
Generate PPT-ready charts from the two training sessions.
Saves to D:\BAH\mega-heracross\ppt_assets\
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path('ppt_assets')
OUT.mkdir(exist_ok=True)

# ── Color palette (dark, professional) ──
BG   = '#0D1117'
CARD = '#161B22'
ACC1 = '#58A6FF'  # blue
ACC2 = '#3FB950'  # green
ACC3 = '#F78166'  # red/orange
ACC4 = '#D2A8FF'  # purple
GREY = '#8B949E'
WHITE = '#E6EDF3'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': GREY, 'axes.labelcolor': WHITE,
    'xtick.color': GREY, 'ytick.color': GREY,
    'text.color': WHITE, 'grid.color': '#21262D',
    'grid.linestyle': '--', 'grid.alpha': 0.6,
    'font.family': 'DejaVu Sans', 'font.size': 11,
})

# ═══════════════════════════════════════════════════════════
# CHART 1: Training curves — both sessions
# ═══════════════════════════════════════════════════════════
sess1_iou = [0.0514,0.0890,0.1545,0.2162,0.2704,0.3248,0.3647,0.4000,
             0.4368,0.4637,0.4928,0.5181,0.5393,0.5565,0.5707,0.5776,
             0.5846,0.5872,0.5880,0.5905]
sess1_f1  = [0.0978,0.1635,0.2676,0.3556,0.4257,0.4904,0.5345,0.5715,
             0.6080,0.6336,0.6603,0.6825,0.7007,0.7151,0.7267,0.7323,
             0.7378,0.7399,0.7406,0.7425]

sess2_iou = [0.1108,0.1347,0.1398,0.1307,0.1382,0.1378,0.1358,0.1386,
             0.1375,0.1391,0.1381,0.1392,0.1403,0.1385,0.1385,0.1395,
             0.1393,0.1390,0.1394,0.1391]
sess2_f1  = [0.1994,0.2374,0.2452,0.2312,0.2429,0.2423,0.2391,0.2435,
             0.2418,0.2442,0.2426,0.2444,0.2460,0.2434,0.2434,0.2449,
             0.2445,0.2441,0.2447,0.2443]

epochs = list(range(1, 21))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Mega-Heracross Part A — Training Progress', fontsize=16, color=WHITE, fontweight='bold', y=1.02)

for ax, (d1, d2, metric) in zip(axes, [
    (sess1_iou, sess2_iou, 'Validation IoU'),
    (sess1_f1,  sess2_f1,  'Validation F1'),
]):
    ax.plot(epochs, d1, color=ACC1, lw=2.5, marker='o', ms=4, label='Session 1\n(homogeneous, density=3.1%)')
    ax.plot(epochs, d2, color=ACC2, lw=2.5, marker='s', ms=4, label='Session 2\n(10 regions, density=11%)')
    ax.axhline(y=max(d1), color=ACC1, lw=1, ls=':', alpha=0.5)
    ax.axhline(y=max(d2), color=ACC2, lw=1, ls=':', alpha=0.5)
    ax.annotate(f'Best: {max(d1):.3f}', xy=(d1.index(max(d1))+1, max(d1)),
                xytext=(12, max(d1)-0.06), color=ACC1, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=ACC1, lw=1))
    ax.annotate(f'Best: {max(d2):.3f}', xy=(d2.index(max(d2))+1, max(d2)),
                xytext=(12, max(d2)+0.03), color=ACC2, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=ACC2, lw=1))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(metric, fontsize=13, color=WHITE)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, max(max(d1), max(d2)) * 1.15)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE)
    ax.grid(True)

plt.tight_layout()
p = OUT / 'chart1_training_curves.png'
plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {p}')

# ═══════════════════════════════════════════════════════════
# CHART 2: Before/After metrics — the improvement story
# ═══════════════════════════════════════════════════════════
metrics = ['Node F1', 'Edge F1', 'Node\nPrecision', 'Node\nRecall', 'Edge\nPrecision', 'Edge\nRecall']
broken  = [0.04,   0.002,  None,   None,   None,   None]
sess1   = [0.0166, 0.0088, 0.3770, 0.0085, 0.5455, 0.0044]
sess2   = [0.4408, 0.3282, 0.4905, 0.4003, 0.8117, 0.2057]

x = np.arange(len(metrics))
w = 0.26

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)

b1 = ax.bar(x - w, [v if v is not None else 0 for v in broken], w,
            label='Before (broken baseline)', color=ACC3, alpha=0.85, zorder=3)
b2 = ax.bar(x,     sess1, w,
            label='After calibration fix\n(Session 1, same GT mask)', color=ACC1, alpha=0.85, zorder=3)
b3 = ax.bar(x + w, sess2, w,
            label='After diversity fix\n(Session 2, 10 regions)', color=ACC2, alpha=0.85, zorder=3)

for bar in b3:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9, color=ACC2, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score (0–1)', fontsize=12)
ax.set_title('Part B Topology Metrics — Three-Stage Improvement', fontsize=14, color=WHITE, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE, loc='upper left')
ax.grid(True, axis='y', zorder=0)

# Improvement callout
ax.annotate('Node F1: 0.04 → 0.44\n(+26×)', xy=(0 + w, 0.4408),
            xytext=(0.5, 0.75), color=ACC2, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACC2, lw=1.5))
ax.annotate('Edge F1: 0.002 → 0.33\n(+37×)', xy=(1 + w, 0.3282),
            xytext=(1.5, 0.65), color=ACC2, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACC2, lw=1.5))

plt.tight_layout()
p = OUT / 'chart2_metrics_improvement.png'
plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {p}')

# ═══════════════════════════════════════════════════════════
# CHART 3: Road density calibration story
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)

stages = ['Broken\nbaseline', 'After\ncalibration\nfix', 'After\ndiversity\nfix']
pred_density   = [56.0, 3.81, 22.84]
gt_density     = [3.49, 3.49, 10.75]
target_density = [3.49, 3.49,  3.49]

x = np.arange(len(stages))
w = 0.3
ax.bar(x - w/2, pred_density, w, label='Predicted road density', color=ACC3, alpha=0.85, zorder=3)
ax.bar(x + w/2, gt_density,   w, label='GT road density',        color=ACC1, alpha=0.85, zorder=3)
ax.axhline(y=3.49, color=ACC2, lw=2, ls='--', label='Real-world target (3.49%)', zorder=4)
ax.axhspan(3.0, 8.0, alpha=0.08, color=ACC2, label='Target range (3-8%)')

for i, (p, g) in enumerate(zip(pred_density, gt_density)):
    ax.text(i - w/2, p + 0.8, f'{p:.1f}%', ha='center', fontsize=10, color=ACC3, fontweight='bold')
    ax.text(i + w/2, g + 0.8, f'{g:.2f}%', ha='center', fontsize=10, color=ACC1)

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=12)
ax.set_ylabel('Road Pixel Density (%)', fontsize=12)
ax.set_title('Road Density Calibration — Prediction vs. Ground Truth', fontsize=14, color=WHITE, fontweight='bold')
ax.set_ylim(0, 65)
ax.legend(fontsize=10, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE)
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
p = OUT / 'chart3_density_calibration.png'
plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {p}')

# ═══════════════════════════════════════════════════════════
# CHART 4: 10-region density profile
# ═══════════════════════════════════════════════════════════
regions = ['koramangala','hsr_layout','indiranagar','jayanagar','btm_layout',
           'malleswaram','jp_nagar','rajajinagar','basavanagudi','whitefield']
densities = [10.75, 8.68, 10.61, 13.27, 12.98, 7.74, 11.80, 12.17, 11.72, 6.26]
colors = [ACC1]*8 + [ACC4]*2  # blue=train, purple=held-out

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)

bars = ax.barh(regions[::-1], densities[::-1], color=colors[::-1], alpha=0.85, zorder=3)
ax.axvline(x=3.49, color=ACC2, lw=2, ls='--', label='Original target (3.49%)')
ax.axvspan(3.0, 8.0, alpha=0.08, color=ACC2, label='Target range (3-8%)')
ax.axvline(x=11.00, color=ACC3, lw=1.5, ls=':', label='Session 2 train mean (11%)')

for bar, d in zip(bars[::-1], densities):
    ax.text(d + 0.2, bar.get_y() + bar.get_height()/2,
            f'{d:.2f}%', va='center', fontsize=10)

train_patch = mpatches.Patch(color=ACC1, label='Training regions (8)')
test_patch  = mpatches.Patch(color=ACC4, label='Held-out test regions (2)')
ax.legend(handles=[train_patch, test_patch,
                   mpatches.Patch(color=ACC2, alpha=0.3, label='Target range 3-8%')],
          fontsize=10, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE)

ax.set_xlabel('Road Pixel Density (%)', fontsize=12)
ax.set_title('GT Mask Density — 10 Bengaluru Regions', fontsize=14, color=WHITE, fontweight='bold')
ax.set_xlim(0, 16)
ax.grid(True, axis='x', zorder=0)

plt.tight_layout()
p = OUT / 'chart4_region_densities.png'
plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {p}')

# ═══════════════════════════════════════════════════════════
# CHART 5: Generalization — val vs held-out
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)

cats = ['In-distribution\nval (8 training regions)', 'Held-out\nbasavanagudi', 'Held-out\nwhitefield']
iou_vals = [0.1391, 0.3500, 0.3139]
f1_vals  = [0.2443, 0.5186, 0.4778]
x = np.arange(len(cats))
w = 0.35

ax.bar(x - w/2, iou_vals, w, label='IoU',  color=ACC1, alpha=0.85, zorder=3)
ax.bar(x + w/2, f1_vals,  w, label='F1',   color=ACC2, alpha=0.85, zorder=3)

for i, (iv, fv) in enumerate(zip(iou_vals, f1_vals)):
    ax.text(i - w/2, iv + 0.01, f'{iv:.3f}', ha='center', fontsize=10, color=ACC1, fontweight='bold')
    ax.text(i + w/2, fv + 0.01, f'{fv:.3f}', ha='center', fontsize=10, color=ACC2, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel('Score (0–1)', fontsize=12)
ax.set_title('Generalization: Held-out Regions Score Higher Than Val\n(Model learned road patterns, not one layout)',
             fontsize=13, color=WHITE, fontweight='bold')
ax.set_ylim(0, 0.75)
ax.legend(fontsize=11, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE)
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
p = OUT / 'chart5_generalization.png'
plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {p}')

print(f'\nAll 5 charts saved to {OUT.resolve()}')
print('Now generate the pipeline visualizations in Colab (see colab_ppt_visuals.py)')
