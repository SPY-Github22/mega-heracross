"""
ITEM 2 — Generate clean training curve PNG.
Data: Session 2 (topology diversity run, 20 epochs, 10-region Bengaluru dataset).
Output: outputs/training_curve.png  (1200x600 px minimum at 150 DPI)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Real epoch data from Session 2 (topology diversity run) ──
VAL_IOU = [
    0.1108, 0.1347, 0.1398, 0.1307, 0.1382,
    0.1378, 0.1358, 0.1386, 0.1375, 0.1391,
    0.1381, 0.1392, 0.1403, 0.1385, 0.1385,
    0.1395, 0.1393, 0.1390, 0.1394, 0.1391,
]
EPOCHS = list(range(1, 21))
BEST_EPOCH = 13
BEST_VAL   = 0.1403

# ── Styling ──
BG    = '#0D1117'
CARD  = '#161B22'
LINE  = '#58A6FF'
BEST  = '#3FB950'
GREY  = '#8B949E'
WHITE = '#E6EDF3'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   CARD,
    'axes.edgecolor':   GREY,
    'axes.labelcolor':  WHITE,
    'xtick.color':      GREY,
    'ytick.color':      GREY,
    'text.color':       WHITE,
    'grid.color':       '#21262D',
    'grid.linestyle':   '--',
    'grid.alpha':       0.7,
    'font.family':      'DejaVu Sans',
    'font.size':        13,
})

fig, ax = plt.subplots(figsize=(8, 4))   # 1200x600 at 150 DPI
fig.patch.set_facecolor(BG)

# Main line
ax.plot(EPOCHS, VAL_IOU, color=LINE, lw=2.5, marker='o', ms=5,
        markerfacecolor=LINE, markeredgecolor=BG, zorder=4, label='Val IoU')

# Best epoch marker
ax.scatter([BEST_EPOCH], [BEST_VAL], color=BEST, s=120, zorder=5,
           edgecolors=BG, linewidths=1.5, label=f'Best  (Ep {BEST_EPOCH}: {BEST_VAL:.4f})')
ax.annotate(f'Best: {BEST_VAL:.4f}\n(Epoch {BEST_EPOCH})',
            xy=(BEST_EPOCH, BEST_VAL),
            xytext=(BEST_EPOCH + 1.5, BEST_VAL + 0.008),
            color=BEST, fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BEST, lw=1.5))

# Reference line
ax.axhline(y=BEST_VAL, color=BEST, lw=1, ls=':', alpha=0.45)

# Axis labels & title
ax.set_xlabel('Epoch', fontsize=13, labelpad=8)
ax.set_ylabel('Validation IoU', fontsize=13, labelpad=8)
ax.set_title('SegFormer B3 — Multi-Region Bengaluru Training', fontsize=14,
             fontweight='bold', color=WHITE, pad=12)

ax.set_xlim(0.5, 20.5)
ax.set_ylim(0.08, 0.18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(True, zorder=0)
ax.legend(fontsize=11, facecolor=CARD, edgecolor=GREY, labelcolor=WHITE,
          loc='lower right')

plt.tight_layout(pad=1.4)

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
out_path = OUT / 'training_curve.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

import os
size_kb = os.path.getsize(out_path) // 1024
actual_w, actual_h = fig.get_size_inches()
print(f'Saved:   {out_path.resolve()}')
print(f'Size:    {size_kb} KB')
print(f'At 150 DPI: ~{int(actual_w*150)}x{int(actual_h*150)} px')
