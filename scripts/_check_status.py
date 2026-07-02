import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

p = 'part_a_vision/models/best_checkpoint.pth'
print(f"Checkpoint exists: {os.path.exists(p)}")
if os.path.exists(p):
    sz = os.path.getsize(p)
    print(f"  Size: {sz:,} bytes")
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    print(f"  epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  best_val_iou: {ckpt.get('best_val_iou', 0):.4f}")
else:
    print("  No checkpoint saved yet - training was interrupted")

# Check training log tail
log = r'C:\Users\sudpy\.gemini\antigravity\brain\ca793c93-4e4e-48cd-9b75-72766039f7b3\.system_generated\tasks\task-105.log'
if os.path.exists(log):
    with open(log, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    print(f"\nTraining log: {len(lines)} lines total. Last 30:")
    for line in lines[-30:]:
        print(line.rstrip())
else:
    print("\nTraining log not found at expected path")

# Check cached tile counts
for d in ['part_a_vision/data/koramangala/train', 'part_a_vision/data/koramangala/val']:
    if os.path.exists(d):
        npz = [f for f in os.listdir(d) if f.endswith('.npz')]
        print(f"\n{d}: {len(npz)} .npz tiles")
