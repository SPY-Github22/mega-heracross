"""
Creates a minimal zip of only the files needed for Colab training.
Total size: ~2-3MB (not the 4GB full zip).
Run from D:\BAH\mega-heracross
"""
import sys, os, zipfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_ZIP = "colab_upload.zip"

INCLUDE_FILES = [
    # Part A core modules
    "part_a_vision/synthetic_tile.py",
    "part_a_vision/dataset.py",
    "part_a_vision/model.py",
    "part_a_vision/loss.py",
    "part_a_vision/part_a_config.py",
    "part_a_vision/output_writer.py",
    "part_a_vision/optical_reader.py",
    "part_a_vision/sar_reader.py",
    "part_a_vision/fusion.py",
    "part_a_vision/cbam.py",
    "part_a_vision/cloud_fusion.py",
    "part_a_vision/postprocess.py",
    "part_a_vision/tta.py",
    "part_a_vision/bengaluru_transforms.py",
    "part_a_vision/__init__.py",
    # Shared modules
    "shared/config.py",
    "shared/schema.py",
    "shared/eval.py",
    # Training script (corrected)
    "calibration_train_corrected.py",
    # Key data files
    "part_a_vision/data/koramangala/osmnx_gt_mask.npy",
    "part_a_vision/data/koramangala/koramangala_osmnx.graphml",
]

INCLUDE_DIRS = [
    "shared",
    "part_a_vision/models",  # empty dir placeholder
    "part_a_vision/outputs",
    "part_a_vision/data/koramangala/train",   # will be empty after our cache clear
    "part_a_vision/data/koramangala/val",
    "outputs",
]

total_bytes = 0
included = []
missing = []

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fpath in INCLUDE_FILES:
        if os.path.exists(fpath):
            zf.write(fpath)
            sz = os.path.getsize(fpath)
            total_bytes += sz
            included.append((fpath, sz))
        else:
            missing.append(fpath)

    # Write placeholder files for empty directories
    for d in INCLUDE_DIRS:
        os.makedirs(d, exist_ok=True)
        placeholder = os.path.join(d, ".gitkeep")
        if not os.path.exists(placeholder):
            open(placeholder, 'w').close()
        if os.path.exists(d):
            # Only add the placeholder, not any .npz cached tiles
            if os.path.exists(placeholder):
                zf.write(placeholder)

print(f"Created: {OUTPUT_ZIP}")
print(f"Total files included: {len(included)}")
print(f"Total size (uncompressed): {total_bytes/1024:.1f} KB")
print(f"\nIncluded:")
for f, sz in included:
    print(f"  {sz:>10,} bytes  {f}")
if missing:
    print(f"\nMissing (skipped):")
    for f in missing:
        print(f"  [SKIP] {f}")

zip_sz = os.path.getsize(OUTPUT_ZIP)
print(f"\nFinal zip size: {zip_sz/1024:.1f} KB ({zip_sz/1024/1024:.2f} MB)")
print(f"\nUpload '{OUTPUT_ZIP}' to your Google Drive, then run the Colab notebook.")
