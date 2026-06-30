#!/usr/bin/env python3
"""
Phase 7 - Task 1 & 2: RoadDataset + Augmentation Pipeline
==========================================================
torch.utils.data.Dataset subclass that generates or loads fused
optical-SAR tiles paired with binary road masks, applies albumentations
augmentation, and serves (fused_tensor, gt_mask) pairs for training.

Place at: part_a_vision/dataset.py

Key design decisions:
  - 200 synthetic tile variants (different seeds, occlusion types)
  - 80/20 train/val split
  - Class-weighted BCE (weight_road=6.0) - roads ~15% of pixels
  - Augmentations on optical channels only (SAR is physics-based)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("road_dataset")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TILES: int = 200
TRAIN_SPLIT: float = 0.80
DEFAULT_TILE_SIZE: int = 512
NUM_OPTICAL_BANDS: int = 4
NUM_SAR_BANDS: int = 2
TOTAL_CHANNELS: int = NUM_OPTICAL_BANDS + NUM_SAR_BANDS  # 6

# Class weighting: roads are ~15% of pixels → weight_road = (1/0.15) ≈ 6.7
CLASS_WEIGHT_ROAD: float = 6.0
CLASS_WEIGHT_BACKGROUND: float = 1.0

# Synthetic seed range - each seed produces a distinct Koramangala tile variant
SEED_START: int = 1000


# ---------------------------------------------------------------------------
# RoadDataset
# ---------------------------------------------------------------------------

class RoadDataset(Dataset):
    """
    PyTorch Dataset for road segmentation on fused optical-SAR tiles.

    Each item is a (fused_tensor, gt_mask) pair:
      - fused_tensor: (C, H, W) float32 - optical + SAR concatenated
      - gt_mask: (H, W) int64 - binary road mask {0, 1}

    Parameters
    ----------
    tiles : list of dict
        Pre-generated tile metadata. If None, synthetic tiles are generated.
    tile_size : int
        Spatial dimension of each tile (square).
    num_tiles : int
        Number of synthetic tiles to generate.
    split : str
        "train" or "val".
    augment : bool
        If True (train mode), apply albumentations augmentation.
    cache_dir : str
        Directory to cache generated tiles on disk.
    """

    def __init__(
        self,
        tiles: Optional[List[dict]] = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        num_tiles: int = NUM_TILES,
        split: str = "train",
        augment: bool = True,
        cache_dir: str = "data/synthetic_tiles",
        force_occlusion: bool = False,
        cloud_level: float = 0.0,
        has_temporal: bool = True,
        force_shadow: bool = False,
        custom_transform=None,
    ):
        self.tile_size = tile_size
        self.split = split
        self.augment = augment and (split == "train")
        self.custom_transform = custom_transform
        self.force_occlusion = force_occlusion
        self.cloud_level = cloud_level
        self.has_temporal = has_temporal
        self.force_shadow = force_shadow
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Generate or load tiles
        if tiles is not None:
            self.tiles = tiles
        else:
            self.tiles = self._generate_synthetic_tiles(num_tiles)

        # Train/val split (deterministic by seed)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(self.tiles))
        split_idx = int(len(indices) * TRAIN_SPLIT)
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        logger.info(
            "RoadDataset (%s): %d tiles, %d used",
            split, len(self.tiles), len(self.indices),
        )

        # Set up augmentations
        self._setup_augmentations()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (fused_tensor, gt_mask):
          fused_tensor: (C, H, W) float32
          gt_mask: (H, W) int64
        """
        real_idx = self.indices[idx]
        tile = self.tiles[real_idx]
        seed = tile["seed"]
        cache_file = os.path.join(self.cache_dir, f"tile_{seed:04d}.npz")

        # Load from cache or generate
        fused, mask = self._load_or_generate(seed, cache_file)

        # Apply augmentation (train only)
        if self.augment:
            fused, mask = self._augment(fused, mask)

        # Apply canopy occlusion (Phase 12)
        if self.augment or self.force_occlusion:
            fused = self._add_canopy_occlusion(fused, seed + idx)

        # Apply cloud occlusion (Phase 13)
        if self.augment or self.cloud_level > 0.0:
            fused = self._add_cloud_occlusion(fused, seed + idx + 1000)

        # Phase 14: Temporal shadow reasoning
        temporal_diff = np.zeros((4, self.tile_size, self.tile_size), dtype=np.float32)
        if self.augment or self.force_shadow:
            fused_base = fused.copy()
            # T has shadows at offset (15, 15)
            fused = self._add_shadow_occlusion(fused_base, seed + idx + 2000, 15, 15)
            
            if self.has_temporal:
                # T-1 has shadows at offset (-10, -5) simulating sun shift
                fused_t_minus_1 = self._add_shadow_occlusion(fused_base, seed + idx + 2000, -10, -5)
                temporal_diff = np.abs(fused[:4] - fused_t_minus_1[:4])
                
        # Stack temporal diff
        fused = np.concatenate([fused, temporal_diff], axis=0)

        # Phase 22: Compute Spectral Indices from optical bands
        # optical channels: 0=Green, 1=Red, 2=NIR, 3=SWIR
        green = fused[0]
        red = fused[1]
        nir = fused[2]
        
        # NDVI = (NIR - Red) / (NIR + Red + eps)
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        # NDWI = (Green - NIR) / (Green + NIR + eps)
        ndwi = (green - nir) / (green + nir + 1e-6)
        
        # Normalize indices to [0, 1] since they naturally span [-1, 1]
        ndvi = (ndvi + 1.0) / 2.0
        ndwi = (ndwi + 1.0) / 2.0
        
        # Add a channel dimension and stack
        ndvi = np.expand_dims(ndvi, axis=0)
        ndwi = np.expand_dims(ndwi, axis=0)
        
        # We place them right after the 4 optical channels so optical becomes 6 channels
        # Current fused structure: 0-3 (Opt), 4-5 (SAR), 6-9 (Temporal)
        # New fused structure: 0-3 (Opt), 4-5 (Indices), 6-7 (SAR), 8-11 (Temporal)
        opt_bands = fused[:4]
        sar_bands = fused[4:6]
        temp_bands = fused[6:10]
        
        fused = np.concatenate([opt_bands, ndvi, ndwi, sar_bands, temp_bands], axis=0)

        # Convert to tensors
        fused_tensor = torch.from_numpy(fused.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return fused_tensor, mask_tensor

    # -------------------------------------------------------------------
    # Tile generation (Task 1 - 200 synthetic variants)
    # -------------------------------------------------------------------

    def _generate_synthetic_tiles(self, num_tiles: int) -> List[dict]:
        """Generate metadata for num_tiles synthetic Koramangala variants."""
        tiles = []
        for i in range(num_tiles):
            seed = SEED_START + i
            tiles.append({
                "seed": seed,
                "index": i,
                "source": "synthetic_koramangala",
            })
        logger.info("Generated %d synthetic tile metadata entries", num_tiles)
        return tiles

    def _load_or_generate(
        self, seed: int, cache_file: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load from disk cache or generate a fused tile + ground-truth mask.

        Returns (fused (C, H, W) float32, mask (H, W) int64).
        """
        if os.path.isfile(cache_file):
            data = np.load(cache_file)
            return data["fused"].astype(np.float32), data["mask"].astype(np.int64)

        # Generate optical + SAR + mask
        fused, mask = self._make_tile(seed)

        # Save to cache
        np.savez_compressed(cache_file, fused=fused, mask=mask)
        return fused, mask

    def _make_tile(
        self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create one synthetic tile: optical (4 bands) + SAR (2 bands) + road mask.

        Uses the same pipelines from Phases 4, 5, and 6 to maintain consistency.
        Falls back to procedural generation if those modules aren't importable.
        """
        # Try full pipeline first
        try:
            return self._make_tile_full_pipeline(seed)
        except Exception as exc:
            logger.debug("Full pipeline unavailable for seed %d: %s - using procedural", seed, exc)
            return self._make_tile_procedural(seed)

    def _make_tile_full_pipeline(
        self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tile using Phase 4+5+6 pipelines."""
        from optical_reader import OpticalPreprocessor
        from sar_reader import SARPreprocessor
        from fusion import FusionModule

        size = self.tile_size

        # Phase 4: optical with variable cloud that varies per seed
        rng = np.random.RandomState(seed)
        cloud_frac = rng.choice([0.0, 0.0, 0.0, 0.15, 0.3, 0.55, 0.8])  # mostly clear

        opt_pp = OpticalPreprocessor(target_size=size)
        opt_tensor, _ = opt_pp.process(None)  # synthetic

        # Phase 5: SAR
        sar_pp = SARPreprocessor(target_size=size)
        sar_tensor, _ = sar_pp.process(None, None)  # synthetic

        # Phase 6: fusion
        fusion = FusionModule()
        tile = fusion.fuse(opt_tensor, sar_tensor, save_debug=False)

        # Ground-truth mask: roads as dark features in both
        mask = self._extract_road_mask(opt_tensor, sar_tensor, seed)
        return tile.tensor.astype(np.float32), mask.astype(np.int64)

    def _make_tile_procedural(
        self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standalone procedural tile generator.
        Produces realistic-looking fused tile + road mask without external deps.
        """
        rng = np.random.RandomState(seed)
        size = self.tile_size

        # -- Optical channels (4 bands: Green, Red, NIR, SWIR) --
        optical = np.zeros((NUM_OPTICAL_BANDS, size, size), dtype=np.float32)
        for b in range(NUM_OPTICAL_BANDS):
            # Perlin-like texture via smoothed noise
            base = rng.uniform(0.15, 0.45, size=(size, size)).astype(np.float32)
            from scipy.ndimage import gaussian_filter
            optical[b] = gaussian_filter(base, sigma=rng.uniform(3, 12))

        # -- SAR channels (2 bands: VV, VH) --
        sar = np.zeros((NUM_SAR_BANDS, size, size), dtype=np.float32)
        for b in range(NUM_SAR_BANDS):
            # Rayleigh speckle texture
            sar[b] = rng.rayleigh(scale=0.3, size=(size, size)).astype(np.float32)
            sar[b] = gaussian_filter(sar[b], sigma=1.0)
            # Normalize
            lo, hi = np.percentile(sar[b], [1, 99])
            sar[b] = np.clip((sar[b] - lo) / max(hi - lo, 1e-8), 0, 1)

        # -- Road mask --
        mask = self._generate_road_mask(seed, size)

        # -- Paint roads into optical and SAR --
        # Roads in optical: dark gray (all bands low)
        for b in range(NUM_OPTICAL_BANDS):
            optical[b][mask == 1] = rng.uniform(0.05, 0.2, size=int(mask.sum()))

        # Roads in SAR: dark (specular reflection → low backscatter)
        for b in range(NUM_SAR_BANDS):
            sar[b][mask == 1] = rng.uniform(0.0, 0.15, size=int(mask.sum()))

        # -- Buildings: bright in SAR (double-bounce), varied in optical --
        n_buildings = rng.randint(8, 25)
        building_mask = np.zeros((size, size), dtype=np.uint8)
        for _ in range(n_buildings):
            cx, cy = rng.randint(size // 8, 7 * size // 8), rng.randint(size // 8, 7 * size // 8)
            bw, bh = rng.randint(3, 10), rng.randint(3, 10)
            x0, x1 = max(0, cx - bw // 2), min(size, cx + bw // 2)
            y0, y1 = max(0, cy - bh // 2), min(size, cy + bh // 2)
            building_mask[y0:y1, x0:x1] = 1
            # Bright in SAR
            for b in range(NUM_SAR_BANDS):
                sar[b][y0:y1, x0:x1] = rng.uniform(0.7, 1.0)

        # -- Cloud occlusion (variable per tile) --
        cloud_type = rng.choice(["none", "cumulus", "thin", "thick"], p=[0.4, 0.2, 0.2, 0.2])
        if cloud_type != "none":
            n_clouds = rng.randint(2, 8)
            cloud_mask = np.zeros((size, size), dtype=np.float32)
            for _ in range(n_clouds):
                cx, cy = rng.randint(0, size), rng.randint(0, size)
                cr = rng.randint(20, 80)
                Y, X = np.ogrid[:size, :size]
                dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                cloud = np.exp(-0.5 * (dist / cr) ** 2)
                opacity = {"thin": 0.4, "cumulus": 0.7, "thick": 0.95}[cloud_type]
                cloud_mask += cloud * opacity
            cloud_mask = np.clip(cloud_mask, 0, 1)

            # Apply cloud only to optical channels (not SAR - C-band penetrates!)
            for b in range(NUM_OPTICAL_BANDS):
                optical[b] = optical[b] * (1 - cloud_mask) + cloud_mask * rng.uniform(0.7, 0.95)

        # -- Normalize optical per-band --
        for b in range(NUM_OPTICAL_BANDS):
            lo, hi = np.percentile(optical[b], [2, 98])
            optical[b] = np.clip((optical[b] - lo) / max(hi - lo, 1e-8), 0, 1)

        fused = np.concatenate([optical, sar], axis=0).astype(np.float32)
        return fused, mask.astype(np.int64)

    def _generate_road_mask(self, seed: int, size: int) -> np.ndarray:
        """Generate a realistic road network mask for Bengaluru."""
        rng = np.random.RandomState(seed)
        mask = np.zeros((size, size), dtype=np.uint8)
        road_width = rng.randint(3, 7)

        # Main roads (grid-like, Bengaluru style)
        n_horizontal = rng.randint(2, 5)
        for _ in range(n_horizontal):
            y = rng.randint(size // 6, 5 * size // 6)
            w = road_width + rng.randint(-1, 2)
            mask[max(0, y - w): min(size, y + w), :] = 1

        n_vertical = rng.randint(2, 5)
        for _ in range(n_vertical):
            x = rng.randint(size // 6, 5 * size // 6)
            w = road_width + rng.randint(-1, 2)
            mask[:, max(0, x - w): min(size, x + w)] = 1

        # Diagonal roads
        n_diag = rng.randint(0, 3)
        for _ in range(n_diag):
            offset = rng.randint(-size // 4, size // 4)
            for i in range(-road_width, road_width + 1):
                diag_idx = np.arange(size)
                row_idx = np.clip(diag_idx + i + offset, 0, size - 1)
                col_idx = np.clip(diag_idx + i - offset, 0, size - 1)
                mask[row_idx, col_idx] = 1

        # Curved roads (sine waves)
        if rng.rand() > 0.3:
            n_curves = rng.randint(1, 3)
            for _ in range(n_curves):
                amplitude = rng.randint(20, 60)
                frequency = rng.uniform(0.005, 0.02)
                phase = rng.uniform(0, 2 * np.pi)
                base_y = rng.randint(size // 4, 3 * size // 4)
                x_vals = np.arange(size)
                y_vals = (base_y + amplitude * np.sin(frequency * x_vals + phase)).astype(int)
                for i in range(-road_width, road_width + 1):
                    y_idx = np.clip(y_vals + i, 0, size - 1)
                    mask[y_idx, x_vals] = 1

        return mask

    @staticmethod
    def _extract_road_mask(
        optical: np.ndarray, sar: np.ndarray, seed: int
    ) -> np.ndarray:
        """
        Heuristic road mask extraction from synthetic tiles.
        Roads = dark in both optical (all bands low) AND SAR VV (low backscatter).
        """
        size = optical.shape[1]
        rng = np.random.RandomState(seed)
        # Use the procedural generator for consistent masks
        mask = np.zeros((size, size), dtype=np.uint8)
        road_width = rng.randint(3, 7)

        n_h = rng.randint(2, 5)
        for _ in range(n_h):
            y = rng.randint(size // 6, 5 * size // 6)
            w = road_width + rng.randint(-1, 2)
            mask[max(0, y - w): min(size, y + w), :] = 1

        n_v = rng.randint(2, 5)
        for _ in range(n_v):
            x = rng.randint(size // 6, 5 * size // 6)
            w = road_width + rng.randint(-1, 2)
            mask[:, max(0, x - w): min(size, x + w)] = 1

        return mask

    # -------------------------------------------------------------------
    # Task 2 - Data Augmentation
    # -------------------------------------------------------------------

    def _setup_augmentations(self) -> None:
        """
        Prepare albumentations pipeline.
        """
        if getattr(self, "custom_transform", None) is not None:
            self.transform = self.custom_transform
            self._albumentations_available = True
            return
            
        try:
            import albumentations as A

            self.transform = A.Compose([
                # Spatial transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Optical-only brightness/contrast (applied to first 4 channels)
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.5,
                ),

                # Elastic deformation - simulates road curvature variation
                A.ElasticTransform(
                    alpha=50, sigma=5, p=0.2,
                ),

                # Coarse dropout - simulates additional occlusion
                A.CoarseDropout(
                    num_holes_range=(2, 8), hole_height_range=(8, 32), hole_width_range=(8, 32),
                    fill_value=0, p=0.2,
                ),
            ])
            self._albumentations_available = True
        except ImportError:
            logger.warning(
                "albumentations not installed - augmentations disabled. "
                "Install with: pip install albumentations"
            )
            self.transform = None
            self._albumentations_available = False

    def _augment(
        self, fused: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply albumentations augmentation.

        NOTE: Brightness/contrast is applied to ALL channels by albumentations.
        For a stricter version, we'd only augment optical channels (0:4).
        This is flagged as a known limitation - Phase 11 can refine.
        """
        if self.transform is None:
            return fused, mask

        # albumentations expects (H, W, C)
        image = np.transpose(fused, (1, 2, 0))
        augmented = self.transform(image=image, mask=mask)
        fused_out = np.transpose(augmented["image"], (2, 0, 1))
        mask_out = augmented["mask"]

        return fused_out.astype(np.float32), mask_out.astype(np.int64)

    def _add_canopy_occlusion(self, fused: np.ndarray, seed: int) -> np.ndarray:
        """
        Phase 12: Add synthetic canopy occlusion to optical channels (0-3).
        Draws 3-8 noisy circular blobs (radius 20-80px) over the tile.
        fused is (C, H, W)
        """
        rng = np.random.RandomState(seed + 100)
        
        c, h, w = fused.shape
        if c < 4:
            return fused
            
        num_blobs = rng.randint(3, 9)
        fused_out = fused.copy()
        
        for _ in range(num_blobs):
            radius = rng.randint(20, 81)
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            
            y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
            
            # Create a soft mask using distance for a more natural edge
            dist = np.sqrt(x**2 + y**2)
            mask = dist <= radius
            
            if not mask.any():
                continue
                
            # Realistic canopy reflectance for LISS-IV (Green, Red, NIR, SWIR)
            # Green=0.4, Red=0.2, NIR=0.8, SWIR=0.3
            base_color = np.array([0.4, 0.2, 0.8, 0.3])
            noise = rng.normal(0, 0.1, size=(4, mask.sum()))
            texture = base_color[:, None] + noise
            texture = np.clip(texture, 0, 1)
            
            # Apply to optical channels
            fused_out[:4, mask] = texture
            
        return fused_out

    def _add_cloud_occlusion(self, fused: np.ndarray, seed: int) -> np.ndarray:
        """
        Phase 13: Add synthetic cloud cover to optical channels.
        If self.cloud_level > 0, deterministically cover that fraction.
        If self.augment is True, randomly apply total or partial cloud cover.
        """
        rng = np.random.RandomState(seed + 200)
        c, h, w = fused.shape
        if c < 4:
            return fused
            
        fused_out = fused.copy()
        
        # Determine target coverage fraction
        if getattr(self, 'cloud_level', 0.0) > 0.0:
            target_fraction = self.cloud_level
        else:
            # Training augmentation: 30% chance total cloud, else partial
            if rng.rand() < 0.3:
                target_fraction = 1.0
            else:
                target_fraction = rng.uniform(0.3, 0.7)
                
        if target_fraction >= 1.0:
            # Total cloud cover: saturate optical channels
            noise = rng.normal(0.9, 0.1, size=(4, h, w))
            fused_out[:4] = np.clip(noise, 0.0, 1.0)
            return fused_out
            
        # Partial cloud cover
        mask = np.zeros((h, w), dtype=bool)
        attempts = 0
        while mask.mean() < target_fraction and attempts < 100:
            radius = rng.randint(40, 150)
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            
            y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
            blob_mask = (x**2 + y**2) <= radius**2
            mask = mask | blob_mask
            attempts += 1
            
        if mask.any():
            # Apply white/gray noise to the masked regions
            noise = rng.normal(0.85, 0.15, size=(4, mask.sum()))
            texture = np.clip(noise, 0, 1)
            fused_out[:4, mask] = texture
            
        return fused_out

    def _add_shadow_occlusion(self, fused: np.ndarray, seed: int, offset_x: int, offset_y: int) -> np.ndarray:
        """
        Phase 14: Add synthetic building shadows to optical channels.
        """
        rng = np.random.RandomState(seed)
        fused_out = fused.copy()
        c, h, w = fused.shape
        if c < 4:
            return fused
            
        for _ in range(8):
            w_rect = rng.randint(20, 70)
            h_rect = rng.randint(20, 70)
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            
            # Apply sun offset
            cx += offset_x
            cy += offset_y
            
            x1 = max(0, cx - w_rect//2)
            x2 = min(w, cx + w_rect//2)
            y1 = max(0, cy - h_rect//2)
            y2 = min(h, cy + h_rect//2)
            
            if x1 < x2 and y1 < y2:
                # Darken the optical channels to simulate shadow
                fused_out[:4, y1:y2, x1:x2] *= 0.3
        return fused_out



# ---------------------------------------------------------------------------
# Convenience: class weights for BCEWithLogitsLoss
# ---------------------------------------------------------------------------

def get_class_weights(
    road_weight: float = CLASS_WEIGHT_ROAD,
    background_weight: float = CLASS_WEIGHT_BACKGROUND,
) -> torch.Tensor:
    """
    Return class weights for pos_weight in BCEWithLogitsLoss.

    Usage:
        pos_weight = get_class_weights()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    """
    return torch.tensor([road_weight], dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    tile_size: int = DEFAULT_TILE_SIZE,
    num_tiles: int = NUM_TILES,
    batch_size: int = 4,
    num_workers: int = 2,
    cache_dir: str = "data/synthetic_tiles",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation DataLoaders.

    Returns (train_loader, val_loader).
    """
    train_ds = RoadDataset(
        tile_size=tile_size, num_tiles=num_tiles,
        split="train", augment=True, cache_dir=cache_dir,
    )
    val_ds = RoadDataset(
        tile_size=tile_size, num_tiles=num_tiles,
        split="val", augment=False, cache_dir=cache_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        "DataLoaders: train=%d batches, val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI: generate tiles to cache
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 7 - Generate synthetic training tiles")
    parser.add_argument("--num-tiles", type=int, default=NUM_TILES, help="Number of tiles")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size")
    parser.add_argument("--cache-dir", type=str, default="data/synthetic_tiles", help="Cache directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = parser.parse_args()

    ds = RoadDataset(
        tile_size=args.tile_size, num_tiles=args.num_tiles,
        split=args.split, cache_dir=args.cache_dir,
    )
    print(f"Dataset ready: {len(ds)} tiles in {args.split} split")

    # Generate one to verify
    fused, mask = ds[0]
    print(f"Sample: fused={fused.shape}, mask={mask.shape}")
    print(f"Road pixels: {mask.sum().item():,d} / {mask.numel():,d} = {mask.float().mean().item():.1%}")