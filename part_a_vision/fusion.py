#!/usr/bin/env python3
"""
Phase 6 - Optical-SAR Fusion Module
=====================================
Early-fusion strategy: concatenate optical + SAR channels into a single
multi-modal input tensor, with occlusion-aware weighting that adapts
when one modality is degraded (clouds → SAR dominant).

Place this file at: part_a_vision/fusion.py

Exit Criterion:
    fusion.py outputs a FusedTile with a (6, 512, 512) float32 tensor
    and logs/fusion_debug.png shows visually aligned optical and SAR channels.

Architecture Decision:
    EARLY FUSION (concatenate before backbone).  This is simpler, uses
    fewer parameters than late fusion (two backbones), and works well
    when modalities are spatially aligned.  The hard-coded occlusion
    heuristic is a placeholder - Phase 13 replaces it with learned
    dynamic weighting.

Key Physics (for judge presentations):
    SAR penetrates clouds → the core technical narrative of Part A.
    When cloud_fraction > 50%, optical channels are suppressed and SAR
    becomes the primary road detector.  ISRO cares about cloud-free
    road monitoring for monsoon-season disaster response.

Usage:
    from part_a_vision.fusion import FusionModule, FusedTile
    fusion = FusionModule()
    tile = fusion.fuse(optical_tensor, sar_tensor, optical_meta, sar_meta)
    # tile.tensor.shape → (6, 512, 512) for 4-band optical + 2-band SAR
    # tile.optical_suppressed → True if clouds > 50%
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("fusion")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cloud fraction threshold for suppressing optical channels
CLOUD_SUPPRESSION_THRESHOLD: float = 0.50
# SAR weight multiplier when optical is suppressed
SAR_OCCLUSION_WEIGHT: float = 2.0
# Output debug image path
DEFAULT_DEBUG_PATH: str = "logs/fusion_debug.png"

# ---------------------------------------------------------------------------
# FusedTile dataclass (Task 4)
# ---------------------------------------------------------------------------

@dataclass
class FusedTile:
    """
    Self-describing fused multi-modal tile.

    Carries the concatenated tensor plus full provenance so downstream
    modules (training, evaluation, visualisation) can make informed
    decisions without re-deriving metadata.
    """

    tensor: np.ndarray                          # (C_total, H, W) float32
    cloud_fraction: float = 0.0
    optical_available: bool = True
    sar_available: bool = True
    optical_suppressed: bool = False
    sar_weight_multiplier: float = 1.0
    num_optical_channels: int = 0
    num_sar_channels: int = 2
    source_optical: str = "unknown"
    source_sar: str = "unknown"
    shape: Optional[Tuple[int, ...]] = None
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        if self.shape is None and self.tensor is not None:
            self.shape = self.tensor.shape
        self.num_optical_channels = (
            self.tensor.shape[0] - self.num_sar_channels
            if self.tensor is not None and self.tensor.ndim == 3
            else 0
        )

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.pop("tensor", None)
        d["tensor_shape"] = self.shape
        return d


# ---------------------------------------------------------------------------
# FusionModule
# ---------------------------------------------------------------------------

class FusionModule:
    """
    Early-fusion module: concatenates optical and SAR into one tensor
    with optional occlusion-aware channel re-weighting.

    Parameters
    ----------
    cloud_threshold : float
        Cloud fraction above which optical channels are suppressed.
    sar_boost : float
        Multiplier applied to SAR channels when optical is suppressed.
    debug_dir : str
        Directory for sanity-check debug images.
    """

    def __init__(
        self,
        cloud_threshold: float = CLOUD_SUPPRESSION_THRESHOLD,
        sar_boost: float = SAR_OCCLUSION_WEIGHT,
        debug_dir: str = "logs",
    ):
        self.cloud_threshold = cloud_threshold
        self.sar_boost = sar_boost
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fuse(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        optical_meta: Optional[object] = None,
        sar_meta: Optional[object] = None,
        save_debug: bool = True,
    ) -> FusedTile:
        """
        Concatenate optical and SAR into a fused multi-modal tensor.

        Parameters
        ----------
        optical : np.ndarray, shape (C_opt, H, W), float32
            Normalised optical tensor from Phase 4.
        sar : np.ndarray, shape (2, H, W), float32
            Normalised SAR tensor from Phase 5.
        optical_meta : PreprocessMeta or None
            Metadata from optical preprocessing (used for cloud fraction).
        sar_meta : SARMeta or None
            Metadata from SAR preprocessing.
        save_debug : bool
            If True, write logs/fusion_debug.png.

        Returns
        -------
        FusedTile with concatenated tensor and full provenance.
        """
        # ---- Task 2: Resolution alignment guarantee ----
        optical, sar = self._align_resolutions(optical, sar)

        # ---- Task 3: Occlusion-aware channel masking ----
        cloud_frac = self._extract_cloud_fraction(optical_meta)
        suppress_optical = cloud_frac > self.cloud_threshold
        sar_weight = self.sar_boost if suppress_optical else 1.0

        optical_mod = optical.copy()
        sar_mod = sar.copy()

        if suppress_optical:
            optical_mod = np.zeros_like(optical_mod)
            sar_mod = sar_mod * self.sar_boost
            logger.info(
                "Optical suppressed (cloud fraction: %.1f%%), SAR dominant (%.1fx boost)",
                cloud_frac * 100, self.sar_boost,
            )

        # ---- Task 1: Channel concatenation ----
        fused_tensor = np.concatenate([optical_mod, sar_mod], axis=0).astype(np.float32)
        C_opt = optical.shape[0]
        C_sar = sar.shape[0]

        logger.info(
            "Fused: %d optical + %d SAR = %d channels, shape=%s",
            C_opt, C_sar, fused_tensor.shape[0], fused_tensor.shape,
        )

        # ---- Build FusedTile ----
        tile = FusedTile(
            tensor=fused_tensor,
            cloud_fraction=cloud_frac,
            optical_available=not suppress_optical,
            sar_available=True,
            optical_suppressed=suppress_optical,
            sar_weight_multiplier=sar_weight,
            num_optical_channels=C_opt,
            num_sar_channels=C_sar,
            source_optical=getattr(optical_meta, "source_name", "unknown"),
            source_sar=getattr(sar_meta, "source_name", "unknown"),
            shape=fused_tensor.shape,
        )

        # ---- Task 5: Sanity check visualization ----
        if save_debug:
            self._save_debug_image(optical, sar, fused_tensor, tile)

        return tile

    # -----------------------------------------------------------------------
    # Task 2 - Resolution alignment
    # -----------------------------------------------------------------------

    def _align_resolutions(
        self, optical: np.ndarray, sar: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure optical and SAR have identical spatial dimensions (H, W).

        If SAR is coarser (smaller), upsample to optical size using
        bicubic interpolation.  If SAR is larger (unlikely), downsample.
        """
        H_opt, W_opt = optical.shape[1], optical.shape[2]
        H_sar, W_sar = sar.shape[1], sar.shape[2]

        if (H_opt, W_opt) == (H_sar, W_sar):
            logger.info("Resolutions already aligned: %dx%d", H_opt, W_opt)
            return optical, sar

        logger.info(
            "Aligning resolutions: optical %dx%d ← SAR %dx%d",
            H_opt, W_opt, H_sar, W_sar,
        )

        try:
            import cv2

            sar_aligned = np.zeros((sar.shape[0], H_opt, W_opt), dtype=np.float32)
            for c in range(sar.shape[0]):
                sar_aligned[c] = cv2.resize(
                    sar[c], (W_opt, H_opt), interpolation=cv2.INTER_CUBIC
                )
            return optical, sar_aligned
        except ImportError:
            logger.warning("cv2 not available - using scipy zoom fallback")
            from scipy.ndimage import zoom

            zoom_factors = (1.0, H_opt / H_sar, W_opt / W_sar)
            sar_aligned = zoom(sar, zoom_factors, order=3)
            return optical, sar_aligned.astype(np.float32)

    # -----------------------------------------------------------------------
    # Task 3 helper - extract cloud fraction
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_cloud_fraction(meta: Optional[object]) -> float:
        """Safely extract cloud_fraction from optical metadata."""
        if meta is None:
            return 0.0
        if hasattr(meta, "cloud_fraction"):
            return float(meta.cloud_fraction)
        if isinstance(meta, dict):
            return float(meta.get("cloud_fraction", 0.0))
        return 0.0

    # -----------------------------------------------------------------------
    # Task 5 - Sanity check visualization
    # -----------------------------------------------------------------------

    def _save_debug_image(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        fused: np.ndarray,
        tile: FusedTile,
    ) -> None:
        """
        Create a side-by-side debug montage:
            [ Optical RGB ]  [ SAR VV ]  [ Fused ch 0,1,2 ]

        Saves to logs/fusion_debug.png.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available - skipping debug image")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Fusion Debug - Optical: {tile.source_optical} | SAR: {tile.source_sar}\n"
            f"Cloud: {tile.cloud_fraction:.1%} | "
            f"Optical suppressed: {tile.optical_suppressed} | "
            f"SAR weight: {tile.sar_weight_multiplier:.1f}x",
            fontsize=10,
        )

        # Panel 1: Optical RGB (first 3 bands, or grayscale if fewer)
        opt_vis = optical[:3] if optical.shape[0] >= 3 else optical[:1]
        if opt_vis.shape[0] == 3:
            opt_rgb = np.transpose(opt_vis, (1, 2, 0))
            # Clip to [0, 1] and brighten for visibility
            opt_rgb = np.clip(opt_rgb, 0, 1)
            axes[0].imshow(opt_rgb)
        elif opt_vis.shape[0] == 1:
            axes[0].imshow(opt_vis[0], cmap="gray")
        axes[0].set_title("Optical (RGB / band 0)")
        axes[0].axis("off")

        # Panel 2: SAR VV
        sar_vv = sar[0]  # VV channel
        axes[1].imshow(sar_vv, cmap="gray")
        axes[1].set_title("SAR VV")
        axes[1].axis("off")

        # Panel 3: Fused first 3 channels
        fused_vis = fused[:3]
        if fused_vis.shape[0] == 3:
            fused_rgb = np.transpose(fused_vis, (1, 2, 0))
            fused_rgb = np.clip(fused_rgb, 0, 1)
            axes[2].imshow(fused_rgb)
        else:
            axes[2].imshow(fused_vis[0], cmap="gray")
        axes[2].set_title("Fused (ch 0,1,2)")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.debug_dir, "fusion_debug.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Debug image saved to %s", out_path)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fuse_optical_sar(
    optical: np.ndarray,
    sar: np.ndarray,
    optical_meta: Optional[object] = None,
    sar_meta: Optional[object] = None,
    cloud_threshold: float = CLOUD_SUPPRESSION_THRESHOLD,
) -> FusedTile:
    """
    One-shot convenience wrapper.

    >>> from part_a_vision.optical_reader import OpticalPreprocessor
    >>> from part_a_vision.sar_reader import SARPreprocessor
    >>> opt_tensor, opt_meta = OpticalPreprocessor().process(None)
    >>> sar_tensor, sar_meta = SARPreprocessor().process(None, None)
    >>> tile = fuse_optical_sar(opt_tensor, sar_tensor, opt_meta, sar_meta)
    >>> print(tile.tensor.shape)  # (6, 512, 512)
    """
    module = FusionModule(cloud_threshold=cloud_threshold)
    return module.fuse(optical, sar, optical_meta, sar_meta)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6 - Optical-SAR Fusion")
    parser.add_argument("--target-size", type=int, default=512, help="Target spatial size (default: 512)")
    parser.add_argument("--cloud-threshold", type=float, default=CLOUD_SUPPRESSION_THRESHOLD,
                        help=f"Cloud fraction threshold for optical suppression (default: {CLOUD_SUPPRESSION_THRESHOLD})")
    parser.add_argument("--no-debug", action="store_true", help="Skip debug image")
    args = parser.parse_args()

    # Generate synthetic inputs from Phase 4 and 5
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from optical_reader import OpticalPreprocessor
    from sar_reader import SARPreprocessor

    print("=== Phase 4: Optical Preprocessing ===")
    opt_pp = OpticalPreprocessor(target_size=args.target_size)
    opt_tensor, opt_meta = opt_pp.process(None)

    print(f"  Optical: {opt_tensor.shape}, synthetic={opt_meta.is_synthetic}")

    print("\n=== Phase 5: SAR Preprocessing ===")
    sar_pp = SARPreprocessor(target_size=args.target_size)
    sar_tensor, sar_meta = sar_pp.process(None, None)

    print(f"  SAR: {sar_tensor.shape}, synthetic={sar_meta.is_synthetic}")

    print("\n=== Phase 6: Fusion ===")
    fusion = FusionModule(cloud_threshold=args.cloud_threshold)
    tile = fusion.fuse(
        opt_tensor, sar_tensor, opt_meta, sar_meta,
        save_debug=not args.no_debug,
    )

    print(f"  Fused tensor:   {tile.tensor.shape}")
    print(f"  Cloud fraction:  {tile.cloud_fraction:.1%}")
    print(f"  Optical active:  {tile.optical_available}")
    print(f"  SAR weight:      {tile.sar_weight_multiplier:.1f}x")
    print(f"  Optical source:  {tile.source_optical}")
    print(f"  SAR source:      {tile.source_sar}")

    if tile.optical_suppressed:
        print("\n  [!]  OPTICAL SUPPRESSED - SAR dominant (cloud > 50%)")

    print("\n=== Phase 6 COMPLETE ===")