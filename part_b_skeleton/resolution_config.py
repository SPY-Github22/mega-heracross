from dataclasses import dataclass

@dataclass
class ResolutionConfig:
    snap_m: float
    effective_resolution_m: float
    second_pass_snap_m: float

def make_config(resolution_m: float, source: str) -> ResolutionConfig:
    """
    Generate dynamic configuration based on the input mask resolution.
    For LISS-IV (5.8m), snap distances scale appropriately.
    """
    return ResolutionConfig(
        snap_m=max(resolution_m * 2.0, 10.0),
        effective_resolution_m=resolution_m,
        second_pass_snap_m=max(resolution_m * 3.0, 15.0)
    )

def print_resolution_config(cfg: ResolutionConfig) -> None:
    print("\n[Phase 18] RESOLUTION AWARE CONFIGURATION")
    print(f"  Effective Resolution : {cfg.effective_resolution_m:.2f} m")
    print(f"  Healing Snap (Pass 1): {cfg.snap_m:.2f} m")
    print(f"  Healing Snap (Pass 2): {cfg.second_pass_snap_m:.2f} m")
