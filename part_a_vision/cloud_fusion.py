import torch
import torch.nn as nn

class CloudEstimator(nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, opt_x):
        """
        opt_x: (B, C_opt, H, W)
        Returns:
            cloud_fraction: (B, 1) scalar for percentage of cloud cover
            cloud_map: (B, 1, H, W) spatial cloud mask
        """
        # Calculate mean across optical channels (assuming first 3 are RGB)
        # Synthetic clouds are white/bright, so mean RGB is high
        b, c, h, w = opt_x.shape
        rgb_mean = opt_x[:, :3].mean(dim=1, keepdim=True)
        
        # Adaptive thresholding: if it's very bright, it's cloud
        cloud_map = (rgb_mean > self.threshold).float()
        
        # Scalar fraction of cloud cover
        cloud_fraction = cloud_map.mean(dim=(1, 2, 3)).unsqueeze(-1)
        
        return cloud_fraction, cloud_map


class FusionGate(nn.Module):
    def __init__(self):
        super().__init__()
        # 2-layer MLP (1 -> 8 -> 2)
        self.mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, cloud_fraction):
        """
        cloud_fraction: (B, 1)
        Returns:
            w_opt: (B, 1, 1, 1) weight for optical channels
            w_sar: (B, 1, 1, 1) weight for SAR channels
        """
        weights = self.mlp(cloud_fraction)  # (B, 2)
        
        w_opt = weights[:, 0].view(-1, 1, 1, 1)
        w_sar = weights[:, 1].view(-1, 1, 1, 1)
        
        return w_opt, w_sar

if __name__ == "__main__":
    dummy_opt = torch.randn(2, 4, 512, 512)
    estimator = CloudEstimator()
    gate = FusionGate()
    
    frac, cmap = estimator(dummy_opt)
    w_opt, w_sar = gate(frac)
    
    print(f"Cloud Fraction: {frac.shape} (Expected: 2, 1)")
    print(f"Cloud Map: {cmap.shape} (Expected: 2, 1, 512, 512)")
    print(f"w_opt: {w_opt.shape}, w_sar: {w_sar.shape}")
