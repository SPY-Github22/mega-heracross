import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Reduction ratio controls bottleneck in the MLP
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Avg and Max Pool
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        
        # Pass through MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        
        # Combine and apply sigmoid
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        
        # Convolution applied to concatenated (avg_pool, max_pool) across channels
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Pool across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        pool_out = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.conv(pool_out))
        return spatial_att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applied sequentially: Channel Attention -> Spatial Attention
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x, return_attention=False):
        # 1. Channel Attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 2. Spatial Attention
        sa = self.spatial_attention(x_ca)
        x_out = x_ca * sa
        
        if return_attention:
            return x_out, sa
        return x_out


if __name__ == "__main__":
    # Sanity check
    dummy = torch.randn(2, 64, 128, 128)
    cbam = CBAM(64)
    out, sa = cbam(dummy, return_attention=True)
    print(f"Input: {dummy.shape}")
    print(f"Output: {out.shape} (Expected: 2, 64, 128, 128)")
    print(f"Spatial Att: {sa.shape} (Expected: 2, 1, 128, 128)")
    assert out.shape == dummy.shape
    assert sa.shape == (2, 1, 128, 128)
    print("CBAM shape checks passed ✅")
