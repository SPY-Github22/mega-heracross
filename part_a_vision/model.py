"""
Phase 11 - SegFormer B3 Integration
====================================
Centralized model factory routing between:
1. SegFormer B3 (Primary architecture)
2. DeepLabV3+ with ResNet-50 (Fallback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SegformerB3Custom(nn.Module):
    """
    SegFormer B3 adapted for 6-channel input and binary road segmentation.
    Uses ImageNet pretrained MiT-B3 encoder, randomly initialized decoder.
    """
    def __init__(self, input_channels=10, num_classes=1):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # Load the base ImageNet pretrained MiT-B3 encoder config
        # We set num_labels to num_classes (1) for binary BCE loss compatibility.
        logger.info(f"Loading SegFormer MiT-B3 (channels={input_channels}, classes={num_classes})")
        
        # ignore_mismatched_sizes=True allows us to load the encoder weights even
        # though the default decoder head expects 150 classes (ADE20K) or 1000 (ImageNet).
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # ── Adapt input to 6 channels ──
        # Stage 1 patch embedding is a Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
        if hasattr(self.model.segformer, 'encoder'):
            old_proj = self.model.segformer.encoder.patch_embeddings[0].proj
        else:
            old_proj = self.model.segformer.stages[0].patch_embeddings.proj
        new_proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        # Transfer ImageNet RGB weights to first 3 channels
        with torch.no_grad():
            if old_proj.weight.shape[1] >= 3:
                new_proj.weight[:, :3] = old_proj.weight[:, :3]
            # Initialize the extra channels (e.g. SAR) with scaled random noise
            for c in range(3, input_channels):
                nn.init.normal_(new_proj.weight[:, c:c+1], std=old_proj.weight.std().item())
            if old_proj.bias is not None:
                new_proj.bias = old_proj.bias

        # Inject the modified layer back into the model
        if hasattr(self.model.segformer, 'encoder'):
            self.model.segformer.encoder.patch_embeddings[0].proj = new_proj
        else:
            self.model.segformer.stages[0].patch_embeddings.proj = new_proj

        # ── Add CBAM layers for Phase 12 ──
        try:
            from cbam import CBAM
            # MiT-B3 output channels: [64, 128, 320, 512]
            self.cbams = nn.ModuleList([
                CBAM(64), CBAM(128), CBAM(320), CBAM(512)
            ])
            logger.info("CBAM modules initialized for SegFormer decoder.")
        except ImportError:
            logger.warning("cbam.py not found. CBAM will not be used.")
            self.cbams = None

        # ── Add Cloud Fusion for Phase 13 ──
        try:
            from cloud_fusion import CloudEstimator, FusionGate
            self.cloud_estimator = CloudEstimator()
            self.fusion_gate = FusionGate()
            logger.info("Dynamic Cloud Fusion modules initialized.")
        except ImportError:
            logger.warning("cloud_fusion.py not found. Cloud Fusion will not be used.")
            self.cloud_estimator = None
            self.fusion_gate = None

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_gate: bool = False):
        """
        Args:
            x: (B, 6, H, W) input tensor
            return_attention: if True, returns (logits, attention_maps)
            return_gate: if True, returns (logits, gate_weights, cloud_frac)
        Returns:
            logits: (B, 1, H, W) upsampled to match input resolution
        """
        gate_weights = None
        cloud_frac = None
        
        # ── Dynamic Cloud Fusion (Phase 13) & Temporal Gating (Phase 14) ──
        if getattr(self, 'cloud_estimator', None) is not None and getattr(self, 'fusion_gate', None) is not None:
            # Phase 22: x has 12 channels: 0-3 (Opt), 4-5 (NDVI/NDWI), 6-7 (SAR), 8-11 (Temporal)
            opt_x = x[:, :4, :, :]
            idx_x = x[:, 4:6, :, :]
            sar_x = x[:, 6:8, :, :]
            
            # Estimate cloud fraction and spatial map
            cloud_frac, cloud_map = self.cloud_estimator(opt_x)
            
            # Predict fusion weights
            w_opt, w_sar = self.fusion_gate(cloud_frac)
            gate_weights = (w_opt, w_sar)
            
            # Apply weights and spatial mask
            opt_x_weighted = opt_x * w_opt * (1.0 - cloud_map)
            idx_x_weighted = idx_x * w_opt * (1.0 - cloud_map) # weight indices like optical
            sar_x_weighted = sar_x * w_sar
            
            if x.shape[1] == 12:
                temporal_diff = x[:, 8:12, :, :]
                temporal_diff_weighted = temporal_diff * w_opt * (1.0 - cloud_map)
                x = torch.cat([opt_x_weighted, idx_x_weighted, sar_x_weighted, temporal_diff_weighted], dim=1)
            else:
                x = torch.cat([opt_x_weighted, idx_x_weighted, sar_x_weighted], dim=1)

        # Forward pass through encoder
        outputs = self.model.segformer(
            x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = list(outputs.hidden_states)
        
        # Apply CBAM to each multiscale feature map
        attention_maps = []
        if getattr(self, 'cbams', None) is not None:
            for i in range(len(hidden_states)):
                h, sa = self.cbams[i](hidden_states[i], return_attention=True)
                hidden_states[i] = h
                attention_maps.append(sa)
                
        # Forward pass through decoder
        logits = self.model.decode_head(hidden_states)
        
        # Upsample back to original HxW using bilinear interpolation
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        
        if return_gate:
            if return_attention:
                return upsampled_logits, attention_maps, gate_weights, cloud_frac
            return upsampled_logits, gate_weights, cloud_frac
            
        if return_attention:
            return upsampled_logits, attention_maps
        return upsampled_logits


def build_deeplabv3plus(input_channels=6, num_classes=1, backbone="resnet50", pretrained=True):
    """
    Fallback: DeepLabV3+ with custom input channels.
    """
    try:
        import segmentation_models_pytorch as smp
        logger.info("Using segmentation-models-pytorch for DeepLabV3+")
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=input_channels,
            classes=num_classes,
        )
        return model
    except ImportError:
        logger.warning("segmentation-models-pytorch not available - falling back to torchvision DeepLabV3")

    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1" if pretrained else None)
    
    model.classifier = DeepLabHead(2048, num_classes)
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] >= 3:
            new_conv.weight[:, :3] = old_conv.weight[:, :3]
        for c in range(3, input_channels):
            nn.init.normal_(new_conv.weight[:, c:c + 1], std=old_conv.weight.std().item())

    model.backbone.conv1 = new_conv
    model.aux_classifier = None
    return model


def build_model(backbone: str = "segformer_b3", input_channels: int = 6, num_classes: int = 1) -> nn.Module:
    """
    Model Factory routing based on backbone string.
    """
    if "segformer" in backbone.lower():
        return SegformerB3Custom(input_channels=input_channels, num_classes=num_classes)
    else:
        # Fallback to DeepLabV3+ with ResNet-50
        return build_deeplabv3plus(input_channels=input_channels, num_classes=num_classes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Phase 11 Model Sanity Check ===")
    
    # Check SegFormer
    print("\nBuilding SegFormer B3...")
    model_seg = build_model("segformer_b3", input_channels=6, num_classes=1)
    
    dummy_input = torch.randn(2, 6, 512, 512)
    print(f"Input shape:  {dummy_input.shape}")
    
    output = model_seg(dummy_input)
    print(f"Output shape: {output.shape} (Expected: 2, 1, 512, 512)")
    assert output.shape == (2, 1, 512, 512)
    
    param_count = sum(p.numel() for p in model_seg.parameters())
    print(f"SegFormer B3 parameters: {param_count:,}")
    
    print("\n=== Sanity Check Passed ✅ ===")
