import torch
import os
from model import build_model

def export_to_onnx(output_path="outputs/model.onnx"):
    print("Loading 12-channel SegFormer model...")
    model = build_model(input_channels=12)
    model.eval()
    
    print("Creating dummy input tensor (1, 12, 512, 512)...")
    dummy_input = torch.randn(1, 12, 512, 512)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print("Export successful!")

if __name__ == "__main__":
    export_to_onnx()
