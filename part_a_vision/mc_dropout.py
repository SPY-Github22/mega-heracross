import torch
import torch.nn as nn

def enable_dropout(model):
    """
    Enables dropout layers during inference for Monte Carlo Dropout.
    Keeps everything else (like BatchNorm/LayerNorm) in eval mode to prevent 
    corrupting running statistics.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_infer(model, x, n_samples=30):
    """
    Monte Carlo Dropout Inference.
    Runs the input through the network `n_samples` times with dropout active.
    
    Args:
        model: The PyTorch model.
        x: Input tensor (B, C, H, W)
        n_samples: Number of forward passes.
        
    Returns:
        mean_prob: Average probability map across samples (B, 1, H, W)
        uncertainty_map: Standard deviation across samples (B, 1, H, W)
    """
    # 1. Put model in eval mode to freeze Normalization stats
    model.eval()
    
    # 2. Re-enable Dropout layers specifically
    enable_dropout(model)
    
    probs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.sigmoid(logits)
            probs.append(prob)
            
    # stack along a new dimension at dim=0: shape (n_samples, B, 1, H, W)
    probs_stack = torch.stack(probs, dim=0)
    
    mean_prob = probs_stack.mean(dim=0)
    uncertainty_map = probs_stack.std(dim=0)
    
    return mean_prob, uncertainty_map
