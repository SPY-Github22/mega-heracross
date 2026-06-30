import torch

def tta_infer(model, x):
    """
    Test-Time Augmentation (TTA) inference.
    x: (B, C, H, W) tensor
    Returns:
        mean_prob: (B, 1, H, W)
        uncertainty: (B, 1, H, W) standard deviation
    """
    transforms = [
        lambda t: t,
        lambda t: torch.rot90(t, k=1, dims=[2, 3]),
        lambda t: torch.rot90(t, k=2, dims=[2, 3]),
        lambda t: torch.rot90(t, k=3, dims=[2, 3]),
        lambda t: torch.flip(t, dims=[3]),
        lambda t: torch.rot90(torch.flip(t, dims=[3]), k=1, dims=[2, 3]),
        lambda t: torch.rot90(torch.flip(t, dims=[3]), k=2, dims=[2, 3]),
        lambda t: torch.rot90(torch.flip(t, dims=[3]), k=3, dims=[2, 3]),
    ]
    
    inverse_transforms = [
        lambda t: t,
        lambda t: torch.rot90(t, k=-1, dims=[2, 3]),
        lambda t: torch.rot90(t, k=-2, dims=[2, 3]),
        lambda t: torch.rot90(t, k=-3, dims=[2, 3]),
        lambda t: torch.flip(t, dims=[3]),
        lambda t: torch.flip(torch.rot90(t, k=-1, dims=[2, 3]), dims=[3]),
        lambda t: torch.flip(torch.rot90(t, k=-2, dims=[2, 3]), dims=[3]),
        lambda t: torch.flip(torch.rot90(t, k=-3, dims=[2, 3]), dims=[3]),
    ]
    
    probs = []
    
    for tfm, inv_tfm in zip(transforms, inverse_transforms):
        x_aug = tfm(x)
        with torch.no_grad():
            logits = model(x_aug)
            # Ensure model output is just logits, unpack if a tuple was accidentally returned
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.sigmoid(logits)
        
        prob_inv = inv_tfm(prob)
        probs.append(prob_inv)
        
    probs_stack = torch.stack(probs, dim=0) # (8, B, 1, H, W)
    mean_prob = probs_stack.mean(dim=0)
    uncertainty = probs_stack.std(dim=0)
    
    return mean_prob, uncertainty
