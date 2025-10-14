# CORRECTED debias_hook for flattened spatial training
# Copy and paste this into a new notebook cell

def debias_hook(module, input, output, alpha=1.0):
    """
    Hook for classifier trained on flattened spatial: [B, S*C].
    Processes whole images per batch element.
    """
    import torch
    import numpy as np
    
    act = output
    original_shape = act.shape
    
    # Flatten spatial dimensions: [B, S, C] → [B, S*C]
    if act.ndim == 3:
        B, S, C = act.shape
        act_flat = act.reshape(B, S * C)
    elif act.ndim == 4:
        B, C, H, W = act.shape
        act_flat = act.reshape(B, C * H * W)
    else:
        act_flat = act.clone()
        if act.ndim == 3:
            B, S, C = act.shape[0], act.shape[1], act.shape[2]
    
    # Create independent tensor for classifier
    act_np = act_flat.detach().cpu().numpy().copy()
    act_tensor = torch.from_numpy(act_np).to(
        device=classifier.device,
        dtype=torch.float32
    )
    
    # Compute gradients through classifier
    with torch.enable_grad():
        act_tensor.requires_grad_(True)
        classifier.classifier.train()
        
        # Forward: [B, S*C] → [B, num_classes]
        logits = classifier.classifier(act_tensor)
        
        # K_Steering unsupervised loss
        mean_logits = logits.mean(dim=1, keepdim=True)
        deviation = logits - mean_logits
        
        avoid_mask = deviation > 0  # Over-represented
        target_mask = deviation < 0  # Under-represented
        
        loss = 0.0
        if avoid_mask.any():
            loss += (logits * avoid_mask.float()).sum() / (avoid_mask.sum() + 1e-8)
        if target_mask.any():
            loss -= (logits * target_mask.float()).sum() / (target_mask.sum() + 1e-8)
        
        # Gradient: [B, S*C]
        grad = torch.autograd.grad(loss, act_tensor)[0]
        steering = -grad
        
        classifier.classifier.eval()
    
    # Apply steering
    steering_np = steering.detach().cpu().numpy()
    steering_tensor = torch.from_numpy(steering_np.copy()).to(
        device=act_flat.device,
        dtype=act_flat.dtype
    )
    steered_flat = act_flat.detach() + alpha * steering_tensor
    
    # Reshape back: [B, S*C] → [B, S, C]
    if len(original_shape) == 3:
        return steered_flat.reshape(B, S, C)
    elif len(original_shape) == 4:
        return steered_flat.reshape(B, C, H, W)
    else:
        return steered_flat


print("="*80)
print("✓ Corrected debias_hook loaded")
print("="*80)
print("\nThis hook:")
print("  - Flattens spatial: [B, S, C] → [B, S*C]")
print("  - Processes per image (B samples)")
print("  - Computes per-image gradients")
print("  - Reshapes back to [B, S, C]")
print("\nMatches training on aggregate_spatial='flatten'")
print("="*80)

