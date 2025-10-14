# CORRECTED: debias_hook for flattened spatial training
# Paste this into a notebook cell and run it

def debias_hook_v2(module, input, output, alpha=1.0):
    """
    Hook for classifier trained on flattened spatial: [B, S*C].
    Processes whole images, not individual positions.
    """
    import torch
    import numpy as np
    
    act = output
    original_shape = act.shape
    
    # Flatten: [B, S, C] → [B, S*C]
    if act.ndim == 3:
        B, S, C = act.shape
        act_flat = act.reshape(B, S * C)
    elif act.ndim == 4:
        B, C, H, W = act.shape
        act_flat = act.reshape(B, C * H * W)
    else:
        act_flat = act.clone()
        B = act.shape[0]
    
    # Independent tensor for classifier
    act_np = act_flat.detach().cpu().numpy().copy()
    act_tensor = torch.from_numpy(act_np).to(classifier.device, dtype=torch.float32)
    
    # Compute gradients
    with torch.enable_grad():
        act_tensor.requires_grad_(True)
        classifier.classifier.train()
        
        logits = classifier.classifier(act_tensor)  # [B, num_classes]
        
        # K_Steering loss: minimize over-represented, maximize under-represented
        mean_logits = logits.mean(dim=1, keepdim=True)
        deviation = logits - mean_logits
        
        avoid_mask = deviation > 0  # Over-represented classes
        target_mask = deviation < 0  # Under-represented classes
        
        loss = 0.0
        if avoid_mask.any():
            loss += (logits * avoid_mask.float()).sum() / (avoid_mask.sum() + 1e-8)
        if target_mask.any():
            loss -= (logits * target_mask.float()).sum() / (target_mask.sum() + 1e-8)
        
        grad = torch.autograd.grad(loss, act_tensor)[0]
        steering = -grad
        
        classifier.classifier.eval()
    
    # Apply steering
    steering_np = steering.detach().cpu().numpy()
    steering_tensor = torch.from_numpy(steering_np.copy()).to(act_flat.device, dtype=act_flat.dtype)
    steered_flat = act_flat.detach() + alpha * steering_tensor
    
    # Reshape back to original shape
    if len(original_shape) == 3:
        return steered_flat.reshape(B, S, C)
    elif len(original_shape) == 4:
        return steered_flat.reshape(B, C, H, W)
    else:
        return steered_flat


# Also update analyze function
def analyze_ethnicity_distribution_v2(images, batch_size=4):
    """Analyze using flattened spatial (matching training)"""
    print(f"\nAnalyzing {len(images)} images...")
    
    activations = batch_cache_image_activations(
        model=model,
        images=images,
        layer_name=layer_to_probe,
        timestep=500,
        batch_size=batch_size,
        encode_batch_size=8,
        aggregate_spatial='flatten',  # Match training
        empty_text_embed=True
    )
    
    # Shape: [num_images, S*C]
    print(f"Extracted activations shape: {activations.shape}")
    
    # Predict
    classifier.classifier.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch_acts = activations[i:i+batch_size]
            batch_tensor = torch.tensor(batch_acts, dtype=torch.float32, device=classifier.device)
            
            logits = classifier.classifier(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    # Count
    counts = np.bincount(predictions, minlength=len(classifier.class_names))
    distribution = {
        classifier.class_names[i]: counts[i] / len(images)
        for i in range(len(classifier.class_names))
    }
    
    return distribution


print("="*80)
print("TO USE THESE CORRECTED FUNCTIONS:")
print("="*80)
print("1. Run: debias_hook = debias_hook_v2")
print("2. Run: analyze_ethnicity_distribution = analyze_ethnicity_distribution_v2")
print("3. Re-train classifier with aggregate_spatial='flatten'")
print("4. Run experiment")
print("="*80)

