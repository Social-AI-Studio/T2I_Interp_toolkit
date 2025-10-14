# Test if steering direction is correct
# Run this in a notebook cell to verify the gradient logic

import torch
import numpy as np

print("="*80)
print("TESTING STEERING DIRECTION")
print("="*80)

# Get classifier input dimension
expected_dim = list(classifier.classifier.children())[0].in_features
print(f"\nClassifier expects: {expected_dim}-dimensional input")

# Create dummy activations
dummy_act = torch.randn(10, expected_dim, device=classifier.device, dtype=torch.float32)

# Test WITHOUT steering (baseline)
print("\n1. BASELINE (no steering):")
classifier.classifier.eval()
with torch.no_grad():
    logits_before = classifier.classifier(dummy_act)
    probs_before = torch.softmax(logits_before, dim=-1)
    
    # Check uniformity
    mean_prob = probs_before.mean(dim=0)
    target_uniform = 1.0 / classifier.num_classes
    uniformity_before = torch.abs(mean_prob - target_uniform).sum().item()
    
    print(f"   Mean probabilities across samples: {mean_prob.cpu().numpy()}")
    print(f"   Target uniform: {target_uniform:.4f}")
    print(f"   Uniformity score: {uniformity_before:.4f}")

# Test WITH steering (using our hook logic)
print("\n2. AFTER STEERING (alpha=1.0):")
dummy_act_tensor = dummy_act.clone()

with torch.enable_grad():
    dummy_act_tensor.requires_grad_(True)
    classifier.classifier.train()
    
    logits = classifier.classifier(dummy_act_tensor)
    
    # OUR LOSS FORMULA
    mean_logits = logits.mean(dim=1, keepdim=True)
    deviation = logits - mean_logits
    
    avoid_mask = deviation > 0
    target_mask = deviation < 0
    
    loss = 0.0
    if avoid_mask.any():
        loss += (logits * avoid_mask.float()).sum() / (avoid_mask.sum() + 1e-8)
    if target_mask.any():
        loss -= (logits * target_mask.float()).sum() / (target_mask.sum() + 1e-8)
    
    print(f"   Loss value: {loss.item():.4f}")
    
    grad = torch.autograd.grad(loss, dummy_act_tensor)[0]
    steering = -grad
    
    classifier.classifier.eval()

# Apply steering
dummy_act_steered = dummy_act.detach() + 1.0 * steering.detach()

# Check uniformity after steering
with torch.no_grad():
    logits_after = classifier.classifier(dummy_act_steered)
    probs_after = torch.softmax(logits_after, dim=-1)
    
    mean_prob_after = probs_after.mean(dim=0)
    uniformity_after = torch.abs(mean_prob_after - target_uniform).sum().item()
    
    print(f"   Mean probabilities after steering: {mean_prob_after.cpu().numpy()}")
    print(f"   Uniformity score: {uniformity_after:.4f}")

# Compare
print("\n3. RESULT:")
if uniformity_after < uniformity_before:
    improvement = ((uniformity_before - uniformity_after) / uniformity_before) * 100
    print(f"   ✓ IMPROVED: {uniformity_before:.4f} → {uniformity_after:.4f} ({improvement:.1f}% better)")
else:
    worsening = ((uniformity_after - uniformity_before) / uniformity_before) * 100
    print(f"   ❌ WORSENED: {uniformity_before:.4f} → {uniformity_after:.4f} ({worsening:.1f}% worse)")
    print(f"   → BUG: Steering direction or loss formula is incorrect!")

print("\n" + "="*80)

