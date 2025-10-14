# Debiasing Fix Guide

## The Problem

You were getting **less uniform** distributions after steering due to **inconsistent aggregation methods** between training, steering, and analysis.

## Root Causes

1. **Training**: Used `aggregate_spatial='flatten'` → [B, S*C] per image
2. **Analysis**: Used `aggregate_spatial='all_positions'` → [B*S, C] per position ❌ MISMATCH
3. **Steering**: Tried to process per-position but classifier expects flattened

## The Solution: Consistent Flattened Approach

### Training (Cell 10):
```python
aggregate_spatial='flatten'  # [B, S, C] → [B, S*C]
# Result: [20 images, 1,310,720 features] per ethnicity
```

### Steering Hook (New Cell):
```python
# Use FIXED_debias_hook.py

def debias_hook(module, input, output, alpha=1.0):
    # Flatten: [B, S, C] → [B, S*C]
    B, S, C = output.shape
    act_flat = output.reshape(B, S * C)
    
    # Compute gradients (per image)
    # grad shape: [B, S*C]
    
    # Apply steering
    steered_flat = act_flat + alpha * steering
    
    # Reshape back: [B, S*C] → [B, S, C]
    return steered_flat.reshape(B, S, C)
```

### Analysis (New Cell):
```python
# Use FIXED_analyze_function.py

def analyze_ethnicity_distribution(images):
    # Extract with SAME aggregation as training
    activations = batch_cache_image_activations(
        aggregate_spatial='flatten'  # ✓ Matches training
    )
    # Shape: [num_images, S*C]
    
    # Predict directly (one prediction per image)
    predictions = classifier(activations).argmax(dim=1)
```

## Steps to Fix:

1. **Re-run Cell 10** (training with `aggregate_spatial='flatten'`)
   - This trains classifier on [20, 1,310,720] per ethnicity

2. **Load fixed hook**: 
   - Copy contents of `FIXED_debias_hook.py` into new cell
   - Run it

3. **Load fixed analysis**:
   - Copy contents of `FIXED_analyze_function.py` into new cell  
   - Run it

4. **Run experiment**:
   - Should now properly debias!

## Why This Works:

```
Image → [S, C] activations
           ↓
      Flatten to [S*C]
           ↓
   Classifier (trained on this)
           ↓
    Gradients w.r.t [S*C]
           ↓
   Steer the flattened repr
           ↓
  Reshape back to [S, C]
           ↓
    Continues generation
```

**Consistent representation throughout!**

