# Debugging Checklist: Why Is Uniformity Getting Worse?

Run these checks in notebook cells:

## Check 1: Verify Analysis Aggregation
```python
# Paste from check_analysis_aggregation.py
# Should show: aggregate_spatial='flatten'
```

## Check 2: Test Steering Direction
```python
# Paste from test_steering_direction.py
# Should show: uniformity IMPROVES after steering
# If it worsens → Bug in loss or gradient direction
```

## Check 3: Verify Training Aggregation
```python
print("Training configuration:")
print(f"  Layer: {layer_to_probe}")
# Check what you used in Cell 10
# Should be: aggregate_spatial='flatten'
```

## Check 4: Verify Classifier Dimensions
```python
expected_dim = list(classifier.classifier.children())[0].in_features
print(f"Classifier expects: {expected_dim} dimensions")

# For flattened approach:
# Should be: S*C = 4096 * 320 = 1,310,720

# Check if it matches:
test_shape = (2, 4096, 320)  # [B, S, C]
flattened_dim = test_shape[1] * test_shape[2]
print(f"Flattened dim: {flattened_dim}")
print(f"Match: {expected_dim == flattened_dim}")
```

## Check 5: Inspect Baseline Distribution
```python
print("Baseline distribution:")
for eth, prob in baseline_dist.items():
    print(f"  {eth}: {prob:.3f}")

# Is baseline already uniform?
# If yes, steering can't improve it!
```

## Check 6: Check Alpha Values
```python
print(f"Alpha values being tested: {alphas_to_try}")

# If alpha too large: might overshoot
# If alpha too small: no effect
# Typical range: 0.01 - 1.0
```

## Common Bugs:

### Bug #1: Dimension Mismatch
**Symptom**: RuntimeError about shapes
**Fix**: Ensure training, steering, analysis all use 'flatten'

### Bug #2: Wrong Gradient Direction
**Symptom**: Uniformity gets worse
**Fix**: Ensure `steering = -grad` and `steered = act + alpha * steering`

### Bug #3: Analysis Mismatch
**Symptom**: Wrong predictions
**Fix**: Analysis must use SAME aggregation as training

### Bug #4: Classifier-Free Guidance Issue
**Symptom**: Batch size is 2 during generation (CFG), but loss expects 1
**Fix**: Handle B=2 case (conditional + unconditional samples)

## Expected Results:

With correct implementation:
- Baseline uniformity: ~1.5-2.0 (biased)
- After steering: ~0.5-1.0 (less biased)
- Improvement: 30-50%

If you see:
- Baseline: 1.7
- After: 2.0
- This means steering is WORSENING bias → BUG!

