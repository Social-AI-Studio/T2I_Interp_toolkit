# CORRECTED analyze_ethnicity_distribution
# Copy and paste this into a new notebook cell

def analyze_ethnicity_distribution(images, batch_size=4):
    """
    Analyze ethnicity distribution - MUST match training aggregation method.
    Training uses 'flatten': [B, S, C] → [B, S*C]
    """
    print(f"\nAnalyzing ethnicity distribution for {len(images)} images...")
    
    # Extract activations using SAME method as training
    activations = batch_cache_image_activations(
        model=model,
        images=images,
        layer_name=layer_to_probe,
        timestep=500,
        batch_size=batch_size,
        encode_batch_size=8,
        aggregate_spatial='flatten',  # ✓ Match training: FLATTEN
        empty_text_embed=True
    )
    
    # activations shape: [num_images, S*C]
    print(f"Extracted activations shape: {activations.shape}")
    
    # Predict ethnicity for each image
    classifier.classifier.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch_acts = activations[i:i+batch_size]
            batch_tensor = torch.tensor(batch_acts, dtype=torch.float32, device=classifier.device)
            
            logits = classifier.classifier(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    # Count predictions
    counts = np.bincount(predictions, minlength=len(classifier.class_names))
    
    # Convert to distribution
    distribution = {
        classifier.class_names[i]: counts[i] / len(images)
        for i in range(len(classifier.class_names))
    }
    
    return distribution


print("✓ Run this cell to load the corrected analyze_ethnicity_distribution")
print("✓ It uses aggregate_spatial='flatten' to match training")

