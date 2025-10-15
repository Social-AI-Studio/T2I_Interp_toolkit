"""
Ethnicity Distribution Evaluation for Text-to-Image Models
This is just an output classifier traine don fairface data that predicts ethnic classes.

Trains a CNN classifier directly on FairFace images to predict ethnicity,
then uses it to analyze ethnicity distributions in generated images.

"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import torchvision.models as models
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from t2Interp.T2I import T2IModel


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def get_image_transforms(image_size=224):
    """Standard ImageNet preprocessing for pretrained models."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# FAIRFACE DATASET LOADING
# ============================================================================

def load_fairface_by_ethnicity(dataset_path, max_images=None):
    """Load FairFace dataset organized by ethnicity."""
    ethnicities = ['Black', 'East_Asian', 'Indian', 'Latino_Hispanic',
                   'Middle_Eastern', 'Southeast_Asian', 'White']
    
    data = {}
    
    for ethnicity in ethnicities:
        pattern = os.path.join(dataset_path, f"**/race_{ethnicity}_*.jpg")
        image_paths = glob(pattern, recursive=True)
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        images = []
        for path in tqdm(image_paths, desc=f"Loading {ethnicity}"):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        data[ethnicity] = images
        print(f"{ethnicity}: {len(images)} images")
    
    return data


# ============================================================================
# ETHNICITY CLASSIFIER (CNN on raw images)
# ============================================================================

class EthnicityClassifier:
    """CNN classifier to predict ethnicity directly from images."""
    
    def __init__(self, num_classes=7, pretrained=True, lr=1e-4, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        self.class_names = None
        
        # Use pretrained ResNet18 as backbone
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for ethnicity classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = get_image_transforms()
    
    def fit(self, images_dict, epochs=50, batch_size=32, val_split=0.2):
        """
        Train classifier on images.
        
        Args:
            images_dict: Dict mapping class names to lists of PIL Images
            epochs: Number of epochs
            batch_size: Batch size
            val_split: Validation split ratio
        """
        from torch.utils.data import Dataset, DataLoader
        
        self.class_names = list(images_dict.keys())
        print(f"\nTraining on {len(self.class_names)} classes: {self.class_names}")
        
        # Create dataset
        class ImageDataset(Dataset):
            def __init__(self, images, labels, transform):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img = self.images[idx]
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]
        
        # Prepare data
        all_images = []
        all_labels = []
        
        for i, class_name in enumerate(self.class_names):
            images = images_dict[class_name]
            all_images.extend(images)
            all_labels.extend([i] * len(images))
        
        print(f"Total samples: {len(all_images)}")
        
        # Split train/val
        n_val = int(len(all_images) * val_split)
        indices = np.random.permutation(len(all_images))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        train_images = [all_images[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        train_dataset = ImageDataset(train_images, train_labels, self.transform)
        val_dataset = ImageDataset(val_images, val_labels, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Training loop
        best_val_acc = 0
        for epoch in range(epochs):
            self.model.train()
            train_correct, train_total = 0, 0
            train_loss = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Validate
            self.model.eval()
            val_correct, val_total = 0, 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    logits = self.model(batch_X)
                    val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_loss = train_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train={train_acc:.1f}%, Val={val_acc:.1f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        print(f"\n✓ Best validation accuracy: {best_val_acc:.1f}%")
    
    def predict(self, images, return_probs=False, batch_size=32):
        """
        Predict ethnicity from images.
        
        Args:
            images: List of PIL Images
            return_probs: If True, also return probabilities
            batch_size: Batch size for prediction
        
        Returns:
            predictions (and probabilities if return_probs=True)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                # Transform images
                batch_tensors = torch.stack([
                    self.transform(img) for img in batch_images
                ]).to(self.device)
                
                # Predict
                logits = self.model(batch_tensors)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                if return_probs:
                    all_probs.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        
        if return_probs:
            probabilities = np.concatenate(all_probs, axis=0)
            return predictions, probabilities
        return predictions
    
    def save(self, path):
        """Save classifier."""
        torch.save({
            'state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }, path)
        print(f"✓ Saved to {path}")
    
    def load(self, path):
        """Load classifier."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        print(f"✓ Loaded from {path}")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def train_ethnicity_classifier(images_dict, save_path=None, epochs=50, batch_size=32):
    """
    Train CNN ethnicity classifier directly on images.
    
    Args:
        images_dict: Dict mapping ethnicity names to lists of PIL Images
                     (e.g., from load_fairface_by_ethnicity)
        save_path: Where to save trained classifier (optional)
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        Trained classifier
    """

    num_classes = len(images_dict)
    print(f"\nTotal classes: {num_classes}")
    for ethnicity, images in images_dict.items():
        print(f"  {ethnicity}: {len(images)} images")
    
    # Initialize classifier (ResNet18)
    classifier = EthnicityClassifier(
        num_classes=num_classes,
        pretrained=True,  # Use ImageNet pretrained weights
        lr=1e-4,
        device='cuda'
    )
    
    # Train directly on images
    classifier.fit(images_dict, epochs=epochs, batch_size=batch_size)
    
    # Save if path provided
    if save_path:
        classifier.save(save_path)
    
    return classifier



