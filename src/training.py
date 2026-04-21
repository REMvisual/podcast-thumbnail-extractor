#!/usr/bin/env python3
"""
Custom Model Trainer for Thumbnail Asset Extractor

This script trains a custom quality scoring model based on YOUR preferences.

Usage:
    python train_custom_model.py --training-data ./my_thumbnails/ --output custom_model.pth

Training Data Structure:
    ./my_thumbnails/
        ├── good/           # Thumbnails or assets you LIKE
        │   ├── thumb_001.jpg
        │   └── thumb_002.jpg
        ├── bad/            # Examples you DON'T like (optional)
        │   └── bad_001.jpg
        └── descriptions.json  # Optional: Natural language descriptions

Requirements:
    pip install torch torchvision pillow numpy scikit-learn
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import argparse
import numpy as np
from pathlib import Path

# ============= DATASET =============

class ThumbnailPreferenceDataset(Dataset):
    """
    Dataset that learns from your thumbnail preferences
    
    Folder structure:
        good/ - Images you like (label: 1.0)
        bad/  - Images you don't like (label: 0.0)
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Load "good" examples (high score)
        good_dir = self.root_dir / 'good'
        if good_dir.exists():
            for img_path in good_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1.0))
        
        # Load "bad" examples (low score)
        bad_dir = self.root_dir / 'bad'
        if bad_dir.exists():
            for img_path in bad_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0.0))
        
        print(f"Loaded {len(self.samples)} training samples")
        print(f"  - Good examples: {sum(1 for _, label in self.samples if label == 1.0)}")
        print(f"  - Bad examples: {sum(1 for _, label in self.samples if label == 0.0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


# ============= MODEL =============

class ThumbnailQualityScorer(nn.Module):
    """
    Neural network that scores thumbnail/asset quality
    Based on ResNet50 pretrained on ImageNet
    """
    
    def __init__(self):
        super(ThumbnailQualityScorer, self).__init__()
        
        # Use pretrained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace final layer with quality scorer (0-1)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1 score
        )
    
    def forward(self, x):
        return self.backbone(x).squeeze()


# ============= TRAINING =============

def train_model(training_data_dir, output_path, epochs=10, batch_size=8, learning_rate=0.001):
    """
    Train custom quality scoring model
    
    Args:
        training_data_dir: Path to training data (good/ and bad/ folders)
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    
    print("\n" + "="*60)
    print("Training Custom Thumbnail Quality Scorer")
    print("="*60 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Data transforms (same as ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ThumbnailPreferenceDataset(training_data_dir, transform=transform)
    
    if len(dataset) < 10:
        print("ERROR: Need at least 10 training examples")
        print("   Add more images to good/ and/or bad/ folders")
        return
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set: {train_size} samples")
    print(f"Validation set: {val_size} samples")
    print()
    
    # Create model
    model = ThumbnailQualityScorer().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for 0-1 scores
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        print()
    
    print("="*60)
    print(f"Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {output_path}")
    print("="*60 + "\n")


# ============= TESTING =============

def test_model(model_path, test_image_path):
    """
    Test trained model on a single image
    
    Args:
        model_path: Path to trained model
        test_image_path: Path to test image
    
    Returns:
        Quality score (0-1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ThumbnailQualityScorer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get score
    with torch.no_grad():
        score = model(image_tensor).item()
    
    return score


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser(
        description='Train custom thumbnail quality scorer'
    )
    parser.add_argument(
        '--training-data',
        type=str,
        required=True,
        help='Path to training data directory (with good/ and bad/ folders)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='custom_model.pth',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Test model on an image after training'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        training_data_dir=args.training_data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Test if requested
    if args.test:
        print(f"\nTesting on: {args.test}")
        score = test_model(args.output, args.test)
        print(f"   Quality score: {score:.4f}")
        if score > 0.7:
            print(f"   HIGH quality (model likes this!)")
        elif score > 0.4:
            print(f"   MEDIUM quality")
        else:
            print(f"   LOW quality (model doesn't like this)")


if __name__ == '__main__':
    main()
