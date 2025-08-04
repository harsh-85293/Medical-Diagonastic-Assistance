#!/usr/bin/env python3
"""
Demo Training Script for Medical Diagnosis Assistant
This script creates a minimal trained model for demonstration purposes.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.model_utils import ChestXRayModel
from utils.data_utils import get_device, save_checkpoint
import argparse

def create_demo_data():
    """Create minimal demo data for training"""
    print("Creating demo training data...")
    
    # Create demo images (random data for demonstration)
    num_samples = 100
    img_size = 224
    
    # Generate random images (simulating chest X-rays)
    demo_images = np.random.randn(num_samples, 3, img_size, img_size).astype(np.float32)
    demo_labels = np.random.randint(0, 2, num_samples)  # Binary classification
    
    # Convert to PyTorch tensors
    images = torch.FloatTensor(demo_images)
    labels = torch.LongTensor(demo_labels)
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    return dataloader

def train_demo_model(data_loader, device, epochs=5):
    """Train a demo model with minimal data"""
    print(f"Training demo model on {device}...")
    
    # Initialize model
    model = ChestXRayModel(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Demo Training for Medical Diagnosis Assistant')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output-path', type=str, default='./models/chest_xray_demo.pth',
                       help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create demo data
    data_loader = create_demo_data()
    
    # Train model
    model = train_demo_model(data_loader, device, epochs=args.epochs)
    
    # Save model
    save_checkpoint(model, args.output_path, device=device)
    print(f"✅ Demo model saved to: {args.output_path}")
    
    # Test model loading
    print("Testing model loading...")
    try:
        loaded_model = ChestXRayModel(num_classes=2)
        loaded_model.load_state_dict(torch.load(args.output_path, map_location=device))
        loaded_model.eval()
        print("✅ Model loading test successful!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")

if __name__ == "__main__":
    main() 