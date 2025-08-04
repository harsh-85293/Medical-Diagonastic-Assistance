#!/usr/bin/env python3
"""
Training script for Medical Diagnosis Assistant
Trains a ResNet50 model on chest X-ray images for pneumonia detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils.data_utils import create_data_loaders, get_device, save_checkpoint
from utils.model_utils import MultiClassChestXRayModel, train_epoch, validate_epoch
from utils.data_utils import plot_training_history

def train_model(args):
    """Main training function"""
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = MultiClassChestXRayModel(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.model_dir, 'best_model.pt')
            )
            print("✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"❌ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs-1, val_loss,
        os.path.join(args.model_dir, 'final_model.pt')
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    np.save(os.path.join(args.model_dir, 'training_history.npy'), history)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Model saved to: {args.model_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Medical Diagnosis Assistant')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing train/val/test folders')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory '{args.data_dir}' not found!")
        print("Please run 'python download_dataset.py' first to setup the dataset.")
        return
    
    # Check if train/val/test directories exist
    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(args.data_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Directory '{dir_path}' not found!")
            print("Please ensure your dataset is properly organized.")
            return
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main() 