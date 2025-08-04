#!/usr/bin/env python3
"""
Multi-Class Medical Diagnosis Assistant Training Script
Supports training on multiple disease classes with advanced features
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from utils.data_utils import (
    create_dataloaders, get_device, save_checkpoint, load_checkpoint,
    plot_training_history, calculate_class_weights, DISEASE_LABELS
)
from utils.model_utils import (
    create_model, train_epoch, validate_epoch, evaluate_model,
    get_loss_function
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Multi-Class Medical Diagnosis Model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data/chest_xray',
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='resnet50',
                       choices=['resnet50', 'vit', 'ensemble'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--loss-type', type=str, default='bce',
                       choices=['bce', 'focal', 'cross_entropy'],
                       help='Loss function type')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced dataset')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'step'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for models')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--save-best-only', action='store_true',
                       help='Save only the best model')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directory and logging"""
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"multiclass_{args.model_type}_{timestamp}"
    
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return experiment_dir

def train_model(args):
    """Main training function"""
    print("🚀 Starting Multi-Class Medical Diagnosis Training")
    print(f"📊 Model Type: {args.model_type}")
    print(f"🏥 Disease Classes: {len(DISEASE_LABELS)}")
    print(f"📋 Classes: {', '.join(DISEASE_LABELS)}")
    
    # Setup device
    device = get_device()
    print(f"💻 Using device: {device}")
    
    # Setup experiment
    experiment_dir = setup_experiment(args)
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Create dataloaders
    print("📂 Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.val_split, args.test_split
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model(args.model_type, len(DISEASE_LABELS), args.pretrained)
    model = model.to(device)
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("⚖️ Calculating class weights...")
        class_weights = calculate_class_weights(train_loader.dataset)
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights}")
    
    # Setup loss function and optimizer
    criterion = get_loss_function(args.loss_type, class_weights)
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
    start_epoch, best_val_loss = load_checkpoint(model, checkpoint_path, device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print("🎯 Starting training...")
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"💾 Saved best model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if not args.save_best_only:
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    checkpoint_path_epoch = os.path.join(experiment_dir, f'epoch_{epoch+1}.pth')
                    save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path_epoch)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping after {args.patience} epochs without improvement")
            break
    
    # Final evaluation
    print("\n🔍 Final evaluation on test set...")
    test_metrics = evaluate_model(
        model, test_loader, device,
        save_path=os.path.join(experiment_dir, 'evaluation_results.png')
    )
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.png')
    plot_training_history(history).savefig(history_path, dpi=300, bbox_inches='tight')
    
    # Save metrics
    metrics_path = os.path.join(experiment_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Print final results
    print("\n🎉 Training completed!")
    print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"📊 Mean F1 Score: {test_metrics['mean_f1']:.4f}")
    print(f"📊 Mean AUC: {test_metrics['mean_auc']:.4f}")
    print(f"📁 Results saved to: {experiment_dir}")
    
    return model, test_metrics

def main():
    """Main function"""
    args = parse_args()
    
    try:
        model, metrics = train_model(args)
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 