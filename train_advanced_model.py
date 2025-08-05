#!/usr/bin/env python3
"""
Advanced Training Script for Multi-Class Medical Diagnosis Assistant
Trains a robust model on real chest X-ray datasets with advanced techniques
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime

from utils.data_utils import get_transforms, get_device, DISEASE_LABELS, DISEASE_DISPLAY_NAMES
from utils.model_utils import create_model, FocalLoss
from utils.data_utils import MultiClassChestXRayDataset, calculate_class_weights

class AdvancedTrainer:
    """Advanced trainer with multiple training strategies"""
    
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup experiment tracking and directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"multiclass_{self.args.model_type}_{timestamp}"
        self.experiment_dir = os.path.join("experiments", self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize wandb if enabled
        if self.args.use_wandb:
            wandb.init(
                project="medical-diagnosis-assistant",
                name=self.experiment_name,
                config=vars(self.args)
            )
    
    def create_advanced_transforms(self):
        """Create advanced data augmentation transforms"""
        train_transform, val_transform = get_transforms(image_size=self.args.img_size)
        
        # Add more aggressive augmentation for training
        if self.args.advanced_augmentation:
            from torchvision import transforms
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.args.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self):
        """Create advanced dataloaders with class balancing"""
        train_transform, val_transform = self.create_advanced_transforms()
        
        # Load data
        if self.args.data_format == 'csv':
            # Load from CSV (NIH dataset format)
            labels_df = pd.read_csv(self.args.data_path)
            
            # Split data
            train_df, temp_df = train_test_split(
                labels_df, test_size=0.3, random_state=42, stratify=labels_df['labels']
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42, stratify=temp_df['labels']
            )
            
            # Create datasets
            train_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, train_df, train_transform, 'train'
            )
            val_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, val_df, val_transform, 'val'
            )
            test_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, test_df, val_transform, 'test'
            )
            
        else:
            # Load from folder structure
            train_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, transform=train_transform, mode='train'
            )
            val_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, transform=val_transform, mode='val'
            )
            test_dataset = MultiClassChestXRayDataset(
                self.args.data_dir, transform=val_transform, mode='test'
            )
        
        # Calculate class weights for imbalanced datasets
        if self.args.use_class_weights:
            class_weights = calculate_class_weights(train_dataset)
            sampler = WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, 
                sampler=sampler, num_workers=self.args.num_workers
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, 
                shuffle=True, num_workers=self.args.num_workers
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, 
            shuffle=False, num_workers=self.args.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, 
            shuffle=False, num_workers=self.args.num_workers
        )
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """Create advanced model with different architectures"""
        model = create_model(
            model_type=self.args.model_type,
            num_classes=len(DISEASE_LABELS),
            pretrained=self.args.pretrained
        )
        model = model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_optimizer(self, model):
        """Create advanced optimizer with different strategies"""
        if self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        return optimizer
    
    def create_scheduler(self, optimizer):
        """Create advanced learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=1e-6
            )
        elif self.args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def create_loss_function(self):
        """Create advanced loss function"""
        if self.args.loss_type == 'focal':
            return FocalLoss(alpha=1, gamma=2)
        elif self.args.loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif self.args.loss_type == 'ce':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch with advanced metrics"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle multi-label vs single-label
            if labels.shape[1] > 1:
                loss = criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += ((preds == labels).sum(dim=1) > 0).sum().item()
            else:
                loss = criterion(outputs, labels.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.squeeze()).sum().item()
            
            total += labels.size(0)
            running_loss += loss.item()
            
            loss.backward()
            
            # Gradient clipping
            if self.args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate one epoch with advanced metrics"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                
                if labels.shape[1] > 1:
                    loss = criterion(outputs, labels.float())
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += ((preds == labels).sum(dim=1) > 0).sum().item()
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    loss = criterion(outputs, labels.squeeze())
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels.squeeze()).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.squeeze().cpu().numpy())
                
                total += labels.size(0)
                running_loss += loss.item()
        
        return running_loss / len(val_loader), 100. * correct / total, all_predictions, all_labels
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'args': self.args
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.experiment_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.experiment_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (Val Loss: {val_loss:.4f})")
    
    def plot_training_history(self, history):
        """Plot advanced training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['lr'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confusion matrix (if available)
        if 'confusion_matrix' in history:
            cm = history['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    def train(self):
        """Main training loop with advanced features"""
        print(f"🚀 Starting advanced training for {self.args.epochs} epochs...")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Create loss function
        criterion = self.create_loss_function()
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.args.epochs):
            print(f"\n📈 Epoch {epoch+1}/{self.args.epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(model, optimizer, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.args.patience:
                print(f"🛑 Early stopping after {self.args.patience} epochs without improvement")
                break
        
        # Final evaluation
        print("\n🔍 Final evaluation on test set...")
        test_loss, test_acc, test_preds, test_labels = self.validate_epoch(model, test_loader, criterion)
        
        # Generate confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        history['confusion_matrix'] = cm
        
        # Save training history
        history_path = os.path.join(self.experiment_dir, 'training_history.png')
        fig = self.plot_training_history(history)
        fig.savefig(history_path, dpi=300, bbox_inches='tight')
        
        # Save final model
        final_path = os.path.join(self.experiment_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_loss': test_loss,
            'test_acc': test_acc,
            'args': self.args
        }, final_path)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
        print(f"📁 Results saved to: {self.experiment_dir}")
        
        return model, history

def main():
    parser = argparse.ArgumentParser(description='Advanced Multi-Class Medical Diagnosis Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--data_path', type=str, help='Path to CSV file (if using CSV format)')
    parser.add_argument('--data_format', type=str, default='folder', choices=['folder', 'csv'], help='Data format')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'vit', 'ensemble'], help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Advanced features
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'plateau', 'step', 'none'], help='Learning rate scheduler')
    parser.add_argument('--loss_type', type=str, default='focal', 
                       choices=['focal', 'bce', 'ce'], help='Loss function')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--advanced_augmentation', action='store_true', help='Use advanced data augmentation')
    parser.add_argument('--gradient_clipping', action='store_true', help='Use gradient clipping')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = AdvancedTrainer(args)
    model, history = trainer.train()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Check results in: {trainer.experiment_dir}")

if __name__ == "__main__":
    main() 