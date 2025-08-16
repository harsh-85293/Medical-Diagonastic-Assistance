"""
Training module for Medical AI Assistant.
Implements training loop with early stopping, mixed precision, and logging.
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
import numpy as np
from tqdm import tqdm

from .model import create_model, export_model
from .data import create_data_loaders, get_class_weights
from .losses import create_loss_function
from .gradcam import GradCAM

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10, delta: float = 0.001, min_epochs: int = 10):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after improvement
            delta: Minimum change to qualify as improvement
            min_epochs: Minimum number of epochs before early stopping
        """
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if epoch < self.min_epochs:
            return False
            
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
            return False
        
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class Trainer:
    """Main trainer class for the CXR classifier."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to train on
        """
        self.config = config
        self.device = device
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
        
        # Create directories
        self.models_dir = Path(config['paths']['models_dir'])
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def setup_training(self, data_dir: str):
        """Setup training components."""
        logger.info("Setting up training components...")
        
        # Create data loaders
        self.train_loader, self.val_loader, _ = create_data_loaders(
            config=self.config,
            data_dir=data_dir,
            batch_size=self.config['training']['batch_size']
        )
        
        # Create model
        self.model = create_model(self.config).to(self.device)
        
        # Calculate class weights
        class_weights = get_class_weights(self.train_loader.dataset)
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Create loss function
        self.criterion = create_loss_function(self.config, class_weights).to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        logger.info("Training setup completed")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        config = self.config['training']
        
        if config['optimizer'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'].lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        config = self.config['training']
        
        if config['scheduler'].lower() == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config['learning_rate'],
                epochs=config['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        elif config['scheduler'].lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif config['scheduler'].lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        else:
            logger.warning(f"Unsupported scheduler: {config['scheduler']}")
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # Update scheduler if using OneCycleLR
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.models_dir / "latest.ckpt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.models_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        logger.info(f"Checkpoint saved to {latest_path}")
    
    def train(self, data_dir: str, resume_from: Optional[str] = None):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup training
        self.setup_training(data_dir)
        
        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            delta=self.config['training']['early_stopping']['delta']
        )
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler (if not OneCycleLR)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Check early stopping
            if early_stopping(val_loss, epoch + 1):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Export final model
        self._export_final_model()
        
        logger.info("Training completed!")
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def _export_final_model(self):
        """Export the final model in various formats."""
        logger.info("Exporting final model...")
        
        export_config = self.config['export']
        
        for export_format in export_config['formats']:
            if export_format.lower() == "torchscript":
                export_path = self.models_dir / "model.ts"
                export_model(
                    self.model,
                    str(export_path),
                    "torchscript",
                    device=self.device
                )
            elif export_format.lower() == "onnx":
                export_path = self.models_dir / "model.onnx"
                export_model(
                    self.model,
                    str(export_path),
                    "onnx",
                    device=self.device
                )
        
        logger.info("Model export completed")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CXR Classifier")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer and start training
    trainer = Trainer(config, device)
    trainer.train(args.data_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
