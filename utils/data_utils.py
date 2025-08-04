#!/usr/bin/env python3
"""
Data utilities for Medical Diagnosis Assistant
Supports multi-class disease classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Multi-class disease labels
DISEASE_LABELS = [
    'NORMAL',
    'PNEUMONIA', 
    'COVID-19',
    'TUBERCULOSIS',
    'PLEURAL_EFFUSION',
    'PNEUMOTHORAX',
    'LUNG_CANCER',
    'CARDIOMEGALY'
]

# Map disease names to display names
DISEASE_DISPLAY_NAMES = {
    'NORMAL': 'Normal',
    'PNEUMONIA': 'Pneumonia',
    'COVID-19': 'COVID-19',
    'TUBERCULOSIS': 'Tuberculosis',
    'PLEURAL_EFFUSION': 'Pleural Effusion',
    'PNEUMOTHORAX': 'Pneumothorax',
    'LUNG_CANCER': 'Lung Cancer',
    'CARDIOMEGALY': 'Cardiomegaly'
}

def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_transforms(image_size=224):
    """Get training and validation transforms"""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

class MultiClassChestXRayDataset(Dataset):
    """Dataset for multi-class chest X-ray classification"""
    
    def __init__(self, data_dir, labels_df=None, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        if labels_df is not None:
            # Use provided labels dataframe
            self.data = labels_df
            self.multi_label = True
        else:
            # Fallback to folder structure
            self.data = self._load_from_folders()
            self.multi_label = False
    
    def _load_from_folders(self):
        """Load data from folder structure (disease_name/image.jpg)"""
        data = []
        for disease in DISEASE_LABELS:
            disease_dir = os.path.join(self.data_dir, disease.lower())
            if os.path.exists(disease_dir):
                for img_name in os.listdir(disease_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        data.append({
                            'image_path': os.path.join(disease_dir, img_name),
                            'labels': [disease]
                        })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.multi_label:
            # Multi-label format
            row = self.data.iloc[idx]
            image_path = row['image_path']
            labels = row['labels'].split(',') if isinstance(row['labels'], str) else row['labels']
        else:
            # Single-label format
            item = self.data[idx]
            image_path = item['image_path']
            labels = item['labels']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to multi-hot encoding
        label_vector = torch.zeros(len(DISEASE_LABELS))
        for label in labels:
            if label in DISEASE_LABELS:
                label_vector[DISEASE_LABELS.index(label)] = 1.0
        
        return image, label_vector

def create_dataloaders(data_dir, batch_size=16, val_split=0.2, test_split=0.1):
    """Create train/val/test dataloaders for multi-class classification"""
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Try to load labels from CSV (NIH dataset format)
    labels_file = os.path.join(data_dir, 'labels.csv')
    if os.path.exists(labels_file):
        print("Loading multi-label dataset from CSV...")
        labels_df = pd.read_csv(labels_file)
        
        # Split data
        train_df, temp_df = train_test_split(labels_df, test_size=val_split+test_split, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_split/(val_split+test_split), random_state=42)
        
        # Create datasets
        train_dataset = MultiClassChestXRayDataset(data_dir, train_df, train_transform, 'train')
        val_dataset = MultiClassChestXRayDataset(data_dir, val_df, val_transform, 'val')
        test_dataset = MultiClassChestXRayDataset(data_dir, test_df, val_transform, 'test')
        
    else:
        print("Loading single-label dataset from folders...")
        # Create datasets from folder structure
        train_dataset = MultiClassChestXRayDataset(data_dir, transform=train_transform, mode='train')
        val_dataset = MultiClassChestXRayDataset(data_dir, transform=val_transform, mode='val')
        test_dataset = MultiClassChestXRayDataset(data_dir, transform=val_transform, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)
            return 0, float('inf')
    return 0, float('inf')

def save_checkpoint(model, optimizer, epoch, val_loss, save_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': val_loss
    }, save_path)

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    return fig

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(len(DISEASE_LABELS))
    
    for _, labels in dataset:
        class_counts += labels
    
    # Calculate weights (inverse frequency)
    total_samples = len(dataset)
    class_weights = total_samples / (len(DISEASE_LABELS) * class_counts)
    class_weights = torch.where(torch.isinf(class_weights), 1.0, class_weights)
    
    return class_weights 