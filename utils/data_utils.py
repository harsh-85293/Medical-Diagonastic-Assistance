"""
Data utilities for Medical Diagnosis Assistant
Handles data loading, preprocessing, and augmentation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ChestXRayDataset:
    """Custom dataset for chest X-ray images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(img_size=224):
    """Get data transforms for training and validation"""
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size=32, img_size=224):
    """Create train, validation, and test data loaders"""
    
    train_transform, val_transform = get_transforms(img_size)
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = ChestXRayDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    test_dataset = ChestXRayDataset(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def plot_sample_images(data_loader, num_samples=8):
    """Plot sample images from the dataset"""
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    
    # Convert to numpy and transpose
    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Clip values to [0, 1]
    images = np.clip(images, 0, 1)
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_device():
    """Get the best available device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss 