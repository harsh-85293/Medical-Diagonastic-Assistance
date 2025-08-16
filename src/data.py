"""
Data handling module for Medical AI Assistant.
Handles dataset loading, DICOM/PDF processing, and data augmentation.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CXRDataset(Dataset):
    """Chest X-ray dataset supporting multiple image formats."""
    
    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        csv_file: Optional[str] = None
    ):
        """
        Initialize the CXR dataset.
        
        Args:
            data_dir: Directory containing the dataset
            classes: List of class names
            transform: Albumentations transform pipeline
            mode: Dataset mode ('train', 'val', 'test')
            csv_file: Optional CSV file with filepath, label columns
        """
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transform
        self.mode = mode
        
        self.samples = self._load_samples(csv_file)
        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_samples(self, csv_file: Optional[str]) -> List[Tuple[str, int]]:
        """Load dataset samples from directory structure or CSV."""
        samples = []
        
        if csv_file and os.path.exists(csv_file):
            # Load from CSV
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filepath = row['filepath']
                    label = row['label']
                    if label in self.class_to_idx:
                        samples.append((filepath, self.class_to_idx[label]))
        else:
            # Load from directory structure
            for class_name in self.classes:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    for img_file in class_dir.glob("*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.pdf']:
                            samples.append((str(img_file), self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        
        # Load and preprocess image
        image = self._load_image(filepath)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def _load_image(self, filepath: str) -> np.ndarray:
        """Load image from various formats (PNG, JPG, DICOM, PDF)."""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        try:
            if suffix in ['.png', '.jpg', '.jpeg']:
                # Standard image formats
                image = cv2.imread(str(filepath))
                if image is None:
                    raise ValueError(f"Failed to load image: {filepath}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif suffix == '.dcm':
                # DICOM format
                image = self._load_dicom(filepath)
                
            elif suffix == '.pdf':
                # PDF format - convert first page
                image = self._load_pdf(filepath)
                
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            # Ensure 3-channel RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return a black image as fallback
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _load_dicom(self, filepath: Path) -> np.ndarray:
        """Load and preprocess DICOM file."""
        try:
            dcm = pydicom.dcmread(str(filepath))
            
            # Get pixel array
            image = dcm.pixel_array
            
            # Apply windowing if available
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                center = dcm.WindowCenter
                width = dcm.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
                
                min_val = center - width // 2
                max_val = center + width // 2
                image = np.clip(image, min_val, max_val)
            
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM {filepath}: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def _load_pdf(self, filepath: Path) -> np.ndarray:
        """Load first page of PDF as image."""
        try:
            # Convert first page to image
            images = convert_from_path(str(filepath), first_page=1, last_page=1)
            if images:
                # Convert PIL to numpy
                image = np.array(images[0])
                return image
            else:
                raise ValueError("No pages found in PDF")
                
        except Exception as e:
            logger.error(f"Error loading PDF {filepath}: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)


def get_transforms(config: Dict[str, Any], mode: str = "train") -> A.Compose:
    """Get Albumentations transform pipeline based on config and mode."""
    
    if mode == "train":
        aug_config = config['augmentation']['train']
        transforms = []
        
        # Add augmentations based on config
        for aug_name, aug_params in aug_config.items():
            if aug_name == "RandomResizedCrop":
                transforms.append(A.RandomResizedCrop(**aug_params))
            elif aug_name == "HorizontalFlip":
                transforms.append(A.HorizontalFlip(**aug_params))
            elif aug_name == "Rotation":
                transforms.append(A.Rotate(**aug_params))
            elif aug_name == "RandomBrightnessContrast":
                transforms.append(A.RandomBrightnessContrast(**aug_params))
            elif aug_name == "CLAHE":
                transforms.append(A.CLAHE(**aug_params))
            elif aug_name == "GaussianNoise":
                transforms.append(A.GaussianNoise(**aug_params))
        
        # Add normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    else:
        # Validation/Test transforms
        aug_config = config['augmentation']['val_test']
        transforms = []
        
        for aug_name, aug_params in aug_config.items():
            if aug_name == "Resize":
                transforms.append(A.Resize(**aug_params))
            elif aug_name == "CenterCrop":
                transforms.append(A.CenterCrop(**aug_params))
        
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return A.Compose(transforms)


def create_data_loaders(
    config: Dict[str, Any],
    data_dir: str,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Get transforms
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")
    test_transform = get_transforms(config, "test")
    
    # Create datasets
    train_dataset = CXRDataset(
        data_dir=data_dir,
        classes=config['dataset']['classes'],
        transform=train_transform,
        mode="train"
    )
    
    val_dataset = CXRDataset(
        data_dir=data_dir,
        classes=config['dataset']['classes'],
        transform=val_transform,
        mode="val"
    )
    
    test_dataset = CXRDataset(
        data_dir=data_dir,
        classes=config['dataset']['classes'],
        transform=test_transform,
        mode="test"
    )
    
    # Calculate class weights for weighted sampling
    if config['training']['use_weighted_sampler']:
        class_weights = _calculate_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=class_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def _calculate_class_weights(dataset: CXRDataset) -> torch.Tensor:
    """Calculate class weights for weighted sampling."""
    class_counts = torch.zeros(len(dataset.classes))
    
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    
    # Create weight tensor for each sample
    sample_weights = torch.zeros(len(dataset))
    for idx, (_, label) in enumerate(dataset.samples):
        sample_weights[idx] = weights[label]
    
    return sample_weights


def get_class_weights(dataset: CXRDataset) -> torch.Tensor:
    """Get class weights for loss function."""
    class_counts = torch.zeros(len(dataset.classes))
    
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Normalize weights
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    
    return weights


def stratified_split(
    data_dir: str,
    classes: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Create stratified train/val/test splits."""
    
    np.random.seed(random_seed)
    
    train_files = []
    val_files = []
    test_files = []
    
    for class_name in classes:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue
            
        # Get all files for this class
        files = [str(f) for f in class_dir.glob("*") 
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.pdf']]
        
        # Shuffle files
        np.random.shuffle(files)
        
        # Calculate split indices
        n_files = len(files)
        n_train = int(n_files * train_split)
        n_val = int(n_files * val_split)
        
        # Split files
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    
    return train_files, val_files, test_files


def save_split_to_csv(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    output_dir: str
):
    """Save train/val/test splits to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train split
    with open(output_dir / "train.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        for filepath in train_files:
            label = Path(filepath).parent.name
            writer.writerow([filepath, label])
    
    # Save validation split
    with open(output_dir / "val.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        for filepath in val_files:
            label = Path(filepath).parent.name
            writer.writerow([filepath, label])
    
    # Save test split
    with open(output_dir / "test.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        for filepath in test_files:
            label = Path(filepath).parent.name
            writer.writerow([filepath, label])
    
    logger.info(f"Saved splits to {output_dir}")
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
