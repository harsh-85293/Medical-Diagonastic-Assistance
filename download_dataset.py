#!/usr/bin/env python3
"""
Dataset Download Script for Medical Diagnosis Assistant
Downloads and extracts the Chest X-Ray Images (Pneumonia) dataset from Kaggle
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def setup_dataset():
    """Download and setup the chest X-ray dataset"""
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Dataset URL (Chest X-Ray Images for Pneumonia Detection)
    # This is a sample URL - in practice, you'd need to download from Kaggle API
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    
    # For demonstration, we'll create a synthetic dataset structure
    # In a real scenario, you would download the actual chest X-ray dataset
    
    print("Setting up dataset structure...")
    
    # Create synthetic dataset structure for demonstration
    train_dir = "data/train"
    test_dir = "data/test"
    val_dir = "data/val"
    
    for directory in [train_dir, test_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(directory, "PNEUMONIA"), exist_ok=True)
    
    print("Dataset directories created successfully!")
    print("\nNote: This script creates the directory structure for the chest X-ray dataset.")
    print("To use real data, you need to:")
    print("1. Download the 'Chest X-Ray Images (Pneumonia)' dataset from Kaggle")
    print("2. Extract it to the 'data' directory")
    print("3. Organize images into train/test/val splits with NORMAL/PNEUMONIA subdirectories")
    
    return True

if __name__ == "__main__":
    print("Medical Diagnosis Assistant - Dataset Setup")
    print("=" * 50)
    
    try:
        setup_dataset()
        print("\n✅ Dataset setup completed successfully!")
    except Exception as e:
        print(f"\n❌ Error setting up dataset: {e}")
        print("Please ensure you have proper internet connection and permissions.") 