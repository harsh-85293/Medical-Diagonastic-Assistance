#!/usr/bin/env python3
"""
Dataset Preparation Script for Medical Diagnosis Assistant
Helps organize and prepare real chest X-ray datasets for training
"""

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

def create_folder_structure(base_dir):
    """Create the standard folder structure for training"""
    folders = ['train', 'val', 'test']
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
        print(f"✅ Created folder: {os.path.join(base_dir, folder)}")

def organize_nih_dataset(nih_dir, output_dir):
    """Organize NIH ChestX-ray14 dataset"""
    print("🏥 Organizing NIH ChestX-ray14 dataset...")
    
    # Read NIH data file
    data_file = os.path.join(nih_dir, 'Data_Entry_2017.csv')
    if not os.path.exists(data_file):
        print(f"❌ NIH data file not found: {data_file}")
        return False
    
    df = pd.read_csv(data_file)
    print(f"📊 Found {len(df)} images in NIH dataset")
    
    # Create folder structure
    create_folder_structure(output_dir)
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = os.path.join(nih_dir, 'images', row['Image Index'])
        
        if not os.path.exists(image_path):
            continue
        
        # Get labels
        labels = row['Finding Labels'].split('|')
        
        # Copy image to appropriate folder
        for label in labels:
            label = label.strip().lower()
            if label in ['normal', 'no finding']:
                dest_folder = 'normal'
            else:
                dest_folder = label.replace(' ', '_')
            
            # Create disease folder
            disease_dir = os.path.join(output_dir, 'train', dest_folder)
            os.makedirs(disease_dir, exist_ok=True)
            
            # Copy image
            dest_path = os.path.join(disease_dir, row['Image Index'])
            shutil.copy2(image_path, dest_path)
    
    print("✅ NIH dataset organized successfully!")
    return True

def organize_kaggle_dataset(kaggle_dir, output_dir):
    """Organize Kaggle Chest X-Ray Images (Pneumonia) dataset"""
    print("🏥 Organizing Kaggle Chest X-Ray dataset...")
    
    # Create folder structure
    create_folder_structure(output_dir)
    
    # Process train folder
    train_src = os.path.join(kaggle_dir, 'train')
    if os.path.exists(train_src):
        for category in ['NORMAL', 'PNEUMONIA']:
            src_folder = os.path.join(train_src, category)
            if os.path.exists(src_folder):
                dest_folder = os.path.join(output_dir, 'train', category.lower())
                shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
                print(f"✅ Copied {category} images to train folder")
    
    # Process test folder
    test_src = os.path.join(kaggle_dir, 'test')
    if os.path.exists(test_src):
        for category in ['NORMAL', 'PNEUMONIA']:
            src_folder = os.path.join(test_src, category)
            if os.path.exists(src_folder):
                dest_folder = os.path.join(output_dir, 'test', category.lower())
                shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
                print(f"✅ Copied {category} images to test folder")
    
    print("✅ Kaggle dataset organized successfully!")
    return True

def create_multi_class_dataset(base_dir, output_dir):
    """Create a multi-class dataset from multiple sources"""
    print("🏥 Creating multi-class dataset...")
    
    # Create folder structure
    create_folder_structure(output_dir)
    
    # Define disease mappings
    disease_mappings = {
        'normal': ['normal', 'no finding', 'healthy'],
        'pneumonia': ['pneumonia', 'consolidation'],
        'covid19': ['covid-19', 'covid19', 'sars-cov-2'],
        'tuberculosis': ['tuberculosis', 'tb'],
        'pleural_effusion': ['pleural effusion', 'effusion'],
        'pneumothorax': ['pneumothorax'],
        'lung_cancer': ['lung cancer', 'mass', 'nodule'],
        'cardiomegaly': ['cardiomegaly', 'enlarged heart']
    }
    
    # Process each source directory
    for source_dir in os.listdir(base_dir):
        source_path = os.path.join(base_dir, source_dir)
        if not os.path.isdir(source_path):
            continue
        
        print(f"📁 Processing source: {source_dir}")
        
        # Find images in source directory
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Determine disease category based on path or filename
                    relative_path = os.path.relpath(root, source_path)
                    file_lower = file.lower()
                    path_lower = relative_path.lower()
                    
                    # Try to match disease
                    matched_disease = None
                    for disease, keywords in disease_mappings.items():
                        for keyword in keywords:
                            if keyword in path_lower or keyword in file_lower:
                                matched_disease = disease
                                break
                        if matched_disease:
                            break
                    
                    if matched_disease:
                        # Copy to appropriate folder
                        dest_folder = os.path.join(output_dir, 'train', matched_disease)
                        os.makedirs(dest_folder, exist_ok=True)
                        
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(dest_folder, f"{source_dir}_{file}")
                        shutil.copy2(src_path, dest_path)
    
    print("✅ Multi-class dataset created successfully!")
    return True

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets"""
    print("📊 Splitting dataset into train/val/test sets...")
    
    # Verify ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("❌ Ratios must sum to 1.0")
        return False
    
    # Create temporary structure
    temp_dir = os.path.join(data_dir, 'temp_split')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all images to temp directory
    for disease_folder in os.listdir(data_dir):
        disease_path = os.path.join(data_dir, disease_folder)
        if os.path.isdir(disease_path) and disease_folder not in ['train', 'val', 'test', 'temp_split']:
            temp_disease_path = os.path.join(temp_dir, disease_folder)
            shutil.move(disease_path, temp_disease_path)
    
    # Create final structure
    create_folder_structure(data_dir)
    
    # Split each disease folder
    for disease_folder in os.listdir(temp_dir):
        disease_path = os.path.join(temp_dir, disease_folder)
        if not os.path.isdir(disease_path):
            continue
        
        # Get all images
        images = [f for f in os.listdir(disease_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            continue
        
        # Split images
        train_imgs, temp_imgs = train_test_split(
            images, train_size=train_ratio, random_state=42
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=val_ratio_adjusted, random_state=42
        )
        
        # Move images to appropriate folders
        for img in train_imgs:
            src = os.path.join(disease_path, img)
            dst = os.path.join(data_dir, 'train', disease_folder, img)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        
        for img in val_imgs:
            src = os.path.join(disease_path, img)
            dst = os.path.join(data_dir, 'val', disease_folder, img)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        
        for img in test_imgs:
            src = os.path.join(disease_path, img)
            dst = os.path.join(data_dir, 'test', disease_folder, img)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        
        print(f"✅ Split {disease_folder}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("✅ Dataset splitting completed!")
    return True

def create_csv_labels(data_dir, output_file):
    """Create CSV file with image paths and labels"""
    print("📝 Creating CSV labels file...")
    
    data = []
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for disease_folder in os.listdir(split_dir):
            disease_path = os.path.join(split_dir, disease_folder)
            if not os.path.isdir(disease_path):
                continue
            
            for img_file in os.listdir(disease_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(split, disease_folder, img_file)
                    data.append({
                        'image_path': img_path,
                        'labels': disease_folder,
                        'split': split
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✅ Created labels file: {output_file}")
    print(f"📊 Total images: {len(df)}")
    print(f"📊 Images per split:")
    print(df['split'].value_counts())
    print(f"📊 Images per disease:")
    print(df['labels'].value_counts())
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for medical diagnosis training')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['nih', 'kaggle', 'multi_class', 'split', 'csv'],
                       help='Dataset preparation mode')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'nih':
        organize_nih_dataset(args.input_dir, args.output_dir)
    elif args.mode == 'kaggle':
        organize_kaggle_dataset(args.input_dir, args.output_dir)
    elif args.mode == 'multi_class':
        create_multi_class_dataset(args.input_dir, args.output_dir)
    elif args.mode == 'split':
        split_dataset(args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.mode == 'csv':
        create_csv_labels(args.input_dir, os.path.join(args.output_dir, 'labels.csv'))
    
    print(f"\n🎉 Dataset preparation completed!")
    print(f"📁 Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 