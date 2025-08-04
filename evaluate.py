#!/usr/bin/env python3
"""
Evaluation script for Medical Diagnosis Assistant
Evaluates trained model on test set and generates comprehensive metrics
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from utils.data_utils import create_data_loaders, get_device, load_checkpoint
from utils.model_utils import MultiClassChestXRayModel, evaluate_model, plot_evaluation_results

def evaluate_model_comprehensive(args):
    """Comprehensive model evaluation"""
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = MultiClassChestXRayModel(num_classes=2, pretrained=False)
    model = model.to(device)
    
    # Load trained model
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        return
    
    print(f"Loading model from: {model_path}")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading
    model, optimizer, epoch, loss = load_checkpoint(model, optimizer, model_path)
    
    print(f"Model loaded from epoch {epoch} with loss {loss:.4f}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print("="*50)
    
    # Classification report
    class_names = ['NORMAL', 'PNEUMONIA']
    print("\nClassification Report:")
    print(classification_report(
        results['labels'], 
        results['predictions'], 
        target_names=class_names
    ))
    
    # Plot evaluation results
    print("\nGenerating evaluation plots...")
    plot_evaluation_results(results['labels'], results['predictions'], results['probabilities'], 
                          os.path.join(args.model_dir, 'evaluation_results.png'))
    
    # Save results
    results_file = os.path.join(args.model_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("MEDICAL DIAGNOSIS ASSISTANT - EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Test Loss: {loss:.4f}\n\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1 Score:  {results['f1']:.4f}\n")
        f.write(f"AUC:       {results['auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            results['labels'], 
            results['predictions'], 
            target_names=class_names
        ))
    
    print(f"\n✅ Evaluation completed!")
    print(f"Results saved to: {results_file}")
    
    # Save predictions for further analysis
    predictions_file = os.path.join(args.model_dir, 'test_predictions.npz')
    np.savez(predictions_file,
             predictions=results['predictions'],
             labels=results['labels'],
             probabilities=results['probabilities'])
    print(f"Predictions saved to: {predictions_file}")

def analyze_predictions(args):
    """Analyze model predictions in detail"""
    
    predictions_file = os.path.join(args.model_dir, 'test_predictions.npz')
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file '{predictions_file}' not found!")
        print("Please run evaluation first.")
        return
    
    # Load predictions
    data = np.load(predictions_file)
    predictions = data['predictions']
    labels = data['labels']
    probabilities = data['probabilities']
    
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    # Confusion matrix analysis
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (TN): {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP): {cm[1, 1]}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value (Precision): {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    # Probability distribution analysis
    print(f"\nProbability Distribution:")
    print(f"Mean probability for NORMAL: {np.mean(probabilities[labels == 0, 1]):.4f}")
    print(f"Mean probability for PNEUMONIA: {np.mean(probabilities[labels == 1, 1]):.4f}")
    print(f"Std probability for NORMAL: {np.std(probabilities[labels == 0, 1]):.4f}")
    print(f"Std probability for PNEUMONIA: {np.std(probabilities[labels == 1, 1]):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Medical Diagnosis Assistant')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing train/val/test folders')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing model checkpoints')
    parser.add_argument('--model_name', type=str, default='best_model.pt',
                       help='Name of the model file to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform detailed prediction analysis')
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory '{args.model_dir}' not found!")
        print("Please train a model first using 'python train.py'")
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory '{args.data_dir}' not found!")
        print("Please run 'python download_dataset.py' first to setup the dataset.")
        return
    
    # Perform evaluation
    evaluate_model_comprehensive(args)
    
    # Perform detailed analysis if requested
    if args.analyze:
        analyze_predictions(args)

if __name__ == "__main__":
    main() 