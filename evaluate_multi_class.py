#!/usr/bin/env python3
"""
Multi-Class Medical Diagnosis Assistant Evaluation Script
Comprehensive evaluation with detailed metrics and visualizations
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import json
from datetime import datetime

from utils.data_utils import create_dataloaders, get_device, DISEASE_LABELS, DISEASE_DISPLAY_NAMES
from utils.model_utils import create_model, evaluate_model, load_checkpoint

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Class Medical Diagnosis Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/chest_xray',
                       help='Path to test dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--model-type', type=str, default='resnet50',
                       choices=['resnet50', 'vit', 'ensemble'],
                       help='Model architecture type')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    
    return parser.parse_args()

def create_detailed_evaluation_plots(y_true, y_pred, y_prob, output_dir):
    """Create comprehensive evaluation plots"""
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS],
                yticklabels=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics
    class_metrics = []
    for i, disease in enumerate(DISEASE_LABELS):
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        
        class_metrics.append({
            'disease': DISEASE_DISPLAY_NAMES[disease],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ap': ap
        })
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(class_metrics)
    
    # Plot per-class metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision
    axes[0,0].bar(metrics_df['disease'], metrics_df['precision'])
    axes[0,0].set_title('Precision by Disease')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[0,1].bar(metrics_df['disease'], metrics_df['recall'])
    axes[0,1].set_title('Recall by Disease')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # F1 Score
    axes[1,0].bar(metrics_df['disease'], metrics_df['f1'])
    axes[1,0].set_title('F1 Score by Disease')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # AUC
    axes[1,1].bar(metrics_df['disease'], metrics_df['auc'])
    axes[1,1].set_title('AUC by Disease')
    axes[1,1].set_ylabel('AUC')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(12, 8))
    for i, disease in enumerate(DISEASE_LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{DISEASE_DISPLAY_NAMES[disease]} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Diseases')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    for i, disease in enumerate(DISEASE_LABELS):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f'{DISEASE_DISPLAY_NAMES[disease]} (AP={ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Diseases')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Model Performance Summary
    plt.figure(figsize=(10, 6))
    metrics_summary = ['precision', 'recall', 'f1', 'auc']
    summary_data = []
    
    for metric in metrics_summary:
        mean_val = metrics_df[metric].mean()
        summary_data.append(mean_val)
    
    plt.bar(['Precision', 'Recall', 'F1', 'AUC'], summary_data)
    plt.title('Overall Model Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(summary_data):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_df

def generate_classification_report(y_true, y_pred, output_dir):
    """Generate detailed classification report"""
    
    # Convert to single-label format for sklearn
    y_true_single = y_true.argmax(axis=1)
    y_pred_single = y_pred.argmax(axis=1)
    
    # Generate classification report
    class_names = [DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS]
    report = classification_report(y_true_single, y_pred_single, 
                                 target_names=class_names, output_dict=True)
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create report visualization
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']].iloc[:-3], 
                annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Classification Report Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return report

def evaluate_model_comprehensive(args):
    """Comprehensive model evaluation"""
    
    print("🔍 Starting comprehensive model evaluation...")
    
    # Setup device
    device = get_device()
    print(f"💻 Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("📂 Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, 0.2, 0.1
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model(args.model_type, len(DISEASE_LABELS), pretrained=False)
    model = model.to(device)
    
    # Load trained model
    print(f"📥 Loading model from: {args.model_path}")
    start_epoch, best_val_loss = load_checkpoint(model, args.model_path, device)
    model.eval()
    
    # Evaluate model
    print("🎯 Running evaluation...")
    metrics = evaluate_model(model, test_loader, device, 
                           save_path=os.path.join(args.output_dir, 'evaluation_results.png'))
    
    # Get detailed predictions
    print("📊 Generating detailed predictions...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            predictions_onehot = torch.nn.functional.one_hot(predictions, num_classes=len(DISEASE_LABELS))
            
            all_predictions.append(predictions_onehot.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenate all batches
    y_pred = torch.cat(all_predictions, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_prob = torch.cat(all_probabilities, dim=0).numpy()
    
    # Generate detailed plots
    print("📈 Creating evaluation plots...")
    metrics_df = create_detailed_evaluation_plots(y_true, y_pred, y_prob, args.output_dir)
    
    # Generate classification report
    print("📋 Generating classification report...")
    report = generate_classification_report(y_true, y_pred, args.output_dir)
    
    # Save detailed metrics
    print("💾 Saving detailed metrics...")
    detailed_metrics = {
        'overall_metrics': metrics,
        'per_class_metrics': metrics_df.to_dict('records'),
        'classification_report': report,
        'evaluation_info': {
            'model_path': args.model_path,
            'model_type': args.model_type,
            'test_samples': len(y_true),
            'num_classes': len(DISEASE_LABELS),
            'evaluation_date': datetime.now().isoformat()
        }
    }
    
    with open(os.path.join(args.output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # Print summary
    print("\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 Mean F1 Score: {metrics['mean_f1']:.4f}")
    print(f"📊 Mean AUC: {metrics['mean_auc']:.4f}")
    
    # Print per-class performance
    print("\n📋 Per-Class Performance:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['disease']}: F1={row['f1']:.3f}, AUC={row['auc']:.3f}")
    
    return detailed_metrics

def main():
    """Main function"""
    args = parse_args()
    
    try:
        metrics = evaluate_model_comprehensive(args)
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 