"""
Evaluation module for Medical AI Assistant.
Implements comprehensive evaluation metrics and model calibration.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import yaml
from tqdm import tqdm

from .model import load_pretrained_model
from .data import create_data_loaders
from .losses import create_loss_function

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator for CXR classification."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
            device: Device to evaluate on
        """
        self.config = config
        self.device = device
        self.model = None
        self.test_loader = None
        self.classes = config['dataset']['classes']
        
        # Results storage
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.features = []
        
        logger.info(f"Evaluator initialized on device: {device}")
    
    def load_model(self, checkpoint_path: str):
        """Load the trained model."""
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = load_pretrained_model(checkpoint_path, self.config, self.device)
        logger.info("Model loaded successfully")
    
    def setup_data(self, data_dir: str):
        """Setup test data loader."""
        logger.info("Setting up test data...")
        _, _, self.test_loader = create_data_loaders(
            config=self.config,
            data_dir=data_dir,
            batch_size=32  # Use smaller batch size for evaluation
        )
        logger.info(f"Test data loaded: {len(self.test_loader.dataset)} samples")
    
    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and data must be loaded before evaluation")
        
        logger.info("Starting evaluation...")
        
        # Run inference
        self._run_inference()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        self._generate_plots()
        
        # Generate report
        self._generate_report(metrics)
        
        logger.info("Evaluation completed!")
        return metrics
    
    def _run_inference(self):
        """Run inference on test set."""
        self.model.eval()
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Running inference"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Get model output
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Store results
                self.predictions.extend(predictions.cpu().numpy())
                self.targets.extend(target.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        self.probabilities = np.array(self.probabilities)
        
        logger.info(f"Inference completed on {len(self.targets)} samples")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        logger.info("Calculating metrics...")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(self.targets, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.targets, self.predictions, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # ROC-AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(
                self.targets, self.probabilities, 
                multi_class='ovr', average='macro'
            )
            metrics['roc_auc_macro'] = roc_auc
        except Exception as e:
            logger.warning(f"ROC-AUC calculation failed: {e}")
            metrics['roc_auc_macro'] = None
        
        # Per-class ROC-AUC
        roc_auc_per_class = []
        for i in range(len(self.classes)):
            try:
                auc = roc_auc_score(
                    (self.targets == i).astype(int),
                    self.probabilities[:, i]
                )
                roc_auc_per_class.append(auc)
            except Exception:
                roc_auc_per_class.append(None)
        
        metrics['roc_auc_per_class'] = roc_auc_per_class
        
        # Confusion matrix
        cm = confusion_matrix(self.targets, self.predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        metrics['per_class_accuracy'] = per_class_accuracy.tolist()
        
        logger.info(f"Overall accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        if metrics['roc_auc_macro']:
            logger.info(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        
        return metrics
    
    def _generate_plots(self):
        """Generate evaluation plots."""
        logger.info("Generating plots...")
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(plots_dir / "confusion_matrix.png")
        
        # 2. ROC Curves
        self._plot_roc_curves(plots_dir / "roc_curves.png")
        
        # 3. Precision-Recall Curves
        self._plot_pr_curves(plots_dir / "pr_curves.png")
        
        # 4. Per-class Metrics
        self._plot_per_class_metrics(plots_dir / "per_class_metrics.png")
        
        # 5. Calibration Plot
        self._plot_calibration(plots_dir / "calibration.png")
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _plot_confusion_matrix(self, save_path: Path):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.targets, self.predictions)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, save_path: Path):
        """Plot ROC curves for all classes."""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.classes):
            try:
                fpr, tpr, _ = roc_curve(
                    (self.targets == i).astype(int),
                    self.probabilities[:, i]
                )
                auc = roc_auc_score(
                    (self.targets == i).astype(int),
                    self.probabilities[:, i]
                )
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
            except Exception:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, save_path: Path):
        """Plot Precision-Recall curves for all classes."""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.classes):
            try:
                precision, recall, _ = precision_recall_curve(
                    (self.targets == i).astype(int),
                    self.probabilities[:, i]
                )
                plt.plot(recall, precision, label=f'{class_name}')
            except Exception:
                continue
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, save_path: Path):
        """Plot per-class metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].bar(self.classes, self.metrics['per_class_accuracy'])
        axes[0, 0].set_title('Per-Class Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision
        axes[0, 1].bar(self.classes, self.metrics['precision_per_class'])
        axes[0, 1].set_title('Per-Class Precision')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1, 0].bar(self.classes, self.metrics['recall_per_class'])
        axes[1, 0].set_title('Per-Class Recall')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 1].bar(self.classes, self.metrics['f1_per_class'])
        axes[1, 1].set_title('Per-Class F1-Score')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration(self, save_path: Path):
        """Plot calibration curves."""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.classes):
            try:
                # Get binary targets for this class
                binary_targets = (self.targets == i).astype(int)
                binary_probs = self.probabilities[:, i]
                
                # Calculate calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_targets, binary_probs, n_bins=10
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', label=f'{class_name}')
            except Exception:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, metrics: Dict[str, Any]):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        report_path = Path("evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MEDICAL AI ASSISTANT - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            if metrics['roc_auc_macro']:
                f.write(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 20 + "\n")
            for i, class_name in enumerate(self.classes):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Accuracy: {metrics['per_class_accuracy'][i]:.4f}\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"  Support: {metrics['support_per_class'][i]}\n")
                if metrics['roc_auc_per_class'][i]:
                    f.write(f"  ROC-AUC: {metrics['roc_auc_per_class'][i]:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully!\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Calculate uncertainty metrics for predictions."""
        logger.info("Calculating uncertainty metrics...")
        
        # Get top-2 probabilities
        top2_probs = np.sort(self.probabilities, axis=1)[:, -2:]
        top1_probs = top2_probs[:, -1]
        top2_probs = top2_probs[:, -2]
        
        # Uncertainty metrics
        uncertainty_metrics = {
            'confidence_gap': top1_probs - top2_probs,
            'entropy': -np.sum(self.probabilities * np.log(self.probabilities + 1e-10), axis=1),
            'max_confidence': top1_probs,
            'prediction_confidence': top1_probs[self.targets == self.predictions],
            'prediction_uncertainty': top1_probs[self.targets != self.predictions]
        }
        
        # Calculate statistics
        stats = {}
        for metric_name, values in uncertainty_metrics.items():
            if len(values) > 0:
                stats[f'{metric_name}_mean'] = np.mean(values)
                stats[f'{metric_name}_std'] = np.std(values)
                stats[f'{metric_name}_min'] = np.min(values)
                stats[f'{metric_name}_max'] = np.max(values)
        
        logger.info("Uncertainty metrics calculated")
        return stats


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CXR Classifier")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
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
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(config, device)
    evaluator.load_model(args.ckpt)
    evaluator.setup_data(args.data_dir)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    if metrics['roc_auc_macro']:
        print(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
