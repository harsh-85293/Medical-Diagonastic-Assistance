#!/usr/bin/env python3
"""
Model utilities for Medical Diagnosis Assistant
Supports multi-class disease classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from utils.data_utils import DISEASE_LABELS, DISEASE_DISPLAY_NAMES

class MultiClassChestXRayModel(nn.Module):
    """Multi-class chest X-ray classification model based on ResNet50"""
    
    def __init__(self, num_classes=len(DISEASE_LABELS), pretrained=True):
        super(MultiClassChestXRayModel, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer for multi-class classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class VisionTransformerModel(nn.Module):
    """Vision Transformer for chest X-ray classification"""
    
    def __init__(self, num_classes=len(DISEASE_LABELS), img_size=224, patch_size=16):
        super(VisionTransformerModel, self).__init__()
        
        # Load pretrained ViT
        self.vit = models.vit_b_16(pretrained=True)
        
        # Replace the final layer
        self.vit.heads.head = nn.Linear(self.vit.hidden_dim, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for better performance"""
    
    def __init__(self, models_list, num_classes=len(DISEASE_LABELS)):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.num_classes = num_classes
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average the outputs
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # For multi-label classification
        if labels.shape[1] > 1:
            loss = criterion(outputs, labels.float())
            # Calculate accuracy (any correct label)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += ((preds == labels).sum(dim=1) > 0).sum().item()
        else:
            loss = criterion(outputs, labels.squeeze())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.squeeze()).sum().item()
        
        total += labels.size(0)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # For multi-label classification
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
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, device, save_path=None):
    """Evaluate model and generate comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities
            if labels.shape[1] > 1:  # Multi-label
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
            else:  # Single-label
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                predictions = F.one_hot(predictions, num_classes=len(DISEASE_LABELS))
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels.numpy(), all_predictions.numpy(), all_probabilities.numpy())
    
    # Plot results
    if save_path:
        plot_evaluation_results(all_labels.numpy(), all_predictions.numpy(), all_probabilities.numpy(), save_path)
    
    return metrics

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Overall accuracy
    if y_true.shape[1] > 1:  # Multi-label
        metrics['accuracy'] = ((y_pred == y_true).sum(axis=1) > 0).mean()
    else:  # Single-label
        metrics['accuracy'] = (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()
    
    # Per-class metrics
    for i, disease in enumerate(DISEASE_LABELS):
        if y_true.shape[1] > 1:  # Multi-label
            metrics[f'{disease}_precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'{disease}_recall'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'{disease}_f1'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            if y_prob.shape[1] > 1:
                metrics[f'{disease}_auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
        else:  # Single-label
            metrics[f'{disease}_precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'{disease}_recall'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'{disease}_f1'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            if y_prob.shape[1] > 1:
                metrics[f'{disease}_auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
    
    # Mean metrics
    metrics['mean_precision'] = np.mean([metrics[f'{d}_precision'] for d in DISEASE_LABELS])
    metrics['mean_recall'] = np.mean([metrics[f'{d}_recall'] for d in DISEASE_LABELS])
    metrics['mean_f1'] = np.mean([metrics[f'{d}_f1'] for d in DISEASE_LABELS])
    metrics['mean_auc'] = np.mean([metrics[f'{d}_auc'] for d in DISEASE_LABELS])
    
    return metrics

def plot_evaluation_results(y_true, y_pred, y_prob, save_path):
    """Plot comprehensive evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS],
                yticklabels=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS],
                ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Per-class F1 scores
    f1_scores = [metrics[f'{d}_f1'] for d in DISEASE_LABELS]
    axes[0,1].bar(range(len(DISEASE_LABELS)), f1_scores)
    axes[0,1].set_title('F1 Scores by Disease')
    axes[0,1].set_xticks(range(len(DISEASE_LABELS)))
    axes[0,1].set_xticklabels([DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS], rotation=45)
    axes[0,1].set_ylabel('F1 Score')
    
    # 3. ROC Curves
    for i, disease in enumerate(DISEASE_LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = metrics[f'{disease}_auc']
        axes[1,0].plot(fpr, tpr, label=f'{DISEASE_DISPLAY_NAMES[disease]} (AUC={auc:.3f})')
    
    axes[1,0].plot([0, 1], [0, 1], 'k--')
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curves')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 4. Precision-Recall curves
    for i, disease in enumerate(DISEASE_LABELS):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        axes[1,1].plot(recall, precision, label=DISEASE_DISPLAY_NAMES[disease])
    
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].set_title('Precision-Recall Curves')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_model(model_type='resnet50', num_classes=len(DISEASE_LABELS), pretrained=True):
    """Create model based on specified architecture"""
    if model_type == 'resnet50':
        return MultiClassChestXRayModel(num_classes, pretrained)
    elif model_type == 'vit':
        return VisionTransformerModel(num_classes)
    elif model_type == 'ensemble':
        models_list = [
            MultiClassChestXRayModel(num_classes, pretrained),
            VisionTransformerModel(num_classes)
        ]
        return EnsembleModel(models_list, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_loss_function(loss_type='bce', class_weights=None):
    """Get loss function for multi-class classification"""
    if loss_type == 'bce':
        if class_weights is not None:
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() 