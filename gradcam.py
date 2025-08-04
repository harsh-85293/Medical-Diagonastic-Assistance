#!/usr/bin/env python3
"""
Grad-CAM implementation for Medical Diagnosis Assistant
Supports multi-class disease classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from utils.data_utils import DISEASE_LABELS, DISEASE_DISPLAY_NAMES

class GradCAM:
    """Grad-CAM implementation for multi-class classification"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        if self.target_layer is None:
            self.target_layer = self._get_target_layer()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def _get_target_layer(self):
        """Get the target layer for Grad-CAM"""
        # For ResNet50
        if hasattr(self.model, 'resnet'):
            return self.model.resnet.layer4[-1]
        # For Vision Transformer
        elif hasattr(self.model, 'vit'):
            return self.model.vit.encoder.layers[-1]
        # For simple models
        elif hasattr(self.model, 'features'):
            return self.model.features[-2]  # Last conv layer
        else:
            # Fallback: find last convolutional layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if conv_layers:
                return conv_layers[-1]
            else:
                raise ValueError("No suitable target layer found for Grad-CAM")
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Grad-CAM for a specific class"""
        # Forward pass
        output = self.model(input_image)
        
        # If no specific class is provided, use the predicted class
        if class_idx is None:
            if output.shape[1] > 1:  # Multi-label
                class_idx = torch.argmax(output, dim=1).item()
            else:  # Single-label
                class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations not captured. Check target layer.")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def generate_multi_class_cam(self, input_image, class_indices=None):
        """Generate Grad-CAM for multiple classes"""
        if class_indices is None:
            # Generate for all classes
            class_indices = list(range(len(DISEASE_LABELS)))
        
        cams = {}
        for class_idx in class_indices:
            try:
                cam = self.generate_cam(input_image, class_idx)
                cams[class_idx] = cam
            except Exception as e:
                print(f"Warning: Could not generate CAM for class {class_idx}: {e}")
                cams[class_idx] = None
        
        return cams

def generate_gradcam_for_single_image(model, image_path, class_idx=None, save_path=None):
    """Generate Grad-CAM for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Preprocess for model
    from utils.data_utils import get_transforms
    _, transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    
    if class_idx is None:
        # Generate for predicted class
        with torch.no_grad():
            output = model(input_tensor)
            if output.shape[1] > 1:  # Multi-label
                class_idx = torch.argmax(output, dim=1).item()
            else:  # Single-label
                class_idx = torch.argmax(output, dim=1).item()
    
    cam = gradcam.generate_cam(input_tensor, class_idx)
    
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = heatmap * 0.7 + original_image * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    if save_path:
        # Save results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Grad-CAM ({DISEASE_DISPLAY_NAMES.get(DISEASE_LABELS[class_idx], DISEASE_LABELS[class_idx])})')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return cam_resized, overlay

def generate_multi_class_gradcam(model, image_path, save_dir=None):
    """Generate Grad-CAM for all classes"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Preprocess for model
    from utils.data_utils import get_transforms
    _, transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0)
    
    # Generate Grad-CAM for all classes
    gradcam = GradCAM(model)
    cams = gradcam.generate_multi_class_cam(input_tensor)
    
    # Create visualizations
    results = {}
    for class_idx, cam in cams.items():
        if cam is not None:
            # Resize CAM
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = heatmap * 0.7 + original_image * 0.3
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            results[class_idx] = {
                'cam': cam_resized,
                'heatmap': heatmap,
                'overlay': overlay,
                'disease_name': DISEASE_DISPLAY_NAMES.get(DISEASE_LABELS[class_idx], DISEASE_LABELS[class_idx])
            }
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create grid visualization
        n_classes = len(results)
        cols = 4
        rows = (n_classes + 2) // cols  # +2 for original image and predicted class
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            if output.shape[1] > 1:  # Multi-label
                predicted_class = torch.argmax(output, dim=1).item()
            else:  # Single-label
                predicted_class = torch.argmax(output, dim=1).item()
        
        # Predicted class CAM
        if predicted_class in results:
            axes[1].imshow(results[predicted_class]['overlay'])
            axes[1].set_title(f'Predicted: {results[predicted_class]["disease_name"]}')
            axes[1].axis('off')
        
        # All class CAMs
        for i, (class_idx, result) in enumerate(results.items()):
            if i + 2 < len(axes):
                axes[i + 2].imshow(result['overlay'])
                axes[i + 2].set_title(f'{result["disease_name"]}')
                axes[i + 2].axis('off')
        
        # Hide unused subplots
        for i in range(len(results) + 2, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'multi_class_gradcam.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual CAMs
        for class_idx, result in results.items():
            disease_name = result['disease_name'].replace(' ', '_').lower()
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(result['cam'], cmap='jet')
            plt.title(f'Grad-CAM: {result["disease_name"]}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(result['overlay'])
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gradcam_{disease_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    return results

def create_gradcam_comparison(model, image_paths, save_path=None):
    """Create Grad-CAM comparison for multiple images"""
    n_images = len(image_paths)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, image_path in enumerate(image_paths):
        # Generate Grad-CAM
        cam, overlay = generate_gradcam_for_single_image(model, image_path)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Display results
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title(f'Grad-CAM {i+1}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay {i+1}')
        axes[i, 2].axis('off')
        
        # Add prediction info
        from utils.data_utils import get_transforms
        _, transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            if output.shape[1] > 1:  # Multi-label
                predicted_class = torch.argmax(output, dim=1).item()
            else:  # Single-label
                predicted_class = torch.argmax(output, dim=1).item()
            
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities[0, predicted_class].item()
        
        axes[i, 3].text(0.1, 0.5, f'Prediction: {DISEASE_DISPLAY_NAMES.get(DISEASE_LABELS[predicted_class], DISEASE_LABELS[predicted_class])}\nConfidence: {confidence:.2%}', 
                        transform=axes[i, 3].transAxes, fontsize=12, verticalalignment='center')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 