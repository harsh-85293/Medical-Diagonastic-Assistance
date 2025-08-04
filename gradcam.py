"""
Grad-CAM implementation for Medical Diagnosis Assistant
Provides model explainability by generating attention heatmaps
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

class GradCAM:
    """Grad-CAM implementation for model explainability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on the target layer
        target_module = self.get_target_module()
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def get_target_module(self):
        """Get the target layer module"""
        # For ResNet50, we typically use the last convolutional layer
        if hasattr(self.model, 'resnet'):
            # Get the last layer of the ResNet backbone
            return self.model.resnet.layer4[-1]
        else:
            # Fallback for other architectures
            return list(self.model.modules())[-2]
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM for the given input image"""
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
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
    
    def overlay_heatmap(self, original_image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        
        # Convert heatmap to RGB
        heatmap_rgb = np.zeros((*heatmap.shape, 3))
        heatmap_rgb[:, :, 0] = heatmap  # Red channel
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap_rgb, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize original image to [0, 1]
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Overlay
        overlay = alpha * heatmap_resized + (1 - alpha) * original_image
        overlay = np.clip(overlay, 0, 1)
        
        return overlay

def generate_gradcam_for_batch(model, images, labels, save_dir='gradcam_results'):
    """Generate Grad-CAM for a batch of images"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=None)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormalized_images = images * std + mean
    denormalized_images = torch.clamp(denormalized_images, 0, 1)
    
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for i in range(len(images)):
        # Get original image
        original_image = denormalized_images[i].permute(1, 2, 0).numpy()
        
        # Generate Grad-CAM
        heatmap = gradcam.generate_cam(images[i:i+1], target_class=labels[i])
        
        # Overlay heatmap
        overlay = gradcam.overlay_heatmap(original_image, heatmap)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image\nClass: {class_names[labels[i]]}')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f'gradcam_sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Clean up hooks
    gradcam.remove_hooks()
    
    print(f"Grad-CAM visualizations saved to: {save_dir}")

def generate_gradcam_for_single_image(model, image_path, model_prediction, save_path=None):
    """Generate Grad-CAM for a single image"""
    
    # Load and preprocess image
    from utils.data_utils import get_transforms
    
    _, val_transform = get_transforms()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image) / 255.0
    
    # Transform image
    input_tensor = val_transform(image).unsqueeze(0)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=None)
    
    # Generate heatmap
    heatmap = gradcam.generate_cam(input_tensor, target_class=model_prediction)
    
    # Overlay heatmap
    overlay = gradcam.overlay_heatmap(original_image, heatmap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nPredicted: {class_names[model_prediction]}')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to: {save_path}")
    else:
        plt.show()
    
    # Clean up hooks
    gradcam.remove_hooks()
    
    return heatmap, overlay

def analyze_attention_patterns(model, test_loader, save_dir='attention_analysis'):
    """Analyze attention patterns across different classes"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=None)
    
    class_names = ['NORMAL', 'PNEUMONIA']
    class_heatmaps = {0: [], 1: []}
    
    # Generate heatmaps for all test samples
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(len(images)):
            heatmap = gradcam.generate_cam(images[i:i+1], target_class=labels[i])
            class_heatmaps[labels[i].item()].append(heatmap)
        
        if batch_idx >= 10:  # Limit for memory
            break
    
    # Analyze attention patterns
    for class_idx, class_name in enumerate(class_names):
        if class_heatmaps[class_idx]:
            heatmaps = np.array(class_heatmaps[class_idx])
            
            # Calculate mean attention map
            mean_attention = np.mean(heatmaps, axis=0)
            std_attention = np.std(heatmaps, axis=0)
            
            # Plot mean attention
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(mean_attention, cmap='jet')
            plt.title(f'Mean Attention - {class_name}')
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(std_attention, cmap='jet')
            plt.title(f'Attention Variability - {class_name}')
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'attention_analysis_{class_name.lower()}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Clean up hooks
    gradcam.remove_hooks()
    
    print(f"Attention analysis saved to: {save_dir}")

if __name__ == "__main__":
    # Example usage
    print("Grad-CAM module for Medical Diagnosis Assistant")
    print("This module provides model explainability through attention heatmaps.")
    print("Use the functions in this module with your trained model to generate visualizations.") 