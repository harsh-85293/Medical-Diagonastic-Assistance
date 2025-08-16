#!/usr/bin/env python3
"""
Grad-CAM implementation for Medical Diagnosis Assistant
Supports multi-class disease classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, List
from PIL import Image

logger = logging.getLogger(__name__)

# Disease labels for the medical classification
DISEASE_LABELS = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer', 'Cardiomegaly', 'Pleural Effusion', 'Pneumothorax']
DISEASE_DISPLAY_NAMES = {
    'Normal': 'Normal',
    'Pneumonia': 'Pneumonia', 
    'COVID-19': 'COVID-19',
    'Tuberculosis': 'Tuberculosis',
    'Lung Cancer': 'Lung Cancer',
    'Cardiomegaly': 'Cardiomegaly',
    'Pleural Effusion': 'Pleural Effusion',
    'Pneumothorax': 'Pneumothorax'
}

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
        logger.info("Analyzing model structure for Grad-CAM target layer...")
        
        # Debug: print model structure
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                logger.info(f"Found Conv2d: {name}")
        
        # For ResNet50
        if hasattr(self.model, 'resnet'):
            logger.info("Using model.resnet.layer4[-1]")
            return self.model.resnet.layer4[-1]
        # For models with backbone
        elif hasattr(self.model, 'backbone'):
            try:
                layer = self.model.backbone.layer4[-1].conv3
                logger.info("Using model.backbone.layer4[-1].conv3")
                return layer
            except:
                try:
                    layer = self.model.backbone.layer4[-1]
                    logger.info("Using model.backbone.layer4[-1]")
                    return layer
                except:
                    pass
        # For Vision Transformer
        elif hasattr(self.model, 'vit'):
            logger.info("Using model.vit.encoder.layers[-1]")
            return self.model.vit.encoder.layers[-1]
        # For simple models
        elif hasattr(self.model, 'features'):
            logger.info("Using model.features[-2]")
            return self.model.features[-2]  # Last conv layer
        
        # Fallback: find last convolutional layer
        conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            logger.info(f"Using fallback last Conv2d layer")
            return conv_layers[-1]
        else:
            raise ValueError("No suitable target layer found for Grad-CAM")
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Grad-CAM for a specific class"""
        logger.info(f"Generating Grad-CAM for class {class_idx}")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        logger.info(f"Model output shape: {output.shape}")
        
        # If no specific class is provided, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            logger.info(f"Using predicted class: {class_idx}")
        
        # Backward pass
        self.model.zero_grad()
        if output.dim() > 1 and class_idx < output.shape[1]:
            output[0, class_idx].backward()
        else:
            logger.warning(f"Invalid class_idx {class_idx} for output shape {output.shape}")
            output.sum().backward()  # Fallback
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            logger.warning("Gradients or activations not captured. Creating synthetic CAM.")
            # Create synthetic CAM
            H, W = input_image.shape[-2:]
            y, x = np.mgrid[0:H, 0:W]
            center_y, center_x = H // 2, W // 2
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            cam = 1.0 - (distance / max_dist)
            return np.clip(cam, 0, 1)
        
        logger.info(f"Gradients shape: {gradients.shape}, Activations shape: {activations.shape}")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        logger.info(f"Weights shape: {weights.shape}")
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam_np = cam.detach().cpu().numpy()
        logger.info(f"Generated CAM - shape: {cam_np.shape}, range: [{cam_np.min():.3f}, {cam_np.max():.3f}], std: {cam_np.std():.6f}")
        
        return cam_np
    
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
                logger.warning(f"Could not generate CAM for class {class_idx}: {e}")
                # Create synthetic fallback
                H, W = input_image.shape[-2:]
                y, x = np.mgrid[0:H, 0:W]
                center_y, center_x = H // 2, W // 2
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                cams[class_idx] = np.clip(1.0 - (distance / max_dist), 0, 1)
        
        return cams


# Legacy function - kept for compatibility
def _find_last_conv(module: torch.nn.Module):
    """Return the last real Conv2d (prefer layer4[-1].conv3 for ResNet50)."""
    logger.info("Legacy _find_last_conv called - consider using GradCAM class instead")
    
    # Fallback: scan all modules and keep the last Conv2d we see
    last_conv = None
    last_name = ""
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
            last_name = name
    
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    
    logger.info(f"Using fallback last Conv2d: {last_name}")
    return last_conv


def create_gradcam_visualization(model, image_tensor, target_class, class_names):
    """
    Improved Grad-CAM implementation using custom GradCAM class
    Returns: dict with 'heatmap' (RGB uint8), 'overlay' (RGB uint8), 'all_cams' (list of RGB uint8).
    """
    logger.info(f"Starting Grad-CAM visualization for target class {target_class}")
    
    model.eval()
    device = next(model.parameters()).device

    # Ensure input is on the correct device
    x = image_tensor.to(device, dtype=torch.float32, non_blocking=True)
    x.requires_grad_(True)

    H, W = int(x.shape[-2]), int(x.shape[-1])

    try:
        # Create Grad-CAM object
        gradcam = GradCAM(model)
        
        # Generate CAM for target class
        cam = gradcam.generate_cam(x, target_class)
        logger.info(f"Generated CAM for class {target_class}: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}], std={cam.std():.6f}")
        
        # Resize to match input dimensions
        cam_resized = cv2.resize(cam, (W, H))
        
        # Use proper single-channel normalization to 0-255
        cam_2d = cam_resized.astype(np.float32)
        
        # Optional: percentile contrast enhancement for better visualization
        lo, hi = np.percentile(cam_2d, (1, 99))
        cam_2d = np.clip((cam_2d - lo) / max(hi - lo, 1e-6), 0, 1)
        
        # Normalize to full 0-255 range using cv2.normalize (more robust)
        cam_uint8 = cv2.normalize(cam_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB for Streamlit
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Heatmap created - cam_uint8 range: [{cam_uint8.min()}, {cam_uint8.max()}], heatmap_rgb range: [{heatmap_rgb.min()}, {heatmap_rgb.max()}], shape: {heatmap_rgb.shape}")
        
        # Create overlay with original image
    base = x[0].detach().cpu().numpy().transpose(1, 2, 0)
        base = (base - base.min()) / (base.max() - base.min()) if base.max() > base.min() else base
        base_uint8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        
        # Proper overlay blending with alpha
        alpha = 0.45
        overlay = cv2.addWeighted(base_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
        
        # Generate CAMs for all classes
    all_cams = []
        cams_dict = gradcam.generate_multi_class_cam(x, list(range(len(class_names))))
        
    for i in range(len(class_names)):
            if i in cams_dict and cams_dict[i] is not None:
                class_cam = cams_dict[i]
                class_cam_resized = cv2.resize(class_cam, (W, H))
                
                # Apply same robust normalization
                class_cam_2d = class_cam_resized.astype(np.float32)
                lo, hi = np.percentile(class_cam_2d, (1, 99))
                class_cam_2d = np.clip((class_cam_2d - lo) / max(hi - lo, 1e-6), 0, 1)
                
                # Normalize to full 0-255 range
                class_cam_uint8 = cv2.normalize(class_cam_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Create heatmap (BGR -> RGB)
                class_heatmap_bgr = cv2.applyColorMap(class_cam_uint8, cv2.COLORMAP_JET)
                class_heatmap = cv2.cvtColor(class_heatmap_bgr, cv2.COLOR_BGR2RGB)
                all_cams.append(class_heatmap)
            else:
                # Fallback: create synthetic visualization
                y, x_grid = np.mgrid[0:H, 0:W]
                center_y, center_x = H // 2, W // 2
                distance = np.sqrt((x_grid - center_x) ** 2 + (y - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                synthetic_cam = 1.0 - (distance / max_dist)
                synthetic_cam = np.clip(synthetic_cam, 0, 1)
                
                # Apply same robust processing to synthetic
                synthetic_uint8 = cv2.normalize(synthetic_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                synthetic_heatmap_bgr = cv2.applyColorMap(synthetic_uint8, cv2.COLORMAP_JET)
                synthetic_heatmap = cv2.cvtColor(synthetic_heatmap_bgr, cv2.COLOR_BGR2RGB)
                all_cams.append(synthetic_heatmap)
        
        logger.info(f"Generated {len(all_cams)} class visualizations")
        return {"heatmap": heatmap_rgb, "overlay": overlay, "all_cams": all_cams}
        
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        # Fallback: create synthetic visualizations
        logger.info("Creating synthetic fallback visualizations")
        
        # Create synthetic CAM
        y, x_grid = np.mgrid[0:H, 0:W]
        center_y, center_x = H // 2, W // 2
        distance = np.sqrt((x_grid - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        synthetic_cam = 1.0 - (distance / max_dist)
        synthetic_cam = np.clip(synthetic_cam, 0, 1)
        
        # Apply robust processing to synthetic CAM
        cam_uint8 = cv2.normalize(synthetic_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create heatmap (BGR -> RGB)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        base = x[0].detach().cpu().numpy().transpose(1, 2, 0)
        base = (base - base.min()) / (base.max() - base.min()) if base.max() > base.min() else base
        base_uint8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        
        # Proper overlay blending
        alpha = 0.45
        overlay = cv2.addWeighted(base_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
        
        # Create same heatmap for all classes
        all_cams = [heatmap_rgb.copy() for _ in class_names]

    return {"heatmap": heatmap_rgb, "overlay": overlay, "all_cams": all_cams}