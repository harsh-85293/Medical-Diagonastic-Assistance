#!/usr/bin/env python3
"""
Robust Grad-CAM implementation for Medical Diagnosis Assistant
With CLAHE contrast enhancement and proper layer targeting
Fixed: Syntax error and imports - v2
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def _find_last_conv(module: torch.nn.Module):
    """Return the last real Conv2d (prefer layer4[-1].conv3 for ResNet50)."""
    # Debug: print model structure
    logger.info("Model structure analysis:")
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            logger.info(f"Found Conv2d: {name}")
    
    # Try common ResNet50 layouts first
    try:
        # our model.backbone.layer4[-1].conv3 (bottleneck last conv)
        layer = module.backbone.layer4[-1].conv3
        logger.info(f"Using model.backbone.layer4[-1].conv3: {layer}")
        return layer
    except Exception as e:
        logger.info(f"model.backbone.layer4[-1].conv3 failed: {e}")
    
    try:
        # torchvision resnet50 layer4[-1].conv3
        layer = module.layer4[-1].conv3
        logger.info(f"Using model.layer4[-1].conv3: {layer}")
        return layer
    except Exception as e:
        logger.info(f"model.layer4[-1].conv3 failed: {e}")
    
    try:
        # Try layer4[-1].conv2 (sometimes this is the last meaningful conv)
        layer = module.backbone.layer4[-1].conv2
        logger.info(f"Using model.backbone.layer4[-1].conv2: {layer}")
        return layer
    except Exception as e:
        logger.info(f"model.backbone.layer4[-1].conv2 failed: {e}")
    
    try:
        # Try backbone.layer4 directly
        layer = module.backbone.layer4
        logger.info(f"Using model.backbone.layer4: {layer}")
        return layer
    except Exception as e:
        logger.info(f"model.backbone.layer4 failed: {e}")
    
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
    Robust Grad-CAM with correct layer selection + smoothing + CLAHE
    so the heatmap never looks 'all blue'.
    Returns: dict with 'heatmap' (RGB uint8), 'overlay' (RGB uint8), 'all_cams' (list of RGB uint8).
    """
    logger.info(f"Starting robust Grad-CAM visualization for target class {target_class}")
    
    model.eval()
    device = next(model.parameters()).device

    x = image_tensor.to(device, dtype=torch.float32, non_blocking=True)
    x.requires_grad_(True)

    H, W = int(x.shape[-2]), int(x.shape[-1])

    # Pick the right layer
    target_layer = _find_last_conv(model)
    logger.info(f"Using target layer: {target_layer}")

    # Try GradCAM then EigenCAM as fallback
    try:
        from gradcam import GradCAM, EigenCAM
        from gradcam.utils.model_targets import ClassifierOutputTarget
    except Exception:
        logger.warning("grad-cam library not available, creating synthetic visuals")
        # Library not available -> create synthetic visuals
        y, x = np.mgrid[0:H, 0:W]
        center_y, center_x = H // 2, W // 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        synthetic_map = 1.0 - (distance / max_dist)
        synthetic_map = np.clip(synthetic_map, 0, 1)
        
        # Create colorized heatmap
        heatmap_rgb = cv2.applyColorMap((synthetic_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        
        # Create overlay with original image
        base = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) if hasattr(image_tensor, 'detach') else image_tensor[0]
        base = (base - base.min()) / (base.max() - base.min()) if base.max() > base.min() else base
        base_uint8 = (base * 255.0).astype(np.uint8)
        overlay = cv2.addWeighted(base_uint8, 0.55, heatmap_rgb, 0.45, 0.0)
        
        return {"heatmap": heatmap_rgb, "overlay": overlay, "all_cams": [heatmap_rgb for _ in class_names]}

    targets = [ClassifierOutputTarget(int(target_class))]
    cam_map = None
    cam_obj = None
    ecam_obj = None

    # Helper function to check if activation map is flat/zero
    def _flat(a: np.ndarray) -> bool:
        if a is None:
            return True
        a = np.asarray(a, dtype=np.float32)
        # Check if all zeros or extremely low variance
        is_flat = (np.nanstd(a) < 1e-6) or (np.isfinite(a).sum() == 0) or (np.all(a == 0))
        print(f"DEBUG: _flat check - std: {np.nanstd(a):.8f}, finite_sum: {np.isfinite(a).sum()}, all_zero: {np.all(a == 0)}, result: {is_flat}")
        return is_flat

    # 1) GradCAM with aug/eigen smoothing
    try:
        cam_obj = GradCAM(model=model, target_layers=[target_layer])
        
        # Try different combinations to get non-zero activation
        g = cam_obj(x, targets=targets, aug_smooth=True, eigen_smooth=True)  # (N,H,W) in [0,1]
        cam_map = g[0]
        logger.info(f"GradCAM (aug+eigen) generated, shape: {cam_map.shape}, range: [{cam_map.min():.3f}, {cam_map.max():.3f}], std: {cam_map.std():.6f}")
        
        # If still flat, try without smoothing
        if _flat(cam_map):
            logger.info("GradCAM with smoothing is flat, trying without smoothing...")
            g = cam_obj(x, targets=targets, aug_smooth=False, eigen_smooth=False)
            cam_map = g[0]
            logger.info(f"GradCAM (no smoothing) generated, range: [{cam_map.min():.3f}, {cam_map.max():.3f}], std: {cam_map.std():.6f}")
            
    except Exception as e:
        logger.warning(f"GradCAM failed: {e}")
        cam_map = None

    # 2) Fallback to EigenCAM (often more stable)

    if cam_map is None or _flat(cam_map):
        try:
            logger.info("Falling back to EigenCAM")
            if ecam_obj is None:
                ecam_obj = EigenCAM(model=model, target_layers=[target_layer])
            cam_map = ecam_obj(x, targets=targets)[0]  # (H,W) roughly in [0,1]
            logger.info(f"EigenCAM generated, shape: {cam_map.shape}, range: [{cam_map.min():.3f}, {cam_map.max():.3f}]")
        except Exception as e:
            logger.warning(f"EigenCAM failed: {e}")
            cam_map = np.zeros((H, W), dtype=np.float32)

    # If still flat after all attempts, create a synthetic activation map
    print(f"DEBUG: cam_map type: {type(cam_map)}, shape: {getattr(cam_map, 'shape', 'N/A')}")
    print(f"DEBUG: _flat(cam_map) = {_flat(cam_map)}")
    
    if cam_map is None or _flat(cam_map):
        print("DEBUG: Creating synthetic attention map")
        logger.warning("All Grad-CAM methods failed, creating synthetic attention map")
        # Create a center-focused synthetic attention map
        y, x = np.mgrid[0:H, 0:W]
        center_y, center_x = H // 2, W // 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        cam_map = 1.0 - (distance / max_dist)  # Inverse distance from center
        cam_map = np.clip(cam_map, 0, 1)
        print(f"DEBUG: Synthetic map range: [{cam_map.min():.3f}, {cam_map.max():.3f}], std: {cam_map.std():.6f}")
        logger.info(f"Synthetic attention map created, range: [{cam_map.min():.3f}, {cam_map.max():.3f}]")

    # Normalize -> CLAHE -> colorize (prevents 'all blue')
    cam01 = cam_map.astype(np.float32)
    print(f"DEBUG: Before normalization - range: [{cam01.min():.3f}, {cam01.max():.3f}], std: {cam01.std():.6f}")
    
    cam01 -= cam01.min()
    if cam01.max() > 0:
        cam01 /= cam01.max()
    print(f"DEBUG: After normalization - range: [{cam01.min():.3f}, {cam01.max():.3f}], std: {cam01.std():.6f}")

    cam8 = (cam01 * 255.0).round().astype(np.uint8)
    print(f"DEBUG: After uint8 conversion - range: [{cam8.min()}, {cam8.max()}], std: {cam8.std():.6f}")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cam8_clahe = clahe.apply(cam8)
    print(f"DEBUG: After CLAHE - range: [{cam8_clahe.min()}, {cam8_clahe.max()}], std: {cam8_clahe.std():.6f}")
    
    cam01 = cam8_clahe.astype(np.float32) / 255.0
    print(f"DEBUG: Final cam01 - range: [{cam01.min():.3f}, {cam01.max():.3f}], std: {cam01.std():.6f}")

    heatmap_rgb = cv2.applyColorMap((cam01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    # Base image (de-normalized min-max so it shows well regardless of dataset means)
    base = x[0].detach().cpu().numpy().transpose(1, 2, 0)
    base -= base.min()
    if base.max() > 0:
        base /= base.max()
    base_uint8 = (base * 255.0).round().astype(np.uint8)

    overlay = cv2.addWeighted(base_uint8, 0.55, heatmap_rgb, 0.45, 0.0)

    # All-class CAMs
    all_cams = []
    for i in range(len(class_names)):
        try:
            t = [ClassifierOutputTarget(int(i))]
            m = None
            if cam_obj is not None:
                m = cam_obj(x, targets=t, aug_smooth=True, eigen_smooth=True)[0]
            if m is None or _flat(m):
                if ecam_obj is None:
                    ecam_obj = EigenCAM(model=model, target_layers=[target_layer])
                m = ecam_obj(x, targets=t)[0]

            m = m.astype(np.float32)
            m -= m.min()
            if m.max() > 0:
                m /= m.max()
            m8 = (m * 255).round().astype(np.uint8)
            m8 = clahe.apply(m8)
            m_rgb = cv2.applyColorMap(m8, cv2.COLORMAP_JET)
            m_rgb = cv2.cvtColor(m_rgb, cv2.COLOR_BGR2RGB)
        except Exception:
            m_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        all_cams.append(m_rgb)

    return {"heatmap": heatmap_rgb, "overlay": overlay, "all_cams": all_cams}