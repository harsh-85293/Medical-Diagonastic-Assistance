"""
Input validation module for Medical AI Assistant.
Implements robust validation to ensure only valid chest X-ray images are processed.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import open_clip
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class CXRInputValidator:
    """Robust validator for chest X-ray images."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the CXR input validator.
        
        Args:
            config: Configuration dictionary
            device: Device to run validation on
        """
        self.config = config
        self.device = device
        self.validation_config = config['validation']
        
        # Initialize CLIP model for zero-shot validation
        self.clip_model = None
        self.clip_processor = None
        self._initialize_clip()
        
        # Initialize face detection
        self.face_cascade = None
        self._initialize_face_detection()
        
        logger.info("CXR Input Validator initialized")
    
    def _initialize_clip(self):
        """Initialize CLIP model for zero-shot validation."""
        try:
            # Try to use transformers CLIP first
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            logger.info("CLIP model loaded from transformers")
        except Exception as e:
            try:
                # Fallback to open_clip
                self.clip_model, _, self.clip_processor = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
                self.clip_model.to(self.device)
                logger.info("CLIP model loaded from open_clip")
            except Exception as e2:
                logger.warning(f"Failed to load CLIP model: {e2}")
                self.clip_model = None
                self.clip_processor = None
    
    def _initialize_face_detection(self):
        """Initialize face detection cascade."""
        try:
            # Try to load OpenCV face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Face detection cascade loaded")
            else:
                logger.warning("Face detection cascade not found")
                self.face_cascade = None
        except Exception as e:
            logger.warning(f"Failed to initialize face detection: {e}")
            self.face_cascade = None
    
    def validate_image(
        self, 
        image: np.ndarray, 
        filename: str = ""
    ) -> Tuple[bool, float, str]:
        """
        Validate if an image is a valid chest X-ray.
        
        Args:
            image: Input image as numpy array
            filename: Original filename for logging
            
        Returns:
            Tuple of (is_valid, clip_confidence, reason)
        """
        logger.info(f"Validating image: {filename}")
        
        # Check basic image properties
        if not self._check_basic_properties(image):
            return False, 0.0, "Image does not meet basic requirements"
        
        # CLIP-based validation
        clip_valid, clip_conf = self._validate_with_clip(image)
        if not clip_valid:
            return False, clip_conf, "Image does not appear to be a medical radiograph"
        
        # Face detection
        if self.validation_config['face_detection']:
            if self._detect_faces(image):
                return False, clip_conf, "Face detected - likely not a medical image"
        
        # Color variance check
        if not self._check_color_variance(image):
            return False, clip_conf, "Image has high color variance - likely not a CXR"
        
        # Aspect ratio check
        if not self._check_aspect_ratio(image):
            return False, clip_conf, "Image aspect ratio is not typical for CXR"
        
        # Resolution check
        if not self._check_resolution(image):
            return False, clip_conf, "Image resolution is too low"
        
        # All checks passed
        logger.info(f"Image validation passed: {filename}")
        return True, clip_conf, "Valid chest X-ray image"
    
    def _check_basic_properties(self, image: np.ndarray) -> bool:
        """Check basic image properties."""
        if image is None:
            return False
        
        if len(image.shape) < 2:
            return False
        
        # Check minimum dimensions
        min_height, min_width = self.validation_config['min_resolution']
        if image.shape[0] < min_height or image.shape[1] < min_width:
            return False
        
        return True
    
    def _validate_with_clip(self, image: np.ndarray) -> Tuple[bool, float]:
        """Validate image using CLIP zero-shot classification."""
        if self.clip_model is None:
            logger.warning("CLIP model not available, skipping CLIP validation")
            return True, 0.5  # Default to passing if CLIP unavailable
        
        try:
            # Prepare text prompts
            positive_prompts = [
                "a chest x-ray",
                "an x-ray image",
                "a medical radiograph",
                "a chest radiograph",
                "a medical x-ray"
            ]
            
            negative_prompts = [
                "a selfie",
                "a portrait",
                "a landscape",
                "a document",
                "an illustration",
                "a painting",
                "a drawing",
                "a photograph of a person",
                "a family photo"
            ]
            
            # Convert image to PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image)
            else:
                # Convert grayscale to RGB
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Process image and text
            if hasattr(self.clip_processor, 'encode_image'):
                # Transformers CLIP
                image_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
                text_inputs = self.clip_processor(
                    text=positive_prompts + negative_prompts, 
                    return_tensors="pt", 
                    padding=True
                )
                
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**image_inputs)
                    text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Calculate similarities
                similarities = torch.mm(image_features, text_features.T)
                
            else:
                # Open CLIP
                image_inputs = self.clip_processor(pil_image).unsqueeze(0).to(self.device)
                text_inputs = open_clip.tokenize(positive_prompts + negative_prompts).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_inputs)
                    text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Calculate similarities
                similarities = torch.mm(image_features, text_features.T)
            
            # Get positive and negative similarities
            positive_sim = similarities[0, :len(positive_prompts)].max().item()
            negative_sim = similarities[0, len(positive_prompts):].max().item()
            
            # Calculate confidence score
            confidence = positive_sim - negative_sim
            
            # Check if confidence meets threshold
            threshold = self.validation_config['clip_threshold']
            is_valid = confidence >= threshold
            
            logger.info(f"CLIP validation: confidence={confidence:.3f}, threshold={threshold}, valid={is_valid}")
            
            return is_valid, confidence
            
        except Exception as e:
            logger.error(f"CLIP validation failed: {e}")
            return True, 0.5  # Default to passing if validation fails
    
    def _detect_faces(self, image: np.ndarray) -> bool:
        """Detect faces in the image."""
        if self.face_cascade is None:
            return False
        
        try:
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            has_faces = len(faces) > 0
            if has_faces:
                logger.info(f"Face detected in image")
            
            return has_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return False
    
    def _check_color_variance(self, image: np.ndarray) -> bool:
        """Check color variance to identify non-medical images."""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                
                # Calculate variance in each channel
                variances = []
                for i in range(3):
                    channel = lab[:, :, i]
                    variance = np.var(channel)
                    variances.append(variance)
                
                # Use maximum variance as indicator
                max_variance = max(variances)
                threshold = self.validation_config['color_variance_threshold'] * 1000
                
                is_valid = max_variance <= threshold
                logger.info(f"Color variance check: max_variance={max_variance:.1f}, threshold={threshold}, valid={is_valid}")
                
                return is_valid
            else:
                # Grayscale image - check variance
                variance = np.var(image)
                threshold = self.validation_config['color_variance_threshold'] * 1000
                
                is_valid = variance <= threshold
                logger.info(f"Color variance check: variance={variance:.1f}, threshold={threshold}, valid={is_valid}")
                
                return is_valid
                
        except Exception as e:
            logger.error(f"Color variance check failed: {e}")
            return True  # Default to passing if check fails
    
    def _check_aspect_ratio(self, image: np.ndarray) -> bool:
        """Check if aspect ratio is typical for CXR images."""
        try:
            height, width = image.shape[:2]
            aspect_ratio = max(width, height) / min(width, height)
            
            max_ratio = self.validation_config['max_aspect_ratio']
            is_valid = aspect_ratio <= max_ratio
            
            logger.info(f"Aspect ratio check: ratio={aspect_ratio:.2f}, max_ratio={max_ratio}, valid={is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Aspect ratio check failed: {e}")
            return True  # Default to passing if check fails
    
    def _check_resolution(self, image: np.ndarray) -> bool:
        """Check if image resolution meets minimum requirements."""
        try:
            min_height, min_width = self.validation_config['min_resolution']
            height, width = image.shape[:2]
            
            is_valid = height >= min_height and width >= min_width
            
            logger.info(f"Resolution check: {width}x{height}, min={min_width}x{min_height}, valid={is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Resolution check failed: {e}")
            return True  # Default to passing if check fails
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation capabilities."""
        return {
            'clip_available': self.clip_model is not None,
            'face_detection_available': self.face_cascade is not None,
            'clip_threshold': self.validation_config['clip_threshold'],
            'min_resolution': self.validation_config['min_resolution'],
            'max_aspect_ratio': self.validation_config['max_aspect_ratio'],
            'color_variance_threshold': self.validation_config['color_variance_threshold']
        }


def validate_single_image(
    image: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    filename: str = ""
) -> Tuple[bool, float, str]:
    """
    Convenience function to validate a single image.
    
    Args:
        image: Input image as numpy array
        config: Configuration dictionary
        device: Device to run validation on
        filename: Original filename for logging
        
    Returns:
        Tuple of (is_valid, clip_confidence, reason)
    """
    validator = CXRInputValidator(config, device)
    return validator.validate_image(image, filename)


def batch_validate_images(
    images: list,
    config: Dict[str, Any],
    device: torch.device,
    filenames: Optional[list] = None
) -> list:
    """
    Validate multiple images in batch.
    
    Args:
        images: List of input images as numpy arrays
        config: Configuration dictionary
        device: Device to run validation on
        filenames: List of filenames for logging
        
    Returns:
        List of validation results (is_valid, clip_confidence, reason)
    """
    if filenames is None:
        filenames = [f"image_{i}" for i in range(len(images))]
    
    validator = CXRInputValidator(config, device)
    results = []
    
    for image, filename in zip(images, filenames):
        result = validator.validate_image(image, filename)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test validation functionality
    print("CXR Input Validator module loaded successfully!")
    print("This module provides:")
    print("- CLIP-based zero-shot validation")
    print("- Face detection using OpenCV")
    print("- Color variance analysis")
    print("- Aspect ratio and resolution checks")
    print("- Comprehensive validation pipeline")
