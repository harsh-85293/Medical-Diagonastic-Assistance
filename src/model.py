"""
Model architecture for Medical AI Assistant.
ResNet50-based classifier with custom head for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CXRClassifier(nn.Module):
    """ResNet50-based chest X-ray classifier."""
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_compile: bool = True
    ):
        """
        Initialize the CXR classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate in the classifier head
            use_compile: Whether to use torch.compile if available
        """
        super().__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the backbone
        in_features = self.backbone.fc.in_features
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, num_classes)
        )
        
        # Replace the original classifier
        self.backbone.fc = self.classifier
        
        # Initialize classifier weights
        self._initialize_weights()
        
        # Try to compile the model if requested and available
        if use_compile and hasattr(torch, 'compile'):
            try:
                self = torch.compile(self)
                logger.info("Model compiled successfully with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        logger.info(f"Initialized CXR classifier with {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (before classifier).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        # Remove the classifier temporarily
        original_fc = self.backbone.fc
        self.backbone.fc = nn.Identity()
        
        # Extract features
        features = self.backbone(x)
        
        # Restore the classifier
        self.backbone.fc = original_fc
        
        return features
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        logger.info(f"Backbone {'frozen' if freeze else 'unfrozen'}")
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> CXRClassifier:
    """
    Create a CXR classifier based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized CXR classifier
    """
    model_config = config['model']
    
    model = CXRClassifier(
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        dropout_rate=model_config['dropout_rate'],
        use_compile=model_config['use_compile']
    )
    
    return model


def load_pretrained_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> CXRClassifier:
    """
    Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded CXR classifier
    """
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {checkpoint_path}")
    
    return model


def export_model(
    model: CXRClassifier,
    export_path: str,
    export_format: str = "torchscript",
    input_shape: tuple = (1, 3, 480, 480),
    device: torch.device = torch.device("cpu")
):
    """
    Export the model to different formats.
    
    Args:
        model: The model to export
        export_path: Path to save the exported model
        export_format: Export format ('torchscript' or 'onnx')
        input_shape: Input tensor shape for export
        device: Device to export from
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    if export_format.lower() == "torchscript":
        # Export to TorchScript
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(export_path)
        logger.info(f"Model exported to TorchScript: {export_path}")
        
    elif export_format.lower() == "onnx":
        # Export to ONNX
        try:
            import onnx
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"Model exported to ONNX: {export_path}")
        except ImportError:
            logger.error("ONNX export failed: onnx package not installed")
    
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def get_model_summary(model: CXRClassifier) -> Dict[str, Any]:
    """
    Get a summary of the model architecture and parameters.
    
    Args:
        model: The model to summarize
        
    Returns:
        Dictionary containing model summary
    """
    summary = {
        'total_parameters': model.get_total_parameters(),
        'trainable_parameters': model.get_trainable_parameters(),
        'architecture': 'ResNet50',
        'num_classes': model.classifier[-1].out_features,
        'input_channels': 3,
        'dropout_rate': 0.2  # This could be made configurable
    }
    
    return summary


if __name__ == "__main__":
    # Test model creation
    import yaml
    
    # Load config
    with open("src/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Print model summary
    summary = get_model_summary(model)
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 480, 480)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: (1, {config['model']['num_classes']})")
