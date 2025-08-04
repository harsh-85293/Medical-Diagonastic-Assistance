#!/usr/bin/env python3
"""
Create a minimal demo model for the Medical Diagnosis Assistant
This creates a simple model checkpoint that the Streamlit app can load.
"""

import os
import torch
import torch.nn as nn
import numpy as np

class SimpleChestXRayModel(nn.Module):
    """Simple model for demo purposes"""
    def __init__(self, num_classes=2):
        super(SimpleChestXRayModel, self).__init__()
        # Simple CNN for demo
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_demo_model():
    """Create and save a demo model"""
    print("Creating demo model...")
    
    # Create model
    model = SimpleChestXRayModel(num_classes=2)
    
    # Create output directory
    os.makedirs('./models', exist_ok=True)
    
    # Save model
    model_path = './models/chest_xray_demo.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ Demo model saved to: {model_path}")
    
    # Test loading
    try:
        loaded_model = SimpleChestXRayModel(num_classes=2)
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        loaded_model.eval()
        print("✅ Model loading test successful!")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded_model(dummy_input)
            print(f"✅ Model inference test successful! Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")

if __name__ == "__main__":
    create_demo_model() 