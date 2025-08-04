#!/usr/bin/env python3
"""
Streamlit App for Medical Diagnosis Assistant
Provides a web interface for chest X-ray analysis
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import cv2
from utils.data_utils import get_transforms, get_device, load_checkpoint
from utils.model_utils import ChestXRayModel
from gradcam import GradCAM, generate_gradcam_for_single_image

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-prediction {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    .normal-confidence {
        background-color: #28a745;
    }
    .pneumonia-confidence {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = get_device()
        model = ChestXRayModel(num_classes=2, pretrained=False)
        model = model.to(device)
        
        # Load trained weights
        model_path = "models/best_model.pt"
        if os.path.exists(model_path):
            optimizer = torch.optim.Adam(model.parameters())
            model, optimizer, epoch, loss = load_checkpoint(model, optimizer, model_path)
            model.eval()
            return model, device
        else:
            st.error("Model file not found. Please train the model first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess uploaded image"""
    _, val_transform = get_transforms()
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply transforms
    input_tensor = val_transform(image).unsqueeze(0)
    return input_tensor

def predict_image(model, device, input_tensor):
    """Make prediction on the input image"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence, probabilities[0].cpu().numpy()

def create_gradcam_visualization(model, image, prediction):
    """Create Grad-CAM visualization"""
    try:
        # Save temporary image
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Generate Grad-CAM
        heatmap, overlay = generate_gradcam_for_single_image(
            model, temp_path, prediction, save_path=None
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return heatmap, overlay
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Diagnosis Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Chest X-Ray Analysis for Pneumonia Detection</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure the model has been trained.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image (JPG, PNG)
    2. The model will analyze the image
    3. View the prediction and confidence
    4. Explore the Grad-CAM visualization
    """)
    
    st.sidebar.markdown("## ⚠️ Disclaimer")
    st.sidebar.markdown("""
    This tool is for educational purposes only.
    It should not replace professional medical diagnosis.
    Always consult with healthcare professionals.
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload X-Ray Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
            
            # Preprocess image
            input_tensor = preprocess_image(image)
            
            # Make prediction
            prediction, confidence, all_probabilities = predict_image(model, device, input_tensor)
            
            # Display results
            st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
            
            class_names = ['NORMAL', 'PNEUMONIA']
            prediction_text = class_names[prediction]
            
            # Prediction box
            if prediction == 0:  # NORMAL
                st.markdown(f"""
                <div class="prediction-box normal-prediction">
                    <h3>✅ Prediction: {prediction_text}</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:  # PNEUMONIA
                st.markdown(f"""
                <div class="prediction-box pneumonia-prediction">
                    <h3>⚠️ Prediction: {prediction_text}</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence bars
            st.markdown("### Confidence Breakdown")
            for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
                color_class = "normal-confidence" if i == 0 else "pneumonia-confidence"
                st.markdown(f"""
                <div>
                    <strong>{class_name}:</strong> {prob:.2%}
                    <div class="confidence-bar">
                        <div class="confidence-fill {color_class}" style="width: {prob*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">🔬 Model Explainability</h2>', unsafe_allow_html=True)
            
            # Generate Grad-CAM
            with st.spinner("Generating Grad-CAM visualization..."):
                heatmap, overlay = create_gradcam_visualization(model, image, prediction)
            
            if heatmap is not None and overlay is not None:
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(overlay)
                axes[2].set_title('Attention Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Display in Streamlit
                st.pyplot(fig)
                
                st.markdown("### 📊 What is Grad-CAM?")
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which regions of the X-ray image 
                the model focuses on when making its prediction. The red areas indicate regions of high attention.
                
                - **Red areas**: Regions the model considers important for the diagnosis
                - **Blue areas**: Regions with lower attention
                - **Overlay**: Shows how the attention map aligns with the original image
                """)
            else:
                st.error("Failed to generate Grad-CAM visualization.")
        else:
            st.markdown("### 📊 Model Explainability")
            st.markdown("""
            Once you upload an image, you'll see:
            
            - **Grad-CAM Heatmap**: Shows which regions the model focuses on
            - **Attention Overlay**: Combines the heatmap with the original image
            - **Explanation**: Helps understand the model's decision-making process
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Medical Diagnosis Assistant | Built with PyTorch & Streamlit</p>
        <p>For educational purposes only - Not for medical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 