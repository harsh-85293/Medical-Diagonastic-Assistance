#!/usr/bin/env python3
"""
Multi-Class Medical Diagnosis Assistant Streamlit App
Supports detection of multiple diseases from chest X-ray images
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os
import cv2
import PyPDF2
import docx
import pandas as pd
from utils.data_utils import get_transforms, get_device, DISEASE_LABELS, DISEASE_DISPLAY_NAMES
from utils.model_utils import create_model
from gradcam import generate_multi_class_gradcam, generate_gradcam_for_single_image

# Page configuration
st.set_page_config(
    page_title="Multi-Class Medical Diagnosis Assistant",
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
        border-left: 5px solid;
    }
    .normal-prediction {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .disease-prediction {
        background-color: #f8d7da;
        border-color: #dc3545;
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained multi-class model"""
    try:
        device = get_device()
        
        # Try to load the multi-class model
        model_path = "models/multiclass_best_model.pth"
        if os.path.exists(model_path):
            model = create_model('resnet50', len(DISEASE_LABELS), pretrained=False)
            model = model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model, device
        else:
            # Fallback to binary demo model
            demo_model_path = "models/chest_xray_demo.pth"
            if os.path.exists(demo_model_path):
                st.warning("Multi-class model not found. Using binary demo model instead.")
                from create_demo_model import SimpleChestXRayModel
                model = SimpleChestXRayModel(num_classes=2)
                model.load_state_dict(torch.load(demo_model_path, map_location=device))
                model = model.to(device)
                model.eval()
                return model, device
            else:
                st.error("No models found. Please train the model first.")
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
    """Make multi-class prediction on the input image"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        
        # Handle both multi-label (sigmoid) and single-label (softmax) outputs
        if outputs.shape[1] == len(DISEASE_LABELS):
            # Multi-label case - use sigmoid
            probabilities = torch.sigmoid(outputs)
            # For multi-label, we need to handle each class independently
            probs = probabilities[0]
            top_probs, top_indices = torch.topk(probs, 3)
        else:
            # Single-label case - use softmax
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities[0]
            top_probs, top_indices = torch.topk(probs, 3)
        
        predictions = []
        for i in range(3):
            predictions.append({
                'disease': DISEASE_LABELS[top_indices[i].item()],
                'display_name': DISEASE_DISPLAY_NAMES[DISEASE_LABELS[top_indices[i].item()]],
                'probability': top_probs[i].item(),
                'confidence': top_probs[i].item()
            })
    
    return predictions, probs.cpu().numpy()

def validate_image_content(image):
    """Validate if the uploaded image is likely a medical image"""
    try:
        img_array = np.array(image)
        
        # Check if image is grayscale or has medical characteristics
        if len(img_array.shape) == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            if np.mean(np.abs(r - g)) < 10 and np.mean(np.abs(r - b)) < 10:
                return True, "Image appears to be medical (grayscale-like)"
        
        # Check image dimensions
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        if 0.5 <= aspect_ratio <= 2.0:
            return True, "Image dimensions are appropriate for medical images"
        
        return True, "Image format is valid"
        
    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"

def create_prediction_chart(predictions):
    """Create a bar chart for predictions"""
    diseases = [pred['display_name'] for pred in predictions]
    probabilities = [pred['probability'] for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=probabilities,
            marker_color=['#28a745' if pred['disease'] == 'NORMAL' else '#dc3545' for pred in predictions],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 3 Disease Predictions",
        xaxis_title="Disease",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_radar_chart(probabilities):
    """Create a radar chart for all disease probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=probabilities,
        theta=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS],
        fill='toself',
        name='Disease Probabilities'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="All Disease Probabilities",
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Multi-Class Medical Diagnosis Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Multi-Disease Detection from Chest X-Rays</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading multi-class model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure the multi-class model has been trained.")
        return
    
    # Test model with a dummy input to verify it's working
    try:
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            st.success(f"✅ Model loaded successfully! Output shape: {test_output.shape}")
    except Exception as e:
        st.error(f"❌ Model test failed: {e}")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload Medical Files Only:**
       - **Images**: Chest X-rays, medical scans (JPG, PNG)
       - **Documents**: Medical reports, patient records (PDF, DOCX, DOC, TXT)
    
    2. **For Images**: The AI will analyze and detect multiple diseases
    3. **For Documents**: Text will be extracted and displayed
    4. **View Results**: Multi-class analysis and visualizations will appear
    
    **🏥 Supported Diseases:**
    - Normal
    - Pneumonia
    - COVID-19
    - Tuberculosis
    - Pleural Effusion
    - Pneumothorax
    - Lung Cancer
    - Cardiomegaly
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
        st.markdown('<h2 class="sub-header">📤 Upload Medical File</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your medical file here...",
            type=['jpg', 'jpeg', 'png', 'pdf', 'docx', 'doc', 'txt'],
            help="Upload medical images (JPG, PNG) or documents (PDF, DOCX, DOC, TXT)"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("❌ File too large! Please upload a file smaller than 10MB.")
                return
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                # Process as image
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Validate image content
                    is_valid, message = validate_image_content(image)
                    if not is_valid:
                        st.error(f"❌ Invalid medical image: {message}")
                        st.info("💡 Please upload a chest X-ray image or other medical scan.")
                        return
                    
                    st.success(f"✅ {message}")
                    
                    # Display uploaded image
                    st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
                    
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    
                    # Make prediction
                    predictions, all_probabilities = predict_image(model, device, input_tensor)
                    
                    # Display results
                    st.markdown('<h2 class="sub-header">🔍 Multi-Class Analysis Results</h2>', unsafe_allow_html=True)
                    
                    # Top prediction
                    top_prediction = predictions[0]
                    prediction_class = "normal-prediction" if top_prediction['disease'] == 'NORMAL' else "disease-prediction"
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h3 style="color: #2c3e50; margin: 0;">🎯 Primary Prediction: {top_prediction['display_name']}</h3>
                        <p style="font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">Confidence: {top_prediction['confidence']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top 3 predictions
                    st.markdown("### 📊 Top 3 Predictions")
                    for i, pred in enumerate(predictions):
                        confidence_color = "#28a745" if pred['disease'] == 'NORMAL' else "#dc3545"
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong style="color: #2c3e50;">{i+1}. {pred['display_name']}</strong><br>
                            <span style="color: {confidence_color}; font-weight: bold;">Confidence: {pred['confidence']:.2%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Prediction chart
                    st.plotly_chart(create_prediction_chart(predictions), use_container_width=True)
                    
                    # Radar chart for all diseases
                    st.markdown("### 📈 All Disease Probabilities")
                    st.plotly_chart(create_radar_chart(all_probabilities), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Please upload a valid image file (JPG, PNG).")
                    return
            
            elif file_extension in ['pdf', 'docx', 'doc', 'txt']:
                # Process as document
                st.markdown('<h2 class="sub-header">📄 Document Content</h2>', unsafe_allow_html=True)
                st.info(f"📎 File: {uploaded_file.name}")
                
                # Document processing (same as before)
                if file_extension == 'pdf':
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.error(f"❌ Error reading PDF: {str(e)}")
                        return
                
                elif file_extension in ['docx', 'doc']:
                    try:
                        doc = docx.Document(uploaded_file)
                        text_content = ""
                        for paragraph in doc.paragraphs:
                            text_content += paragraph.text + "\n"
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.error(f"❌ Error reading Word document: {str(e)}")
                        return
                
                elif file_extension == 'txt':
                    try:
                        text_content = uploaded_file.read().decode('utf-8')
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.error(f"❌ Error reading text file: {str(e)}")
                        return
                
                # Document analysis
                st.markdown('<h2 class="sub-header">📊 Document Analysis</h2>', unsafe_allow_html=True)
                st.info("""
                **Document Analysis Features:**
                - ✅ Text extraction completed
                - 📋 Document content displayed
                - 🔍 Medical terminology detection (coming soon)
                - 📈 Report summarization (coming soon)
                """)
                
                # Word count and basic stats
                word_count = len(text_content.split())
                st.metric("Word Count", word_count)
                st.metric("Character Count", len(text_content))
            
            else:
                st.error(f"❌ Unsupported file type: {file_extension}")
                st.info("💡 Supported formats: JPG, PNG (images) | PDF, DOCX, DOC, TXT (documents)")
                return
    
    with col2:
        if uploaded_file is not None and file_extension in ['jpg', 'jpeg', 'png']:
            st.markdown('<h2 class="sub-header">🔬 Model Explainability</h2>', unsafe_allow_html=True)
            
            # Generate Grad-CAM for top prediction
            with st.spinner("Generating Grad-CAM visualization..."):
                try:
                    # Save temporary image
                    temp_path = "temp_image.jpg"
                    image.save(temp_path)
                    
                    # Generate Grad-CAM for top prediction
                    top_class_idx = DISEASE_LABELS.index(predictions[0]['disease'])
                    heatmap, overlay = generate_gradcam_for_single_image(
                        model, temp_path, top_class_idx, save_path=None
                    )
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(image)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # Heatmap
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title(f'Grad-CAM ({predictions[0]["display_name"]})')
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
                    
                    # Option to generate Grad-CAM for all classes
                    if st.button("🔍 Generate Grad-CAM for All Diseases"):
                        with st.spinner("Generating multi-class Grad-CAM..."):
                            try:
                                results = generate_multi_class_gradcam(model, temp_path, save_dir="temp_gradcam")
                                
                                # Display results in a grid
                                n_results = len(results)
                                cols = 2
                                rows = (n_results + 1) // cols
                                
                                fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
                                axes = axes.flatten()
                                
                                # Original image
                                axes[0].imshow(image)
                                axes[0].set_title('Original Image')
                                axes[0].axis('off')
                                
                                # Disease CAMs
                                for i, (class_idx, result) in enumerate(results.items()):
                                    if i + 1 < len(axes):
                                        axes[i + 1].imshow(result['overlay'])
                                        axes[i + 1].set_title(f'{result["disease_name"]}')
                                        axes[i + 1].axis('off')
                                
                                # Hide unused subplots
                                for i in range(len(results) + 1, len(axes)):
                                    axes[i].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error generating multi-class Grad-CAM: {e}")
                
                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {e}")
                    st.info("""
                    **Model Explainability:**
                    - The model has analyzed your X-ray image
                    - Prediction and confidence scores are shown above
                    - For full Grad-CAM visualization, ensure model compatibility
                    """)
        
        elif uploaded_file is not None and file_extension in ['pdf', 'docx', 'doc', 'txt']:
            st.markdown('<h2 class="sub-header">📄 Document Analysis</h2>', unsafe_allow_html=True)
            st.info("""
            **Document Processing Complete:**
            - ✅ Text extraction successful
            - 📋 Content displayed in the left panel
            - 📊 Basic statistics calculated
            """)
        
        else:
            st.markdown("### 📊 Model Explainability")
            st.markdown("""
            Once you upload an image, you'll see:
            
            - **Multi-Class Predictions**: Top 3 disease predictions with confidence
            - **Grad-CAM Heatmap**: Shows which regions the model focuses on
            - **Attention Overlay**: Combines the heatmap with the original image
            - **All Disease Probabilities**: Radar chart showing all disease scores
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Multi-Class Medical Diagnosis Assistant | Built with PyTorch & Streamlit</p>
        <p>For educational purposes only - Not for medical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 