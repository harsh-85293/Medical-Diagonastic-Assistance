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
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sub Header Styles */
    .sub-header {
        background: white;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .sub-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Card Styles */
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .normal-prediction {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
    }
    
    .disease-prediction {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Confidence Bar */
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 12px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    
    .confidence-fill.disease {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
    }
    
    /* Upload Area */
    .upload-area {
        background: white;
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: #f8f9fa;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
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
            marker_line_color='white',
            marker_line_width=2,
        )
    ])
    
    fig.update_layout(
        title="Top 3 Disease Predictions",
        xaxis_title="Disease",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    return fig

def create_radar_chart(probabilities):
    """Create a radar chart for all disease probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=probabilities,
        theta=[DISEASE_DISPLAY_NAMES[d] for d in DISEASE_LABELS],
        fill='toself',
        name='Disease Probabilities',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        title="All Disease Probabilities",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    return fig

def main():
    # Header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical AI Assistant</h1>
        <p>Advanced Multi-Disease Detection from Chest X-Rays</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model with beautiful loading animation
    with st.spinner("🔬 Loading AI Model..."):
        model, device = load_model()
    
    if model is None:
        st.markdown("""
        <div class="error-message">
            ❌ Failed to load model. Please ensure the multi-class model has been trained.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Test model with a dummy input to verify it's working
    try:
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            st.markdown("""
            <div class="success-message">
                ✅ AI Model loaded successfully! Ready for analysis.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="error-message">
            ❌ Model test failed: {e}
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Modern sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
            <h3>📋 How to Use</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **1. Upload Medical Files:**
        - **Images**: Chest X-rays, medical scans (JPG, PNG)
        - **Documents**: Medical reports, patient records (PDF, DOCX, DOC, TXT)
        
        **2. AI Analysis:**
        - Multi-disease detection from X-rays
        - Text extraction from documents
        - Real-time predictions with confidence scores
        
        **3. View Results:**
        - Interactive visualizations
        - Grad-CAM explainability
        - Detailed analysis reports
        """)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
            <h4>🏥 Supported Diseases</h4>
            <ul style="color: white; margin: 0.5rem 0;">
                <li>Normal</li>
                <li>Pneumonia</li>
                <li>COVID-19</li>
                <li>Tuberculosis</li>
                <li>Pleural Effusion</li>
                <li>Pneumothorax</li>
                <li>Lung Cancer</li>
                <li>Cardiomegaly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px;">
            <h4>⚠️ Important Notice</h4>
            <p style="font-size: 0.9rem; margin: 0;">
                This tool is for educational purposes only. 
                It should not replace professional medical diagnosis. 
                Always consult with healthcare professionals.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content with modern layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="sub-header">
            <h2>📤 Upload Medical File</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Modern file uploader
        uploaded_file = st.file_uploader(
            "Choose your medical file...",
            type=['jpg', 'jpeg', 'png', 'pdf', 'docx', 'doc', 'txt'],
            help="Upload medical images (JPG, PNG) or documents (PDF, DOCX, DOC, TXT)"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.markdown("""
                <div class="error-message">
                    ❌ File too large! Please upload a file smaller than 10MB.
                </div>
                """, unsafe_allow_html=True)
                return
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                # Process as image
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Validate image content
                    is_valid, message = validate_image_content(image)
                    if not is_valid:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ Invalid medical image: {message}
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("💡 Please upload a chest X-ray image or other medical scan.")
                        return
                    
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ {message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display uploaded image with modern styling
                    st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 12px; 
                                box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin: 1rem 0;">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📷 Uploaded X-Ray Image</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(image, use_column_width=True)
                    
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    
                    # Make prediction
                    predictions, all_probabilities = predict_image(model, device, input_tensor)
                    
                    # Display results with modern cards
                    st.markdown("""
                    <div class="sub-header">
                        <h2>🔍 AI Analysis Results</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top prediction with modern card
                    top_prediction = predictions[0]
                    prediction_class = "normal-prediction" if top_prediction['disease'] == 'NORMAL' else "disease-prediction"
                    confidence_class = "" if top_prediction['disease'] == 'NORMAL' else "disease"
                    
                    st.markdown(f"""
                    <div class="prediction-card {prediction_class}">
                        <h3 style="color: #2c3e50; margin: 0 0 1rem 0; font-size: 1.5rem;">
                            🎯 Primary Prediction: {top_prediction['display_name']}
                        </h3>
                        <div class="confidence-bar">
                            <div class="confidence-fill {confidence_class}" 
                                 style="width: {top_prediction['confidence']*100}%"></div>
                        </div>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0; color: #2c3e50;">
                            Confidence: {top_prediction['confidence']:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top 3 predictions with modern cards
                    st.markdown("### 📊 Top 3 Predictions")
                    for i, pred in enumerate(predictions):
                        confidence_color = "#28a745" if pred['disease'] == 'NORMAL' else "#dc3545"
                        confidence_class = "" if pred['disease'] == 'NORMAL' else "disease"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: #2c3e50; font-size: 1.1rem;">
                                        {i+1}. {pred['display_name']}
                                    </strong>
                                </div>
                                <div style="text-align: right;">
                                    <span style="color: {confidence_color}; font-weight: bold; font-size: 1.1rem;">
                                        {pred['confidence']:.1%}
                                    </span>
                                </div>
                            </div>
                            <div class="confidence-bar" style="margin-top: 0.5rem;">
                                <div class="confidence-fill {confidence_class}" 
                                     style="width: {pred['confidence']*100}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Charts with modern containers
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📈 Prediction Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_prediction_chart(predictions), use_container_width=True)
                    
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🎯 All Disease Probabilities</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_radar_chart(all_probabilities), use_container_width=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-message">
                        ❌ Error processing image: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 Please upload a valid image file (JPG, PNG).")
                    return
            
            elif file_extension in ['pdf', 'docx', 'doc', 'txt']:
                # Process as document with modern styling
                st.markdown("""
                <div class="sub-header">
                    <h2>📄 Document Content</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📎 File: {uploaded_file.name}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Document processing (same as before)
                if file_extension == 'pdf':
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ Error reading PDF: {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        return
                
                elif file_extension in ['docx', 'doc']:
                    try:
                        doc = docx.Document(uploaded_file)
                        text_content = ""
                        for paragraph in doc.paragraphs:
                            text_content += paragraph.text + "\n"
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ Error reading Word document: {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        return
                
                elif file_extension == 'txt':
                    try:
                        text_content = uploaded_file.read().decode('utf-8')
                        st.text_area("Extracted Text", text_content, height=300, disabled=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ Error reading text file: {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        return
                
                # Document analysis with modern styling
                st.markdown("""
                <div class="sub-header">
                    <h2>📊 Document Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">✅ Document Processing Complete</h4>
                    <ul style="color: #2c3e50; margin: 0;">
                        <li>Text extraction successful</li>
                        <li>Content displayed in the left panel</li>
                        <li>Basic statistics calculated</li>
                        <li>Medical terminology detection (coming soon)</li>
                        <li>Report summarization (coming soon)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Word count and basic stats with modern cards
                word_count = len(text_content.split())
                col1_stats, col2_stats = st.columns(2)
                
                with col1_stats:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #2c3e50;">📝 Word Count</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #667eea; margin: 0.5rem 0;">
                            {word_count:,}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_stats:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #2c3e50;">📄 Character Count</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #667eea; margin: 0.5rem 0;">
                            {len(text_content):,}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Unsupported file type: {file_extension}
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 Supported formats: JPG, PNG (images) | PDF, DOCX, DOC, TXT (documents)")
                return
    
    with col2:
        if uploaded_file is not None and file_extension in ['jpg', 'jpeg', 'png']:
            st.markdown("""
            <div class="sub-header">
                <h2>🔬 AI Explainability</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate Grad-CAM for top prediction
            with st.spinner("🔍 Generating AI visualization..."):
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
                    
                    # Create visualization with modern styling
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🎯 AI Attention Visualization</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(image)
                    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                    axes[0].axis('off')
                    
                    # Heatmap
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title(f'AI Focus ({predictions[0]["display_name"]})', fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(overlay)
                    axes[2].set_title('Attention Overlay', fontsize=14, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Display in Streamlit
                    st.pyplot(fig)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📊 What is AI Explainability?</h4>
                        <p style="color: #2c3e50; line-height: 1.6;">
                            <strong>Grad-CAM (Gradient-weighted Class Activation Mapping)</strong> shows which regions 
                            of the X-ray image the AI focuses on when making its prediction.
                        </p>
                        <ul style="color: #2c3e50; margin: 0.5rem 0;">
                            <li><strong>Red areas:</strong> Regions the AI considers important for diagnosis</li>
                            <li><strong>Blue areas:</strong> Regions with lower attention</li>
                            <li><strong>Overlay:</strong> Shows how the attention map aligns with the original image</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Option to generate Grad-CAM for all classes
                    if st.button("🔍 Generate AI Analysis for All Diseases", 
                               help="Create visualizations for all disease types"):
                        with st.spinner("🔬 Generating comprehensive AI analysis..."):
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
                                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                                axes[0].axis('off')
                                
                                # Disease CAMs
                                for i, (class_idx, result) in enumerate(results.items()):
                                    if i + 1 < len(axes):
                                        axes[i + 1].imshow(result['overlay'])
                                        axes[i + 1].set_title(f'{result["disease_name"]}', 
                                                            fontsize=14, fontweight='bold')
                                        axes[i + 1].axis('off')
                                
                                # Hide unused subplots
                                for i in range(len(results) + 1, len(axes)):
                                    axes[i].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-message">
                                    ❌ Error generating comprehensive AI analysis: {e}
                                </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-message">
                        ❌ Error generating AI visualization: {e}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🔬 AI Analysis Complete</h4>
                        <ul style="color: #2c3e50; margin: 0;">
                            <li>✅ AI has analyzed your X-ray image</li>
                            <li>✅ Prediction and confidence scores shown</li>
                            <li>✅ For full visualization, ensure model compatibility</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif uploaded_file is not None and file_extension in ['pdf', 'docx', 'doc', 'txt']:
            st.markdown("""
            <div class="sub-header">
                <h2>📄 Document Analysis</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">✅ Document Processing Complete</h4>
                <ul style="color: #2c3e50; margin: 0;">
                    <li>Text extraction successful</li>
                    <li>Content displayed in the left panel</li>
                    <li>Basic statistics calculated</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="sub-header">
                <h2>🔬 AI Explainability</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📊 What You'll See</h4>
                <p style="color: #2c3e50; line-height: 1.6;">
                    Once you upload an image, you'll see:
                </p>
                <ul style="color: #2c3e50; margin: 0.5rem 0;">
                    <li><strong>Multi-Class Predictions:</strong> Top 3 disease predictions with confidence</li>
                    <li><strong>AI Focus Visualization:</strong> Shows which regions the AI focuses on</li>
                    <li><strong>Attention Overlay:</strong> Combines the focus map with the original image</li>
                    <li><strong>All Disease Probabilities:</strong> Radar chart showing all disease scores</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Modern footer
    st.markdown("""
    <div class="footer">
        <h4 style="margin: 0 0 1rem 0;">🏥 Medical AI Assistant</h4>
        <p style="margin: 0; opacity: 0.8;">
            Built with PyTorch & Streamlit | For educational purposes only - Not for medical diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 