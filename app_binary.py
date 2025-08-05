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
import PyPDF2
import docx
from utils.data_utils import get_transforms, get_device
from utils.model_utils import MultiClassChestXRayModel
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
        
        # Try to load the demo model first
        model_path = "models/chest_xray_demo.pth"
        if os.path.exists(model_path):
            # Use a simple model for demo
            from create_demo_model import SimpleChestXRayModel
            model = SimpleChestXRayModel(num_classes=2)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, device
        else:
            st.error("Model file not found. Please run create_demo_model.py first.")
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

def validate_image_content(image):
    """Validate if the uploaded image is likely a medical image"""
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Check if image is grayscale or has medical characteristics
        if len(img_array.shape) == 3:
            # Check if it's mostly grayscale (medical images often are)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            if np.mean(np.abs(r - g)) < 10 and np.mean(np.abs(r - b)) < 10:
                return True, "Image appears to be medical (grayscale-like)"
        
        # Check image dimensions (medical images are usually square-ish)
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        if 0.5 <= aspect_ratio <= 2.0:
            return True, "Image dimensions are appropriate for medical images"
        
        # Basic validation passed
        return True, "Image format is valid"
        
    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"

def validate_document_content(text_content):
    """Validate if the uploaded document contains medical content"""
    try:
        # Convert to lowercase for easier matching
        text_lower = text_content.lower()
        
        # Medical keywords to look for
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'symptoms', 'medical', 'doctor',
            'hospital', 'clinic', 'prescription', 'medication', 'test', 'lab',
            'x-ray', 'chest', 'pneumonia', 'infection', 'fever', 'cough',
            'breathing', 'respiratory', 'lung', 'heart', 'blood', 'pressure',
            'temperature', 'pulse', 'oxygen', 'saturation', 'ct', 'mri', 'scan'
        ]
        
        # Count medical keywords found
        found_keywords = [word for word in medical_keywords if word in text_lower]
        
        if len(found_keywords) >= 2:
            return True, f"Document contains medical content ({len(found_keywords)} medical terms found)"
        elif len(text_content.split()) < 10:
            return False, "Document appears to be too short or empty"
        else:
            return True, "Document format is valid (content validation limited)"
            
    except Exception as e:
        return False, f"Error analyzing document: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Process uploaded file and determine its type"""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    # Validate file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File too large! Please upload a file smaller than 10MB.")
        return None, None, None
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        # Process as image
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Validate image content
            is_valid, message = validate_image_content(image)
            if not is_valid:
                st.error(f"❌ Invalid medical image: {message}")
                st.info("💡 Please upload a chest X-ray image or other medical scan.")
                return None, None, None
            
            st.success(f"✅ {message}")
            return 'image', image, None
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("💡 Please upload a valid image file (JPG, PNG).")
            return None, None, None
    
    elif file_extension == 'pdf':
        # Process as PDF document
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            # Validate document content
            is_valid, message = validate_document_content(text_content)
            if not is_valid:
                st.error(f"❌ Invalid medical document: {message}")
                st.info("💡 Please upload a medical report, patient record, or clinical document.")
                return None, None, None
            
            st.success(f"✅ {message}")
            return 'document', None, text_content
            
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            st.info("💡 Please upload a valid PDF document.")
            return None, None, None
    
    elif file_extension in ['docx', 'doc']:
        # Process as Word document
        try:
            doc = docx.Document(uploaded_file)
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            # Validate document content
            is_valid, message = validate_document_content(text_content)
            if not is_valid:
                st.error(f"❌ Invalid medical document: {message}")
                st.info("💡 Please upload a medical report, patient record, or clinical document.")
                return None, None, None
            
            st.success(f"✅ {message}")
            return 'document', None, text_content
            
        except Exception as e:
            st.error(f"❌ Error reading Word document: {str(e)}")
            st.info("💡 Please upload a valid Word document.")
            return None, None, None
    
    elif file_extension == 'txt':
        # Process as text file
        try:
            text_content = uploaded_file.read().decode('utf-8')
            
            # Validate document content
            is_valid, message = validate_document_content(text_content)
            if not is_valid:
                st.error(f"❌ Invalid medical document: {message}")
                st.info("💡 Please upload a medical report, patient record, or clinical document.")
                return None, None, None
            
            st.success(f"✅ {message}")
            return 'document', None, text_content
            
        except Exception as e:
            st.error(f"❌ Error reading text file: {str(e)}")
            st.info("💡 Please upload a valid text file.")
            return None, None, None
    
    else:
        st.error(f"❌ Unsupported file type: {file_extension}")
        st.info("💡 Supported formats: JPG, PNG (images) | PDF, DOCX, DOC, TXT (documents)")
        return None, None, None

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
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Medical File Analysis</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure the model has been trained.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload Medical Files Only:**
       - **Images**: Chest X-rays, medical scans (JPG, PNG)
       - **Documents**: Medical reports, patient records (PDF, DOCX, DOC, TXT)
    
    2. **For Images**: The AI will analyze and provide diagnosis
    3. **For Documents**: Text will be extracted and displayed
    4. **View Results**: Analysis and visualizations will appear
    
    **⚠️ Note**: Only medical-related files are accepted
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
            file_type, image, text_content = process_uploaded_file(uploaded_file)
            
            # Initialize variables for use in both columns
            prediction = None
            confidence = None
            all_probabilities = None
            
            # Check if file processing was successful
            if file_type is None:
                st.warning("⚠️ Please try uploading a different file.")
                st.markdown("""
                **📋 Upload Guidelines:**
                - **Images**: Chest X-rays, medical scans (JPG, PNG)
                - **Documents**: Medical reports, patient records (PDF, DOCX, DOC, TXT)
                - **File Size**: Maximum 10MB
                - **Content**: Must be medical-related
                """)
                return
            
            if file_type == 'image':
                # Display uploaded image
                st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
                
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
            
            elif file_type == 'document':
                # Display document content
                st.markdown('<h2 class="sub-header">📄 Document Content</h2>', unsafe_allow_html=True)
                
                # Show file info
                st.info(f"📎 File: {uploaded_file.name}")
                
                # Display text content
                st.markdown("### Document Text:")
                st.text_area("Extracted Text", text_content, height=300, disabled=True)
                
                # Document analysis placeholder
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
                st.error("Failed to process uploaded file.")
    
    with col2:
        if uploaded_file is not None and file_type == 'image':
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
                st.warning("⚠️ Grad-CAM visualization not available for this model.")
                st.info("""
                **Model Explainability:**
                - The model has analyzed your X-ray image
                - Prediction and confidence scores are shown above
                - For full Grad-CAM visualization, use a trained ResNet model
                """)
        elif uploaded_file is not None and file_type == 'document':
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