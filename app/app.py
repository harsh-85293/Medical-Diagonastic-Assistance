"""
Medical AI Assistant - Streamlit App
Chest X-ray Classification with robust input validation and explainability.
Version: 1.0.2 - Python 3.10 + OpenCV headless deployment fix
"""

import os
import sys
import logging
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
# Robust OpenCV import with fallbacks for deployment
cv2 = None
try:
    import cv2
    print("OpenCV imported successfully")
except ImportError as e:
    print(f"OpenCV import failed: {e}")
    st.error("⚠️ OpenCV not found. Attempting to install...")
    
    # Try multiple OpenCV versions for better compatibility
    opencv_versions = [
        "opencv-python-headless>=4.8.0,<5.0.0",
        "opencv-python-headless==4.9.0.80", 
        "opencv-python-headless==4.8.0.76",
        "opencv-python-headless"  # Latest stable
    ]
    
    installation_success = False
    for version in opencv_versions:
        try:
            import subprocess
            import sys
            st.info(f"🔄 Trying to install {version}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", version, 
                "--no-cache-dir", "--force-reinstall"
            ], timeout=300)
            import cv2
            st.success(f"✅ OpenCV installed successfully with {version}!")
            installation_success = True
            st.rerun()
            break
        except Exception as install_error:
            st.warning(f"⚠️ Failed to install {version}: {str(install_error)[:100]}...")
            continue
    
    if not installation_success:
        st.error("❌ Failed to install OpenCV with any version. Please check your deployment environment.")
        st.error("**Deployment Requirements:**")
        st.code("""
# Add to your deployment requirements:
opencv-python-headless>=4.8.0,<5.0.0

# Or try alternative packages:
opencv-contrib-python-headless>=4.8.0
        """)
        st.error("**Alternative Solutions:**")
        st.markdown("""
        1. **Streamlit Cloud/Heroku**: Add `opencv-python-headless` to requirements.txt
        2. **Docker**: Install system dependencies: `apt-get install libgl1-mesa-glx`
        3. **Manual Install**: `pip install opencv-python-headless --no-cache-dir`
        """)
        st.stop()
from PIL import Image
import yaml
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import load_pretrained_model
from data import CXRDataset
from validate_input import CXRInputValidator
from gradcam import create_gradcam_visualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant - CXR Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_theme(dark: bool):
    """Inject theme-based CSS variables and component styles."""
    if dark:
        # Dark theme variables
        theme_vars = """
        --bg: #0f172a; --panel: #111827; --panel-2: #0b1220; 
        --text: #e5e7eb; --muted: #9ca3af; --border: #313a46;
        --brand: #6d28d9; --brand-2: #0ea5e9; --radius: 16px; 
        --shadow: 0 8px 24px rgba(16,24,40,.08);
        --warn-bg: #451a03; --warn-border: #92400e; --warn-text: #fbbf24;
        --good: #10b981; --bad: #ef4444;
        """
        uploader_bg = "#1e293b"
        uploader_border = "rgba(109,40,217,.6)"
        uploader_text = "#e5e7eb"
    else:
        # Light theme variables (default)
        theme_vars = """
        --bg: #ffffff; --panel: #ffffff; --panel-2: #f6f7fb; 
        --text: #0f172a; --muted: #5b6472; --border: #e5e7eb;
        --brand: #6d28d9; --brand-2: #0ea5e9; --radius: 16px; 
        --shadow: 0 8px 24px rgba(16,24,40,.08);
        --warn-bg: #fff7e6; --warn-border: #ffcf66; --warn-text: #5f4700;
        --good: #10b981; --bad: #ef4444;
        """
        uploader_bg = "#f5f3ff"
        uploader_border = "rgba(109,40,217,.35)"
        uploader_text = "#3b3b3b"

    st.markdown(f"""<style>
    :root {{ {theme_vars} }}
    
    /* Global Layout */
    .stApp {{ background: var(--bg) !important; color: var(--text) !important; }}
    .main .block-container {{ 
        max-width: 1100px !important; margin: 0 auto !important; 
        padding: 24px !important; background: var(--bg) !important; 
    }}
    
    /* Hero Section */
    .hero {{
        background: linear-gradient(180deg, rgba(109,40,217,.06), rgba(14,165,233,.04));
        padding: 32px 20px; border-radius: 20px; margin-bottom: 28px; text-align: center;
    }}
    .hero h1 {{
        font-size: 2.2rem; font-weight: 800; margin: 0 0 6px; color: var(--text);
    }}
    .hero p {{ color: var(--muted); margin: 0; font-size: 1.1rem; }}
    
    /* Card System */
    .card {{
        background: var(--panel); border: 1px solid var(--border); 
        border-radius: var(--radius); box-shadow: var(--shadow); 
        padding: 20px; margin: 20px 0;
    }}
    .card-header {{
        font-size: 1.25rem; font-weight: 700; color: var(--text); 
        margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
    }}
    
    /* Alert/Disclaimer */
    .alert {{
        background: var(--warn-bg); border: 1px solid var(--warn-border); 
        color: var(--warn-text); border-radius: 14px; padding: 14px 16px;
        display: flex; align-items: flex-start; gap: 12px; line-height: 1.55;
    }}
    .alert-icon {{ font-size: 18px; margin-top: 2px; flex-shrink: 0; }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background: {uploader_bg}; border: 2px dashed {uploader_border}; 
        border-radius: 14px; padding: 18px; transition: .25s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(109,40,217,.6);
    }}
    [data-testid="stFileUploader"] * {{ color: {uploader_text} !important; }}
    [data-testid="stFileUploader"] section {{ background: transparent !important; }}
    [data-testid="stFileUploaderDropzone"] {{ 
        background: transparent !important; border: none !important; padding: 0 !important; 
    }}
    
    /* Badges */
    .badge {{
        display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; 
        padding: 6px 12px; font-weight: 600; font-size: 0.875rem; margin: 4px 0;
    }}
    .badge.ok {{ background: rgba(16,185,129,.1); color: var(--good); border: 1px solid rgba(16,185,129,.3); }}
    .badge.err {{ background: rgba(239,68,68,.1); color: var(--bad); border: 1px solid rgba(239,68,68,.3); }}
    .badge.warn {{ background: var(--warn-bg); color: var(--warn-text); border: 1px solid var(--warn-border); }}
    
    /* Predictions */
    .prediction-card {{
        background: var(--panel-2); border: 1px solid var(--border); 
        border-radius: 12px; padding: 16px; margin: 12px 0; 
        border-left: 4px solid var(--brand); transition: all 0.2s ease;
    }}
    .prediction-card:hover {{ transform: translateX(4px); }}
    .prediction-title {{ 
        font-weight: 700; color: var(--text); margin-bottom: 8px; font-size: 1.1rem; 
    }}
    .confidence-text {{ color: var(--muted); font-size: 0.875rem; }}
    
    /* Metrics */
    .metric-container {{
        background: var(--panel-2); border: 1px solid var(--border); 
        border-radius: 12px; padding: 16px; margin: 8px 0; text-align: center;
    }}
    .metric-value {{ 
        font-size: 1.5rem; font-weight: 700; color: var(--text); margin-bottom: 4px; 
    }}
    .metric-label {{ font-size: 0.875rem; color: var(--muted); }}
    
    /* Image Preview */
    .image-preview {{
        background: var(--panel); border: 1px solid var(--border); 
        border-radius: var(--radius); padding: 20px; margin: 16px 0; text-align: center;
    }}
    .image-preview h4 {{ 
        font-size: 1.125rem; font-weight: 600; color: var(--text); margin-bottom: 16px; 
    }}
    
    /* Grad-CAM Section */
    .gradcam-section {{
        background: var(--panel); border: 1px solid var(--border); 
        border-radius: var(--radius); padding: 24px; margin: 24px 0;
    }}
    .gradcam-title {{
        font-size: 1.5rem; font-weight: 700; color: var(--text); 
        text-align: center; margin-bottom: 20px; padding-bottom: 8px; 
        border-bottom: 2px solid var(--brand);
    }}
    
    /* Sidebar */
    .sidebar-card {{
        background: var(--panel-2); border: 1px solid var(--border); 
        border-radius: 12px; padding: 16px; margin-bottom: 16px;
    }}
    .sidebar-header {{ 
        font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 12px; 
    }}
    .sidebar-text {{ 
        font-size: 14px; line-height: 1.55; color: var(--muted); 
        white-space: normal; word-wrap: break-word;
    }}
    .sidebar-divider {{ 
        height: 1px; background: var(--border); margin: 16px 0; opacity: 0.5; 
    }}
    
    /* Progress Bars */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--brand) 0%, var(--brand-2) 100%);
    }}
    
    /* Skeleton Loader */
    .skeleton {{
        background: linear-gradient(90deg, var(--panel-2) 25%, var(--border) 50%, var(--panel-2) 75%);
        background-size: 200% 100%; animation: skeleton-loading 1.5s infinite;
        border-radius: 8px; height: 200px; margin: 16px 0;
    }}
    @keyframes skeleton-loading {{
        0% {{ background-position: 200% 0; }} 
        100% {{ background-position: -200% 0; }}
    }}
    
    /* Focus States */
    .card:focus-visible, .prediction-card:focus-visible {{
        outline: 2px solid var(--brand); outline-offset: 2px;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main .block-container {{ padding: 16px !important; }}
        .card {{ padding: 14px; margin: 16px 0; }}
        .hero {{ padding: 24px 16px; }}
        .hero h1 {{ font-size: 1.8rem; }}
        .hero p {{ font-size: 1rem; }}
        [data-testid="stFileUploader"] {{ padding: 14px; }}
    }}
    
    /* Text Utilities */
    .text-muted {{ color: var(--muted); }}
    .text-secondary {{ color: var(--muted); }}
    
    /* Grad-CAM Section */
    .gradcam-card{{ background:var(--panel); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); padding:18px; margin:24px 0; }}
    .gradcam-header{{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.2rem; color:var(--text); margin:4px 4px 12px; }}
    .gradcam-underline{{ height:2px; background:linear-gradient(90deg, var(--brand), var(--brand-2)); border-radius:999px; margin:0 4px 14px; }}
    .gradcam-img{{ width:100%; height:auto; border-radius:12px; border:1px solid var(--border); }}
    /* Kill ghost rounded bar that appears before the header */
    .gradcam-card > :empty {{ display:none !important; }}
    /* Ensure no default Streamlit input/dropzone styling leaks here */
    .gradcam-card [data-testid="stFileUploader"],
    .gradcam-card [data-baseweb]{{ border:none !important; background:transparent !important; }}
    @media (max-width:768px){{ .gradcam-header{{ font-size:1.05rem }} }}
    </style>""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration file."""
    try:
        config_path = Path(__file__).parent.parent / "src" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None


@st.cache_resource
def load_model(config, device):
    """Load the trained model."""
    try:
        # Try to load from models directory
        model_path = Path(__file__).parent.parent / "models" / "best.ckpt"
        if model_path.exists():
            model = load_pretrained_model(str(model_path), config, device)
            return model
        else:
            return None
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


@st.cache_resource
def load_validator(config, device):
    """Load the input validator."""
    try:
        validator = CXRInputValidator(config, device)
        return validator
    except Exception as e:
        print(f"Failed to load validator: {e}")
        return None


def load_and_preprocess_image(uploaded_file):
    """Load and preprocess uploaded image."""
    try:
        # Read image
        if uploaded_file.type == "application/pdf":
            # Handle PDF files
            import pdf2image
            images = pdf2image.convert_from_path(uploaded_file.name, first_page=1, last_page=1)
            if images:
                image = np.array(images[0])
            else:
                return None, "No pages found in PDF"
        else:
            # Handle image files
            image = Image.open(uploaded_file)
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=2)
        elif image.shape[2] > 3:
            image = image[:, :, :3]
        
        return image, None
        
    except Exception as e:
        return None, f"Error loading image: {e}"


def preprocess_for_model(image, config):
    """Preprocess image for model input."""
    try:
        # Resize to input size
        input_size = config['dataset']['input_size']
        image_resized = cv2.resize(image, (input_size, input_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Ensure tensor is float32 and on the same device as the model
        image_tensor = image_tensor.float()
        
        return image_tensor
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


def get_prediction_with_uncertainty(model, image_tensor, config):
    """Get model prediction with uncertainty analysis."""
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
        
        # Calculate uncertainty metrics
        top1_prob = top_probs[0, 0].item()
        top2_prob = top_probs[0, 1].item()
        confidence_gap = top1_prob - top2_prob
        
        # Determine if prediction is uncertain
        is_uncertain = (top1_prob >= 0.85 and top2_prob >= 0.75) or (confidence_gap < 0.15)
        
        return {
            'top_indices': top_indices[0].cpu().numpy(),
            'top_probabilities': top_probs[0].cpu().numpy(),
            'all_probabilities': probabilities[0].cpu().numpy(),
            'is_uncertain': is_uncertain,
            'confidence_gap': confidence_gap
        }


def display_results(prediction_results, config, image, image_tensor, model):
    """Display prediction results and visualizations."""
    classes = config['dataset']['classes']
    
    # Display predictions in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🎯 AI Predictions</div>', unsafe_allow_html=True)
    
    # Check for uncertainty
    if prediction_results['is_uncertain']:
        st.markdown('<div class="warning-card">⚠️ **Uncertain Results** — Consider uploading a clearer image or alternate view.</div>', unsafe_allow_html=True)
    
    # Create columns for results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top predictions
        st.markdown('<div class="card-header">Top 3 Predictions</div>', unsafe_allow_html=True)
        for i in range(3):
            idx = prediction_results['top_indices'][i]
            prob = prediction_results['top_probabilities'][i]
            class_name = classes[idx]
            
            # Create prediction card
            st.markdown(f'''
            <div class="prediction-card">
                <div class="prediction-title">{i+1}. {class_name}</div>
                <div class="confidence-text">Confidence: {prob:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress bar
            st.progress(float(prob))
    
    with col2:
        # Confidence metrics
        st.markdown('<div class="card-header">Confidence Metrics</div>', unsafe_allow_html=True)
        
        # Top prediction metric
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-value">{prediction_results['top_probabilities'][0]:.1%}</div>
            <div class="metric-label">Top Prediction</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Confidence gap metric
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-value">{prediction_results['confidence_gap']:.3f}</div>
            <div class="metric-label">Confidence Gap</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Status metric
        if prediction_results['is_uncertain']:
            st.markdown('''
            <div class="warning-card">
                <div class="metric-value">⚠️ Uncertain</div>
                <div class="metric-label">Low confidence</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="success-card">
                <div class="metric-value">✅ Confident</div>
                <div class="metric-label">High confidence</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Grad-CAM section (clean, no ghost box) ---
    with st.container():
        st.markdown('<div class="gradcam-card">', unsafe_allow_html=True)
        st.markdown('<div class="gradcam-header">🔎 AI Attention (Grad-CAM)</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradcam-underline"></div>', unsafe_allow_html=True)

        try:
            gradcam_results = create_gradcam_visualization(
                model, image_tensor, target_class=prediction_results['top_indices'][0],
                class_names=classes
            )

            tab1, tab2, tab3 = st.tabs(["Heatmap", "Overlay", "All classes"])

            with tab1:
                hm = gradcam_results['heatmap']
                if hm is not None:
                    # Debug info
                    st.write(f"Debug: Heatmap shape: {hm.shape}, dtype: {hm.dtype}")
                    st.write(f"Debug: Heatmap range: [{hm.min():.3f}, {hm.max():.3f}]")
                    
                    # Check if heatmap has variation
                    hm_var = np.var(hm) if hasattr(hm, 'var') else np.var(hm)
                    st.write(f"Debug: Heatmap variance: {hm_var:.6f}")
                    
                    if hm_var < 1e-6:  # Very low variance
                        st.info("Grad-CAM returned a low-signal map for this sample. Try another image or check the target layer.")
                    else:
                        # Ensure proper format for Streamlit
                        if hm.dtype != np.uint8:
                            hm_display = np.clip(hm * 255, 0, 255).astype(np.uint8) if hm.max() <= 1.0 else np.clip(hm, 0, 255).astype(np.uint8)
                        else:
                            hm_display = hm
                        
                        st.image(hm_display, use_column_width=True, caption="Grad-CAM Heatmap", output_format="PNG")
                else:
                    st.error("Failed to generate heatmap")
            with tab2:
                overlay = gradcam_results['overlay']
                if overlay is not None:
                    # Ensure proper format for Streamlit
                    if overlay.dtype != np.uint8:
                        overlay_display = np.clip(overlay, 0, 255).astype(np.uint8)
                    else:
                        overlay_display = overlay
                    
                    st.image(overlay_display, use_column_width=True, caption="Overlay Visualization", output_format="PNG")
                else:
                    st.warning("Overlay generation failed")
            with tab3:
                # grid render; falls back if matplotlib is missing
                cols = st.columns(4)
                for i, (cam, cname) in enumerate(zip(gradcam_results['all_cams'], classes)):
                    with cols[i % 4]:
                        if cam is not None:
                            # Ensure proper format for Streamlit
                            if cam.dtype != np.uint8:
                                cam_display = np.clip(cam, 0, 255).astype(np.uint8)
                            else:
                                cam_display = cam
                            
                            st.image(cam_display, use_column_width=True, caption=cname, output_format="PNG")
                        else:
                            st.warning(f"CAM for {cname} failed")

        except Exception as e:
            st.warning(f"Grad-CAM is unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)


def render_hero():
    """Render the hero section."""
    st.markdown("""
    <div class="hero">
        <h1>🫁 Medical AI Assistant</h1>
        <p>Advanced Chest X-ray Classification & Analysis with AI Explainability</p>
    </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    """Render the disclaimer section."""
    st.markdown("""
    <div class="card">
        <div class="alert">
            <div class="alert-icon">⚠️</div>
            <div>
                <strong>IMPORTANT DISCLAIMER:</strong> This is an educational demonstration tool and is NOT intended for clinical use. 
                The predictions provided are for research and educational purposes only. Always consult qualified healthcare professionals 
                for medical diagnosis and treatment decisions.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_uploader():
    """Render the single file uploader."""
    with st.container():
        st.markdown('<div class="card"><div class="card-header">📁 Upload Chest X-ray Image</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            label="Drag & drop your chest X-ray files here or click to browse",
            type=["png", "jpg", "jpeg", "dcm", "pdf"],
            accept_multiple_files=True,
            key="main_uploader",
            help="Supported: PNG, JPG, JPEG, DICOM (.dcm), PDF · Max 25 MB each",
        )

        st.markdown('</div>', unsafe_allow_html=True)
        return uploaded_files

def render_footer():
    """Render the footer section."""
    st.markdown("""
    <div class="card" style="text-align: center; margin-top: 2rem;">
        <div style="color: var(--text);">
            <h4 style="margin-bottom: 12px;">Medical AI Assistant</h4>
            <p style="color: var(--muted); margin-bottom: 8px;">Educational Research Tool</p>
            <p style="color: var(--muted); font-size: 0.875rem;">
                Built with PyTorch, Streamlit, and advanced AI techniques
            </p>
            <div class="badge warn" style="margin-top: 16px;">
                ⚠️ Not for clinical use - Always consult healthcare professionals
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    # Load configuration first
    config = load_config()
    if config is None:
        st.error("❌ Failed to load configuration. Please check the config file.")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sidebar with dark mode toggle
    with st.sidebar:
        # Dark mode toggle
        dark_mode = st.toggle("Dark mode", value=False)
        
        # Inject theme based on toggle
        inject_theme(dark_mode)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-header">📋 Instructions</div>
            <div class="sidebar-text">
                1. <strong>Upload</strong> a chest X-ray image<br>
                2. <strong>Wait</strong> for AI validation<br>
                3. <strong>Review</strong> predictions & attention maps<br>
                4. <strong>Understand</strong> AI reasoning via Grad-CAM
            </div>
            <div class="sidebar-divider"></div>
            <div class="sidebar-text">
                <div style="font-weight: 600; margin-bottom: 6px;">📂 Supported Formats:</div>
                <div style="margin-left: 4px; margin-bottom: 8px;">PNG, JPG, JPEG, DICOM (.dcm), PDF</div>
                <div style="font-weight: 600; margin-bottom: 6px;">File Size Limit:</div>
                <div style="margin-left: 4px;">25 MB maximum per file</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-header">🔧 Settings</div>
        </div>
        """, unsafe_allow_html=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-header">📊 Model Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Load model
        if 'model' not in st.session_state:
            with st.spinner("Loading model..."):
                st.session_state.model = load_model(config, device)
        
        # Load validator
        if 'validator' not in st.session_state:
            with st.spinner("Loading validator..."):
                st.session_state.validator = load_validator(config, device)
        
        if 'model' in st.session_state and st.session_state.model is not None:
            st.markdown('<div class="badge ok">✅ Model Loaded</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sidebar-text">
                <strong>Architecture:</strong> ResNet50<br>
                <strong>Classes:</strong> 8<br>
                <strong>Input Size:</strong> 512×512<br>
                <strong>Device:</strong> {device}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge err">❌ Model Not Loaded</div>', unsafe_allow_html=True)
    
    # Main content sections in order: Hero → Disclaimer → Upload → Results → Footer
    render_hero()
    render_disclaimer()
    
    # Check if model is available
    if st.session_state.get('model') is None:
        st.markdown("""
        <div class="card">
            <div class="alert">
                <div class="alert-icon">❌</div>
                <div>
                    <strong>Model Not Available</strong><br>
                    Please ensure you have trained the model and the checkpoint exists in the models/ directory.
                    For testing, you can use the toy dataset provided in the project.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Upload section
    uploaded_files = render_uploader()
    
    # Optional status chip
    if uploaded_files:
        st.markdown(
            f'''
            <div style="max-width:700px;margin:1rem auto;padding:0 1rem;">
              <div class="badge ok" style="font-size:1rem;padding:10px 14px;">
                ✅ {len(uploaded_files)} file(s) selected
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    # Process uploaded files (results and analysis)
    if uploaded_files:
        render_results_and_gradcam(uploaded_files, config, st.session_state.model, st.session_state.validator)
    else:
        # Show empty state with instructions
        st.markdown("""
        <div class="card">
            <div style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🫁</div>
                <h3 style="color: var(--text); margin-bottom: 16px;">Ready to Analyze Chest X-rays</h3>
                <p style="color: var(--muted); margin-bottom: 24px;">
                    Upload your medical images using the file selector above to get started with AI-powered analysis.
                </p>
                <div class="badge warn">⬆️ Use the upload button above to begin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    render_footer()

def render_results_and_gradcam(uploaded_files, config, model, validator):
    """Render results and Grad-CAM for uploaded files."""
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"""
        <div class="card">
            <div class="card-header">📊 Analysis Results - {uploaded_file.name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create expander for each file
        with st.expander(f"View details for {uploaded_file.name}", expanded=True):
            # Load and preprocess image
            with st.spinner(f"Processing {uploaded_file.name}..."):
                image, error = load_and_preprocess_image(uploaded_file)
                
                if error:
                    st.markdown(f'<div class="badge err">❌ {error}</div>', unsafe_allow_html=True)
                    continue
                
                if image is None:
                    st.markdown('<div class="badge err">❌ Failed to load image</div>', unsafe_allow_html=True)
                    continue
            
            # Display original image in a card
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.markdown('<h4>Original Image</h4>', unsafe_allow_html=True)
            st.image(image, use_column_width=True, caption=f"Original: {uploaded_file.name}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input validation
            with st.container():
                st.markdown("""
                <div class="card">
                    <div class="card-header">🔍 Input Validation</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Validating input..."):
                    if st.session_state.validator:
                        try:
                            is_valid, clip_conf, reason = st.session_state.validator.validate_image(
                                image, uploaded_file.name
                            )
                            
                            if is_valid:
                                st.markdown(f'<div class="badge ok">✅ Valid Input - {reason}</div>', unsafe_allow_html=True)
                                st.metric("CLIP Confidence", f"{clip_conf:.3f}")
                                
                                # Preprocess for model
                                image_tensor = preprocess_for_model(image, config)
                                if image_tensor is None:
                                    continue
                                
                                # Get predictions
                                st.markdown('<div class="badge ok">🤖 Running AI Analysis...</div>', unsafe_allow_html=True)
                                
                                # Show skeleton loader
                                skeleton_placeholder = st.empty()
                                skeleton_placeholder.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
                                
                                with st.spinner("Generating predictions..."):
                                    prediction_results = get_prediction_with_uncertainty(
                                        st.session_state.model, image_tensor, config
                                    )
                                
                                # Clear skeleton and display results
                                skeleton_placeholder.empty()
                                display_results(prediction_results, config, image, image_tensor, st.session_state.model)
                                
                            else:
                                st.markdown(f'<div class="badge err">❌ Invalid Input - {reason}</div>', unsafe_allow_html=True)
                                st.metric("CLIP Confidence", f"{clip_conf:.3f}")
                                
                                st.markdown("""
                                <div class="card">
                                    <div class="alert">
                                        <div class="alert-icon">⚠️</div>
                                        <div>
                                            <strong>Invalid Input Detected</strong><br>
                                            Please upload a valid chest X-ray image. The uploaded file appears to be a non-medical image, portrait, document, or doesn't meet CXR requirements.
                                            <br><br>
                                            <strong>Tips for valid CXR images:</strong><br>
                                            • Ensure the image shows a chest X-ray<br>
                                            • Avoid images with faces or people<br>
                                            • Use clear, high-resolution images<br>
                                            • Ensure proper contrast and brightness
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="badge err">❌ Validation error: {e}</div>', unsafe_allow_html=True)
                            # Continue without validation
                            image_tensor = preprocess_for_model(image, config)
                            if image_tensor is None:
                                continue
                            
                            # Get predictions
                            with st.spinner("Generating predictions..."):
                                prediction_results = get_prediction_with_uncertainty(
                                    st.session_state.model, image_tensor, config
                                )
                            
                            # Display results
                            display_results(prediction_results, config, image, image_tensor, st.session_state.model)
                    else:
                        st.markdown('<div class="badge warn">⚠️ Input validation not available</div>', unsafe_allow_html=True)
                        # Continue without validation
                        image_tensor = preprocess_for_model(image, config)
                        if image_tensor is None:
                            continue
                        
                        # Get predictions
                        with st.spinner("Generating predictions..."):
                            prediction_results = get_prediction_with_uncertainty(
                                st.session_state.model, image_tensor, config
                            )
                        
                        # Display results
                        display_results(prediction_results, config, image, image_tensor, st.session_state.model)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show empty state with instructions
        st.markdown("""
        <div class="card">
            <div style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🫁</div>
                <h3 style="color: var(--text); margin-bottom: 16px;">Ready to Analyze Chest X-rays</h3>
                <p style="color: var(--muted); margin-bottom: 24px;">
                    Upload your medical images using the file selector above to get started with AI-powered analysis.
                </p>
                <div class="badge warn">⬆️ Use the upload button above to begin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="card" style="text-align: center; margin-top: 2rem;">
        <div style="color: var(--text);">
            <h4 style="margin-bottom: 12px;">Medical AI Assistant</h4>
            <p style="color: var(--muted); margin-bottom: 8px;">Educational Research Tool</p>
            <p style="color: var(--muted); font-size: 0.875rem;">
                Built with PyTorch, Streamlit, and advanced AI techniques
            </p>
            <div class="badge warn" style="margin-top: 16px;">
                ⚠️ Not for clinical use - Always consult healthcare professionals
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
