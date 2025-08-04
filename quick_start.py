#!/usr/bin/env python3
"""
Quick Start Guide for Medical Diagnosis Assistant
This script provides a guided tour of the project and helps with initial setup
"""

import os
import sys
import subprocess
import torch

def print_banner():
    """Print project banner"""
    print("=" * 70)
    print("🏥 MEDICAL DIAGNOSIS ASSISTANT - QUICK START GUIDE")
    print("=" * 70)
    print()

def check_python_version():
    """Check Python version"""
    print("📋 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required. Current version:", sys.version)
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'streamlit', 'opencv-python',
        'matplotlib', 'seaborn', 'scikit-learn', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are installed!")
        return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🚀 Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA is not available. Training will use CPU (slower).")
    print()

def check_project_structure():
    """Check if project structure is correct"""
    print("📁 Checking project structure...")
    
    required_files = [
        'requirements.txt',
        'download_dataset.py',
        'train.py',
        'evaluate.py',
        'gradcam.py',
        'app.py',
        'README.md',
        'utils/__init__.py',
        'utils/data_utils.py',
        'utils/model_utils.py'
    ]
    
    required_dirs = [
        'data',
        'models',
        'utils'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
        else:
            print(f"✅ {directory}/")
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure is correct!")
        return True

def setup_directories():
    """Create necessary directories"""
    print("\n📂 Setting up directories...")
    
    directories = [
        'data/train/NORMAL',
        'data/train/PNEUMONIA',
        'data/val/NORMAL',
        'data/val/PNEUMONIA',
        'data/test/NORMAL',
        'data/test/PNEUMONIA',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}/")

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print()
    print("1. 📊 Dataset Setup:")
    print("   python download_dataset.py")
    print("   # Then manually organize your chest X-ray images into the data/ directory")
    print()
    print("2. 🚀 Train the Model:")
    print("   python train.py --epochs 50 --batch_size 32")
    print()
    print("3. 📈 Evaluate the Model:")
    print("   python evaluate.py --model_name best_model.pt")
    print()
    print("4. 🌐 Run the Web App:")
    print("   streamlit run app.py")
    print()
    print("5. 🐳 Docker (Optional):")
    print("   docker build -t medical-diagnosis-assistant .")
    print("   docker run -p 8501:8501 medical-diagnosis-assistant")
    print()

def show_project_info():
    """Show project information"""
    print("\n📚 PROJECT INFORMATION")
    print("-" * 40)
    print("• Model: ResNet50 with transfer learning")
    print("• Task: Binary classification (Normal vs Pneumonia)")
    print("• Input: Chest X-ray images (224x224)")
    print("• Framework: PyTorch + Streamlit")
    print("• Explainability: Grad-CAM visualizations")
    print("• GPU Support: CUDA acceleration")
    print()

def main():
    """Main function"""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        return
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    check_cuda()
    
    if not check_project_structure():
        print("\n❌ Project structure is incomplete.")
        return
    
    # Setup directories
    setup_directories()
    
    # Show project info
    show_project_info()
    
    # Show next steps
    show_next_steps()
    
    print("🎉 Setup complete! You're ready to start training your model.")
    print("📖 For detailed instructions, see README.md")

if __name__ == "__main__":
    main() 