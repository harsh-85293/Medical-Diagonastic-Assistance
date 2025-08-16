#!/usr/bin/env python3
"""
Quick OpenCV deployment fix script
Run this if you encounter OpenCV installation issues during deployment
"""

import subprocess
import sys
import importlib

def install_opencv():
    """Try to install OpenCV with multiple fallback options"""
    
    opencv_packages = [
        "opencv-python-headless>=4.8.0,<5.0.0",
        "opencv-python-headless==4.9.0.80",
        "opencv-python-headless==4.8.0.76", 
        "opencv-python-headless",
        "opencv-contrib-python-headless>=4.8.0"
    ]
    
    print("🔍 Checking if OpenCV is already available...")
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} is already installed!")
        return True
    except ImportError:
        print("⚠️ OpenCV not found. Attempting installation...")
    
    for package in opencv_packages:
        try:
            print(f"🔄 Trying to install: {package}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package,
                "--no-cache-dir", "--force-reinstall"
            ], timeout=300)
            
            # Test the installation
            importlib.invalidate_caches()
            import cv2
            print(f"✅ Successfully installed OpenCV {cv2.__version__} with package: {package}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install {package}: {str(e)[:100]}...")
            continue
    
    print("❌ All OpenCV installation attempts failed.")
    print("\n🛠️ Manual troubleshooting steps:")
    print("1. Update pip: python -m pip install --upgrade pip")
    print("2. Clear cache: pip cache purge")
    print("3. Install system dependencies (Linux): apt-get install libgl1-mesa-glx")
    print("4. Try manual install: pip install opencv-python-headless --no-cache-dir")
    
    return False

if __name__ == "__main__":
    success = install_opencv()
    sys.exit(0 if success else 1)
