# 🫁 Medical AI Assistant - Chest X-ray Classifier

A comprehensive, production-ready medical AI system for chest X-ray classification with robust input validation, explainability, and a modern Streamlit interface.

## 🎯 Project Overview

This project implements a multi-class chest X-ray classifier using PyTorch and ResNet50, designed for educational and research purposes. It includes:

- **Multi-class Classification**: 8 chest X-ray conditions
- **Robust Input Validation**: CLIP-based validation + heuristics
- **Explainability**: Grad-CAM attention visualization
- **Modern UI**: Streamlit-based web application
- **Production Ready**: Comprehensive training, evaluation, and deployment pipeline

## 🚀 Features

### Core Functionality
- **8-Class Classification**: Normal, Pneumonia, COVID-19, Tuberculosis, Lung Cancer, Cardiomegaly, Pleural Effusion, Pneumothorax
- **Multiple Input Formats**: PNG, JPG, DICOM, PDF support
- **Advanced Validation**: CLIP zero-shot validation, face detection, color analysis
- **Explainability**: Grad-CAM heatmaps and attention visualization
- **Uncertainty Quantification**: Confidence metrics and uncertainty detection

### Technical Features
- **ResNet50 Architecture**: ImageNet pretrained with custom head
- **Focal Loss**: Class-balanced training with focal loss
- **Mixed Precision**: Training acceleration with AMP
- **Data Augmentation**: Albumentations-based augmentation pipeline
- **Early Stopping**: Intelligent training termination
- **Model Export**: TorchScript and ONNX support

### User Interface
- **Modern Streamlit App**: Clean, responsive design
- **Batch Processing**: Multiple file upload support
- **Real-time Validation**: Instant input validation feedback
- **Interactive Results**: Expandable result cards
- **Professional Styling**: Medical-grade UI/UX

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd medical-ai-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
```

## 🏗️ Project Structure

```
medical-ai-assistant/
├── src/                    # Core source code
│   ├── config.yaml        # Configuration file
│   ├── data.py            # Dataset and data loading
│   ├── model.py           # ResNet50 model architecture
│   ├── losses.py          # Focal loss implementation
│   ├── train.py           # Training pipeline
│   ├── eval.py            # Evaluation and metrics
│   ├── gradcam.py         # Grad-CAM explainability
│   └── validate_input.py  # Input validation
├── app/                    # Streamlit application
│   └── app.py             # Main Streamlit app
├── data/                   # Dataset directory
│   └── cxr_dataset/       # Chest X-ray dataset
├── models/                 # Model checkpoints
├── logs/                   # Training logs
├── plots/                  # Evaluation plots
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Prepare Dataset
Organize your chest X-ray images in the following structure:
```
data/cxr_dataset/
├── Normal/
│   ├── normal_001.png
│   └── normal_002.png
├── Pneumonia/
│   ├── pneumonia_001.png
│   └── pneumonia_002.png
└── ... (other classes)
```

### 2. Train the Model
```bash
python -m src.train --config src/config.yaml --data_dir data/cxr_dataset
```

### 3. Evaluate the Model
```bash
python -m src.eval --config src/config.yaml --ckpt models/best.ckpt --data_dir data/cxr_dataset
```

### 4. Launch Streamlit App
```bash
streamlit run app/app.py
```

## 📚 Usage Examples

### Training Commands

#### Basic Training
```bash
python -m src.train --config src/config.yaml --data_dir data/cxr_dataset
```

#### Resume Training
```bash
python -m src.train --config src/config.yaml --data_dir data/cxr_dataset --resume models/latest.ckpt
```

#### Custom Device
```bash
python -m src.train --config src/config.yaml --data_dir data/cxr_dataset --device cuda
```

### Evaluation Commands

#### Full Evaluation
```bash
python -m src.eval --config src/config.yaml --ckpt models/best.ckpt --data_dir data/cxr_dataset
```

#### Generate Reports
```bash
python -m src.eval --config src/config.yaml --ckpt models/best.ckpt --data_dir data/cxr_dataset
# This will generate plots/ and evaluation_report.txt
```

### Streamlit App

#### Local Development
```bash
streamlit run app/app.py
```

#### Production Deployment
```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

## ⚙️ Configuration

The `src/config.yaml` file contains all configurable parameters:

### Dataset Configuration
```yaml
dataset:
  classes: ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer", "Cardiomegaly", "Pleural Effusion", "Pneumothorax"]
  data_dir: "data/cxr_dataset"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

### Training Configuration
```yaml
training:
  epochs: 60
  batch_size: 32
  learning_rate: 3e-4
  mixed_precision: true
  early_stopping:
    patience: 10
    delta: 0.001
```

### Validation Configuration
```yaml
validation:
  clip_threshold: 0.90
  min_resolution: [256, 256]
  face_detection: true
  color_variance_threshold: 0.3
```

## 🔧 Customization

### Adding New Classes
1. Update `src/config.yaml`:
   ```yaml
   dataset:
     classes:
       - "Normal"
       - "Pneumonia"
       - "COVID-19"
       - "Your_New_Class"
   ```

2. Add corresponding dataset folders
3. Retrain the model

### Modifying Model Architecture
Edit `src/model.py` to change the backbone or classifier head:

```python
# Change backbone
self.backbone = models.resnet101(pretrained=True)  # ResNet101 instead of ResNet50

# Modify classifier
self.classifier = nn.Sequential(
    nn.BatchNorm1d(in_features),
    nn.Dropout(0.3),  # Different dropout rate
    nn.Linear(in_features, 1024),  # Different hidden size
    nn.ReLU(inplace=True),
    nn.Linear(1024, num_classes)
)
```

### Custom Data Augmentation
Modify `src/config.yaml`:

```yaml
augmentation:
  train:
    RandomResizedCrop:
      height: 512
      width: 512
    # Add your custom augmentations
    YourCustomAug:
      param1: value1
      param2: value2
```

## 📊 Model Performance

### Expected Results
- **Validation Accuracy**: ≥92% (with sufficient data)
- **Training Time**: 2-4 hours on RTX 3080 (100K images)
- **Inference Speed**: ~50ms per image on GPU

### Performance Optimization
- **Mixed Precision**: 2x training speed improvement
- **Data Loading**: Multi-worker DataLoader with pin_memory
- **Model Compilation**: torch.compile for inference acceleration

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Smoke Test
```bash
# Test with small dataset
python -m src.train --config src/config.yaml --data_dir data/toy_dataset --epochs 2
```

### Validation Test
```bash
# Test input validation
python -c "
from src.validate_input import validate_single_image
import numpy as np
# Test with sample images
"

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501"]
```

### Local Production
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y poppler-utils

# Run with production settings
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔒 Security & Safety

### Input Validation
- **CLIP-based Validation**: Zero-shot classification for medical vs non-medical
- **Face Detection**: Prevents processing of personal photos
- **Format Validation**: Strict file type and size limits
- **Sanitization**: Secure file handling and processing

### Medical Disclaimer
⚠️ **IMPORTANT**: This tool is for educational and research purposes only. It is NOT intended for clinical use. Always consult qualified healthcare professionals for medical decisions.

### Data Privacy
- No data is stored permanently
- All processing is done in memory
- No external API calls for validation
- Local model inference only

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature-name`
6. Create Pull Request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Run black formatter: `black src/`

### Testing
- Add tests for new features
- Ensure all tests pass: `pytest`
- Maintain test coverage

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ResNet Architecture**: Microsoft Research
- **Focal Loss**: Facebook AI Research
- **Grad-CAM**: Georgia Tech
- **CLIP**: OpenAI
- **Streamlit**: Streamlit Inc.

## 📞 Support

### Issues
- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Documentation: Check this README and inline code comments

### Community
- Discussions: GitHub Discussions
- Wiki: Project Wiki for detailed guides

### Contact
- Email: your-email@example.com
- Twitter: @your-handle

---

**Made with ❤️ for the medical AI community**

*This project demonstrates advanced AI techniques for medical image analysis. Remember: AI is a tool to assist healthcare professionals, not replace them.*
