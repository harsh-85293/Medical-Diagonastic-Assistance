# 🏥 Multi-Class Medical Diagnosis Assistant

A comprehensive AI-powered medical diagnosis system that can detect multiple diseases from chest X-ray images using deep learning. This project supports both binary (Normal vs Pneumonia) and multi-class disease classification.

## 🎯 **Features**

### **Multi-Class Disease Detection**
- **Normal** - Healthy chest X-rays
- **Pneumonia** - Bacterial and viral pneumonia
- **COVID-19** - Coronavirus-related lung abnormalities
- **Tuberculosis** - TB-related lung changes
- **Pleural Effusion** - Fluid around the lungs
- **Pneumothorax** - Air between lung and chest wall
- **Lung Cancer** - Malignant lung lesions
- **Cardiomegaly** - Enlarged heart

### **Advanced Model Architectures**
- **ResNet50** - Transfer learning with ImageNet weights
- **Vision Transformer (ViT)** - State-of-the-art transformer architecture
- **Ensemble Models** - Combination of multiple architectures
- **Custom CNNs** - Lightweight models for deployment

### **Comprehensive Evaluation**
- **Per-class metrics** (Precision, Recall, F1, AUC)
- **Confusion matrices** and classification reports
- **ROC curves** and precision-recall curves
- **Model explainability** with Grad-CAM visualizations

### **User-Friendly Interface**
- **Streamlit Web App** - Interactive medical file analysis
- **Document Support** - PDF, DOCX, DOC, TXT processing
- **Image Validation** - Medical content verification
- **Real-time Predictions** - Instant disease detection

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/your-username/medical-diagnosis-assistant.git
cd medical-diagnosis-assistant

# Install dependencies
pip install -r requirements.txt
```

### **2. Setup Project Structure**
```bash
# Create necessary directories
python download_dataset.py
```

### **3. Train Multi-Class Model**
```bash
# Train with default settings
python train_multi_class.py --data-dir ./data/chest_xray --epochs 50

# Train with custom parameters
python train_multi_class.py \
    --data-dir ./data/chest_xray \
    --model-type resnet50 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --use-class-weights \
    --experiment-name my_experiment
```

### **4. Evaluate Model**
```bash
# Comprehensive evaluation
python evaluate_multi_class.py \
    --model-path ./models/multiclass_best_model.pth \
    --data-dir ./data/chest_xray \
    --output-dir ./evaluation_results
```

### **5. Run Streamlit App**
```bash
# Original binary classification app
streamlit run app.py

# Multi-class classification app
streamlit run app_multi_class.py
```

## 📊 **Model Performance**

### **Target Metrics**
- **Mean Average Precision**: >80% across all disease classes
- **Per-class F1 Score**: >75% for each disease
- **Overall Accuracy**: >85% on test set
- **Mean AUC**: >0.90 for all diseases

### **Supported Datasets**
- **NIH ChestX-ray14** - 112,120 images, 14 diseases
- **CheXpert** - 224,316 images, 14 observations
- **COVID-19 Chest X-ray** - COVID-19 specific dataset
- **Custom Datasets** - Your own labeled data

## 🏗️ **Architecture**

### **Data Pipeline**
```
Raw X-ray Images → Preprocessing → Augmentation → DataLoader → Model
```

### **Model Architecture**
```
Input (224x224x3) → ResNet50/ViT → Feature Extraction → Classification Head → Output (8 classes)
```

### **Training Pipeline**
```
Data Loading → Model Forward Pass → Loss Calculation → Backward Pass → Optimization → Validation
```

## 📁 **Project Structure**

```
medical-diagnosis-assistant/
├── data/                           # Dataset directory
│   ├── chest_xray/                # Chest X-ray images
│   └── labels.csv                 # Multi-label annotations
├── models/                        # Trained models
│   ├── multiclass_best_model.pth  # Multi-class model
│   └── chest_xray_demo.pth       # Demo model
├── utils/                         # Utility modules
│   ├── data_utils.py             # Data loading and preprocessing
│   └── model_utils.py            # Model definitions and training
├── evaluation_results/            # Evaluation outputs
├── app.py                        # Binary classification Streamlit app
├── app_multi_class.py            # Multi-class Streamlit app
├── train_multi_class.py          # Multi-class training script
├── evaluate_multi_class.py       # Comprehensive evaluation
├── gradcam.py                    # Model explainability
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 🔧 **Advanced Usage**

### **Custom Training**
```bash
# Train with specific disease classes
python train_multi_class.py \
    --model-type vit \
    --loss-type focal \
    --optimizer adamw \
    --scheduler cosine \
    --use-class-weights \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5
```

### **Model Evaluation**
```bash
# Detailed evaluation with visualizations
python evaluate_multi_class.py \
    --model-path ./models/my_model.pth \
    --output-dir ./my_evaluation \
    --batch-size 32
```

### **Grad-CAM Analysis**
```python
from gradcam import generate_multi_class_gradcam

# Generate attention maps for all diseases
results = generate_multi_class_gradcam(model, "image.jpg", save_dir="gradcam_results")
```

## 📈 **Training Options**

### **Model Types**
- `resnet50` - ResNet50 with transfer learning
- `vit` - Vision Transformer (B/16)
- `ensemble` - Combination of multiple models

### **Loss Functions**
- `bce` - Binary Cross Entropy (multi-label)
- `focal` - Focal Loss for class imbalance
- `cross_entropy` - Cross Entropy Loss (single-label)

### **Optimizers**
- `adam` - Adam optimizer (default)
- `sgd` - Stochastic Gradient Descent
- `adamw` - Adam with weight decay

### **Schedulers**
- `plateau` - ReduceLROnPlateau
- `cosine` - Cosine Annealing
- `step` - Step LR scheduler

## 🎨 **Streamlit App Features**

### **Multi-Class App (`app_multi_class.py`)**
- **Top 3 Predictions** with confidence scores
- **Radar Chart** showing all disease probabilities
- **Grad-CAM Visualizations** for model explainability
- **Interactive Charts** using Plotly
- **Document Processing** for medical reports

### **Binary App (`app.py`)**
- **Normal vs Pneumonia** classification
- **Confidence Bars** and prediction boxes
- **Grad-CAM Overlays** for attention visualization
- **File Validation** for medical content

## 🔬 **Model Explainability**

### **Grad-CAM Features**
- **Per-class attention maps** for each disease
- **Multi-class Grad-CAM** for all diseases simultaneously
- **Attention overlays** showing model focus areas
- **Comparative analysis** across different diseases

### **Visualization Types**
- **Heatmaps** - Raw attention scores
- **Overlays** - Attention maps on original images
- **Grid views** - Multiple disease attention maps
- **Comparative plots** - Side-by-side analysis

## 📊 **Evaluation Metrics**

### **Overall Metrics**
- **Accuracy** - Overall classification accuracy
- **Mean F1 Score** - Average F1 across all classes
- **Mean AUC** - Average AUC across all classes
- **Mean Precision** - Average precision across all classes

### **Per-Class Metrics**
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall
- **AUC** - Area under ROC curve
- **AP** - Average precision

### **Visualizations**
- **Confusion Matrix** - Classification performance
- **ROC Curves** - Per-class ROC analysis
- **Precision-Recall Curves** - Per-class PR analysis
- **Per-class Metrics** - Bar charts of metrics

## 🚀 **Deployment**

### **Local Deployment**
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app_multi_class.py
```

### **Cloud Deployment**
```bash
# Deploy to Streamlit Cloud
git push origin main

# Deploy to Heroku
heroku create my-medical-app
git push heroku main
```

### **Docker Deployment**
```bash
# Build Docker image
docker build -t medical-diagnosis-app .

# Run container
docker run -p 8501:8501 medical-diagnosis-app
```

## ⚠️ **Important Notes**

### **Medical Disclaimer**
- This tool is for **educational purposes only**
- **Not for clinical diagnosis** or medical use
- Always consult healthcare professionals
- Results should not replace medical expertise

### **Data Privacy**
- Use **anonymized/de-identified** medical data
- Follow **HIPAA** and other privacy regulations
- Implement proper **data security** measures
- Use **local processing** when possible

### **Model Limitations**
- **Training data dependent** performance
- **Domain shift** considerations
- **Class imbalance** challenges
- **Clinical validation** required for medical use

## 🤝 **Contributing**

### **How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📚 **References**

### **Datasets**
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [COVID-19 Chest X-ray](https://github.com/ieee8023/covid-chestxray-dataset)

### **Papers**
- [ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases](https://arxiv.org/abs/1705.02315)
- [CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison](https://arxiv.org/abs/1901.07031)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NIH** for the ChestX-ray14 dataset
- **Stanford ML Group** for CheXpert dataset
- **PyTorch** and **Streamlit** communities
- **Medical imaging** research community

---

**🏥 Built with ❤️ for medical AI research and education** 