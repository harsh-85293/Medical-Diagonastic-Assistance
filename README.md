# Medical Diagnosis Assistant

A comprehensive machine learning project for chest X-ray analysis and pneumonia detection using deep learning with PyTorch and Streamlit.

## 🏥 Project Overview

This project implements a complete medical diagnosis assistant that can analyze chest X-ray images to detect pneumonia. The system uses transfer learning with ResNet50 and provides explainability through Grad-CAM visualizations.

### Key Features

- **Deep Learning Model**: ResNet50-based architecture with transfer learning
- **Data Preprocessing**: Comprehensive image preprocessing and augmentation
- **Model Training**: Complete training pipeline with early stopping and checkpointing
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and AUC
- **Explainability**: Grad-CAM visualizations for model interpretability
- **Web Interface**: Streamlit app for easy interaction
- **CUDA Support**: GPU acceleration for faster training and inference

## 📁 Project Structure

```
Medical Diagnosis Assistant/
├── data/                          # Dataset directory
│   ├── train/                     # Training data
│   │   ├── NORMAL/               # Normal chest X-rays
│   │   └── PNEUMONIA/            # Pneumonia chest X-rays
│   ├── val/                      # Validation data
│   └── test/                     # Test data
├── models/                        # Trained model checkpoints
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_utils.py             # Data loading and preprocessing
│   └── model_utils.py            # Model definition and training
├── download_dataset.py            # Dataset download script
├── train.py                      # Training script
├── evaluate.py                    # Evaluation script
├── gradcam.py                    # Grad-CAM implementation
├── app.py                        # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional but recommended)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Medical-Diagnosis-Assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup dataset**
   ```bash
   python download_dataset.py
   ```

## 📊 Dataset

The project uses the "Chest X-Ray Images (Pneumonia)" dataset, which contains:

- **Normal**: 1,349 chest X-ray images
- **Pneumonia**: 3,883 chest X-ray images (bacterial and viral)

### Dataset Organization

After running `download_dataset.py`, organize your data as follows:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🎯 Usage

### 1. Training the Model

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--patience`: Early stopping patience (default: 10)

### 2. Evaluating the Model

```bash
python evaluate.py --model_name best_model.pt
```

**Evaluation Options:**
- `--model_name`: Model file to evaluate (default: best_model.pt)
- `--analyze`: Perform detailed prediction analysis

### 3. Running the Web Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🔧 Model Architecture

### ResNet50 with Transfer Learning

- **Backbone**: ResNet50 pretrained on ImageNet
- **Classification Head**: Custom layers with dropout
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (Normal/Pneumonia)

### Training Configuration

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with weight decay
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: Random flip, rotation, color jitter
- **Normalization**: ImageNet mean and std

## 📈 Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## 🔬 Explainability

### Grad-CAM Implementation

The project includes a complete Grad-CAM implementation that:

- Shows which regions of the X-ray the model focuses on
- Provides attention heatmaps for model interpretability
- Overlays attention maps on original images
- Analyzes attention patterns across different classes

## 🌐 Web Interface

### Streamlit App Features

- **File Upload**: Drag and drop X-ray images
- **Real-time Prediction**: Instant classification results
- **Confidence Display**: Probability scores for each class
- **Grad-CAM Visualization**: Interactive attention maps
- **Responsive Design**: Works on desktop and mobile

### App Usage

1. Upload a chest X-ray image
2. View the prediction and confidence
3. Explore the Grad-CAM visualization
4. Understand which regions influenced the decision

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t medical-diagnosis-assistant .
docker run -p 8501:8501 medical-diagnosis-assistant
```

## 📋 Requirements

### Core Dependencies

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB+ RAM, CUDA GPU
- **Storage**: 5GB+ for dataset and models

## ⚠️ Important Notes

### Medical Disclaimer

⚠️ **This tool is for educational and research purposes only.**

- Not intended for clinical diagnosis
- Always consult healthcare professionals
- Results should not replace medical expertise
- Use at your own risk

### Dataset Considerations

- Ensure proper data privacy and consent
- Follow medical data handling guidelines
- Consider dataset bias and limitations
- Validate results with domain experts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Chest X-Ray Images (Pneumonia) dataset contributors
- PyTorch and Streamlit communities
- Medical imaging research community

## 📞 Support

For questions or issues:

1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Contact the maintainers

---

**Built with ❤️ for medical AI research and education** 