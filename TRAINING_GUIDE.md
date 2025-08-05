# 🏥 Advanced Training Guide for Medical Diagnosis Assistant

This guide will help you train a robust multi-class medical diagnosis model using real chest X-ray datasets.

## 📋 Prerequisites

### 1. Install Additional Dependencies
```bash
pip install wandb tqdm scikit-learn pandas
```

### 2. Prepare Your Dataset

#### Option A: Use NIH ChestX-ray14 Dataset (Recommended)
```bash
# Download NIH dataset from: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Extract to: data/nih_chestxray14/

# Organize the dataset
python prepare_dataset.py --mode nih \
    --input_dir data/nih_chestxray14 \
    --output_dir data/processed_nih

# Split into train/val/test
python prepare_dataset.py --mode split \
    --output_dir data/processed_nih \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# Create CSV labels
python prepare_dataset.py --mode csv \
    --input_dir data/processed_nih \
    --output_dir data/processed_nih
```

#### Option B: Use Kaggle Chest X-Ray Dataset
```bash
# Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to: data/kaggle_chestxray/

# Organize the dataset
python prepare_dataset.py --mode kaggle \
    --input_dir data/kaggle_chestxray \
    --output_dir data/processed_kaggle
```

#### Option C: Create Multi-Class Dataset from Multiple Sources
```bash
# Combine multiple datasets
python prepare_dataset.py --mode multi_class \
    --input_dir data/multiple_sources \
    --output_dir data/processed_multiclass
```

## 🚀 Training Commands

### 1. Basic Training (ResNet50)
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --model_type resnet50 \
    --pretrained \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type focal \
    --use_class_weights \
    --advanced_augmentation \
    --gradient_clipping
```

### 2. Advanced Training (Vision Transformer)
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --model_type vit \
    --pretrained \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type focal \
    --use_class_weights \
    --advanced_augmentation \
    --gradient_clipping \
    --use_wandb
```

### 3. Ensemble Training
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --epochs 75 \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --model_type ensemble \
    --pretrained \
    --optimizer adamw \
    --scheduler plateau \
    --loss_type focal \
    --use_class_weights \
    --advanced_augmentation
```

## 📊 Training Parameters Explained

### Model Architecture
- **`resnet50`**: Fast training, good performance
- **`resnet101`**: Better performance, slower training
- **`vit`**: Vision Transformer, excellent performance
- **`ensemble`**: Multiple models combined

### Optimizers
- **`adamw`**: Best for most cases (recommended)
- **`adam`**: Good for smaller datasets
- **`sgd`**: Good with momentum and scheduling

### Loss Functions
- **`focal`**: Handles class imbalance (recommended)
- **`bce`**: Binary cross-entropy for multi-label
- **`ce`**: Cross-entropy for single-label

### Schedulers
- **`cosine`**: Smooth learning rate decay (recommended)
- **`plateau`**: Reduces LR when validation plateaus
- **`step`**: Step-wise LR reduction

## 🎯 Advanced Features

### 1. Class Weights
Use `--use_class_weights` to handle imbalanced datasets:
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --use_class_weights \
    # ... other args
```

### 2. Advanced Augmentation
Use `--advanced_augmentation` for better generalization:
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --advanced_augmentation \
    # ... other args
```

### 3. Gradient Clipping
Use `--gradient_clipping` to prevent exploding gradients:
```bash
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --gradient_clipping \
    # ... other args
```

### 4. Weights & Biases Logging
Use `--use_wandb` for experiment tracking:
```bash
# First, login to wandb
wandb login

# Then train with logging
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --use_wandb \
    # ... other args
```

## 📈 Expected Performance

### With NIH ChestX-ray14 Dataset:
- **ResNet50**: 85-90% accuracy
- **ResNet101**: 87-92% accuracy  
- **Vision Transformer**: 90-95% accuracy
- **Ensemble**: 92-96% accuracy

### Training Time (on RTX 3080):
- **ResNet50**: 2-4 hours
- **ResNet101**: 4-6 hours
- **Vision Transformer**: 6-8 hours
- **Ensemble**: 8-12 hours

## 🔍 Monitoring Training

### 1. Check Training Progress
The script will show:
- Epoch progress
- Training/validation loss and accuracy
- Learning rate schedule
- Best model saves

### 2. View Training History
After training, check:
- `experiments/[experiment_name]/training_history.png`
- `experiments/[experiment_name]/best_model.pth`
- `experiments/[experiment_name]/final_model.pth`

### 3. Weights & Biases Dashboard
If using `--use_wandb`, view:
- Real-time training curves
- Model performance metrics
- Confusion matrices
- Grad-CAM visualizations

## 🎯 Tips for Better Training

### 1. Data Quality
- Ensure high-quality, properly labeled images
- Remove corrupted or low-quality images
- Balance classes if possible

### 2. Hyperparameter Tuning
- Start with recommended settings
- Adjust learning rate based on loss curves
- Increase epochs if validation loss is still decreasing
- Use early stopping to prevent overfitting

### 3. Hardware Optimization
- Use GPU with sufficient VRAM (8GB+ recommended)
- Adjust batch size based on GPU memory
- Use multiple workers for data loading

### 4. Model Selection
- Start with ResNet50 for quick experiments
- Use Vision Transformer for best performance
- Try ensemble for production deployment

## 🚀 Quick Start Example

```bash
# 1. Prepare NIH dataset
python prepare_dataset.py --mode nih \
    --input_dir data/nih_chestxray14 \
    --output_dir data/processed_nih

python prepare_dataset.py --mode split \
    --output_dir data/processed_nih

# 2. Train ResNet50 model
python train_advanced_model.py \
    --data_dir data/processed_nih \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --model_type resnet50 \
    --pretrained \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type focal \
    --use_class_weights \
    --advanced_augmentation \
    --gradient_clipping

# 3. Check results
ls experiments/
```

## 📁 Output Structure

After training, you'll get:
```
experiments/
└── multiclass_resnet50_20231201_143022/
    ├── best_model.pth          # Best model checkpoint
    ├── final_model.pth         # Final model checkpoint
    ├── latest_checkpoint.pth   # Latest checkpoint
    └── training_history.png    # Training curves
```

## 🔧 Troubleshooting

### Common Issues:

1. **Out of Memory**
   - Reduce batch size
   - Use smaller model (ResNet50 instead of ResNet101)
   - Reduce image size

2. **Poor Performance**
   - Check data quality
   - Increase training epochs
   - Try different learning rate
   - Use class weights for imbalanced data

3. **Overfitting**
   - Reduce model complexity
   - Increase data augmentation
   - Use early stopping
   - Add dropout layers

4. **Slow Training**
   - Use GPU acceleration
   - Increase batch size if memory allows
   - Reduce image size
   - Use fewer workers

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Verify dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory usage
5. Review the training logs

Happy training! 🎉 