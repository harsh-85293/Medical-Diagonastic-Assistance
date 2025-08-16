# Toy Dataset for Medical AI Assistant

This directory contains a small toy dataset for testing and development purposes.

## Structure

```
toy_dataset/
├── Normal/
│   ├── normal_001.png
│   └── normal_002.png
├── Pneumonia/
│   ├── pneumonia_001.png
│   └── pneumonia_002.png
├── COVID-19/
│   ├── covid_001.png
│   └── covid_002.png
├── Tuberculosis/
│   ├── tb_001.png
│   └── tb_002.png
├── Lung Cancer/
│   ├── cancer_001.png
│   └── cancer_002.png
├── Cardiomegaly/
│   ├── cardio_001.png
│   └── cardio_002.png
├── Pleural Effusion/
│   ├── pleural_001.png
│   └── pleural_002.png
└── Pneumothorax/
    ├── pneumo_001.png
    └── pneumo_002.png
```

## Usage

### For Testing
Use this dataset to test the training pipeline with minimal data:

```bash
python -m src.train --config src/config.yaml --data_dir data/toy_dataset --epochs 2
```

### For Development
Use this dataset to test data loading, augmentation, and model components:

```bash
python -c "
from src.data import CXRDataset
dataset = CXRDataset('data/toy_dataset', ['Normal', 'Pneumonia'])
print(f'Dataset size: {len(dataset)}')
"
```

## Notes

- **Purpose**: Development and testing only
- **Size**: 2 images per class (16 total)
- **Format**: PNG images
- **Content**: Synthetic/dummy images for testing
- **Not for training**: Too small for meaningful model training

## Creating Your Own Toy Dataset

If you need to create a toy dataset with different characteristics:

1. Create the directory structure above
2. Add 2-3 images per class
3. Use any PNG/JPG images (they don't need to be real CXR images for testing)
4. Ensure file names are unique within each class

## Example Commands

```bash
# Test data loading
python -c "from src.data import CXRDataset; print('Data loading works!')"

# Test model creation
python -c "from src.model import CXRClassifier; print('Model creation works!')"

# Test training setup (will fail due to small dataset, but tests pipeline)
python -m src.train --config src/config.yaml --data_dir data/toy_dataset --epochs 1
```
