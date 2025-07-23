# CapsNet Feature Extractor for Air Quality Prediction

A comprehensive deep learning pipeline that uses Capsule Networks (CapsNet) and CNN baselines to extract spatial features from satellite imagery for PM2.5 air quality prediction. The system is designed for trunk-based training to handle large datasets efficiently and integrates with LSTM temporal features for final prediction using LightGBM.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Memory Management](#memory-management)
- [Integration with LSTM and LightGBM](#integration-with-lstm-and-lightgbm)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Overview

### What is CapsNet?

**Capsule Networks (CapsNet)** are a novel neural network architecture that addresses limitations of traditional CNNs:

- **Spatial Relationships**: Unlike CNNs that lose spatial information through pooling, CapsNet preserves spatial relationships between features through dynamic routing
- **Viewpoint Invariance**: CapsNet can recognize spatial patterns from different viewpoints and orientations
- **Part-Whole Relationships**: CapsNet explicitly models how local features combine to form global spatial patterns
- **Dynamic Routing**: Uses iterative routing algorithms to determine which spatial features should be active

### Why CapsNet for Air Quality?

Air quality prediction from satellite imagery benefits from CapsNet's spatial awareness:

1. **Spatial Context**: Air pollution patterns have complex spatial relationships that CapsNet can capture
2. **Multi-scale Features**: Pollution sources affect air quality at different spatial scales
3. **Robust Representations**: CapsNet features are more robust to variations in lighting, weather, and seasonal changes
4. **Hierarchical Understanding**: CapsNet models how local pollution sources contribute to regional air quality patterns

### CNN Baselines

We implement CNN baselines using pretrained models for comparison:

- **ResNet18/50**: Deep residual networks with skip connections
- **EfficientNet-B0**: Efficient architecture optimized for accuracy/efficiency trade-off
- **MobileNet-V2**: Lightweight architecture for fast inference
- **VGG16**: Classic architecture for baseline comparison

## ️ Architecture

### CapsNet Feature Extractor

```plaintext
Input Image (3, 256, 256)
↓
Initial CNN Layer (256 channels, 248x248)
↓
Primary Capsules (115,200 capsules of 32D each)
↓
Dynamic Routing Algorithm (3 iterations)
↓
Digit Capsules (10 capsules of 16D each)
↓
Feature Projection Head (128D output)
```

**Key Components:**

1. **Initial CNN Layer**: Basic feature extraction with 9x9 convolution
2. **Primary Capsules**: Extract local spatial features organized into 8 capsule types across 120x120 spatial locations
3. **Dynamic Routing**: Iteratively determines spatial relationships between primary and digit capsules
4. **Digit Capsules**: High-level spatial pattern representations (10 different spatial patterns)
5. **Feature Head**: Maps capsule outputs to fixed-size feature vectors for downstream tasks

### CNN Baseline Architecture

```plaintext
Input Image (3, 256, 256)
↓
Pretrained Backbone (ResNet/EfficientNet/etc.)
↓
Adaptive Pooling
↓
Feature Head (128D output)
```

### Simplified CapsNet (for Testing)

```plaintext
Input Image (3, 256, 256)
↓
CNN Backbone (64→128→256 channels)
↓
Capsule Transform (8 capsules of 16D)
↓
Spatial Attention
↓
Feature Head (128D output)
```

## Data Pipeline

### Data Flow with Memory Management

```plaintext
Raw Satellite Images
↓
Preprocessing (dataset/e_preprocessed_img/)
↓
80/20 Split (dataset/d_data_split/)
↓
Trunk-based Loading (configurable trunk size)
↓
Patch Extraction & Augmentation (max 50 patches per image)
↓
CapsNet/CNN Feature Extraction
↓
Feature Vectors (128D)
↓
Integration with LSTM Features
↓
LightGBM Prediction
```

### Memory-Efficient Data Structure

```plaintext
dataset/
├── d_data_split/           # Pre-split data (80/20)
│   ├── 7_24_data/
│   │   ├── learning.csv    # Training data (80%)
│   │   └── test.csv        # Test data (20%)
│   ├── 10_19_data/
│   └── 11_10_data/
└── e_preprocessed_img/     # Processed image patches
    ├── patch_metadata.csv  # Mapping file
    └── [day]/[patches].npy # Preprocessed image patches
```

### Trunk-Based Training System

The system handles large datasets (2.8M+ samples) through trunk-based training:

- **Trunk Size**: Configurable (default: 10,000 samples per trunk)
- **Memory Management**: Loads one trunk at a time, garbage collection between trunks
- **Sampling Strategy**: Intelligent sampling to reduce redundancy (max 50 patches per image)
- **Progress Tracking**: Detailed progress reporting across trunks

## Installation

### Prerequisites

```shellscript
# Python 3.8+
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn tqdm
pip install optuna  # For hyperparameter tuning (optional)
pip install psutil  # For memory monitoring
```

### Project Setup

```shellscript
git clone <repository>
cd capsnet-feature-extractor
```

## Quick Start

### 1. Test the Pipeline (Small Scale)

```shellscript
# Test CapsNet pipeline with simplified model
python scripts/run_capsnet.py --mode test --day 7_24_data --test_samples 10 --test_epochs 1

# Test trunk-based training
python scripts/run_capsnet.py --mode test --day 7_24_data --test_samples 10 --test_epochs 1 --use_trunk --trunk_size 1000

# Test CNN baseline
python scripts/run_cnn_baseline.py --mode test --backbone resnet50 --day 7_24_data --test_samples 50 --test_epochs 3
```

### 2. Train Models (Full Scale)

```shellscript
# Train CapsNet with trunk-based training (for large datasets)
python scripts/run_capsnet.py --mode train --day 7_24_data --epochs 30 --batch_size 8 --use_trunk --trunk_size 10000

# Train CapsNet standard (for smaller datasets)
python scripts/run_capsnet.py --mode train --day 7_24_data --epochs 30 --batch_size 8

# Train CNN baseline
python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day 7_24_data --epochs 30 --batch_size 8
```

### 3. Extract Features

```shellscript
# Extract CapsNet features
python scripts/run_capsnet.py --mode extract --day 7_24_data

# Extract CNN features
python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day 7_24_data
```

## Command Reference

### CapsNet Commands

| Command          | Description                 | Example                                                                  |
| ---------------- | --------------------------- | ------------------------------------------------------------------------ |
| `--mode test`    | Quick pipeline test         | `python scripts/run_capsnet.py --mode test --day 7_24_data`              |
| `--mode train`   | Train CapsNet model         | `python scripts/run_capsnet.py --mode train --day 7_24_data --epochs 30` |
| `--mode extract` | Extract features            | `python scripts/run_capsnet.py --mode extract --day 7_24_data`           |
| `--use_trunk`    | Enable trunk-based training | `--use_trunk --trunk_size 10000`                                         |
| `--test_samples` | Limit samples for testing   | `--test_samples 50`                                                      |
| `--test_epochs`  | Limit epochs for testing    | `--test_epochs 3`                                                        |

### Memory Management Options

| Option                    | Description           | Default | Recommended   |
| ------------------------- | --------------------- | ------- | ------------- |
| `--trunk_size`            | Samples per trunk     | 10,000  | 10,000-50,000 |
| `--max_patches_per_image` | Max patches per image | 50      | 20-100        |
| `--batch_size`            | Training batch size   | 8       | 4-16          |
| `--epochs_per_trunk`      | Epochs per trunk      | 5       | 3-10          |

### CNN Baseline Commands

| Command          | Description         | Example                                                                                 |
| ---------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `--mode test`    | Quick pipeline test | `python scripts/run_cnn_baseline.py --mode test --backbone resnet50 --day 7_24_data`    |
| `--mode train`   | Train CNN model     | `python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day 7_24_data`   |
| `--mode extract` | Extract features    | `python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day 7_24_data` |
| `--mode compare` | Compare backbones   | `python scripts/run_cnn_baseline.py --mode compare --day 7_24_data`                     |

### Available Backbones

- `resnet18` - Lightweight ResNet (11M parameters)
- `resnet50` - Standard ResNet (25M parameters)
- `efficientnet_b0` - Efficient architecture (5M parameters)
- `mobilenet_v2` - Mobile-optimized (3M parameters)
- `vgg16` - Classic CNN architecture (138M parameters)

### Available Days

- `7_24_data` - July 24th dataset (~56K samples → 2.8M patches)
- `10_19_data` - October 19th dataset
- `11_10_data` - November 10th dataset

## Memory Management

### Understanding Dataset Scale

```plaintext
7_24_data example:
├── Learning samples: 56,510
├── Patches per image: ~50 (max, intelligently sampled)
├── Total training samples: ~2,825,500
├── Validation samples: ~15,878,410
├── Memory per trunk: ~2-4 GB
└── Estimated training time: ~1,400 hours (with 30min/trunk)
```

### Trunk-Based Training Strategy

1. **Data Sampling**: Reduces redundancy by limiting patches per image
2. **Memory Monitoring**: Tracks memory usage and triggers garbage collection
3. **Progressive Loading**: Loads one trunk at a time to manage memory
4. **Checkpoint System**: Saves progress after each trunk for resumability

### Memory Optimization Tips

```shellscript
# For limited memory (< 16GB RAM)
--trunk_size 5000 --batch_size 4 --max_patches_per_image 20

# For standard memory (16-32GB RAM)
--trunk_size 10000 --batch_size 8 --max_patches_per_image 50

# For high memory (> 32GB RAM)
--trunk_size 50000 --batch_size 16 --max_patches_per_image 100
```

## Complete Workflow for Production

### Step 1: Test All Pipelines

```shellscript
# Test CapsNet on all days (quick validation)
for day in 7_24_data 10_19_data 11_10_data; do
    echo "Testing CapsNet on $day"
    python scripts/run_capsnet.py --mode test --day $day --test_samples 10 --test_epochs 1 --use_trunk --trunk_size 1000
done

# Test CNN baselines
for day in 7_24_data 10_19_data 11_10_data; do
    for backbone in resnet18 resnet50 efficientnet_b0; do
        echo "Testing CNN $backbone on $day"
        python scripts/run_cnn_baseline.py --mode test --backbone $backbone --day $day --test_samples 50 --test_epochs 3
    done
done
```

### Step 2: Train Models (Production Scale)

```shellscript
# Train CapsNet with trunk-based training
for day in 7_24_data 10_19_data 11_10_data; do
    echo "Training CapsNet on $day"
    python scripts/run_capsnet.py --mode train --day $day --epochs 30 --batch_size 8 --use_trunk --trunk_size 10000
done

# Train best CNN backbone
for day in 7_24_data 10_19_data 11_10_data; do
    echo "Training CNN ResNet50 on $day"
    python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day $day --epochs 30 --batch_size 8
done
```

### Step 3: Extract Features for Integration

```shellscript
# Extract CapsNet features
for day in 7_24_data 10_19_data 11_10_data; do
    echo "Extracting CapsNet features for $day"
    python scripts/run_capsnet.py --mode extract --day $day
done

# Extract CNN features
for day in 7_24_data 10_19_data 11_10_data; do
    echo "Extracting CNN features for $day"
    python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day $day
done
```

## Integration with LSTM and LightGBM

### Organized Output Structure

The system automatically creates organized outputs:

```plaintext
outputs/
└── capsnet/
    ├── models/
    │   ├── best/
    │   │   └── 7_24_data/
    │   │       └── best_capsnet_7_24_data_20250123_141500.pth
    │   └── final/
    ├── features/
    │   ├── train/
    │   │   └── 7_24_data/
    │   │       └── capsnet_train_features_7_24_data_20250123_141500.npy
    │   └── val/
    ├── plots/
    │   └── training/
    ├── logs/
    ├── checkpoints/
    │   └── trunk/
    ├── hyperparameters/
    ├── metadata/
    └── experiments/
```

### Feature Integration Pipeline

```python
# Example integration code
import pandas as pd
import numpy as np
from pathlib import Path

def integrate_spatial_features(day_folder, split='train'):
    """
    Integrate CapsNet spatial features with LSTM temporal features
    """
    # 1. Load CapsNet spatial features
    capsnet_features_path = f'outputs/capsnet/features/{split}/{day_folder}/capsnet_{split}_features_{day_folder}_*.npy'
    capsnet_metadata_path = f'outputs/capsnet/metadata/{split}/{day_folder}/capsnet_{split}_metadata_{day_folder}_*.csv'

    capsnet_features = np.load(capsnet_features_path)
    capsnet_metadata = pd.read_csv(capsnet_metadata_path)

    # 2. Load LSTM temporal features (from your existing pipeline)
    lstm_features = pd.read_csv(f'lstm_features_{day_folder}_{split}.csv')

    # 3. Load CNN features (optional, for ensemble)
    cnn_features_path = f'outputs/cnn_baseline/features/{split}/{day_folder}/cnn_resnet50_{split}_features_{day_folder}_*.npy'
    cnn_features = np.load(cnn_features_path)

    # 4. Create feature dataframe
    spatial_df = pd.DataFrame(capsnet_features, columns=[f'capsnet_f{i}' for i in range(128)])
    spatial_df = pd.concat([capsnet_metadata, spatial_df], axis=1)

    # 5. Merge with temporal features
    merged_features = spatial_df.merge(
        lstm_features,
        on=['image_filename', 'timestamp'],
        how='inner'
    )

    # 6. Add CNN features for ensemble
    cnn_df = pd.DataFrame(cnn_features, columns=[f'cnn_f{i}' for i in range(128)])
    merged_features = pd.concat([merged_features, cnn_df], axis=1)

    return merged_features

# Usage for all days
for day in ['7_24_data', '10_19_data', '11_10_data']:
    for split in ['train', 'val']:
        integrated_features = integrate_spatial_features(day, split)
        integrated_features.to_csv(f'integrated_features_{day}_{split}.csv', index=False)
        print(f"Integrated features for {day} ({split}): {integrated_features.shape}")
```

### LightGBM Integration

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def train_lightgbm_with_spatial_features(day_folder):
    """
    Train LightGBM with integrated spatial and temporal features
    """
    # Load integrated features
    train_features = pd.read_csv(f'integrated_features_{day_folder}_train.csv')
    val_features = pd.read_csv(f'integrated_features_{day_folder}_val.csv')

    # Separate features and targets
    feature_cols = [col for col in train_features.columns if col not in ['pm2.5', 'image_filename', 'timestamp']]

    X_train = train_features[feature_cols]
    y_train = train_features['pm2.5']
    X_val = val_features[feature_cols]
    y_val = val_features['pm2.5']

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)]
    )

    # Evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"LightGBM RMSE for {day_folder}: {rmse:.4f}")

    return model, rmse
```

## Troubleshooting

### Common Issues

1. **Memory Errors**

```shellscript
# Reduce trunk size and batch size
--trunk_size 5000 --batch_size 4
```

2. **CUDA Out of Memory**

```shellscript
# Use CPU or reduce batch size
--device cpu --batch_size 2
```

3. **Dimension Mismatch in CapsNet**

1. The system automatically handles dimension mismatches in dynamic routing
1. Check debug prints for tensor shapes during training

1. **Long Training Times**

```shellscript
# Use simplified CapsNet for testing
# The system automatically uses simplified model in test mode
```

### Performance Monitoring

The system provides detailed monitoring:

- Memory usage tracking
- Training progress across trunks
- Automatic checkpoint saving
- Organized output structure with timestamps

### Best Practices

1. **Start Small**: Always test with `--mode test` first
2. **Monitor Memory**: Use `--trunk_size` appropriate for your system
3. **Use Checkpoints**: Training automatically saves checkpoints for resumability
4. **Organized Outputs**: All outputs are automatically organized by day and timestamp

This comprehensive system provides a robust pipeline for extracting spatial features from satellite imagery using both CapsNet and CNN approaches, with efficient memory management for large-scale datasets.
