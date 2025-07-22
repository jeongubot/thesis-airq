# CapsNet Feature Extractor for Air Quality Prediction

A comprehensive deep learning pipeline that uses Capsule Networks (CapsNet) and CNN baselines to extract features from satellite imagery for PM2.5 air quality prediction. The extracted features are designed to be integrated with LSTM temporal features and used in LightGBM for final prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Integration with LSTM and LightGBM](#integration-with-lstm-and-lightgbm)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

### What is CapsNet?

**Capsule Networks (CapsNet)** are a novel neural network architecture that addresses limitations of traditional CNNs:

- **Spatial Relationships**: Unlike CNNs that lose spatial information through pooling, CapsNet preserves spatial relationships between features
- **Viewpoint Invariance**: CapsNet can recognize objects from different viewpoints and orientations
- **Part-Whole Relationships**: CapsNet explicitly models how parts combine to form wholes
- **Dynamic Routing**: Uses iterative routing algorithms to determine feature relationships

### Why CapsNet for Air Quality?

Air quality prediction from satellite imagery benefits from CapsNet's capabilities:

1. **Spatial Context**: Air pollution patterns have complex spatial relationships
2. **Multi-scale Features**: Pollution sources affect air quality at different scales
3. **Robust Representations**: CapsNet features are more robust to variations in lighting, weather, and seasonal changes
4. **Hierarchical Understanding**: CapsNet can model how local pollution sources contribute to regional air quality

### CNN Baselines

We also implement CNN baselines using pretrained models for comparison:

- **ResNet18/50**: Deep residual networks with skip connections
- **EfficientNet-B0**: Efficient architecture optimized for accuracy/efficiency trade-off
- **MobileNet-V2**: Lightweight architecture for fast inference
- **VGG16**: Classic architecture for baseline comparison

## 🏗️ Architecture

### CapsNet Feature Extractor

\`\`\`
Input Image (3, 256, 256)
↓
Convolutional Layer (256 channels)
↓
Primary Capsules (8 capsules, 8D each)
↓
Digit Capsules (10 capsules, 16D each)
↓
Feature Head (128D output)
\`\`\`

**Key Components:**

1. **Primary Capsules**: Extract low-level features and organize them into capsules
2. **Dynamic Routing**: Iteratively determines which capsules should be active
3. **Digit Capsules**: High-level feature representations
4. **Feature Head**: Maps capsule outputs to fixed-size feature vectors

### CNN Baseline Architecture

\`\`\`
Input Image (3, 256, 256)
↓
Pretrained Backbone (ResNet/EfficientNet/etc.)
↓
Adaptive Pooling
↓
Feature Head (128D output)
\`\`\`

## 📊 Data Pipeline

### Data Flow

\`\`\`
Raw Satellite Images
↓
Preprocessing (dataset/e_preprocessed_img/)
↓
80/20 Split (dataset/d_data_split/)
↓
Patch Extraction & Augmentation
↓
CapsNet/CNN Feature Extraction
↓
Feature Vectors (128D)
↓
Integration with LSTM Features
↓
LightGBM Prediction
\`\`\`

### Data Structure

\`\`\`
dataset/
├── d_data_split/ # Pre-split data (80/20)
│ ├── 7_24_data/
│ │ ├── learning.csv # Training data (80%)
│ │ └── test.csv # Test data (20%)
│ ├── 10_19_data/
│ └── 11_10_data/
└── e_preprocessed_img/ # Processed image patches
├── patch_metadata.csv # Mapping file
└── [day]/[patches].npy # Preprocessed image patches
\`\`\`

## 🚀 Installation

### Prerequisites

\`\`\`bash

# Python 3.8+

pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn tqdm
pip install optuna # For hyperparameter tuning (optional)
\`\`\`

### Project Setup

\`\`\`bash
git clone <repository>
cd capsnet-feature-extractor
\`\`\`

## ⚡ Quick Start

### 1. Test the Pipeline

\`\`\`bash

# Test CapsNet pipeline

python scripts/run_capsnet.py --mode test --day 7_24_data --test_samples 50 --test_epochs 3

# Test CNN baseline

python scripts/run_cnn_baseline.py --mode test --backbone resnet50 --day 7_24_data --test_samples 50 --test_epochs 3
\`\`\`

### 2. Train Models

\`\`\`bash

# Train CapsNet

python scripts/run_capsnet.py --mode train --day 7_24_data --epochs 30 --batch_size 8

# Train CNN baseline

python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day 7_24_data --epochs 30 --batch_size 8
\`\`\`

### 3. Extract Features

\`\`\`bash

# Extract CapsNet features

python scripts/run_capsnet.py --mode extract --day 7_24_data --split learning
python scripts/run_capsnet.py --mode extract --day 7_24_data --split test

# Extract CNN features

python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day 7_24_data --split learning
python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day 7_24_data --split test
\`\`\`

## 📚 Command Reference

### CapsNet Commands

| Command                 | Description           | Example                                                                         |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------- |
| `--mode test`           | Quick pipeline test   | `python scripts/run_capsnet.py --mode test --day 7_24_data`                     |
| `--mode train`          | Train CapsNet model   | `python scripts/run_capsnet.py --mode train --day 7_24_data --epochs 30`        |
| `--mode extract`        | Extract features      | `python scripts/run_capsnet.py --mode extract --day 7_24_data --split learning` |
| `--mode tune`           | Hyperparameter tuning | `python scripts/run_capsnet.py --mode tune --day 7_24_data --trials 50`         |
| `--mode tune_and_train` | Tune + final training | `python scripts/run_capsnet.py --mode tune_and_train --day 7_24_data`           |
| `--mode cv`             | Cross-validation      | `python scripts/run_capsnet.py --mode cv --day 7_24_data`                       |

### CNN Baseline Commands

| Command              | Description         | Example                                                                                 |
| -------------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `--mode test`        | Quick pipeline test | `python scripts/run_cnn_baseline.py --mode test --backbone resnet50 --day 7_24_data`    |
| `--mode train`       | Train CNN model     | `python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day 7_24_data`   |
| `--mode extract`     | Extract features    | `python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day 7_24_data` |
| `--mode compare`     | Compare backbones   | `python scripts/run_cnn_baseline.py --mode compare --day 7_24_data`                     |
| `--mode compare_all` | Compare on all days | `python scripts/run_cnn_baseline.py --mode compare_all`                                 |

### Model Comparison

| Command           | Description            | Example                                                                |
| ----------------- | ---------------------- | ---------------------------------------------------------------------- |
| Direct comparison | Compare CapsNet vs CNN | `python scripts/compare_capsnet_vs_cnn.py --day 7_24_data --epochs 10` |

### Available Backbones

- `resnet18` - Lightweight ResNet
- `resnet50` - Standard ResNet
- `efficientnet_b0` - Efficient architecture
- `mobilenet_v2` - Mobile-optimized
- `vgg16` - Classic CNN architecture

### Available Days

- `7_24_data` - July 24th dataset
- `10_19_data` - October 19th dataset
- `11_10_data` - November 10th dataset

## 🔄 Complete Workflow for All Days

### Step 1: Test All Pipelines

\`\`\`bash

# Test CapsNet on all days

for day in 7_24_data 10_19_data 11_10_data; do
echo "Testing CapsNet on $day"
python scripts/run_capsnet.py --mode test --day $day --test_samples 50 --test_epochs 3
done

# Test CNN baselines on all days

for day in 7_24_data 10_19_data 11_10_data; do
for backbone in resnet18 resnet50 efficientnet_b0; do
echo "Testing CNN $backbone on $day"
python scripts/run_cnn_baseline.py --mode test --backbone $backbone --day $day --test_samples 50 --test_epochs 3
done
done
\`\`\`

### Step 2: Train Models on All Days

\`\`\`bash

# Train CapsNet on all days

for day in 7_24_data 10_19_data 11_10_data; do
echo "Training CapsNet on $day"
python scripts/run_capsnet.py --mode train --day $day --epochs 30 --batch_size 8
done

# Train best CNN backbone on all days

for day in 7_24_data 10_19_data 11_10_data; do
echo "Training CNN ResNet50 on $day"
python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day $day --epochs 30 --batch_size 8
done
\`\`\`

### Step 3: Extract Features for All Days

\`\`\`bash

# Extract CapsNet features

for day in 7_24_data 10_19_data 11_10_data; do
for split in learning test; do
echo "Extracting CapsNet features for $day ($split)"
python scripts/run_capsnet.py --mode extract --day $day --split $split
done
done

# Extract CNN features

for day in 7_24_data 10_19_data 11_10_data; do
for split in learning test; do
echo "Extracting CNN features for $day ($split)"
python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day $day --split $split
done
done
\`\`\`

### Step 4: Compare Models

\`\`\`bash

# Compare CapsNet vs CNN on all days

for day in 7_24_data 10_19_data 11_10_data; do
echo "Comparing models on $day"
python scripts/compare_capsnet_vs_cnn.py --day $day --epochs 10 --max_samples 200
done
\`\`\`

## 🔗 Integration with LSTM and LightGBM

### Feature Integration Pipeline

```python
# Example integration code
import pandas as pd
import numpy as np

def integrate_features(day_folder, split='learning'):
    """
    Integrate CapsNet features with LSTM features for LightGBM
    """

    # 1. Load CapsNet features
    capsnet_features = pd.read_csv(f'capsnet_features_{day_folder}_{split}.csv')

    # 2. Load LSTM features (from your existing pipeline)
    lstm_features = pd.read_csv(f'lstm_features_{day_folder}_{split}.csv')

    # 3. Load CNN features (optional, for ensemble)
    cnn_features = pd.read_csv(f'cnn_resnet50_features_{day_folder}_{split}.csv')

    # 4. Merge features on common keys (image_filename, timestamp)
    merged_features = capsnet_features.merge(
        lstm_features,
        on=['image_filename', 'timestamp'],
        how='inner'
    )

    # 5. Add CNN features if using ensemble
    if cnn_features is not None:
        merged_features = merged_features.merge(
            cnn_features,
            on=['image_filename', 'timestamp'],
            how='inner'
        )

    return merged_features

# Usage for all days
for day in ['7_24_data', '10_19_data', '11_10_data']:
    for split in ['learning', 'test']:
        integrated_features = integrate_features(day, split)
        integrated_features.to_csv(f'integrated_features_{day}_{split}.csv', index=False)
```
