#!/bin/bash

# Create virtual environment (optional but recommended)
python -m venv capsnet_env
source capsnet_env/bin/activate  # On Windows: capsnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p model/feature_extract/features/7_24_data
mkdir -p model/feature_extract/features/10_19_data
mkdir -p model/feature_extract/features/11_10_data
mkdir -p model/feature_extract/capsnet/__pycache__
mkdir -p logs
mkdir -p checkpoints
