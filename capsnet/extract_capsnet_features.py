import torch
from torch.utils.data import DataLoader
from .model import CapsNetFeatureExtractor
from .dataset import PatchRegressionDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# === CONFIG ===
INPUT_CSV = 'data_matching/matched_image_tabular.csv'
OUTPUT_FEATURES_CSV = 'data_matching/capsnet_features.csv'
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load model ===
model = CapsNetFeatureExtractor().to(DEVICE)
model.eval()

# === Dataset loader ===
dataset = PatchRegressionDataset(INPUT_CSV, npy_root='preprocess_img/img_preprocess_output')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Feature extraction ===
features_list = []

for batch_idx, (x, label) in enumerate(tqdm(loader, desc="Extracting features")):
    x = x.to(DEVICE)  # [B, 1, 256, 256] or [B, 3, 256, 256] depending on your npy

    with torch.no_grad():
        feats = model(x)  # [B, feature_dim]

    feats_np = feats.cpu().numpy()

    start = batch_idx * BATCH_SIZE
    end = start + x.shape[0]
    batch_df = dataset.df.iloc[start:end].copy()

    for i in range(len(batch_df)):
        fvec = feats_np[i]
        meta = batch_df.iloc[i]
        row = dict(meta)
        for j, val in enumerate(fvec):
            row[f'caps_feat_{j}'] = val
        features_list.append(row)

# === Save all features to a new CSV ===
features_df = pd.DataFrame(features_list)
features_df.to_csv(OUTPUT_FEATURES_CSV, index=False)
print(f"\n✅ Saved CapsNet features to {OUTPUT_FEATURES_CSV}")
