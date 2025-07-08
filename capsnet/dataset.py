# capsnet/dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class PatchRegressionDataset(Dataset):
    def __init__(self, csv_path, npy_root='preprocess_img/img_preprocess_output'):
        self.df = pd.read_csv(csv_path)
        self.npy_root = npy_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct the full path to the .npy file
        npy_path = os.path.join(self.npy_root, row['npy_path'])

        # Optional: check if file exists to catch issues early
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NPY file not found: {npy_path}")

        # Load image patch and label
        npy = np.load(npy_path)  # Assumes shape [1, 256, 256]
        patch = torch.tensor(npy).float()
        label = torch.tensor(row['pm2.5']).float()

        return patch, label
