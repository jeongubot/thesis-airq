# models/feature_extract/extract_features.py
import os
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from capsnet.model import CapsNet  # your model.py
# no need for loss.py here


# ─────── CONFIG ─────── #
RAW_SPLIT   = 'dataset/d_data_split'
NPY_ROOT    = 'dataset/e_preprocessed_img'
PATCH_META  = os.path.join(NPY_ROOT, 'patch_metadata.csv')
RESIZE_META = os.path.join(NPY_ROOT, 'resize_metadata.csv')


# ─────── DATASET ─────── #
class NpyImageDataset(Dataset):
    def __init__(self, paths, labels, fnames):
        self.paths  = paths
        self.labels = labels
        self.fnames = fnames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.load(self.paths[i])
        # ensure (C,H,W)
        if arr.ndim == 2:
            arr = arr[None,...]
        img   = torch.from_numpy(arr).float()
        label = torch.tensor(self.labels[i],dtype=torch.long)
        fname = self.fnames[i]
        return img, label, fname


# ─────── UTILS ─────── #
def build_dataset(split_csv, meta_df, day, npy_root):
    """
    Merge split CSV (filename,label) with patch/resize metadata,
    then build flat lists of full .npy paths, labels, and filenames.
    """
    df_split = pd.read_csv(split_csv)
    df_meta  = meta_df[meta_df['day']==day]
    df = pd.merge(df_split,
                  df_meta,
                  left_on='filename',
                  right_on='image_filename',
                  how='inner')\
           .reset_index(drop=True)

    # full path to each .npy
    paths  = df['npy_path'].apply(lambda p: os.path.join(npy_root,p)).tolist()
    labels = df['label'].tolist()
    fnames = df['npy_path'].tolist()  # or df['filename'] if you prefer
    return NpyImageDataset(paths, labels, fnames)


# ─────── MAIN ─────── #
def extract_for(day, method, ckpt, out_dir, batch_size=32):
    # pick correct metadata
    patch_meta  = pd.read_csv(PATCH_META)
    resize_meta = pd.read_csv(RESIZE_META)
    meta_df     = patch_meta if method=='patch' else resize_meta

    # locate CSV splits for this day
    day_folder = os.path.join(RAW_SPLIT, day)
    train_csv  = os.path.join(day_folder, 'train.csv')
    val_csv    = os.path.join(day_folder, 'val.csv')
    if not os.path.isfile(train_csv) or not os.path.isfile(val_csv):
        raise FileNotFoundError(f"Missing splits for day {day}")

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CapsNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # build datasets + loaders
    ds_train = build_dataset(train_csv, meta_df, day, NPY_ROOT)
    ds_val   = build_dataset(val_csv,   meta_df, day, NPY_ROOT)
    loader   = DataLoader(ds_train + ds_val,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True)

    feats_list, labs_list, fn_list = [], [], []

    with torch.no_grad():
        for imgs, labs, fnames in loader:
            imgs = imgs.to(device, non_blocking=True)
            caps = model(imgs)                # [B, num_classes, dim]
            # get capsule-lengths (one feature per class)
            feats = torch.norm(caps, dim=-1)  # [B, num_classes]
            feats_list .append(feats.cpu().numpy())
            labs_list  .extend(labs.tolist())
            fn_list    .extend(fnames)

    # stack and save
    features = np.vstack(feats_list)
    labels   = np.array(labs_list)
    fnames   = np.array(fn_list, dtype='<U')  # string array

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{day}_{method}_features.npz")
    np.savez_compressed(out_file,
                        features=features,
                        labels=labels,
                        filenames=fnames)
    print(f"✔️  Saved features to {out_file}")


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--day',       required=True,
                    help="Day folder name, e.g. '2025-07-10'")
    p.add_argument('--method',    required=True,
                    choices=['patch','resize'])
    p.add_argument('--ckpt',      required=True,
                    help="path to trained CapsNet .pth")
    p.add_argument('--out-dir',   default='outputs/features',
                    help="where to save the .npz files")
    p.add_argument('--batch-size',type=int, default=32)
    args = p.parse_args()

    extract_for(args.day,
                args.method,
                args.ckpt,
                args.out_dir,
                args.batch_size)
