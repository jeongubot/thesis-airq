import os, re, tarfile, random
import numpy as np, pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from datetime import datetime

# Configuration
PATCH_SIZE = 256
STRIDE     = 128
AUGMENT    = True

DATASET_ROOT = 'dataset'
OUTPUT_ROOT  = 'preprocess_img/img_preprocess_output'
METADATA_CSV = os.path.join(OUTPUT_ROOT, 'patch_metadata.csv')

# Normalization
def z_score_normalize(t):
    m, s = t.mean(), t.std()+1e-6
    return (t-m)/s

# Augmentation 
def augment_image(img):
    if random.random()>0.5: img = TF.hflip(img)
    if random.random()>0.5: img = TF.rotate(img, random.uniform(-15,15), fill=0)
    if random.random()>0.5: img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
    if random.random()>0.5: img = TF.adjust_contrast(img, random.uniform(0.8,1.2))
    return img

# patch extraction
def extract_patches(img):
    arr = np.array(img)
    patches = []
    h,w = arr.shape
    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):
            patches.append(Image.fromarray(arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]))
    return patches

# main processing function
def process_dataset_folder(folder, metadata):
    # parse date
    try:
        m,d,_ = folder.split('_')
        date_str = f"2019-{int(m):02d}-{int(d):02d}"
    except:
        return

    # check if images are available
    img_dir = os.path.join(DATASET_ROOT, folder, 'images', 'pictures')
    tarball = os.path.join(DATASET_ROOT, folder, 'images.tar.gz')
    if os.path.exists(tarball) and not os.path.isdir(img_dir):
        os.makedirs(img_dir, exist_ok=True)
        with tarfile.open(tarball,'r:gz') as tf: tf.extractall(img_dir)
    if not os.path.isdir(img_dir): return

    out_dir = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)

    # process each image
    for fname in sorted(os.listdir(img_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png','.jpg'): continue

        # unified timestamp
        if folder=='7_24_data' and len(name)==6 and name.isdigit():
            ts = f"{date_str}_{name}"
        elif folder=='10_19_data':
            ts = name.replace(' ','_')
        elif folder=='11_10_data':
            m = re.match(r'IMG_\d{8}_(\d{6})', name)
            if not m: continue
            ts = f"{date_str}_{m.group(1)}"
        else:
            continue

        img = Image.open(os.path.join(img_dir,fname)).convert('L')
        for idx, patch in enumerate(extract_patches(img)):
            if AUGMENT: patch = augment_image(patch)

            # skip blank
            t = transforms.ToTensor()(patch)
            if t.mean()<0.05 or t.std()<0.01: continue

            # save .npy
            tr = z_score_normalize(t)
            npy_name = f"{folder}_{ts}_p{idx}.npy"
            np.save(os.path.join(out_dir, npy_name), tr.numpy())

            # record metadata
            metadata.append({
                'folder':    folder,
                'timestamp': datetime.strptime(ts, "%Y-%m-%d_%H%M%S"),
                'patch_idx': idx,
                'npy_path':  os.path.join(folder, npy_name)
            })

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    metadata = []
    for f in sorted(os.listdir(DATASET_ROOT)):
        if os.path.isdir(os.path.join(DATASET_ROOT,f)):
            process_dataset_folder(f, metadata)

    pd.DataFrame(metadata).to_csv(METADATA_CSV, index=False)
    print(f"Metadata → {METADATA_CSV}, {len(metadata)} entries")

if __name__=='__main__':
    main()
