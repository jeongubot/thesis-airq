import os, re, random
import numpy as np, pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

PATCH_SIZE = 256
STRIDE     = 128
RESIZE_TO  = 256
AUGMENT    = True

RAW_ROOT   = 'dataset/a_raw'
OUTPUT_ROOT= 'dataset/e_preprocessed_img'
PATCH_META = os.path.join(OUTPUT_ROOT, 'patch_metadata.csv')
RESIZE_META= os.path.join(OUTPUT_ROOT, 'resize_metadata.csv')

def z_score_normalize(t):
    m, s = t.mean(), t.std()+1e-6
    return (t-m)/s

def augment_image(img):
    if random.random()>0.5: img = TF.hflip(img)
    if random.random()>0.5: img = TF.rotate(img, random.uniform(-15,15), fill=0)
    if random.random()>0.5: img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
    if random.random()>0.5: img = TF.adjust_contrast(img, random.uniform(0.8,1.2))
    return img

def extract_patches(img):
    arr = np.array(img)
    patches = []
    h,w = arr.shape
    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):
            patches.append(Image.fromarray(arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]))
    return patches

def process_day(day_folder, patch_meta, resize_meta):
    img_dir = os.path.join(RAW_ROOT, day_folder, 'images', 'pictures')
    if not os.path.isdir(img_dir): return

    patch_out_dir = os.path.join(OUTPUT_ROOT, day_folder, 'patch')
    resize_out_dir = os.path.join(OUTPUT_ROOT, day_folder, 'resize')
    os.makedirs(patch_out_dir, exist_ok=True)
    os.makedirs(resize_out_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png','.jpg','.jpeg'): continue

        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert('L')

        # PATCHING
        for idx, patch in enumerate(extract_patches(img)):
            proc_patch = augment_image(patch) if AUGMENT else patch
            t = transforms.ToTensor()(proc_patch)
            if t.mean()<0.05 or t.std()<0.01: continue
            tr = z_score_normalize(t)
            npy_name = f"{fname}_p{idx}.npy"
            np.save(os.path.join(patch_out_dir, npy_name), tr.numpy())
            patch_meta.append({
                'day': day_folder,
                'image_filename': fname,
                'patch_idx': idx,
                'npy_path': os.path.join(day_folder, 'patch', npy_name)
            })

        # RESIZING
        resized = img.resize((RESIZE_TO, RESIZE_TO))
        proc_resized = augment_image(resized) if AUGMENT else resized
        t = transforms.ToTensor()(proc_resized)
        tr = z_score_normalize(t)
        npy_name = f"{fname}_resize.npy"
        np.save(os.path.join(resize_out_dir, npy_name), tr.numpy())
        resize_meta.append({
            'day': day_folder,
            'image_filename': fname,
            'npy_path': os.path.join(day_folder, 'resize', npy_name)
        })

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    patch_meta, resize_meta = [], []
    for day_folder in sorted(os.listdir(RAW_ROOT)):
        if os.path.isdir(os.path.join(RAW_ROOT, day_folder)):
            process_day(day_folder, patch_meta, resize_meta)
    pd.DataFrame(patch_meta).to_csv(PATCH_META, index=False)
    pd.DataFrame(resize_meta).to_csv(RESIZE_META, index=False)

if __name__=='__main__':
    main()