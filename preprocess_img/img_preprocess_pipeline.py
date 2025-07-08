import os, numpy as np, pandas as pd, random, tarfile, re
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

# Configuration
PATCH_SIZE = 256
STRIDE = 128
AUGMENT = True
TIME_WINDOW_MINUTES = 20  # ±10 minutes window

# Root directories
DATASET_ROOT = 'dataset'
OUTPUT_ROOT = 'preprocess_img/img_preprocess_output'

# Normalization
def z_score_normalize(img_tensor):
    mean = img_tensor.mean()
    std = img_tensor.std()
    return (img_tensor - mean) / std

# Augmentation (can be dropped or modified)
def augment_image(img):
    if random.random() > 0.5:
        img = TF.hflip(img)
    if random.random() > 0.5:
        img = TF.rotate(img, random.uniform(-15, 15))
    if random.random() > 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
    return img

# Patch Extraction (patch = 256x256 pixels, stride = 128 pixels)
# returns as a list of PIL Image objects
def extract_patches(img, patch_size=PATCH_SIZE, stride=STRIDE):
    img_array = np.array(img)
    patches = []
    h, w = img_array.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            patches.append(Image.fromarray(patch))
    return patches

# Process each dataset folder
def process_dataset_folder(folder_name):
    saved_count = 0

    print(f"\n🚀 Processing folder: {folder_name}")

    # parse the date from folder name (eg. "7_24_data" -> "2019-07-24")
    try:
        m, d, _ = folder_name.split('_')
        date_str = f"2019-{int(m):02d}-{int(d):02d}"
        print(f"📅 Parsed date: {date_str}")
    except Exception as e:
        print(f"❌ Failed to parse folder name '{folder_name}': {e}")
        return
    
    # path to image folder (eg. "dataset/7_24_data/images/pictures")
    image_folder = os.path.join(DATASET_ROOT, folder_name, 'images', 'pictures')
    tarball = os.path.join(DATASET_ROOT, folder_name, 'images.tar.gz')

    # if tarball exists, extract it to the image folder
    if os.path.exists(tarball) and not os.path.isdir(image_folder):
        os.makedirs(image_folder, exist_ok=True)
        with tarfile.open(tarball, 'r:gz') as tf:
            tf.extractall(image_folder)

    # check if the image folder exists and contains images
    if not os.path.exists(image_folder):
        print(f"❌ Skipping {folder_name} — missing image folder or CSV.")
        return

    # path to save the output patches (eg. "preprocess_img/img_preprocess_output/7_24_data")
    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # list all files in the image folder
    for filename in os.listdir(image_folder):
        name, ext = os.path.splitext(filename)
        
        # skip non-images
        if ext.lower() not in ('.png','.jpg'):
            print(f"⚠️ Skipping {filename} — not an image")
            continue

        # parse timestamp from filename (each dataset has its own format)
        if folder_name == '7_24_data' and len(name)==6 and name.isdigit(): # eg. "123456.png"
            hh, mm, ss = name[:2], name[2:4], name[4:6]
            ts = f"{date_str}_{hh}{mm}{ss}"
        elif folder_name == '10_19_data': # eg. "2019-10-19_123456.png"
            ts = name.replace(' ', '_')           
        elif folder_name == '11_10_data': # eg. "IMG_20191110_123456.png"
            m = re.match(r'IMG_(\d{8})_(\d{6})', name)
            if not m:
                continue
            ts = f"{m.group(1)}_{m.group(2)}"
        else:
            continue

        # Load image, convert to grayscale, and extract patches
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert('L')
        patches = extract_patches(img)

        # apply transformations 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(z_score_normalize)
        ])

        for i, patch in enumerate(patches):
            if AUGMENT:
                patch = augment_image(patch)
            out_name = f"{folder_name}_{ts}_patch{i}.png"
            patch.save(os.path.join(output_dir, out_name))
            saved_count += 1

    if saved_count:
        print(f"✅ Saved {saved_count} patches to {output_dir}")
    else:
        print(f"⚠️  No patches saved for {folder_name}. Output folder is empty.")

# Main driver function
def main():
    # ensure output directory exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # collect all dataset folders
    folders = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])

    print(f"\n▶ Found {len(folders)} folders to process: {folders}\n")

    for folder_name in folders:
        process_dataset_folder(folder_name)

if __name__ == '__main__':
    main()

