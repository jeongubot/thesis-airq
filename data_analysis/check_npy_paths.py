import pandas as pd
import os
import numpy as np

def check_npy_paths(patch_metadata_path="dataset/e_preprocessed_img/patch_metadata.csv", 
                    base_image_dir="dataset/e_preprocessed_img"):
    """
    Checks for empty or invalid 'npy_path' entries in patch_metadata.csv
    and verifies if the corresponding .npy files exist on disk.

    Args:
        patch_metadata_path (str): Path to the patch_metadata.csv file.
        base_image_dir (str): Base directory where preprocessed images are stored.
                              (e.g., 'dataset/e_preprocessed_img')
    """
    print(f"🔍 Checking NPY paths in: {patch_metadata_path}")
    print(f"   Base image directory: {base_image_dir}")
    print("=" * 60)

    if not os.path.exists(patch_metadata_path):
        print(f"❌ Error: patch_metadata.csv not found at {patch_metadata_path}")
        return

    try:
        df = pd.read_csv(patch_metadata_path)
        print(f"📊 Loaded {len(df)} entries from patch_metadata.csv")
    except Exception as e:
        print(f"❌ Error loading patch_metadata.csv: {e}")
        return

    # 1. Check for empty/null npy_path entries
    empty_path_mask = df['npy_path'].isnull() | (df['npy_path'].astype(str).str.strip() == '')
    empty_paths_df = df[empty_path_mask]

    if not empty_paths_df.empty:
        print(f"\n🚨 Found {len(empty_paths_df)} entries with empty or null 'npy_path':")
        print(empty_paths_df.head())
        print("-" * 30)
    else:
        print("\n✅ No empty or null 'npy_path' entries found.")

    # 2. Check for non-existent files for valid npy_path entries
    valid_paths_df = df[~empty_path_mask].copy()
    
    if valid_paths_df.empty:
        print("\nNo valid 'npy_path' entries to check for file existence.")
        return

    print(f"\nChecking existence for {len(valid_paths_df)} non-empty NPY paths...")
    
    valid_paths_df['full_npy_path'] = valid_paths_df['npy_path'].apply(
        lambda x: os.path.join(base_image_dir, x) if not os.path.isabs(x) else x
    )
    
    # Check if files exist
    valid_paths_df['exists'] = valid_paths_df['full_npy_path'].apply(os.path.exists)
    
    non_existent_files_df = valid_paths_df[~valid_paths_df['exists']]

    if not non_existent_files_df.empty:
        print(f"\n🚨 Found {len(non_existent_files_df)} 'npy_path' entries pointing to non-existent files:")
        print(non_existent_files_df[['npy_path', 'full_npy_path']].head())
        print("-" * 30)
    else:
        print("\n✅ All non-empty 'npy_path' entries point to existing files.")

    # 3. Optional: Check if NPY files are loadable and have expected shape/channels
    print("\n🔬 Performing a quick check on a few existing NPY files (first 5)...")
    checked_count = 0
    for idx, row in valid_paths_df[valid_paths_df['exists']].head(5).iterrows():
        npy_file = row['full_npy_path']
        try:
            data = np.load(npy_file)
            print(f"   - {os.path.basename(npy_file)}: Shape={data.shape}, Dtype={data.dtype}")
            if data.shape[0] != 3 or data.shape[1] != 256 or data.shape[2] != 256:
                print(f"     ⚠️  Warning: Unexpected shape. Expected (3, 256, 256).")
        except Exception as e:
            print(f"   - ❌ Error loading {os.path.basename(npy_file)}: {e}")
        checked_count += 1
    
    if checked_count == 0:
        print("   No existing NPY files to check.")

    print("\nSummary:")
    print(f"   Total entries in metadata: {len(df)}")
    print(f"   Entries with empty/null npy_path: {len(empty_paths_df)}")
    print(f"   Entries with non-existent NPY files: {len(non_existent_files_df)}")
    print("=" * 60)

if __name__ == "__main__":
    # You can run this script directly from your terminal:
    # python scripts/check_npy_paths.py
    # Or modify the paths below if your setup is different
    check_npy_paths()
