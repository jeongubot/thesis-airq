import os
import pandas as pd
from sklearn.model_selection import train_test_split

MATCHED_DIR = 'dataset/c_matched_spatio_temporal_data'
SPLIT_DIR   = 'dataset/d_data_split'

csv_files = [f for f in os.listdir(MATCHED_DIR) if f.endswith('.csv')]

for csv_filename in csv_files:
    day_short = csv_filename.replace('matched_', '').replace('.csv', '')
    day_folder = f"{day_short}_data" # e.g., '7_24_data'
    subfolder = os.path.join(SPLIT_DIR, day_folder)
    os.makedirs(subfolder, exist_ok=True)

    input_csv = os.path.join(MATCHED_DIR, csv_filename)
    df = pd.read_csv(input_csv)
    print(f"{day_short}: Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    if 'image_filename' not in df.columns or 'timestamp' not in df.columns:
        print(f"Skipping {csv_filename}: Missing 'image_filename' or 'timestamp' column.")
        continue

    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])

    # Get the earliest timestamp for each unique image
    image_first_timestamps = df.groupby('image_filename')['timestamp_dt'].min().reset_index()
    image_first_timestamps = image_first_timestamps.sort_values('timestamp_dt').reset_index(drop=True)

    unique_images_sorted = image_first_timestamps['image_filename'].values
    
    # Perform a temporal split on the sorted unique images
    # This ensures that images in the test set are chronologically later than images in the learning set.
    # And crucially, all entries for a given image stay together.
    test_size_ratio = 0.2
    num_test_images = int(len(unique_images_sorted) * test_size_ratio)
    
    # Take the last `num_test_images` for the test set
    test_images = unique_images_sorted[-num_test_images:]
    learning_images = unique_images_sorted[:-num_test_images]

    # Filter the original DataFrame based on these image sets
    learning_df = df[df['image_filename'].isin(learning_images)].copy()
    test_df     = df[df['image_filename'].isin(test_images)].copy()

    # Sort by timestamp within each split for consistency (optional, but good practice)
    learning_df = learning_df.sort_values('timestamp_dt').reset_index(drop=True)
    test_df = test_df.sort_values('timestamp_dt').reset_index(drop=True)

    # Drop the temporary datetime column
    learning_df = learning_df.drop(columns=['timestamp_dt'])
    test_df = test_df.drop(columns=['timestamp_dt'])

    # Save learning and test sets
    learning_df.to_csv(os.path.join(subfolder, "learning.csv"), index=False)
    test_df.to_csv(os.path.join(subfolder, "test.csv"), index=False)
    
    print(f"{day_short}: Image-based temporal split complete.")
    print(f"  Learning: {len(learning_df)} entries ({len(learning_images)} unique images)")
    print(f"  Test: {len(test_df)} entries ({len(test_images)} unique images)")
    
    # Verify no overlap in images
    overlap_check = set(learning_df['image_filename'].unique()).intersection(set(test_df['image_filename'].unique()))
    if overlap_check:
        print(f"WARNING: Overlapping images found after split: {len(overlap_check)} images. This should not happen with image-based split.")
    else:
        print("No image overlap between learning and test sets.")

print("\nData splitting process finished.")
