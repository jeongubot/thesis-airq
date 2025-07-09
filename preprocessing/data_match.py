import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

TABULAR_FOLDER   = 'dataset/b_merged_tabular_data'
SPATIO_FOLDER    = 'dataset/a_raw'
OUTPUT_DIR       = 'dataset/c_matched_spatio_temporal_data'
TIME_TOLERANCE   = timedelta(minutes=20)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# iterate over all tabular CSV files
csv_files = [f for f in os.listdir(TABULAR_FOLDER)
            if f.startswith('merged_') and f.endswith('.csv')]

for csv_filename in tqdm(csv_files, desc="Tabular files"):
    folder = csv_filename.replace('merged_','').replace('.csv','')
    tab_path = os.path.join(TABULAR_FOLDER, csv_filename)
    image_folder = os.path.join(SPATIO_FOLDER, f"{folder}_data", "images", "pictures")

    # load tabular data
    tab = (pd.read_csv(tab_path, parse_dates=['time'])
            .rename(columns={'time':'timestamp'})
            .sort_values('timestamp')
        )
    print(f"{folder}: Loaded {len(tab)} tabular rows from {tab_path}")

    # collect image info
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{folder}: Found {len(image_files)} images in {image_folder}")
    
    image_df = pd.DataFrame({'image_filename': image_files})
    # Try to extract timestamp from filename (customize this as needed)
    def extract_timestamp(fname):
        # Try several formats based on your samples
        import re
        from datetime import datetime
        # Example: 2019-10-19 104440.jpg
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{6})', fname)
        if match:
            return pd.to_datetime(match.group(1), format='%Y-%m-%d %H%M%S', errors='coerce')
        # Example: IMG_20191110_103543.jpg
        match = re.search(r'IMG_(\d{8})_(\d{6})', fname)
        if match:
            dt = match.group(1) + match.group(2)
            return pd.to_datetime(dt, format='%Y%m%d%H%M%S', errors='coerce')
        # Example: 123456.jpg (fallback: treat as HHMMSS)
        match = re.match(r'(\d{6})', fname)
        if match:
            # Use a dummy date, or skip if not useful
            if folder == '7_24':
                date_str = '2019-07-24'
                time_str = match.group(1)
                dt_str = f"{date_str} {time_str}"
                return pd.to_datetime(dt_str, format='%Y-%m-%d %H%M%S', errors='coerce')
        return pd.NaT

    image_df['timestamp'] = image_df['image_filename'].apply(extract_timestamp)
    image_df = image_df.dropna(subset=['timestamp']).sort_values('timestamp')

    image_df = image_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # 1. Match images to tabular data within 20 min window
    matched = pd.merge_asof(
        image_df,
        tab,
        on='timestamp',
        tolerance=TIME_TOLERANCE,
        direction='nearest'
    )

    # 2. Find tabular rows with no matched image (i.e., not in matched['timestamp'])
    matched_tab_timestamps = matched['timestamp'].dropna().unique()
    unmatched_tab = tab[~tab['timestamp'].isin(matched_tab_timestamps)]

    # 3. For unmatched tabular rows, assign nearest image (no tolerance)
    if not unmatched_tab.empty and not image_df.empty:
        unmatched = pd.merge_asof(
            unmatched_tab.sort_values('timestamp'),
            image_df[['image_filename', 'timestamp']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        # Combine matched and unmatched
        final = pd.concat([matched, unmatched], ignore_index=True)
    else:
        final = matched
        
    # Save matched CSV for this day
    output_csv = os.path.join(OUTPUT_DIR, f"matched_{folder}.csv")
    final.to_csv(output_csv, index=False)

print("Done saving per-day matched CSVs.")