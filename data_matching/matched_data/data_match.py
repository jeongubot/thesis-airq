import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

TABULAR_FOLDER   = 'data_matching/merged_tabular_data'
METADATA_CSV     = 'preprocess_img/img_preprocess_output/patch_metadata.csv'
OUTPUT_CSV       = 'data_matching/matched_image_tabular.csv'
TIME_TOLERANCE   = timedelta(minutes=20)

# load patch metadata
meta = pd.read_csv(METADATA_CSV, parse_dates=['timestamp'])
meta = meta.sort_values('timestamp')

matched_dfs = []

# iterate over all tabular CSV files
csv_files = [f for f in os.listdir(TABULAR_FOLDER)
            if f.startswith('merged_') and f.endswith('.csv')]

for csv_filename in tqdm(csv_files, desc="Tabular files"):
    folder = csv_filename.replace('merged_','').replace('.csv','')
    tab_path = os.path.join(TABULAR_FOLDER, csv_filename)

    # load & rename for merge_asof
    tab = (pd.read_csv(tab_path, parse_dates=['time'])
            .rename(columns={'time':'timestamp'})
            .sort_values('timestamp')
        )

    # select only the patches from this folder
    meta_sub = meta[meta['folder']==folder]
    if meta_sub.empty:
        continue

    # nearest‐timestamp join
    merged = pd.merge_asof(
        meta_sub,
        tab,
        on='timestamp',
        tolerance=TIME_TOLERANCE,
        direction='nearest'
    )

    # drop patches with no valid match
    merged = merged.dropna(subset=['pm2.5'])

    matched_dfs.append(merged)

# concatenate & save under data_matching
if matched_dfs:
    final = pd.concat(matched_dfs, ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Matched {len(final)} patches → {OUTPUT_CSV}")
else:
    print("⚠️ No matches found.")

