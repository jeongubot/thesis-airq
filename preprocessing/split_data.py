import os
import pandas as pd

MATCHED_DIR = 'dataset/c_matched_spatio_temporal_data'
SPLIT_DIR   = 'dataset/d_data_split'

csv_files = [f for f in os.listdir(MATCHED_DIR) if f.endswith('.csv')]

for csv_filename in csv_files:
    day = csv_filename.replace('matched_', '').replace('.csv', '')
    subfolder = os.path.join(SPLIT_DIR, f"{day}_data")
    os.makedirs(subfolder, exist_ok=True)

    input_csv = os.path.join(MATCHED_DIR, csv_filename)
    df = pd.read_csv(input_csv)
    print(f"{day}: Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # Sort by location and timestamp for proper time series split
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['location', 'timestamp']).reset_index(drop=True)

    n = len(df)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train+n_val]
    test_df  = df.iloc[n_train+n_val:]

    train_df.to_csv(os.path.join(subfolder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(subfolder, "val.csv"), index=False)
    test_df.to_csv(os.path.join(subfolder, "test.csv"), index=False)

    print(f"{day}: Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")