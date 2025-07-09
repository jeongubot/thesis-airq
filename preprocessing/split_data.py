import os
import pandas as pd
from sklearn.model_selection import train_test_split

MATCHED_DIR = 'dataset/c_matched_spatio_temporal_data'
SPLIT_DIR   = 'dataset/d_data_split'
RANDOM_SEED = 42

csv_files = [f for f in os.listdir(MATCHED_DIR) if f.endswith('.csv')]

for csv_filename in csv_files:
    day = csv_filename.replace('matched_', '').replace('.csv', '')
    subfolder = os.path.join(SPLIT_DIR, f"{day}_data")
    os.makedirs(subfolder, exist_ok=True)

    input_csv = os.path.join(MATCHED_DIR, csv_filename)
    df = pd.read_csv(input_csv)
    print(f"{day}: Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # 1st split: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_SEED
    )

    # 2nd split: val (10%) and test (20%) from temp (which is 30% of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,  # 2/3 of 30% = 20% of total, 1/3 = 10%
        random_state=RANDOM_SEED
    )

    # Save splits in the subfolder
    train_df.to_csv(os.path.join(subfolder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(subfolder, "val.csv"), index=False)
    test_df.to_csv(os.path.join(subfolder, "test.csv"), index=False)

    print(f"{day}: Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")