import os
import pandas as pd

MATCHED_DIR = 'dataset/c_matched_spatio_temporal_data'
SPLIT_DIR   = 'dataset/d_data_split'

def split_data_corrected():
    """
    CORRECTED: Split data according to new methodology
    - 80% learning set (for time series expanding window)
    - 20% hold-out test set
    """
    print("="*60)
    print("CORRECTED DATA SPLITTING")
    print("="*60)
    print("New methodology:")
    print("80% learning set (for time series split)")
    print("20% hold-out test set")
    print("Chronological order maintained")
    print("="*60)
    
    csv_files = [f for f in os.listdir(MATCHED_DIR) if f.endswith('.csv')]

    for csv_filename in csv_files:
        day = csv_filename.replace('matched_', '').replace('.csv', '')
        subfolder = os.path.join(SPLIT_DIR, f"{day}_data")
        os.makedirs(subfolder, exist_ok=True)

        input_csv = os.path.join(MATCHED_DIR, csv_filename)
        df = pd.read_csv(input_csv)
        print(f"\n{day}: Loaded {len(df)} rows")

        # Sort by timestamp for proper time series handling
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # CORRECTED SPLIT: 80% learning, 20% test (chronological)
        n_total = len(df)
        n_learning = int(n_total * 0.8)  # 80% for learning (will be split further by time series)
        
        learning_df = df.iloc[:n_learning].copy()      # First 80% chronologically
        test_df = df.iloc[n_learning:].copy()          # Last 20% chronologically

        # Save the corrected splits
        learning_df.to_csv(os.path.join(subfolder, "learning.csv"), index=False)
        test_df.to_csv(os.path.join(subfolder, "test.csv"), index=False)

        print(f"{day}: Learning set: {len(learning_df)} rows (80%)")
        print(f"{day}: Test set: {len(test_df)} rows (20%)")
        print(f"{day}: Time series split will be applied to learning set")

    print(f"\n{'='*60}")
    print("CORRECTED DATA SPLITTING COMPLETE")
    print("Learning sets ready for time series expanding window")
    print("Test sets reserved for final evaluation")
    print("="*60)

if __name__ == "__main__":
    split_data_corrected()
