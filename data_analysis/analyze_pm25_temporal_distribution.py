import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_pm25_temporal_distribution(day_folder='11_10_data'):
    """
    Analyze the temporal distribution of PM2.5 values in learning/test splits
    """
    print(f"🔍 Analyzing PM2.5 Temporal Distribution for {day_folder}")
    print("=" * 60)

    # Load data
    learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
    test_path = f"dataset/d_data_split/{day_folder}/test.csv"
    
    learning_df = pd.read_csv(learning_path)
    test_df = pd.read_csv(test_path)

    # Parse timestamps
    learning_df['timestamp'] = pd.to_datetime(learning_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    # Add split label
    learning_df['split'] = 'learning'
    test_df['split'] = 'test'

    # Combine
    full_df = pd.concat([learning_df, test_df]).sort_values('timestamp').reset_index(drop=True)

    # Display basic info
    print(f"📊 Entry counts:")
    print(f"   Learning: {len(learning_df)}")
    print(f"   Test: {len(test_df)}")
    print(f"   Total: {len(full_df)}")

    print(f"\n📈 PM2.5 Stats:")
    for split_name, df in [('Learning', learning_df), ('Test', test_df)]:
        print(f"   {split_name} → Min: {df['pm2.5'].min():.2f}, Max: {df['pm2.5'].max():.2f}, Mean: {df['pm2.5'].mean():.2f}")

    # Temporal range
    print(f"\n⏰ Temporal Ranges:")
    for split_name, df in [('Learning', learning_df), ('Test', test_df)]:
        print(f"   {split_name} → {df['timestamp'].min()} to {df['timestamp'].max()}")

    return full_df

def plot_pm25_temporal_distribution(full_df, day_folder):
    """
    Plot time series distribution of PM2.5
    """
    try:
        plt.figure(figsize=(16, 8))

        # Line plot: PM2.5 over time
        plt.subplot(2, 1, 1)
        for split, df in full_df.groupby('split'):
            plt.plot(df['timestamp'], df['pm2.5'], label=f"{split.capitalize()} (n={len(df)})", alpha=0.8)
        plt.xlabel("Timestamp")
        plt.ylabel("PM2.5")
        plt.title(f"PM2.5 Over Time for {day_folder}")
        plt.legend()
        plt.grid(True)

        # Histogram
        plt.subplot(2, 1, 2)
        learning_pm25 = full_df[full_df['split'] == 'learning']['pm2.5']
        test_pm25 = full_df[full_df['split'] == 'test']['pm2.5']
        plt.hist(learning_pm25, bins=30, alpha=0.7, label='Learning', color='blue')
        plt.hist(test_pm25, bins=30, alpha=0.7, label='Test', color='red')
        plt.xlabel("PM2.5")
        plt.ylabel("Frequency")
        plt.title("Histogram of PM2.5 Distribution")
        plt.legend()

        plt.tight_layout()
        out_path = f"pm25_temporal_distribution_{day_folder}.png"
        plt.savefig(out_path, dpi=300)
        plt.show()

        print(f"📊 Plot saved as {out_path}")
    except Exception as e:
        print(f"❌ Failed to plot PM2.5 distribution: {e}")

if __name__ == "__main__":
    folder = '11_10_data'  # Change this if needed
    full_df = analyze_pm25_temporal_distribution(folder)
    plot_pm25_temporal_distribution(full_df, folder)
