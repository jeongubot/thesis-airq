import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_temporal_split(day_folder='11_10_data'):
    """
    Analyze the temporal distribution of images
    """
    print(f"🔍 Analyzing Temporal Split for {day_folder}")
    print("=" * 60)
    
    # Load the split data
    learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
    test_path = f"dataset/d_data_split/{day_folder}/test.csv"
    
    learning_df = pd.read_csv(learning_path)
    test_df = pd.read_csv(test_path)
    
    # Convert timestamps
    learning_df['timestamp'] = pd.to_datetime(learning_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Combine for full analysis
    full_df = pd.concat([
        learning_df.assign(split='learning'),
        test_df.assign(split='test')
    ]).sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Split Summary:")
    print(f"   Learning entries: {len(learning_df):,}")
    print(f"   Test entries: {len(test_df):,}")
    print(f"   Total entries: {len(full_df):,}")
    
    print(f"\n🖼️  Image Distribution:")
    learning_images = learning_df['image_filename'].nunique()
    test_images = test_df['image_filename'].nunique()
    total_images = full_df['image_filename'].nunique()
    
    print(f"   Learning images: {learning_images}")
    print(f"   Test images: {test_images}")
    print(f"   Total unique images: {total_images}")
    
    # Check for overlap
    learning_img_set = set(learning_df['image_filename'].unique())
    test_img_set = set(test_df['image_filename'].unique())
    overlap = learning_img_set.intersection(test_img_set)
    
    if overlap:
        print(f"   🚨 Image overlap: {len(overlap)} images appear in both splits!")
        print(f"      Overlapping images: {list(overlap)[:5]}")
    else:
        print(f"   ✅ No image overlap - clean temporal split")
    
    # Temporal analysis
    print(f"\n⏰ Temporal Analysis:")
    time_range = full_df['timestamp'].max() - full_df['timestamp'].min()
    learning_time_range = learning_df['timestamp'].max() - learning_df['timestamp'].min()
    test_time_range = test_df['timestamp'].max() - test_df['timestamp'].min()
    
    print(f"   Full time range: {time_range}")
    print(f"   Learning time range: {learning_time_range}")
    print(f"   Test time range: {test_time_range}")
    
    print(f"\n   Time boundaries:")
    print(f"   Learning: {learning_df['timestamp'].min()} to {learning_df['timestamp'].max()}")
    print(f"   Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Image distribution over time
    print(f"\n📈 Image Distribution Over Time:")
    
    # Group by image and get time stats
    img_time_stats = full_df.groupby('image_filename')['timestamp'].agg(['min', 'max', 'count']).reset_index()
    img_time_stats['duration'] = img_time_stats['max'] - img_time_stats['min']
    img_time_stats = img_time_stats.sort_values('min')
    
    print(f"   Images sorted by first appearance:")
    for i, row in img_time_stats.head(10).iterrows():
        split_info = "learning" if row['image_filename'] in learning_img_set else "test"
        print(f"   {row['image_filename']}: {row['min']} ({row['count']} entries) → {split_info}")
    
    # Check if images are clustered in time
    learning_first_times = []
    test_first_times = []
    
    for img in learning_img_set:
        first_time = full_df[full_df['image_filename'] == img]['timestamp'].min()
        learning_first_times.append(first_time)
    
    for img in test_img_set:
        first_time = full_df[full_df['image_filename'] == img]['timestamp'].min()
        test_first_times.append(first_time)
    
    if learning_first_times and test_first_times:
        learning_median_time = pd.Series(learning_first_times).median()
        test_median_time = pd.Series(test_first_times).median()
        
        print(f"\n🎯 Temporal Clustering:")
        print(f"   Learning images median first appearance: {learning_median_time}")
        print(f"   Test images median first appearance: {test_median_time}")
        
        if test_median_time > learning_median_time:
            print(f"   ✅ Clean temporal split: test images appear later")
        else:
            print(f"   ⚠️  Mixed temporal distribution")
    
    # Entries per image analysis
    print(f"\n📊 Entries per Image Analysis:")
    learning_entries_per_img = learning_df.groupby('image_filename').size()
    test_entries_per_img = test_df.groupby('image_filename').size()
    
    print(f"   Learning - Avg entries per image: {learning_entries_per_img.mean():.1f}")
    print(f"   Learning - Max entries per image: {learning_entries_per_img.max()}")
    print(f"   Test - Avg entries per image: {test_entries_per_img.mean():.1f}")
    print(f"   Test - Max entries per image: {test_entries_per_img.max()}")
    
    # Top images by entry count
    print(f"\n🔝 Top Images by Entry Count:")
    all_img_counts = full_df.groupby('image_filename').size().sort_values(ascending=False)
    for i, (img, count) in enumerate(all_img_counts.head(5).items()):
        split_info = "learning" if img in learning_img_set else "test"
        print(f"   {i+1}. {img}: {count:,} entries → {split_info}")
    
    return {
        'learning_df': learning_df,
        'test_df': test_df,
        'full_df': full_df,
        'learning_images': learning_images,
        'test_images': test_images,
        'overlap': overlap
    }

def plot_temporal_distribution(data_info):
    """
    Plot the temporal distribution of images
    """
    full_df = data_info['full_df']
    
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Timeline of all entries
        plt.subplot(2, 2, 1)
        learning_data = full_df[full_df['split'] == 'learning']
        test_data = full_df[full_df['split'] == 'test']
        
        plt.scatter(learning_data['timestamp'], range(len(learning_data)), 
                   alpha=0.6, s=1, label='Learning', color='blue')
        plt.scatter(test_data['timestamp'], range(len(learning_data), len(full_df)), 
                   alpha=0.6, s=1, label='Test', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Entry Index')
        plt.title('Temporal Distribution of All Entries')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 2: Images over time
        plt.subplot(2, 2, 2)
        img_first_appearance = full_df.groupby('image_filename')['timestamp'].min().reset_index()
        img_first_appearance['split'] = img_first_appearance['image_filename'].apply(
            lambda x: 'learning' if x in set(learning_data['image_filename']) else 'test'
        )
        
        learning_imgs = img_first_appearance[img_first_appearance['split'] == 'learning']
        test_imgs = img_first_appearance[img_first_appearance['split'] == 'test']
        
        plt.scatter(learning_imgs['timestamp'], range(len(learning_imgs)), 
                   alpha=0.8, s=50, label='Learning Images', color='blue')
        plt.scatter(test_imgs['timestamp'], range(len(learning_imgs), len(img_first_appearance)), 
                   alpha=0.8, s=50, label='Test Images', color='red')
        plt.xlabel('First Appearance Time')
        plt.ylabel('Image Index')
        plt.title('First Appearance of Each Image')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 3: Entries per image
        plt.subplot(2, 2, 3)
        img_counts = full_df.groupby(['image_filename', 'split']).size().unstack(fill_value=0)
        if 'learning' in img_counts.columns and 'test' in img_counts.columns:
            plt.scatter(img_counts['learning'], img_counts['test'], alpha=0.7)
            plt.xlabel('Learning Entries')
            plt.ylabel('Test Entries')
            plt.title('Entries per Image: Learning vs Test')
            
            # Add diagonal line
            max_val = max(img_counts['learning'].max(), img_counts['test'].max())
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Plot 4: Timeline histogram
        plt.subplot(2, 2, 4)
        plt.hist(learning_data['timestamp'], bins=20, alpha=0.7, label='Learning', color='blue')
        plt.hist(test_data['timestamp'], bins=20, alpha=0.7, label='Test', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Entry Count')
        plt.title('Temporal Distribution Histogram')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'temporal_split_analysis_{data_info["full_df"]["split"].iloc[0]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Plot saved as temporal_split_analysis.png")
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

def recommend_split_strategy(data_info):
    """
    Recommend the best split strategy based on analysis
    """
    learning_images = data_info['learning_images']
    test_images = data_info['test_images']
    overlap = data_info['overlap']
    
    print(f"\n💡 Split Strategy Recommendations:")
    
    if len(overlap) > 0:
        print(f"   🚨 PROBLEM: Images appear in both splits!")
        print(f"   🔧 FIX: Use image-based split instead of temporal split")
    
    if learning_images < 20:
        print(f"   🚨 PROBLEM: Too few learning images ({learning_images})")
        print(f"   🔧 SOLUTIONS:")
        print(f"      1. Use cross-validation instead of single split")
        print(f"      2. Combine learning + test for cross-validation")
        print(f"      3. Use Leave-One-Out validation")
    
    if test_images < 5:
        print(f"   🚨 PROBLEM: Too few test images ({test_images})")
        print(f"   🔧 SOLUTION: Use larger test set or cross-validation")
    
    print(f"\n🎯 Recommended Approach:")
    total_images = learning_images + test_images
    if total_images < 50:
        print(f"   Use 5-Fold Cross-Validation on all {total_images} images")
        print(f"   This gives more reliable results than single 80/20 split")
    else:
        print(f"   Current temporal split is reasonable")
        print(f"   But consider stratified split by location if available")

if __name__ == "__main__":
    data_info = analyze_temporal_split('11_10_data')
    plot_temporal_distribution(data_info)
    recommend_split_strategy(data_info)
