import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

def analyze_matching_strategy(day_folder='7_24_data'):
    """
    Analyze the 20-minute matching strategy and its impact on data splitting
    """
    print(f"🔍 Analyzing 20-Minute Matching Strategy for {day_folder}")
    print("=" * 70)
    
    # Load the data
    learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
    test_path = f"dataset/d_data_split/{day_folder}/test.csv"
    
    learning_df = pd.read_csv(learning_path)
    test_df = pd.read_csv(test_path)
    
    # Convert timestamps
    learning_df['timestamp'] = pd.to_datetime(learning_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Combine for analysis
    full_df = pd.concat([
        learning_df.assign(split='learning'),
        test_df.assign(split='test')
    ]).sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total entries: {len(full_df):,}")
    print(f"   Learning entries: {len(learning_df):,} ({len(learning_df)/len(full_df)*100:.1f}%)")
    print(f"   Test entries: {len(test_df):,} ({len(test_df)/len(full_df)*100:.1f}%)")
    print(f"   Time range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
    
    # Analyze locations
    print(f"\n📍 Location Analysis:")
    if 'location' in full_df.columns:
        total_locations = full_df['location'].nunique()
        learning_locations = learning_df['location'].nunique()
        test_locations = test_df['location'].nunique()
        
        print(f"   Total unique locations: {total_locations}")
        print(f"   Learning locations: {learning_locations}")
        print(f"   Test locations: {test_locations}")
        
        # Check location overlap
        learning_locs = set(learning_df['location'].unique())
        test_locs = set(test_df['location'].unique())
        overlap_locs = learning_locs.intersection(test_locs)
        
        if overlap_locs:
            print(f"   🔄 Location overlap: {len(overlap_locs)} locations in both splits")
            print(f"      This means: Same locations at different times")
        else:
            print(f"   🚫 No location overlap: Different locations in each split")
        
        # Location distribution
        print(f"\n   Top locations by entry count:")
        location_counts = full_df['location'].value_counts()
        for i, (loc, count) in enumerate(location_counts.head(5).items()):
            learning_count = len(learning_df[learning_df['location'] == loc])
            test_count = len(test_df[test_df['location'] == loc])
            print(f"     {i+1}. {loc}: {count:,} total ({learning_count:,} learning, {test_count:,} test)")
    
    # Analyze images per location and time
    print(f"\n🖼️  Image-Location-Time Analysis:")
    
    # Group by location and analyze temporal distribution
    if 'location' in full_df.columns:
        location_time_analysis = []
        
        for location in full_df['location'].unique():
            loc_data = full_df[full_df['location'] == location]
            loc_learning = loc_data[loc_data['split'] == 'learning']
            loc_test = loc_data[loc_data['split'] == 'test']
            
            location_time_analysis.append({
                'location': location,
                'total_entries': len(loc_data),
                'learning_entries': len(loc_learning),
                'test_entries': len(loc_test),
                'unique_images': loc_data['image_filename'].nunique(),
                'learning_images': loc_learning['image_filename'].nunique(),
                'test_images': loc_test['image_filename'].nunique(),
                'time_span': loc_data['timestamp'].max() - loc_data['timestamp'].min(),
                'first_time': loc_data['timestamp'].min(),
                'last_time': loc_data['timestamp'].max()
            })
        
        location_df = pd.DataFrame(location_time_analysis)
        location_df = location_df.sort_values('total_entries', ascending=False)
        
        print(f"   Location-wise breakdown:")
        for _, row in location_df.head(5).iterrows():
            print(f"     {row['location']}:")
            print(f"       Entries: {row['total_entries']:,} ({row['learning_entries']:,} learning, {row['test_entries']:,} test)")
            print(f"       Images: {row['unique_images']} ({row['learning_images']} learning, {row['test_images']} test)")
            print(f"       Time span: {row['time_span']}")
    
    # Analyze temporal gaps
    print(f"\n⏰ Temporal Gap Analysis:")
    
    # Check gap between learning and test
    learning_end = learning_df['timestamp'].max()
    test_start = test_df['timestamp'].min()
    temporal_gap = test_start - learning_end
    
    print(f"   Learning period ends: {learning_end}")
    print(f"   Test period starts: {test_start}")
    print(f"   Temporal gap: {temporal_gap}")
    
    if temporal_gap > timedelta(hours=1):
        print(f"   ✅ Clean temporal separation (gap > 1 hour)")
    elif temporal_gap > timedelta(minutes=0):
        print(f"   ⚠️  Small temporal gap ({temporal_gap})")
    else:
        print(f"   🚨 Temporal overlap! Test starts before learning ends")
    
    # Check for potential data leakage
    print(f"\n🔍 Data Leakage Analysis:")
    
    # Check if same image appears in both splits
    learning_images = set(learning_df['image_filename'].unique())
    test_images = set(test_df['image_filename'].unique())
    image_overlap = learning_images.intersection(test_images)
    
    if image_overlap:
        print(f"   🚨 IMAGE LEAKAGE: {len(image_overlap)} images in both splits!")
        print(f"      Overlapping images: {list(image_overlap)[:5]}")
    else:
        print(f"   ✅ No image overlap")
    
    # Check for temporal proximity issues (within 20 minutes)
    print(f"\n   Checking 20-minute proximity issues...")
    proximity_issues = 0
    
    for _, test_row in test_df.head(100).iterrows():  # Check first 100 test entries
        test_time = test_row['timestamp']
        test_location = test_row.get('location', 'unknown')
        
        # Find learning entries within 20 minutes at same location
        if 'location' in learning_df.columns:
            nearby_learning = learning_df[
                (learning_df['location'] == test_location) &
                (abs(learning_df['timestamp'] - test_time) <= timedelta(minutes=20))
            ]
        else:
            nearby_learning = learning_df[
                abs(learning_df['timestamp'] - test_time) <= timedelta(minutes=20)
            ]
        
        if len(nearby_learning) > 0:
            proximity_issues += 1
    
    if proximity_issues > 0:
        print(f"   ⚠️  Found {proximity_issues} test entries with learning data within 20 minutes")
        print(f"      This might cause data leakage")
    else:
        print(f"   ✅ No temporal proximity issues found")
    
    return {
        'full_df': full_df,
        'learning_df': learning_df,
        'test_df': test_df,
        'temporal_gap': temporal_gap,
        'image_overlap': image_overlap,
        'proximity_issues': proximity_issues
    }

def recommend_split_modifications(analysis_results):
    """
    Recommend modifications to the splitting strategy
    """
    print(f"\n💡 Split Strategy Recommendations:")
    print("=" * 50)
    
    temporal_gap = analysis_results['temporal_gap']
    image_overlap = analysis_results['image_overlap']
    proximity_issues = analysis_results['proximity_issues']
    learning_df = analysis_results['learning_df']
    test_df = analysis_results['test_df']
    
    # Assess current split quality
    learning_images = learning_df['image_filename'].nunique()
    test_images = test_df['image_filename'].nunique()
    
    print(f"🎯 Current Split Assessment:")
    
    # Temporal separation
    if temporal_gap > timedelta(hours=24):
        print(f"   ✅ Excellent temporal separation ({temporal_gap})")
        temporal_score = "excellent"
    elif temporal_gap > timedelta(hours=1):
        print(f"   ✅ Good temporal separation ({temporal_gap})")
        temporal_score = "good"
    elif temporal_gap > timedelta(minutes=0):
        print(f"   ⚠️  Marginal temporal separation ({temporal_gap})")
        temporal_score = "marginal"
    else:
        print(f"   🚨 Poor temporal separation ({temporal_gap})")
        temporal_score = "poor"
    
    # Data leakage
    if len(image_overlap) == 0 and proximity_issues == 0:
        print(f"   ✅ No data leakage detected")
        leakage_score = "none"
    elif len(image_overlap) == 0 and proximity_issues < 10:
        print(f"   ⚠️  Minor temporal proximity issues")
        leakage_score = "minor"
    else:
        print(f"   🚨 Significant data leakage risk")
        leakage_score = "major"
    
    # Dataset size
    if learning_images >= 30:
        print(f"   ✅ Sufficient training images ({learning_images})")
        size_score = "sufficient"
    elif learning_images >= 15:
        print(f"   ⚠️  Limited training images ({learning_images})")
        size_score = "limited"
    else:
        print(f"   🚨 Very few training images ({learning_images})")
        size_score = "insufficient"
    
    # Recommendations based on assessment
    print(f"\n🔧 Recommended Actions:")
    
    if temporal_score in ["excellent", "good"] and leakage_score == "none" and size_score != "insufficient":
        print(f"   ✅ KEEP CURRENT SPLIT")
        print(f"      Your temporal split is working well!")
        print(f"      Good for time-series air quality prediction")
        
    elif size_score == "insufficient":
        print(f"   🔄 SWITCH TO CROSS-VALIDATION")
        print(f"      Too few training images for reliable single split")
        print(f"      Recommended: 5-fold cross-validation on all data")
        
    elif leakage_score == "major":
        print(f"   🔧 FIX DATA LEAKAGE")
        print(f"      Remove overlapping images or increase temporal gap")
        print(f"      Consider 30-minute minimum gap instead of 20 minutes")
        
    elif temporal_score == "poor":
        print(f"   📍 CONSIDER SPATIAL SPLIT")
        print(f"      Poor temporal separation suggests spatial split might be better")
        print(f"      Split by location instead of time")
    
    # Specific recommendations
    print(f"\n🎯 Specific Modifications:")
    
    if size_score in ["limited", "insufficient"]:
        print(f"   1. 🔄 Implement Cross-Validation:")
        print(f"      - Combine learning + test data")
        print(f"      - Use 5-fold CV with temporal awareness")
        print(f"      - Each fold: chronological split within fold")
    
    if 'location' in learning_df.columns:
        unique_locations = pd.concat([learning_df, test_df])['location'].nunique()
        if unique_locations >= 5:
            print(f"   2. 📍 Try Location-Based Split:")
            print(f"      - {unique_locations} locations available")
            print(f"      - Split by location for spatial generalization")
            print(f"      - 80% locations for training, 20% for testing")
    
    if proximity_issues > 0:
        print(f"   3. ⏰ Increase Temporal Buffer:")
        print(f"      - Current: 20-minute matching window")
        print(f"      - Suggested: 30-60 minute minimum gap between splits")
        print(f"      - This reduces temporal data leakage")
    
    if learning_images < 20:
        print(f"   4. 📈 Data Augmentation Strategy:")
        print(f"      - You already have patch-based augmentation")
        print(f"      - Consider temporal augmentation (slight time shifts)")
        print(f"      - Use more aggressive spatial augmentation")

def create_improved_split_strategy(day_folder='7_24_data'):
    """
    Create an improved split strategy based on analysis
    """
    print(f"\n🚀 Creating Improved Split Strategy for {day_folder}")
    print("=" * 50)
    
    # Load original matched data
    day_short = day_folder.replace('_data', '')
    matched_path = f"dataset/c_matched_spatio_temporal_data/matched_{day_short}.csv"
    
    if not os.path.exists(matched_path):
        print(f"❌ Matched data not found: {matched_path}")
        return
    
    df = pd.read_csv(matched_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Original Data: {len(df)} entries, {df['image_filename'].nunique()} unique images")
    
    # Strategy 1: Cross-validation approach
    print(f"\n🔄 Strategy 1: Temporal Cross-Validation")
    unique_images = df['image_filename'].unique()
    
    if len(unique_images) < 25:
        print(f"   Recommended for your dataset ({len(unique_images)} images)")
        print(f"   Implementation:")
        print(f"     - 5 folds with temporal awareness")
        print(f"     - Each fold maintains chronological order")
        print(f"     - Better statistical reliability")
    
    # Strategy 2: Location-based split
    if 'location' in df.columns:
        unique_locations = df['location'].nunique()
        print(f"\n📍 Strategy 2: Location-Based Split")
        print(f"   Available locations: {unique_locations}")
        
        if unique_locations >= 5:
            print(f"   Recommended split:")
            location_counts = df['location'].value_counts()
            train_locations = int(unique_locations * 0.8)
            print(f"     - Train locations: {train_locations}")
            print(f"     - Test locations: {unique_locations - train_locations}")
            print(f"     - Better for spatial generalization")
    
    # Strategy 3: Hybrid approach
    print(f"\n🎯 Strategy 3: Hybrid Temporal-Spatial")
    print(f"   Best of both worlds:")
    print(f"     - Primary split by location (spatial generalization)")
    print(f"     - Secondary split by time within locations")
    print(f"     - Cross-validation for robust evaluation")

if __name__ == "__main__":
    analysis_results = analyze_matching_strategy('7_24_data')
    recommend_split_modifications(analysis_results)
    create_improved_split_strategy('7_24_data')
