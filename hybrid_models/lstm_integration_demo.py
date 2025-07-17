#!/usr/bin/env python3
"""
SIMPLIFIED LSTM INTEGRATION EXAMPLE
===================================

This shows exactly how your LSTM component works in the 5-fold pipeline
Focus: Understanding the LSTM component's role in expanding window CV

Your LSTM component does:
1. Takes temporal data (pm10, temperature, humidity) 
2. Creates sequences of length 'timesteps' (e.g., 60)
3. Trains LSTM to predict pm2.5
4. Extracts 32-dimensional temporal features
5. Returns features for meta-learner (LightGBM)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Import your LSTM component
from i_components.lstm.lstm_temporal_feature_generator import (
    LSTMTemporalFeatureGenerator, 
    TemporalDataLoader
)

def demonstrate_lstm_in_5fold():
    """
    Simple demonstration of how your LSTM component works in 5-fold CV
    """
    print("🔍 LSTM COMPONENT IN 5-FOLD CV - STEP BY STEP")
    print("="*60)
    
    # Step 1: Load data
    print("\n STEP 1: Data Loading")
    print("-" * 30)
    
    data_loader = TemporalDataLoader()
    temporal_data, targets, feature_names = data_loader.load_temporal_data('7_24')
    
    print(f"Raw data shape: {temporal_data.shape}")
    print(f"Features: {feature_names}")
    print(f"Total samples: {len(temporal_data)}")
    
    # Step 2: 80/20 split
    print("\n📊 STEP 2: 80/20 Split")
    print("-" * 30)
    
    n_total = len(temporal_data)
    n_learning = int(n_total * 0.8)
    
    learning_temporal = temporal_data[:n_learning]
    learning_targets = targets[:n_learning]
    holdout_temporal = temporal_data[n_learning:]
    holdout_targets = targets[n_learning:]
    
    print(f"Learning set (80%): {len(learning_temporal)} samples")
    print(f"Hold-out test (20%): {len(holdout_temporal)} samples")
    
    # Step 3: Initialize your LSTM component
    print("\n🧠 STEP 3: Initialize LSTM Component")
    print("-" * 30)
    
    # Use default parameters (in real usage, load from hyperparameter tuning)
    lstm_params = {
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,  # Reduced for demo
        'timesteps': 30,  # Reduced for demo
    }
    
    lstm_generator = LSTMTemporalFeatureGenerator(lstm_params)
    print(f"LSTM parameters: {lstm_params}")
    
    # Step 4: 5-fold cross-validation
    print("\n🔄 STEP 4: 5-Fold Cross-Validation")
    print("-" * 30)
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(learning_temporal)):
        print(f"\n📈 FOLD {fold + 1}/5")
        print(f"{'='*20}")
        
        # Get fold data
        train_temporal = learning_temporal[train_idx]
        val_temporal = learning_temporal[val_idx]
        train_targets = learning_targets[train_idx]
        val_targets = learning_targets[val_idx]
        
        print(f"Train samples: {len(train_temporal)} (expanding window)")
        print(f"Val samples: {len(val_temporal)}")
        
        # KEY STEP: Use your LSTM component
        print(f"🧠 Running LSTM component...")
        
        # This is where your LSTM component does its magic:
        # 1. Creates temporal sequences
        # 2. Trains LSTM model
        # 3. Extracts 32D temporal features
        train_temp_features, val_temp_features, _, _ = lstm_generator.train_and_extract_features(
            train_temporal, train_targets, val_temporal, val_targets
        )
        
        print(f"✅ LSTM extracted features:")
        print(f"  Train temporal features: {train_temp_features.shape}")
        print(f"  Val temporal features: {val_temp_features.shape}")
        
        # In a real hybrid model, you'd combine with spatial features:
        # train_combined = np.concatenate([train_temp_features, train_spatial_features], axis=1)
        # val_combined = np.concatenate([val_temp_features, val_spatial_features], axis=1)
        
        # For demo, use only temporal features
        train_combined = train_temp_features
        val_combined = val_temp_features
        
        # Align targets (account for sequence reduction)
        timesteps = lstm_params['timesteps']
        train_targets_aligned = train_targets[timesteps:]
        val_targets_aligned = val_targets[timesteps:]
        
        # Meta-learner (LightGBM)
        print(f"🌳 Training LightGBM meta-learner...")
        
        meta_learner = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        meta_learner.fit(train_combined, train_targets_aligned)
        val_predictions = meta_learner.predict(val_combined)
        
        # Calculate performance
        fold_rmse = np.sqrt(mean_squared_error(val_targets_aligned, val_predictions))
        fold_results.append(fold_rmse)
        
        print(f"📊 Fold {fold + 1} RMSE: {fold_rmse:.4f}")
        
        # Show what's happening inside
        print(f"  🔍 Internal details:")
        print(f"    Original train size: {len(train_temporal)}")
        print(f"    After sequence creation: {len(train_combined)} (-{timesteps} for sequences)")
        print(f"    LSTM feature dimension: {train_combined.shape[1]} (32D temporal)")
        print(f"    Meta-learner input: {train_combined.shape}")
    
    # Step 5: Results
    print("\n📊 STEP 5: Cross-Validation Results")
    print("-" * 30)
    
    cv_rmse_mean = np.mean(fold_results)
    cv_rmse_std = np.std(fold_results)
    
    print(f"📈 5-Fold CV Performance:")
    print(f"  RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    print(f"  Individual fold RMSEs: {[f'{rmse:.4f}' for rmse in fold_results]}")
    
    # Summary
    print("\n🎯 SUMMARY: What your LSTM component does")
    print("="*60)
    print("""
Your LSTM component (lstm_temporal_feature_generator.py) is a TOOL that:

1. 📥 INPUT: Takes temporal data (pm10, temperature, humidity)
2. 🔄 PROCESS: Creates sequences, trains LSTM, extracts features  
3. 📤 OUTPUT: Returns 32-dimensional temporal features

In the 5-fold pipeline:
- Each fold calls your LSTM component
- LSTM creates sequences from temporal data
- LSTM trains on that fold's training data
- LSTM extracts 32D features for meta-learner
- Features get combined with spatial features
- LightGBM uses combined features for final prediction

Your LSTM component is like a specialized feature extractor!
It converts raw temporal data into meaningful 32D representations.
""")
    
    return fold_results

def show_expanding_window_behavior():
    """
    Show how expanding window works in TimeSeriesSplit
    """
    print("\n🔍 EXPANDING WINDOW BEHAVIOR")
    print("="*60)
    
    # Demo with small dataset
    demo_data = np.arange(100)  # 100 samples
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Sample data: 100 time-ordered samples")
    print("5-fold TimeSeriesSplit with expanding window:")
    print()
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(demo_data)):
        train_size = len(train_idx)
        val_size = len(val_idx)
        train_start = train_idx[0]
        train_end = train_idx[-1]
        val_start = val_idx[0]
        val_end = val_idx[-1]
        
        print(f"Fold {fold + 1}:")
        print(f"  Train: samples {train_start:2d}-{train_end:2d} (size: {train_size:2d}) - EXPANDING")
        print(f"  Val:   samples {val_start:2d}-{val_end:2d} (size: {val_size:2d}) - FUTURE DATA")
        print()
    
    print("📈 Notice how training set GROWS each fold (expanding window)")
    print("🔮 Validation set is always future data (realistic prediction)")

def main():
    """
    Main demonstration
    """
    print("🚀 UNDERSTANDING YOUR LSTM COMPONENT IN 5-FOLD CV")
    print("="*80)
    
    # Show expanding window first
    show_expanding_window_behavior()
    
    # Demonstrate LSTM integration
    demonstrate_lstm_in_5fold()
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("="*80)
    print("Now you understand how your LSTM component fits into the bigger picture!")

if __name__ == "__main__":
    main()
