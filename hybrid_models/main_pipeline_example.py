#!/usr/bin/env python3
"""
COMPLETE 5-FOLD TIMESERIES PIPELINE EXAMPLE
============================================

This script demonstrates how your LSTM component integrates with:
- 5-fold TimeSeriesSplit with expanding window
- 80% learning set / 20% hold-out test set
- Placeholder spatial component (since CapsNet not ready)
- LightGBM meta-learner
- Full hybrid model evaluation

Usage:
    python main_pipeline_example.py
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Placeholder for CapsNet
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your LSTM component
from i_components.lstm.lstm_temporal_feature_generator import (
    LSTMTemporalFeatureGenerator, 
    TemporalDataLoader, 
    hyperparameter_tuning_lstm
)

class PlaceholderSpatialFeatureGenerator:
    """
    Placeholder spatial feature generator until CapsNet is ready
    Simulates spatial feature extraction from images
    """
    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        
    def train_and_extract_features(self, train_images, train_targets, val_images, val_targets):
        """
        Placeholder for spatial feature extraction
        In real implementation, this would be CapsNet or CNN
        """
        # Simulate spatial feature extraction
        # In reality, this would process actual satellite images
        np.random.seed(42)  # For reproducible results
        
        # Generate synthetic spatial features based on data size
        train_spatial_features = np.random.randn(len(train_images), self.feature_dim)
        val_spatial_features = np.random.randn(len(val_images), self.feature_dim)
        
        # Add some correlation with targets to make it realistic
        train_spatial_features[:, 0] += train_targets[:len(train_spatial_features)] * 0.1
        val_spatial_features[:, 0] += val_targets[:len(val_spatial_features)] * 0.1
        
        print(f"  Spatial Debug - Train features shape: {train_spatial_features.shape}")
        print(f"  Spatial Debug - Val features shape: {val_spatial_features.shape}")
        
        return train_spatial_features, val_spatial_features

class HybridModelPipeline:
    """
    Complete pipeline for hybrid air quality prediction models
    Integrates LSTM temporal + spatial + LightGBM meta-learner
    """
    def __init__(self, model_type='lstm_spatial_lightgbm'):
        self.model_type = model_type
        self.results = {}
        
    def load_image_data_placeholder(self, day, indices):
        """
        Placeholder for loading image data
        In real implementation, this would load actual satellite images
        """
        # Return dummy image identifiers
        return np.arange(len(indices))
    
    def run_5_fold_cv(self, day='7_24', max_lstm_combinations=5):
        """
        Run complete 5-fold cross-validation pipeline
        """
        print("="*80)
        print(f"HYBRID MODEL PIPELINE: {self.model_type.upper()}")
        print("="*80)
        print(f"Day: {day}")
        print(f"Pipeline: LSTM Temporal + Spatial + LightGBM Meta-learner")
        print("="*80)
        
        # PHASE 1: Hyperparameter Tuning
        print("\n🔧 PHASE 1: HYPERPARAMETER TUNING")
        print("-" * 50)
        
        print(f"Tuning LSTM parameters for {day}...")
        best_lstm_params = hyperparameter_tuning_lstm(day, max_combinations=max_lstm_combinations)
        print(f"✅ LSTM tuning complete")
        
        # PHASE 2: Data Loading and Splitting
        print("\n📊 PHASE 2: DATA LOADING AND SPLITTING")
        print("-" * 50)
        
        # Load temporal data using your component
        data_loader = TemporalDataLoader()
        temporal_data, targets, feature_names = data_loader.load_temporal_data(day)
        
        print(f"📈 Loaded {day} data:")
        print(f"  Total samples: {len(temporal_data)}")
        print(f"  Temporal features: {feature_names}")
        print(f"  Target: pm2.5")
        
        # Split into 80% learning set, 20% hold-out test (chronological)
        n_total = len(temporal_data)
        n_learning = int(n_total * 0.8)   # 80% for learning (5-fold CV)
        # Remaining 20% is holdout test
        
        learning_temporal = temporal_data[:n_learning]
        learning_targets = targets[:n_learning]
        holdout_temporal = temporal_data[n_learning:]
        holdout_targets = targets[n_learning:]
        
        print(f"📊 Data split (chronological):")
        print(f"  Learning set (80%): {len(learning_temporal)} samples [0:{n_learning}]")
        print(f"  Hold-out test (20%): {len(holdout_temporal)} samples [{n_learning}:{n_total}]")
        print(f"  → 5-fold CV will be applied to the 80% learning set")
        print(f"  → Each fold will internally split into ~70% train + ~10% val")
        
        # PHASE 3: 5-Fold Cross-Validation
        print("\n🔄 PHASE 3: 5-FOLD CROSS-VALIDATION")
        print("-" * 50)
        
        # Initialize components
        lstm_generator = LSTMTemporalFeatureGenerator(best_lstm_params)
        spatial_generator = PlaceholderSpatialFeatureGenerator(feature_dim=64)
        
        # TimeSeriesSplit with expanding window on 80% learning set
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        
        print(f"🎯 Running 5-fold TimeSeriesSplit on {len(learning_temporal)} learning samples...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(learning_temporal)):
            print(f"\n📈 FOLD {fold + 1}/5")
            print(f"{'='*30}")
            
            # Get data for this fold FROM 80% LEARNING SET
            fold_train_temporal = learning_temporal[train_idx]
            fold_val_temporal = learning_temporal[val_idx]  # From learning set (expanding window)
            fold_train_targets = learning_targets[train_idx]
            fold_val_targets = learning_targets[val_idx]
            
            # Load corresponding image data (placeholder)
            fold_train_images = self.load_image_data_placeholder(day, train_idx)
            fold_val_images = self.load_image_data_placeholder(day, val_idx)
            
            print(f"  📊 Fold {fold + 1} data:")
            print(f"    Train samples: {len(fold_train_temporal)} (expanding window)")
            print(f"    Val samples: {len(fold_val_temporal)}")
            
            # STEP 1: Get timesteps for proper alignment
            print(f"  🔗 Step 1: Calculate Feature Alignment")
            timesteps = best_lstm_params.get('timesteps', 60)
            
            # CRITICAL: Use same indices for both LSTM and spatial features
            # LSTM reduces data by 'timesteps' due to sequence creation
            # So we need to align all data to match LSTM's effective indices
            
            # Align input data first (before feature extraction)
            fold_train_temporal_aligned = fold_train_temporal[timesteps:]
            fold_val_temporal_aligned = fold_val_temporal[timesteps:]
            fold_train_targets_aligned = fold_train_targets[timesteps:]
            fold_val_targets_aligned = fold_val_targets[timesteps:]
            fold_train_images_aligned = fold_train_images[timesteps:]
            fold_val_images_aligned = fold_val_images[timesteps:]
            
            print(f"    Original train samples: {len(train_temporal)}")
            print(f"    Aligned train samples: {len(train_temporal_aligned)} (after -{timesteps} for sequences)")
            print(f"    Original val samples: {len(val_temporal)}")
            print(f"    Aligned val samples: {len(val_temporal_aligned)} (after -{timesteps} for sequences)")
            
            # STEP 2: Extract temporal features using your LSTM component
            print(f"  🧠 Step 2: LSTM Temporal Feature Extraction")
            # Pass original data to LSTM (it will create sequences internally)
            train_temp_features, val_temp_features, _, _ = lstm_generator.train_and_extract_features(
                train_temporal, train_targets, val_temporal, val_targets
            )
            
            # STEP 3: Extract spatial features using aligned data
            print(f"  �️  Step 3: Spatial Feature Extraction (using aligned indices)")
            # Pass aligned data to spatial extractor (same effective indices as LSTM output)
            train_spatial_features, val_spatial_features = spatial_generator.train_and_extract_features(
                fold_train_images_aligned, fold_train_targets_aligned, fold_val_images_aligned, fold_val_targets_aligned
            )
            
            print(f"    Aligned train features: {len(train_temp_features)} samples")
            print(f"    Aligned val features: {len(val_temp_features)} samples")
            
            # STEP 4: Feature fusion (both features are now properly aligned)
            print(f"  🔄 Step 4: Feature Fusion")
            train_combined = np.concatenate([train_temp_features, train_spatial_features], axis=1)
            val_combined = np.concatenate([val_temp_features, val_spatial_features], axis=1)
            
            print(f"    Combined train features: {train_combined.shape}")
            print(f"    Combined val features: {val_combined.shape}")
            
            # Verify alignment
            print(f"    ✅ Alignment verification:")
            print(f"      LSTM features: {train_temp_features.shape[0]} samples")
            print(f"      Spatial features: {train_spatial_features.shape[0]} samples")
            print(f"      Aligned targets: {len(fold_train_targets_aligned)} samples")
            print(f"      All should match: {train_temp_features.shape[0] == train_spatial_features.shape[0] == len(fold_train_targets_aligned)}")
            
            # STEP 5: Meta-learner training (LightGBM)
            print(f"  🌳 Step 5: LightGBM Meta-learner Training")
            lightgbm_model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            lightgbm_model.fit(train_combined, fold_train_targets_aligned)
            
            # STEP 6: Validation
            print(f"  📊 Step 6: Fold Validation")
            val_predictions = lightgbm_model.predict(val_combined)
            
            # Calculate metrics
            fold_rmse = np.sqrt(mean_squared_error(fold_val_targets_aligned, val_predictions))
            fold_mae = mean_absolute_error(fold_val_targets_aligned, val_predictions)
            fold_r2 = r2_score(fold_val_targets_aligned, val_predictions)
            
            fold_result = {
                'fold': fold + 1,
                'train_samples': len(train_temp_features),
                'val_samples': len(val_temp_features),
                'rmse': fold_rmse,
                'mae': fold_mae,
                'r2': fold_r2
            }
            
            fold_results.append(fold_result)
            
            print(f"    ✅ Fold {fold + 1} Results:")
            print(f"       RMSE: {fold_rmse:.4f}")
            print(f"       MAE: {fold_mae:.4f}")
            print(f"       R²: {fold_r2:.4f}")
        
        # PHASE 4: Cross-validation results
        print("\n📈 PHASE 4: CROSS-VALIDATION RESULTS")
        print("-" * 50)
        
        # Calculate statistics
        fold_rmses = [result['rmse'] for result in fold_results]
        fold_maes = [result['mae'] for result in fold_results]
        fold_r2s = [result['r2'] for result in fold_results]
        
        cv_rmse_mean = np.mean(fold_rmses)
        cv_rmse_std = np.std(fold_rmses)
        cv_mae_mean = np.mean(fold_maes)
        cv_mae_std = np.std(fold_maes)
        cv_r2_mean = np.mean(fold_r2s)
        cv_r2_std = np.std(fold_r2s)
        
        print(f"📊 {self.model_type.upper()} Performance on {day}:")
        print(f"┌{'─'*60}┐")
        print(f"│ FOLD │ TRAIN SAMPLES │ VAL SAMPLES │   RMSE   │   MAE    │   R²     │")
        print(f"├{'─'*60}┤")
        for result in fold_results:
            print(f"│  {result['fold']}   │    {result['train_samples']:5d}     │    {result['val_samples']:4d}     │ {result['rmse']:8.4f} │ {result['mae']:8.4f} │ {result['r2']:8.4f} │")
        print(f"├{'─'*60}┤")
        print(f"│ AVG  │       -       │      -      │ {cv_rmse_mean:8.4f} │ {cv_mae_mean:8.4f} │ {cv_r2_mean:8.4f} │")
        print(f"│ STD  │       -       │      -      │ {cv_rmse_std:8.4f} │ {cv_mae_std:8.4f} │ {cv_r2_std:8.4f} │")
        print(f"└{'─'*60}┘")
        
        # Store results
        self.results[day] = {
            'cv_results': fold_results,
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std,
            'cv_mae_mean': cv_mae_mean,
            'cv_mae_std': cv_mae_std,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'best_lstm_params': best_lstm_params
        }
        
        # PHASE 5: Final evaluation on hold-out test set
        print("\n🔐 PHASE 5: HOLD-OUT TEST SET EVALUATION")
        print("-" * 50)
        
        print("🎯 Training final model on full 80% learning set...")
        
        # Train final model on full learning set
        final_lstm_generator = LSTMTemporalFeatureGenerator(best_lstm_params)
        final_spatial_generator = PlaceholderSpatialFeatureGenerator(feature_dim=64)
        
        # Use full learning set for final training
        holdout_images = self.load_image_data_placeholder(day, np.arange(len(holdout_temporal)))
        learning_images = self.load_image_data_placeholder(day, np.arange(len(learning_temporal)))
        
        # Align data first (same approach as CV folds)
        timesteps = best_lstm_params.get('timesteps', 60)
        learning_images_aligned = learning_images[timesteps:]
        holdout_images_aligned = holdout_images[timesteps:]
        learning_targets_aligned = learning_targets[timesteps:]
        holdout_targets_aligned = holdout_targets[timesteps:]
        
        # Extract features using aligned data
        final_train_temp_features, holdout_temp_features, _, _ = final_lstm_generator.train_and_extract_features(
            learning_temporal, learning_targets, holdout_temporal, holdout_targets
        )
        
        final_train_spatial_features, holdout_spatial_features = final_spatial_generator.train_and_extract_features(
            learning_images_aligned, learning_targets_aligned, holdout_images_aligned, holdout_targets_aligned
        )
        
        # Features are now properly aligned (no need for additional alignment)
        print(f"    ✅ Final alignment verification:")
        print(f"      LSTM features: {final_train_temp_features.shape[0]} samples")
        print(f"      Spatial features: {final_train_spatial_features.shape[0]} samples")
        print(f"      Aligned targets: {len(learning_targets_aligned)} samples")
        
        # Combine features
        final_train_combined = np.concatenate([final_train_temp_features, final_train_spatial_features], axis=1)
        holdout_combined = np.concatenate([holdout_temp_features, holdout_spatial_features], axis=1)
        
        # Train final model
        final_model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        final_model.fit(final_train_combined, learning_targets_aligned)
        
        # Evaluate on hold-out test set
        holdout_predictions = final_model.predict(holdout_combined)
        
        holdout_rmse = np.sqrt(mean_squared_error(holdout_targets_aligned, holdout_predictions))
        holdout_mae = mean_absolute_error(holdout_targets_aligned, holdout_predictions)
        holdout_r2 = r2_score(holdout_targets_aligned, holdout_predictions)
        
        print(f"🎯 Final Hold-out Test Results:")
        print(f"  RMSE: {holdout_rmse:.4f}")
        print(f"  MAE: {holdout_mae:.4f}")
        print(f"  R²: {holdout_r2:.4f}")
        
        # Store final results
        self.results[day]['holdout_rmse'] = holdout_rmse
        self.results[day]['holdout_mae'] = holdout_mae
        self.results[day]['holdout_r2'] = holdout_r2
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        
        return self.results[day]

def main():
    """
    Main execution function
    """
    print("🚀 STARTING HYBRID AIR QUALITY PREDICTION PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = HybridModelPipeline(model_type='lstm_spatial_lightgbm')
    
    # Run for each day
    days = ['7_24', '10_19', '11_10']
    
    for day in days:
        print(f"\n🗓️  Processing {day} data...")
        try:
            results = pipeline.run_5_fold_cv(day=day, max_lstm_combinations=2)
            print(f"✅ {day} processing complete")
            
        except Exception as e:
            print(f"❌ Error processing {day}: {str(e)}")
            continue
    
    # Summary results
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY RESULTS")
    print("="*80)
    
    if pipeline.results:
        print(f"{'DAY':<10} {'CV RMSE':<15} {'CV MAE':<15} {'CV R²':<15} {'HOLDOUT RMSE':<15}")
        print("-" * 70)
        
        for day, results in pipeline.results.items():
            cv_rmse = f"{results['cv_rmse_mean']:.4f}±{results['cv_rmse_std']:.4f}"
            cv_mae = f"{results['cv_mae_mean']:.4f}±{results['cv_mae_std']:.4f}"
            cv_r2 = f"{results['cv_r2_mean']:.4f}±{results['cv_r2_std']:.4f}"
            holdout_rmse = f"{results['holdout_rmse']:.4f}"
            
            print(f"{day:<10} {cv_rmse:<15} {cv_mae:<15} {cv_r2:<15} {holdout_rmse:<15}")
    
    print("\n🎉 ALL PROCESSING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
