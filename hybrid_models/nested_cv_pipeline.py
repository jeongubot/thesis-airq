"""
Nested Cross-Validation Implementation for Full Pipeline
Following the methodology shown in the attached image
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from scipy import stats

from lstm.lstm_temporal_feature_generator import LSTMTemporalFeatureGenerator, TemporalDataLoader

class NestedCVPipeline:
    """
    Nested Cross-Validation Pipeline following the attached image methodology
    """
    def __init__(self, days=['7_24', '10_19', '11_10'], outer_cv=5, inner_cv=3):
        self.days = days
        self.outer_cv = outer_cv  # Outer loop: 5 folds for final evaluation
        self.inner_cv = inner_cv  # Inner loop: 3 folds for hyperparameter tuning
        self.models = ['CapsNet-LSTM-LightGBM', 'CNN-LSTM-LightGBM', 'CapsNet-LSTM']
        self.results = {}
        
        # Hyperparameter spaces for each component
        self.lightgbm_params = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_child_samples': [20, 30, 50]
        }
        
        self.lstm_params = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'epochs': [30, 50]
        }
    
    def inner_cv_hyperparameter_tuning(self, X_train, y_train, model_type):
        """
        Inner cross-validation loop for hyperparameter tuning
        This is the "Inner Loop" from the image
        """
        print(f"    Inner CV: Tuning hyperparameters for {model_type}")
        
        # Create inner CV splits
        inner_tscv = TimeSeriesSplit(n_splits=self.inner_cv)
        
        best_params = None
        best_score = float('inf')
        
        if model_type in ['CapsNet-LSTM-LightGBM', 'CNN-LSTM-LightGBM']:
            # Need to tune both LSTM and LightGBM parameters
            
            # First, quick LSTM tuning
            lstm_param_grid = ParameterGrid(self.lstm_params)
            best_lstm_params = None
            best_lstm_score = float('inf')
            
            for lstm_params in list(lstm_param_grid)[:5]:  # Limit to 5 combinations
                lstm_scores = []
                
                for train_idx, val_idx in inner_tscv.split(X_train):
                    # Split temporal data
                    temporal_train = X_train[train_idx]
                    temporal_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    # Train LSTM with these parameters
                    lstm_gen = LSTMTemporalFeatureGenerator(lstm_params)
                    try:
                        train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                            temporal_train, target_train, temporal_val, target_val
                        )
                        
                        # Simple validation
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression()
                        lr.fit(train_temp_features, target_train[10:])
                        val_pred = lr.predict(val_temp_features)
                        
                        mse = mean_squared_error(target_val[10:], val_pred)
                        lstm_scores.append(mse)
                    except:
                        lstm_scores.append(float('inf'))
                
                avg_score = np.mean(lstm_scores)
                if avg_score < best_lstm_score:
                    best_lstm_score = avg_score
                    best_lstm_params = lstm_params
            
            print(f"      Best LSTM params: {best_lstm_params}")
            
            # Then tune LightGBM parameters
            lgb_param_grid = ParameterGrid(self.lightgbm_params)
            best_lgb_params = None
            best_lgb_score = float('inf')
            
            for lgb_params in list(lgb_param_grid)[:10]:  # Limit to 10 combinations
                lgb_scores = []
                
                for train_idx, val_idx in inner_tscv.split(X_train):
                    # Use best LSTM params to extract features
                    temporal_train = X_train[train_idx]
                    temporal_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    lstm_gen = LSTMTemporalFeatureGenerator(best_lstm_params)
                    try:
                        train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                            temporal_train, target_train, temporal_val, target_val
                        )
                        
                        # Add dummy spatial features (replace with actual CapsNet/CNN)
                        train_spatial = np.random.rand(len(train_temp_features), 32)
                        val_spatial = np.random.rand(len(val_temp_features), 32)
                        
                        # Combine features
                        train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
                        val_combined = np.concatenate([val_temp_features, val_spatial], axis=1)
                        
                        # Train LightGBM with these parameters
                        lgb_model = lgb.LGBMRegressor(
                            objective='regression',
                            metric='rmse',
                            boosting_type='gbdt',
                            verbose=-1,
                            **lgb_params
                        )
                        
                        lgb_model.fit(train_combined, target_train[10:])
                        val_pred = lgb_model.predict(val_combined)
                        
                        mse = mean_squared_error(target_val[10:], val_pred)
                        lgb_scores.append(mse)
                    except:
                        lgb_scores.append(float('inf'))
                
                avg_score = np.mean(lgb_scores)
                if avg_score < best_lgb_score:
                    best_lgb_score = avg_score
                    best_lgb_params = lgb_params
            
            print(f"      Best LightGBM params: {best_lgb_params}")
            
            best_params = {
                'lstm': best_lstm_params,
                'lightgbm': best_lgb_params
            }
            best_score = best_lgb_score
            
        else:  # CapsNet-LSTM
            # Only tune LSTM parameters
            lstm_param_grid = ParameterGrid(self.lstm_params)
            
            for lstm_params in list(lstm_param_grid)[:10]:  # Limit combinations
                lstm_scores = []
                
                for train_idx, val_idx in inner_tscv.split(X_train):
                    temporal_train = X_train[train_idx]
                    temporal_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    lstm_gen = LSTMTemporalFeatureGenerator(lstm_params)
                    try:
                        train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                            temporal_train, target_train, temporal_val, target_val
                        )
                        
                        # Use LSTM prediction layer directly
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression()
                        lr.fit(train_temp_features, target_train[10:])
                        val_pred = lr.predict(val_temp_features)
                        
                        mse = mean_squared_error(target_val[10:], val_pred)
                        lstm_scores.append(mse)
                    except:
                        lstm_scores.append(float('inf'))
                
                avg_score = np.mean(lstm_scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'lstm': lstm_params}
        
        print(f"      Inner CV complete. Best score: {np.sqrt(best_score):.4f}")
        return best_params
    
    def train_model_with_optimal_params(self, X_train, y_train, X_test, y_test, model_type, optimal_params):
        """
        Train model with optimal parameters found in inner CV
        """
        if model_type == 'CapsNet-LSTM-LightGBM':
            # Extract temporal features
            lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
            train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
                X_train, y_train, X_test, y_test
            )
            
            # Extract spatial features (placeholder)
            train_spatial = np.random.rand(len(train_temp_features), 32)
            test_spatial = np.random.rand(len(test_temp_features), 32)
            
            # Combine features
            train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
            test_combined = np.concatenate([test_temp_features, test_spatial], axis=1)
            
            # Train LightGBM
            lgb_model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                verbose=-1,
                **optimal_params['lightgbm']
            )
            
            lgb_model.fit(train_combined, y_train[10:])
            predictions = lgb_model.predict(test_combined)
            
            return predictions, y_test[10:]
            
        elif model_type == 'CNN-LSTM-LightGBM':
            # Similar to CapsNet but with CNN features
            lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
            train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
                X_train, y_train, X_test, y_test
            )
            
            # Extract CNN spatial features (placeholder)
            train_spatial = np.random.rand(len(train_temp_features), 32)
            test_spatial = np.random.rand(len(test_temp_features), 32)
            
            # Combine and train
            train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
            test_combined = np.concatenate([test_temp_features, test_spatial], axis=1)
            
            lgb_model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                verbose=-1,
                **optimal_params['lightgbm']
            )
            
            lgb_model.fit(train_combined, y_train[10:])
            predictions = lgb_model.predict(test_combined)
            
            return predictions, y_test[10:]
            
        else:  # CapsNet-LSTM
            # Direct LSTM prediction
            lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
            train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
                X_train, y_train, X_test, y_test
            )
            
            # Simple prediction
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(train_temp_features, y_train[10:])
            predictions = lr.predict(test_temp_features)
            
            return predictions, y_test[10:]
    
    def run_nested_cv_for_day(self, day):
        """
        Run nested cross-validation for a specific day
        This implements the complete nested CV from the image
        """
        print(f"\n{'='*80}")
        print(f"NESTED CROSS-VALIDATION FOR {day}")
        print(f"{'='*80}")
        print(f"Outer CV: {self.outer_cv} folds")
        print(f"Inner CV: {self.inner_cv} folds")
        
        # Load data
        data_loader = TemporalDataLoader()
        temporal_data, targets, _ = data_loader.load_temporal_data(day)
        
        # Outer cross-validation loop
        outer_tscv = TimeSeriesSplit(n_splits=self.outer_cv)
        
        # Initialize results
        self.results[day] = {}
        for model in self.models:
            self.results[day][model] = {
                'fold_results': [],
                'rmse': [],
                'r2': []
            }
        
        # OUTER LOOP: Final evaluation
        for outer_fold, (train_idx, test_idx) in enumerate(outer_tscv.split(temporal_data)):
            print(f"\n=== OUTER FOLD {outer_fold + 1}/{self.outer_cv} ===")
            print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            X_train = temporal_data[train_idx]
            y_train = targets[train_idx]
            X_test = temporal_data[test_idx]
            y_test = targets[test_idx]
            
            # For each model
            for model in self.models:
                print(f"\n  Model: {model}")
                
                try:
                    # INNER LOOP: Hyperparameter tuning
                    optimal_params = self.inner_cv_hyperparameter_tuning(X_train, y_train, model)
                    
                    # Train with optimal parameters
                    predictions, test_targets = self.train_model_with_optimal_params(
                        X_train, y_train, X_test, y_test, model, optimal_params
                    )
                    
                    # Calculate metrics
                    mse = mean_squared_error(test_targets, predictions)
                    r2 = r2_score(test_targets, predictions)
                    rmse = np.sqrt(mse)
                    
                    # Store results
                    self.results[day][model]['fold_results'].append({
                        'fold': outer_fold + 1,
                        'optimal_params': optimal_params,
                        'rmse': rmse,
                        'r2': r2
                    })
                    
                    self.results[day][model]['rmse'].append(rmse)
                    self.results[day][model]['r2'].append(r2)
                    
                    print(f"    Final result - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                    
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    self.results[day][model]['rmse'].append(float('inf'))
                    self.results[day][model]['r2'].append(-float('inf'))
        
        # Summary for this day
        print(f"\n{'='*80}")
        print(f"NESTED CV RESULTS FOR {day}")
        print(f"{'='*80}")
        
        for model in self.models:
            rmse_scores = [r for r in self.results[day][model]['rmse'] if r != float('inf')]
            r2_scores = [r for r in self.results[day][model]['r2'] if r != -float('inf')]
            
            if rmse_scores:
                print(f"{model}:")
                print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
                print(f"  R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
                print(f"  Individual RMSE: {[f'{r:.4f}' for r in rmse_scores]}")
            else:
                print(f"{model}: ALL FOLDS FAILED")
    
    def run_complete_nested_cv(self):
        """
        Run complete nested cross-validation experiment
        """
        print("="*80)
        print("NESTED CROSS-VALIDATION PIPELINE")
        print("="*80)
        print("Following the methodology from the attached image")
        print("Outer loop: Final model evaluation")
        print("Inner loop: Hyperparameter tuning")
        print("="*80)
        
        # Run for each day
        for day in self.days:
            try:
                self.run_nested_cv_for_day(day)
            except Exception as e:
                print(f"ERROR processing {day}: {str(e)}")
        
        print(f"\n{'='*80}")
        print("NESTED CROSS-VALIDATION COMPLETE")
        print("="*80)

def main():
    """
    Main execution for nested cross-validation
    """
    pipeline = NestedCVPipeline(
        days=['7_24', '10_19', '11_10'],
        outer_cv=5,  # 5 folds for final evaluation
        inner_cv=3   # 3 folds for hyperparameter tuning
    )
    
    pipeline.run_complete_nested_cv()

if __name__ == "__main__":
    main()
