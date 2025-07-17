"""
Individual Model Evaluators for Separate Execution
Each model can be run independently with 5-fold cross-validation
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


class BaseModelEvaluator:
    """
    Base class for individual model evaluation with 5-fold CV
    """
    def __init__(self, model_name, days=['7_24', '10_19', '11_10'], cv_folds=5):
        self.model_name = model_name
        self.days = days
        self.cv_folds = cv_folds
        self.results = {}
        
        # Common hyperparameter spaces
        self.lstm_params = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'epochs': [30, 50],
            'batch_size': [16, 32],
            'timesteps': [10, 15, 20]
        }
        
        self.lightgbm_params = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_child_samples': [20, 30, 50]
        }
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning using inner cross-validation
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement hyperparameter_tuning")
    
    def train_and_predict(self, X_train, y_train, X_test, y_test, optimal_params):
        """
        Train model with optimal parameters and make predictions
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement train_and_predict")
    
    def evaluate_model_for_day(self, day):
        """
        Evaluate model for a specific day using 5-fold time series CV
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {self.model_name} FOR {day}")
        print(f"{'='*80}")
        print(f"Cross-validation folds: {self.cv_folds}")
        
        # Load data
        data_loader = TemporalDataLoader()
        temporal_data, targets, _ = data_loader.load_temporal_data(day)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Initialize results for this day
        self.results[day] = {
            'fold_results': [],
            'rmse_scores': [],
            'r2_scores': [],
            'mae_scores': []
        }
        
        # 5-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(temporal_data)):
            print(f"\n--- FOLD {fold + 1}/{self.cv_folds} ---")
            print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            try:
                # Split data
                X_train = temporal_data[train_idx]
                y_train = targets[train_idx]
                X_test = temporal_data[test_idx]
                y_test = targets[test_idx]
                
                # Hyperparameter tuning
                print("  Hyperparameter tuning...")
                optimal_params = self.hyperparameter_tuning(X_train, y_train)
                print(f"  Optimal parameters: {optimal_params}")
                
                # Train and predict
                print("  Training and predicting...")
                predictions, test_targets = self.train_and_predict(
                    X_train, y_train, X_test, y_test, optimal_params
                )
                
                # Calculate metrics
                mse = mean_squared_error(test_targets, predictions)
                mae = mean_absolute_error(test_targets, predictions)
                r2 = r2_score(test_targets, predictions)
                rmse = np.sqrt(mse)
                
                # Store results
                fold_result = {
                    'fold': fold + 1,
                    'optimal_params': optimal_params,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': predictions.tolist(),
                    'test_targets': test_targets.tolist()
                }
                
                self.results[day]['fold_results'].append(fold_result)
                self.results[day]['rmse_scores'].append(rmse)
                self.results[day]['mae_scores'].append(mae)
                self.results[day]['r2_scores'].append(r2)
                
                print(f"  Fold {fold + 1} results:")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    MAE: {mae:.4f}")
                print(f"    R²: {r2:.4f}")
                
            except Exception as e:
                print(f"  ERROR in fold {fold + 1}: {str(e)}")
                # Store error results
                self.results[day]['rmse_scores'].append(float('inf'))
                self.results[day]['mae_scores'].append(float('inf'))
                self.results[day]['r2_scores'].append(-float('inf'))
        
        # Calculate summary statistics
        self._calculate_summary_statistics(day)
    
    def _calculate_summary_statistics(self, day):
        """
        Calculate summary statistics for all folds
        """
        rmse_scores = [r for r in self.results[day]['rmse_scores'] if r != float('inf')]
        mae_scores = [r for r in self.results[day]['mae_scores'] if r != float('inf')]
        r2_scores = [r for r in self.results[day]['r2_scores'] if r != -float('inf')]
        
        if rmse_scores:
            self.results[day]['summary'] = {
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'rmse_scores': rmse_scores,
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'mae_scores': mae_scores,
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'r2_scores': r2_scores,
                'successful_folds': len(rmse_scores)
            }
            
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {self.model_name} - {day}")
            print(f"{'='*60}")
            print(f"Successful folds: {len(rmse_scores)}/{self.cv_folds}")
            print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
            print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
            print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
            print(f"Individual RMSE scores: {[f'{r:.4f}' for r in rmse_scores]}")
        else:
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {self.model_name} - {day}")
            print(f"{'='*60}")
            print("ALL FOLDS FAILED")
    
    def run_complete_evaluation(self):
        """
        Run complete evaluation across all days
        """
        print(f"\n{'='*100}")
        print(f"COMPLETE EVALUATION: {self.model_name}")
        print(f"{'='*100}")
        
        for day in self.days:
            try:
                self.evaluate_model_for_day(day)
            except Exception as e:
                print(f"ERROR processing {day}: {str(e)}")
        
        # Save results
        self._save_results()
        
        # Print final summary
        self._print_final_summary()
    
    def _save_results(self):
        """
        Save results to JSON file
        """
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'{self.model_name.lower().replace("-", "_")}_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def _print_final_summary(self):
        """
        Print final summary across all days
        """
        print(f"\n{'='*100}")
        print(f"FINAL SUMMARY: {self.model_name}")
        print(f"{'='*100}")
        
        for day in self.days:
            if day in self.results and 'summary' in self.results[day]:
                summary = self.results[day]['summary']
                print(f"{day}:")
                print(f"  RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
                print(f"  MAE:  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
                print(f"  R²:   {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
                print(f"  Successful folds: {summary['successful_folds']}/{self.cv_folds}")
            else:
                print(f"{day}: FAILED")


class CapsNetLSTMLightGBMEvaluator(BaseModelEvaluator):
    """
    Evaluator for CapsNet-LSTM-LightGBM model
    """
    def __init__(self, days=['7_24', '10_19', '11_10'], cv_folds=5):
        super().__init__('CapsNet-LSTM-LightGBM', days, cv_folds)
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning for CapsNet-LSTM-LightGBM
        """
        # Inner CV for hyperparameter tuning
        inner_cv = TimeSeriesSplit(n_splits=3)
        
        # Tune LSTM parameters first
        lstm_param_grid = ParameterGrid(self.lstm_params)
        best_lstm_params = None
        best_lstm_score = float('inf')
        
        for lstm_params in list(lstm_param_grid)[:5]:  # Limit to 5 combinations
            lstm_scores = []
            
            for train_idx, val_idx in inner_cv.split(X_train):
                try:
                    # Split data
                    temp_train = X_train[train_idx]
                    temp_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    # Train LSTM
                    lstm_gen = LSTMTemporalFeatureGenerator(lstm_params)
                    train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                        temp_train, target_train, temp_val, target_val
                    )
                    
                    # Quick validation with simple model
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(train_temp_features, target_train[lstm_params.get('timesteps', 10):])
                    val_pred = lr.predict(val_temp_features)
                    
                    mse = mean_squared_error(target_val[lstm_params.get('timesteps', 10):], val_pred)
                    lstm_scores.append(mse)
                except:
                    lstm_scores.append(float('inf'))
            
            avg_score = np.mean(lstm_scores)
            if avg_score < best_lstm_score:
                best_lstm_score = avg_score
                best_lstm_params = lstm_params
        
        # Tune LightGBM parameters
        lgb_param_grid = ParameterGrid(self.lightgbm_params)
        best_lgb_params = None
        best_lgb_score = float('inf')
        
        for lgb_params in list(lgb_param_grid)[:8]:  # Limit to 8 combinations
            lgb_scores = []
            
            for train_idx, val_idx in inner_cv.split(X_train):
                try:
                    # Use best LSTM params to extract features
                    temp_train = X_train[train_idx]
                    temp_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    lstm_gen = LSTMTemporalFeatureGenerator(best_lstm_params)
                    train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                        temp_train, target_train, temp_val, target_val
                    )
                    
                    # Add dummy CapsNet spatial features (32D)
                    train_spatial = np.random.rand(len(train_temp_features), 32)
                    val_spatial = np.random.rand(len(val_temp_features), 32)
                    
                    # Combine features
                    train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
                    val_combined = np.concatenate([val_temp_features, val_spatial], axis=1)
                    
                    # Train LightGBM
                    lgb_model = lgb.LGBMRegressor(
                        objective='regression',
                        metric='rmse',
                        boosting_type='gbdt',
                        verbose=-1,
                        **lgb_params
                    )
                    
                    lgb_model.fit(train_combined, target_train[best_lstm_params.get('timesteps', 10):])
                    val_pred = lgb_model.predict(val_combined)
                    
                    mse = mean_squared_error(target_val[best_lstm_params.get('timesteps', 10):], val_pred)
                    lgb_scores.append(mse)
                except:
                    lgb_scores.append(float('inf'))
            
            avg_score = np.mean(lgb_scores)
            if avg_score < best_lgb_score:
                best_lgb_score = avg_score
                best_lgb_params = lgb_params
        
        return {
            'lstm': best_lstm_params,
            'lightgbm': best_lgb_params
        }
    
    def train_and_predict(self, X_train, y_train, X_test, y_test, optimal_params):
        """
        Train CapsNet-LSTM-LightGBM and make predictions
        """
        # Extract temporal features with LSTM
        lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
        train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
            X_train, y_train, X_test, y_test
        )
        
        # Add dummy CapsNet spatial features (32D)
        # TODO: Replace with actual CapsNet feature extraction
        train_spatial = np.random.rand(len(train_temp_features), 32)
        test_spatial = np.random.rand(len(test_temp_features), 32)
        
        # Combine temporal and spatial features
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
        
        timesteps = optimal_params['lstm'].get('timesteps', 10)
        lgb_model.fit(train_combined, y_train[timesteps:])
        predictions = lgb_model.predict(test_combined)
        
        return predictions, y_test[timesteps:]


class CNNLSTMLightGBMEvaluator(BaseModelEvaluator):
    """
    Evaluator for CNN-LSTM-LightGBM model
    """
    def __init__(self, days=['7_24', '10_19', '11_10'], cv_folds=5):
        super().__init__('CNN-LSTM-LightGBM', days, cv_folds)
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning for CNN-LSTM-LightGBM
        """
        # Inner CV for hyperparameter tuning
        inner_cv = TimeSeriesSplit(n_splits=3)
        
        # Tune LSTM parameters first
        lstm_param_grid = ParameterGrid(self.lstm_params)
        best_lstm_params = None
        best_lstm_score = float('inf')
        
        for lstm_params in list(lstm_param_grid)[:5]:  # Limit to 5 combinations
            lstm_scores = []
            
            for train_idx, val_idx in inner_cv.split(X_train):
                try:
                    # Split data
                    temp_train = X_train[train_idx]
                    temp_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    # Train LSTM
                    lstm_gen = LSTMTemporalFeatureGenerator(lstm_params)
                    train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                        temp_train, target_train, temp_val, target_val
                    )
                    
                    # Quick validation with simple model
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(train_temp_features, target_train[lstm_params.get('timesteps', 10):])
                    val_pred = lr.predict(val_temp_features)
                    
                    mse = mean_squared_error(target_val[lstm_params.get('timesteps', 10):], val_pred)
                    lstm_scores.append(mse)
                except:
                    lstm_scores.append(float('inf'))
            
            avg_score = np.mean(lstm_scores)
            if avg_score < best_lstm_score:
                best_lstm_score = avg_score
                best_lstm_params = lstm_params
        
        # Tune LightGBM parameters
        lgb_param_grid = ParameterGrid(self.lightgbm_params)
        best_lgb_params = None
        best_lgb_score = float('inf')
        
        for lgb_params in list(lgb_param_grid)[:8]:  # Limit to 8 combinations
            lgb_scores = []
            
            for train_idx, val_idx in inner_cv.split(X_train):
                try:
                    # Use best LSTM params to extract features
                    temp_train = X_train[train_idx]
                    temp_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    lstm_gen = LSTMTemporalFeatureGenerator(best_lstm_params)
                    train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                        temp_train, target_train, temp_val, target_val
                    )
                    
                    # Add dummy CNN spatial features (32D)
                    train_spatial = np.random.rand(len(train_temp_features), 32)
                    val_spatial = np.random.rand(len(val_temp_features), 32)
                    
                    # Combine features
                    train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
                    val_combined = np.concatenate([val_temp_features, val_spatial], axis=1)
                    
                    # Train LightGBM
                    lgb_model = lgb.LGBMRegressor(
                        objective='regression',
                        metric='rmse',
                        boosting_type='gbdt',
                        verbose=-1,
                        **lgb_params
                    )
                    
                    lgb_model.fit(train_combined, target_train[best_lstm_params.get('timesteps', 10):])
                    val_pred = lgb_model.predict(val_combined)
                    
                    mse = mean_squared_error(target_val[best_lstm_params.get('timesteps', 10):], val_pred)
                    lgb_scores.append(mse)
                except:
                    lgb_scores.append(float('inf'))
            
            avg_score = np.mean(lgb_scores)
            if avg_score < best_lgb_score:
                best_lgb_score = avg_score
                best_lgb_params = lgb_params
        
        return {
            'lstm': best_lstm_params,
            'lightgbm': best_lgb_params
        }
    
    def train_and_predict(self, X_train, y_train, X_test, y_test, optimal_params):
        """
        Train CNN-LSTM-LightGBM and make predictions
        """
        # Extract temporal features with LSTM
        lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
        train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
            X_train, y_train, X_test, y_test
        )
        
        # Add dummy CNN spatial features (32D)
        # TODO: Replace with actual CNN feature extraction
        train_spatial = np.random.rand(len(train_temp_features), 32)
        test_spatial = np.random.rand(len(test_temp_features), 32)
        
        # Combine temporal and spatial features
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
        
        timesteps = optimal_params['lstm'].get('timesteps', 10)
        lgb_model.fit(train_combined, y_train[timesteps:])
        predictions = lgb_model.predict(test_combined)
        
        return predictions, y_test[timesteps:]


class CapsNetLSTMEvaluator(BaseModelEvaluator):
    """
    Evaluator for CapsNet-LSTM model (without LightGBM)
    """
    def __init__(self, days=['7_24', '10_19', '11_10'], cv_folds=5):
        super().__init__('CapsNet-LSTM', days, cv_folds)
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning for CapsNet-LSTM
        """
        # Inner CV for hyperparameter tuning
        inner_cv = TimeSeriesSplit(n_splits=3)
        
        # Tune LSTM parameters
        lstm_param_grid = ParameterGrid(self.lstm_params)
        best_lstm_params = None
        best_lstm_score = float('inf')
        
        for lstm_params in list(lstm_param_grid)[:8]:  # Limit to 8 combinations
            lstm_scores = []
            
            for train_idx, val_idx in inner_cv.split(X_train):
                try:
                    # Split data
                    temp_train = X_train[train_idx]
                    temp_val = X_train[val_idx]
                    target_train = y_train[train_idx]
                    target_val = y_train[val_idx]
                    
                    # Train LSTM
                    lstm_gen = LSTMTemporalFeatureGenerator(lstm_params)
                    train_temp_features, val_temp_features, _, _ = lstm_gen.train_and_extract_features(
                        temp_train, target_train, temp_val, target_val
                    )
                    
                    # Add dummy CapsNet spatial features (32D)
                    train_spatial = np.random.rand(len(train_temp_features), 32)
                    val_spatial = np.random.rand(len(val_temp_features), 32)
                    
                    # Combine features
                    train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
                    val_combined = np.concatenate([val_temp_features, val_spatial], axis=1)
                    
                    # Simple prediction layer
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(train_combined, target_train[lstm_params.get('timesteps', 10):])
                    val_pred = lr.predict(val_combined)
                    
                    mse = mean_squared_error(target_val[lstm_params.get('timesteps', 10):], val_pred)
                    lstm_scores.append(mse)
                except:
                    lstm_scores.append(float('inf'))
            
            avg_score = np.mean(lstm_scores)
            if avg_score < best_lstm_score:
                best_lstm_score = avg_score
                best_lstm_params = lstm_params
        
        return {
            'lstm': best_lstm_params
        }
    
    def train_and_predict(self, X_train, y_train, X_test, y_test, optimal_params):
        """
        Train CapsNet-LSTM and make predictions
        """
        # Extract temporal features with LSTM
        lstm_gen = LSTMTemporalFeatureGenerator(optimal_params['lstm'])
        train_temp_features, test_temp_features, _, _ = lstm_gen.train_and_extract_features(
            X_train, y_train, X_test, y_test
        )
        
        # Add dummy CapsNet spatial features (32D)
        # TODO: Replace with actual CapsNet feature extraction
        train_spatial = np.random.rand(len(train_temp_features), 32)
        test_spatial = np.random.rand(len(test_temp_features), 32)
        
        # Combine temporal and spatial features
        train_combined = np.concatenate([train_temp_features, train_spatial], axis=1)
        test_combined = np.concatenate([test_temp_features, test_spatial], axis=1)
        
        # Simple prediction layer
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        timesteps = optimal_params['lstm'].get('timesteps', 10)
        lr.fit(train_combined, y_train[timesteps:])
        predictions = lr.predict(test_combined)
        
        return predictions, y_test[timesteps:]


def main():
    """
    Main function to demonstrate individual model evaluation
    """
    print("="*100)
    print("INDIVIDUAL MODEL EVALUATORS")
    print("="*100)
    print("Each model will be evaluated separately with 5-fold cross-validation")
    print("Choose which model to evaluate:")
    print("1. CapsNet-LSTM-LightGBM")
    print("2. CNN-LSTM-LightGBM")
    print("3. CapsNet-LSTM")
    print("4. All models")
    print("="*100)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    # Available days
    days = ['7_24', '10_19', '11_10']
    
    if choice == '1':
        print("\nEvaluating CapsNet-LSTM-LightGBM...")
        evaluator = CapsNetLSTMLightGBMEvaluator(days=days, cv_folds=5)
        evaluator.run_complete_evaluation()
    
    elif choice == '2':
        print("\nEvaluating CNN-LSTM-LightGBM...")
        evaluator = CNNLSTMLightGBMEvaluator(days=days, cv_folds=5)
        evaluator.run_complete_evaluation()
    
    elif choice == '3':
        print("\nEvaluating CapsNet-LSTM...")
        evaluator = CapsNetLSTMEvaluator(days=days, cv_folds=5)
        evaluator.run_complete_evaluation()
    
    elif choice == '4':
        print("\nEvaluating all models...")
        
        # Evaluate CapsNet-LSTM-LightGBM
        print("\n" + "="*50)
        print("EVALUATING CapsNet-LSTM-LightGBM")
        print("="*50)
        evaluator1 = CapsNetLSTMLightGBMEvaluator(days=days, cv_folds=5)
        evaluator1.run_complete_evaluation()
        
        # Evaluate CNN-LSTM-LightGBM
        print("\n" + "="*50)
        print("EVALUATING CNN-LSTM-LightGBM")
        print("="*50)
        evaluator2 = CNNLSTMLightGBMEvaluator(days=days, cv_folds=5)
        evaluator2.run_complete_evaluation()
        
        # Evaluate CapsNet-LSTM
        print("\n" + "="*50)
        print("EVALUATING CapsNet-LSTM")
        print("="*50)
        evaluator3 = CapsNetLSTMEvaluator(days=days, cv_folds=5)
        evaluator3.run_complete_evaluation()
        
        # Print comparison summary
        print("\n" + "="*100)
        print("COMPARISON SUMMARY")
        print("="*100)
        
        all_evaluators = [
            ('CapsNet-LSTM-LightGBM', evaluator1),
            ('CNN-LSTM-LightGBM', evaluator2),
            ('CapsNet-LSTM', evaluator3)
        ]
        
        for day in days:
            print(f"\n{day}:")
            print("-" * 60)
            for model_name, evaluator in all_evaluators:
                if day in evaluator.results and 'summary' in evaluator.results[day]:
                    summary = evaluator.results[day]['summary']
                    print(f"{model_name:25} | RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f} | R²: {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
                else:
                    print(f"{model_name:25} | FAILED")
    
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
