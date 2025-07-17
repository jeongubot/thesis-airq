"""
FOR TESTING ONLY the script for LSTM component
This verifies that your LSTM component works correctly
"""

import numpy as np
import json
import os
from i_components.lstm.lstm_temporal_feature_generator import LSTMTemporalFeatureGenerator, TemporalDataLoader

def test_lstm_component():
    """Test the LSTM component functionality"""
    print("="*60)
    print("TESTING LSTM COMPONENT")
    print("="*60)
    
    # Test for each day
    days = ['7_24', '10_19', '11_10']
    
    for day in days:
        print(f"\nTesting LSTM component for {day}...")
        
        try:
            # Load data
            data_loader = TemporalDataLoader()
            temporal_data, targets, feature_names = data_loader.load_temporal_data(day)
            
            print(f"Data loaded: {temporal_data.shape} temporal data, {len(targets)} targets")
            print(f"Features: {feature_names}")
            
            # Check if best parameters exist
            params_file = f'models/lstm_temporal/{day}_best_params.json'
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    best_params = json.load(f)
                print(f"Best parameters loaded: {best_params}")
            else:
                print(f"No best parameters found, using defaults")
                best_params = None
            
            # Test train/test split (simulate CV fold)
            n_total = len(temporal_data)
            n_train = int(n_total * 0.7)
            
            train_temporal = temporal_data[:n_train]
            test_temporal = temporal_data[n_train:]
            train_targets = targets[:n_train]
            test_targets = targets[n_train:]
            
            print(f"Simulated CV split: {len(train_temporal)} train, {len(test_temporal)} test")
            
            # Test LSTM component
            lstm_generator = LSTMTemporalFeatureGenerator(best_params)
            
            train_features, test_features, train_y, test_len = lstm_generator.train_and_extract_features(
                train_temporal, train_targets, test_temporal, test_targets
            )
            
            print(f"LSTM training completed")
            print(f"Train features shape: {train_features.shape}")
            print(f"Test features shape: {test_features.shape}")
            print(f"Features are 32-dimensional: {train_features.shape[1] == 32}")
            
            # Test feature quality
            print(f" Feature statistics:")
            print(f"    Train features - mean: {np.mean(train_features):.4f}, std: {np.std(train_features):.4f}")
            print(f"    Test features - mean: {np.mean(test_features):.4f}, std: {np.std(test_features):.4f}")
            
            # Simple validation test
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            lr = LinearRegression()
            lr.fit(train_features, train_y)
            test_pred = lr.predict(test_features)
            test_rmse = np.sqrt(mean_squared_error(test_targets[10:], test_pred))
            
            print(f"Simple validation RMSE: {test_rmse:.4f}")
            print(f"{day} component test PASSED")
            
        except Exception as e:
            print(f"✗ {day} component test FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("LSTM COMPONENT TEST COMPLETE")
    print("="*60)
    print("If all tests passed, your LSTM component is ready for integration!")

if __name__ == "__main__":
    test_lstm_component()
