import numpy as np
import torch
import torch.nn as nn
import os
import json # Import json to load hyperparameters

# Import your new EnhancedLSTM model class
from lstm_model import EnhancedLSTM # Assuming the file is named lstm_model_enhanced_fixed.py

DAY_PREFIXES = ['11_10_', '7_24_', '10_19_']

def extract_lstm_features_pytorch(model_path, X_test, input_size, hyperparams):
    """
    Extracts 32-dimensional temporal features from the fc1 layer of the trained LSTM model.
    
    Args:
        model_path (str): Path to the trained PyTorch model (.pth file).
        X_test (np.ndarray): Preprocessed test sequences (NumPy array).
        input_size (int): Number of features in each timestep.
        hyperparams (dict): Dictionary of hyperparameters used to train the model.
                            Expected keys: 'hidden_size', 'num_layers', 'dropout', 'activation'.
                                           If None, default parameters will be used.
    Returns:
        np.ndarray: Extracted temporal features (batch_size, 32).
    """
    
    # Initialize the model with the loaded hyperparameters
    # Provide default values if hyperparams is None or a key is missing
    model = EnhancedLSTM(
        input_size=input_size,
        hidden_size=hyperparams.get('hidden_size', 64),
        num_layers=hyperparams.get('num_layers', 1),
        dropout=hyperparams.get('dropout', 0.2),
        activation=hyperparams.get('activation', 'relu')
    )
    
    # Load the trained model's state dictionary
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # Set model to evaluation mode
    
    # Convert numpy to tensor
    X_tensor = torch.FloatTensor(X_test)
    
    # Extract features from the fc1 layer (32 dimensions)
    features_list = []
    
    with torch.no_grad(): # Disable gradient calculation for inference
        # Process in batches to handle memory efficiently
        batch_size = 32 # Can be adjusted
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            
            # Forward pass through LSTM
            lstm_out, (hn, cn) = model.lstm(batch)
            last_output = lstm_out[:, -1, :] # Get the output from the last timestep
            
            # Get features from fc1 layer (before final prediction)
            # Use model.activation for consistency with EnhancedLSTM
            features = model.activation(model.fc1(last_output)) # 32-dimensional features
            
            features_list.append(features.numpy())
    
    # Combine all batches
    all_features = np.vstack(features_list)
    
    return all_features

def main():
    
    print("="*60)
    print("EXTRACTING LSTM TEMPORAL FEATURES")
    print("="*60)
    print("Purpose: Extract temporal features for LightGBM integration")
    print("Framework: Process -> LSTM Module -> Extract temporal features")
    print("="*60)
    
    for prefix in DAY_PREFIXES:
        print(f"\nProcessing {prefix}:")
        
        # Define paths for optimized model and hyperparameters
        optimized_model_path = f'{prefix}lstm_model_optimized.pth'
        default_model_path = f'{prefix}lstm_model_default.pth' # Fallback for default training
        hyperparams_path = f'{prefix}best_hyperparameters.json'
        
        model_to_load = None
        hyperparams = {}

        # Check if optimized model exists, otherwise check for default model
        if os.path.exists(optimized_model_path):
            model_to_load = optimized_model_path
            print(f"   Loading optimized model: {model_to_load}")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    hyperparams = json.load(f)
                print(f"   Loaded hyperparameters: {hyperparams}")
            else:
                print(f"   Warning: Hyperparameters file not found for optimized model. Using default model params.")
                # Fallback to default params if JSON is missing for optimized model
                hyperparams = {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2, 'activation': 'relu'}
        elif os.path.exists(default_model_path):
            model_to_load = default_model_path
            print(f"   Loading default model: {model_to_load}")
            # For default model, use hardcoded default hyperparameters
            hyperparams = {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2, 'activation': 'relu'}
        else:
            print(f"   Error: Neither optimized model ({optimized_model_path}) nor default model ({default_model_path}) found.")
            print(f"   Run lstm_model_enhanced_fixed.py first to train the model.")
            continue
        
        # Load preprocessed test data
        npz_path = f'{prefix}lstm_preprocessed_data.npz'
        if not os.path.exists(npz_path):
            print(f"   Preprocessed data not found: {npz_path}")
            print(f"   Run lstm_preprocessing.py first")
            continue
        
        # Load the data
        data = np.load(npz_path, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Get input size from the data
        input_size = X_test.shape[2] # Number of features
        
        print(f"   Test data shape: {X_test.shape}")
        print(f"   Input features: {input_size}")
        
        try:
            # Extract temporal features
            print(f"   Extracting temporal features...")
            temporal_features = extract_lstm_features_pytorch(model_to_load, X_test, input_size, hyperparams)
            
            # Save temporal features for LightGBM
            feature_output_path = f'{prefix}lstm_temporal_features.npy'
            np.save(feature_output_path, temporal_features)
            
            # Also save the corresponding targets
            target_output_path = f'{prefix}lstm_targets.npy'
            np.save(target_output_path, y_test)
            
            print(f"   Features extracted: {temporal_features.shape}")
            print(f"   Saved temporal features: {feature_output_path}")
            print(f"   Saved targets: {target_output_path}")
            print(f"   Feature dimensions: 32 (ready for LightGBM)")
            
        except Exception as e:
            print(f"   Error extracting features: {str(e)}")
            continue
    
    print(f"\n" + "="*60)
    print("LSTM FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print("Temporal features ready for hybrid model integration")
    print("Next steps:")
    print("   1. Extract CapsNet spatial features")
    print("   2. Concatenate LSTM + CapsNet features")
    print("   3. Train LightGBM with combined features")

if __name__ == "__main__":
    main()
