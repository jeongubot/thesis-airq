import numpy as np
import torch
import torch.nn as nn
import os

# Import your LSTM model class
from lstm_model import SimpleLSTM

DAY_PREFIXES = ['11_10_', '7_24_', '10_19_']

def extract_lstm_features_pytorch(model_path, X_test, input_size):

    
    # Load the trained model
    model = SimpleLSTM(input_size=input_size, hidden_size=64, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Convert numpy to tensor
    X_tensor = torch.FloatTensor(X_test)
    
    # Extract features from the fc1 layer (32 dimensions)
    features_list = []
    
    with torch.no_grad():
        # Process in batches to handle memory efficiently
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            
            # Forward pass through LSTM
            lstm_out, (hn, cn) = model.lstm(batch)
            last_output = lstm_out[:, -1, :]  # Get last timestep output
            
            # Get features from fc1 layer (before final prediction)
            features = model.relu(model.fc1(last_output))  # 32-dimensional features
            
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
        
        # Check if trained model exists
        model_path = f'{prefix}lstm_model.pth'
        if not os.path.exists(model_path):
            print(f"   Model not found: {model_path}")
            print(f"   Run lstm_model.py first to train the model")
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
        input_size = X_test.shape[2]  # Number of features
        
        print(f"   Test data shape: {X_test.shape}")
        print(f"   Input features: {input_size}")
        
        try:
            # Extract temporal features
            print(f"   Extracting temporal features...")
            temporal_features = extract_lstm_features_pytorch(model_path, X_test, input_size)
            
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
