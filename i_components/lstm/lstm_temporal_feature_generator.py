import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
import json
import time
from itertools import product

class LSTMTemporalFeatureExtractor(nn.Module):
    """
    LSTM component for hybrid models that produces 32-dimensional temporal features
    Compatible with: CapsNet-LSTM-LightGBM, CNN-LSTM-LightGBM, CapsNet-LSTM
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, 
                 activation='relu', lstm_dropout=0.0):
        super(LSTMTemporalFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Feature extraction layer (32 dimensions for hybrid integration)
        self.temporal_feature_layer = nn.Linear(hidden_size, 32)
        
        # Optional prediction layer (for CapsNet-LSTM model without LightGBM)
        self.prediction_layer = nn.Linear(32, 1)
        
    def forward(self, x, return_features_only=False):
        """
        Forward pass with option to return only temporal features or full prediction
        
        Args:
            x: Input sequences (batch_size, seq_len, features)
            return_features_only: If True, returns only 32-dim temporal features
                                If False, returns both features and prediction
        
        Returns:
            temporal_features: 32-dimensional temporal features for hybrid models
            prediction: PM2.5 prediction (only if return_features_only=False)
        """
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Use last timestep output
        
        # Extract 32-dimensional temporal features
        temporal_features = self.activation(self.temporal_feature_layer(last_output))
        temporal_features = self.dropout(temporal_features)
        
        if return_features_only:
            return temporal_features
        else:
            # For CapsNet-LSTM model (without LightGBM)
            prediction = self.prediction_layer(temporal_features)
            return temporal_features, prediction

class LSTMTemporalFeatureGenerator:
    """
    LSTM component for temporal feature extraction in hybrid models
    Designed to work within full pipeline cross-validation
    """
    def __init__(self, best_params=None):
        # Default hyperparameters (should be determined through separate tuning)
        self.default_params = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'timesteps': 60,  # Default temporal window size
            'weight_decay': 0.0,
            'grad_clip': 1.0,
            'lstm_dropout': 0.0
        }
        
        self.params = best_params if best_params else self.default_params
        self.scaler = None
        self.model = None
        
    def prepare_temporal_sequences(self, temporal_data, targets, timesteps=10):
        """
        Prepare temporal sequences for a given fold
        Called during each CV fold by the full pipeline
        """
        # Scale temporal features
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            temporal_data_scaled = self.scaler.fit_transform(temporal_data)
        else:
            temporal_data_scaled = self.scaler.transform(temporal_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(temporal_data_scaled) - timesteps):
            X.append(temporal_data_scaled[i:i+timesteps])
            y.append(targets[i+timesteps])
        
        return np.array(X), np.array(y)
    
    def train_and_extract_features(self, train_temporal_data, train_targets, val_temporal_data, val_targets, timesteps=None):
        """
        Train LSTM and extract temporal features for current CV fold
        This is called by the full pipeline during each fold
        
        IMPORTANT: val_temporal_data is validation data from the 80% learning set,
                   NOT the 20% hold-out test set (to prevent data leakage)
        
        Returns:
            train_features: 32D temporal features for training data
            val_features: 32D temporal features for validation data (within learning set)
        """
        # Use timesteps from params or default
        if timesteps is None:
            timesteps = self.params.get('timesteps', 60)
            
        # Prepare sequences for training
        X_train, y_train = self.prepare_temporal_sequences(train_temporal_data, train_targets, timesteps)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        # Initialize LSTM model
        self.model = LSTMTemporalFeatureExtractor(
            input_size=train_temporal_data.shape[1],
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            activation=self.params['activation'],
            lstm_dropout=self.params.get('lstm_dropout', 0.0)
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.params['learning_rate'],
            weight_decay=self.params.get('weight_decay', 0.0)
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                _, predictions = self.model(batch_X, return_features_only=False)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.params.get('grad_clip', 1.0)
                )
                
                optimizer.step()
                total_loss += loss.item()
        
        # Extract features from training data
        self.model.eval()
        with torch.no_grad():
            train_features = self.model(X_train_tensor, return_features_only=True).numpy()
        
        # Prepare test sequences and extract features
        X_val, _ = self.prepare_temporal_sequences(val_temporal_data, val_targets, timesteps)
        X_val_tensor = torch.FloatTensor(X_val)
        
        with torch.no_grad():
            val_features = self.model(X_val_tensor, return_features_only=True).numpy()
        
        # Debug: Verify numpy arrays are generated correctly
        print(f"  LSTM Debug - Train features shape: {train_features.shape}, type: {type(train_features)}")
        print(f"  LSTM Debug - Val features shape: {val_features.shape}, type: {type(val_features)}")
        print(f"  LSTM Debug - Feature sample: {train_features[0][:5]}...")  # First 5 values
        
        return train_features, val_features, y_train, X_val.shape[0]
    
class TemporalDataLoader:
    """
    Data loader for temporal features across all days
    Supports full pipeline cross-validation
    """
    def __init__(self, days=['7_24', '10_19', '11_10']):
        self.days = days
        self.temporal_features = ['pm10', 'temperature', 'humidity']
    
    def load_temporal_data(self, day):
        """
        Load temporal data for a specific day
        Returns raw temporal data and targets for CV splitting
        """
        matched_file = f'dataset/c_matched_spatio_temporal_data/matched_{day}.csv'
        if not os.path.exists(matched_file):
            raise FileNotFoundError(f"Matched data not found: {matched_file}")
        
        df = pd.read_csv(matched_file)
        
        # Sort by timestamp for proper time series handling
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get available temporal features
        available_features = [col for col in self.temporal_features if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No temporal features found in {day}")
        
        # Remove rows with missing values
        required_cols = available_features + ['pm2.5']
        df_clean = df.dropna(subset=required_cols)
        
        temporal_data = df_clean[available_features].values
        targets = df_clean['pm2.5'].values
        
        return temporal_data, targets, available_features

def hyperparameter_tuning_lstm(day, max_combinations=10):
    """
    Separate hyperparameter tuning for LSTM component
    This should be run once per day to find optimal parameters
    
    IMPORTANT: Uses only the 80% learning set for tuning
    The 20% hold-out test set remains completely untouched
    """
    hyperparameter_space = {
        # LSTM Architecture
        'hidden_size': [32, 64, 128, 256],           # More options for LSTM capacity
        'num_layers': [1, 2, 3],                     # Try deeper networks
        'dropout': [0.1, 0.2, 0.3, 0.4],           # Regularization strength
        'activation': ['relu', 'tanh', 'gelu'],      # Different activation functions
        
        # Training Parameters
        'learning_rate': [0.0001, 0.001, 0.01],     # Learning rate range
        'batch_size': [16, 32, 64],                  # Batch size options
        'epochs': [10, 30, 50],                     # Proper training duration
        
        # Sequence Parameters
        'timesteps': [10, 30, 40, 60],               # Sequence length for temporal patterns
        
        # Regularization
        'weight_decay': [0.0, 1e-5, 1e-4],         # L2 regularization
        'grad_clip': [0.5, 1.0, 2.0],              # Gradient clipping
        
        # LSTM-specific
        'lstm_dropout': [0.0, 0.1, 0.2],           # LSTM internal dropout
    }
    
    print(f"LSTM Hyperparameter Tuning for {day}")
    print("=" * 50)
    
    # Load data
    data_loader = TemporalDataLoader()
    temporal_data, targets, feature_names = data_loader.load_temporal_data(day)
    
    # CRITICAL: Use only the 80% learning set for hyperparameter tuning
    # The 20% hold-out test set remains completely untouched
    n_total = len(temporal_data)
    n_learning = int(n_total * 0.8)  # Only use 80% learning set
    
    learning_temporal = temporal_data[:n_learning]  # 80% learning set only
    learning_targets = targets[:n_learning]
    
    # Split the 80% learning set into train/validation for hyperparameter tuning
    # Use 70% of total for training, 10% of total for validation (as per pipeline requirements)
    n_train = int(n_total * 0.7)  # 70% of total data
    n_val = int(n_total * 0.1)    # 10% of total data
    
    train_temporal = learning_temporal[:n_train]           # 70% of total
    train_targets = learning_targets[:n_train]
    val_temporal = learning_temporal[n_train:n_train + n_val]  # 10% of total
    val_targets = learning_targets[n_train:n_train + n_val]
    
    print(f"Data split for hyperparameter tuning:")
    print(f"  Total samples: {n_total}")
    print(f"  Learning set (used): {n_learning} ({n_learning/n_total*100:.1f}%)")
    print(f"  Hold-out test (untouched): {n_total - n_learning} ({(n_total - n_learning)/n_total*100:.1f}%)")
    print(f"  Train for tuning: {len(train_temporal)} ({len(train_temporal)/n_total*100:.1f}%)")
    print(f"  Val for tuning: {len(val_temporal)} ({len(val_temporal)/n_total*100:.1f}%)")
    
    # Generate parameter combinations
    keys = list(hyperparameter_space.keys())
    values = list(hyperparameter_space.values())
    all_combinations = list(product(*values))
    
    if len(all_combinations) > max_combinations:
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), size=max_combinations, replace=False)
        combinations = [all_combinations[i] for i in selected_indices]
    else:
        combinations = all_combinations
    
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    
    best_params = None
    best_score = float('inf')
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"  Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create LSTM generator with current params
        lstm_gen = LSTMTemporalFeatureGenerator(params)
        
        try:
            # Train and get validation features (using only learning set)
            train_features, val_features, train_y, val_len = lstm_gen.train_and_extract_features(
                train_temporal, train_targets, val_temporal, val_targets, timesteps=params.get('timesteps', 60)
            )
            
            # Simple validation: predict using temporal features
            val_y = val_targets[params.get('timesteps', 60):]  # Account for sequence length
            if len(val_y) == len(val_features):
                # Use simple linear regression for validation
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(train_features, train_y)
                val_pred = lr.predict(val_features)
                
                mse = mean_squared_error(val_y, val_pred)
                rmse = np.sqrt(mse)
                
                print(f"    Validation RMSE: {rmse:.4f}")
                
                if mse < best_score:
                    best_score = mse
                    best_params = params.copy()
                    print(f"     New best score: {rmse:.4f}")
                
        except Exception as e:
            print(f"     Error with params: {e}")
            continue
    
    # Save best parameters
    os.makedirs('models/lstm_temporal', exist_ok=True)
    with open(f'models/lstm_temporal/{day}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters for {day}: {best_params}")
    print(f"Best validation RMSE: {np.sqrt(best_score):.4f}")
    
    return best_params
    

def main():
    """
    Main execution for LSTM temporal feature extraction
    Focused on component preparation for full pipeline CV
    """
    print("="*80)
    print("LSTM TEMPORAL FEATURE EXTRACTION - PIPELINE READY")
    print("="*80)
    print("Purpose: Prepare LSTM component for full pipeline cross-validation")
    print("- Hyperparameter tuning for each day")
    print("- Component ready for CV integration")
    print("- No standalone evaluation (done in full pipeline)")
    print("="*80)
    
    days = ['7_24', '10_19', '11_10']
    
    # Step 1: Hyperparameter tuning for each day
    print("\nStep 1: LSTM Hyperparameter Tuning")
    print("-" * 50)
    
    for day in days:
        try:
            print(f"\nTuning LSTM parameters for {day}...")
            best_params = hyperparameter_tuning_lstm(day, max_combinations=1)
            print(f" {day} tuning complete")
            
        except Exception as e:
            print(f" Error tuning {day}: {str(e)}")
    
    print(f"\n{'='*80}")
    print("LSTM COMPONENT PREPARATION COMPLETE")
    print("="*80)
    print("Best parameters saved for each day")
    print("LSTMTemporalFeatureGenerator class ready")
    print("TemporalDataLoader class ready")
    print("\nNext Steps:")
    print("1. Integration team: Use LSTMTemporalFeatureGenerator in full pipeline")
    print("2. Full pipeline CV: Wrap entire models in 5-fold cross-validation")
    print("3. Models to evaluate: CapsNet-LSTM-LightGBM, CNN-LSTM-LightGBM, CapsNet-LSTM")
    print("4. Statistical analysis: Compare 5-fold results across models")
    print("="*80)
    
    # Example usage for integration team
    print("\n" + "="*80)
    print("EXAMPLE USAGE FOR INTEGRATION TEAM:")
    print("="*80)
    print("""
# In your full pipeline cross-validation script:

from lstm_temporal_feature_generator import LSTMTemporalFeatureGenerator, TemporalDataLoader

# Load best parameters
with open('models/lstm_temporal/7_24_best_params.json', 'r') as f:
    best_params = json.load(f)

# Initialize components
data_loader = TemporalDataLoader()
lstm_generator = LSTMTemporalFeatureGenerator(best_params)

# For each CV fold:
for fold in range(5):
    # Get temporal data for this fold
    temporal_data, targets, _ = data_loader.load_temporal_data('7_24')
    
    # Split data for this fold (train/val indices from CV within learning set)
    train_temporal = temporal_data[train_idx]
    val_temporal = temporal_data[val_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    
    # Extract temporal features
    train_temp_features, val_temp_features, _, _ = lstm_generator.train_and_extract_features(
        train_temporal, train_targets, val_temporal, val_targets
    )
    
    # Combine with spatial features and train full model
    # ... (CapsNet/CNN + LightGBM integration)
    
    # Evaluate full model performance for this fold
    # ... (this gives you one performance score per fold)
""")
    print("="*80)

if __name__ == "__main__":
    main()
