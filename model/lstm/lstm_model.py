import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from itertools import product
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DAY_PREFIXES = ['11_10_', '7_24_', '10_19_']

class EnhancedLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, 
                 activation='relu', output_size=1):
        super(EnhancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with configurable parameters
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only for multi-layer
            batch_first=True
        )
        
        # Configurable activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # Default
        
        # Fully connected layers with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers with activation and dropout
        out = self.activation(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class LSTMHyperparameterTuner:
    def __init__(self):
        # Define hyperparameter search space
        self.hyperparameter_space = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'activation': ['relu', 'tanh'],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'batch_size': [8, 16, 32, 64],
            'epochs': [3,4]
        }
        
        self.best_params = {}
        self.best_score = float('inf')
        self.tuning_results = []
        
    def create_hyperparameter_combinations(self, max_combinations=50):

        # Get all possible combinations
        keys = list(self.hyperparameter_space.keys())
        values = list(self.hyperparameter_space.values())
        all_combinations = list(product(*values))
        
        # If too many combinations, sample intelligently
        if len(all_combinations) > max_combinations:
            # Prioritize certain combinations
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(
                len(all_combinations), 
                size=max_combinations, 
                replace=False
            )
            combinations = [all_combinations[i] for i in selected_indices]
        else:
            combinations = all_combinations
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def validate_hyperparameters(self, params, X_train, y_train, X_val, y_val, input_size):
 
        try:
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False
            )
            
            # Initialize model with hyperparameters
            model = EnhancedLSTM(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                activation=params['activation']
            )
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Calculate additional metrics
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.squeeze().numpy())
                    val_targets.extend(batch_y.squeeze().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            # Calculate comprehensive metrics
            mse = mean_squared_error(val_targets, val_predictions)
            mae = mean_absolute_error(val_targets, val_predictions)
            r2 = r2_score(val_targets, val_predictions)
            rmse = np.sqrt(mse)
            
            # Convert all numpy.float32 to standard Python float for JSON serialization
            return {
                'val_loss': float(best_val_loss),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'model': model,
                'epochs_trained': int(epoch + 1) # Ensure integer
            }
            
        except Exception as e:
            print(f"Error with hyperparameters {params}: {str(e)}")
            return {
                'val_loss': float('inf'),
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'model': None,
                'epochs_trained': 0
            }
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, input_size, 
                           day_prefix, max_combinations=3):

        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING FOR {day_prefix}")
        print(f"{'='*60}")
        print(f"Framework: Train/Val split only (Test set untouched)")
        print(f"Search space: {max_combinations} combinations")
        print(f"Validation strategy: Time series split compliance")
        print(f"{'='*60}")
        
        # Generate hyperparameter combinations
        param_combinations = self.create_hyperparameter_combinations(max_combinations)
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations...")
        
        best_model = None
        best_params = None
        best_score = float('inf')
        
        for i, params in enumerate(param_combinations):
            print(f"\nCombination {i+1}/{len(param_combinations)}")
            print(f"Params: {params}")
            
            start_time = time.time()
            
            # Validate hyperparameters (ONLY using train/val data)
            results = self.validate_hyperparameters(
                params, X_train, y_train, X_val, y_val, input_size
            )
            
            end_time = time.time()
            
            # Store results
            result_entry = {
                'combination': i + 1,
                'params': params,
                'val_loss': results['val_loss'],
                'mse': results['mse'],
                'mae': results['mae'],
                'rmse': results['rmse'],
                'r2': results['r2'],
                'epochs_trained': results['epochs_trained'],
                'training_time': end_time - start_time
            }
            
            self.tuning_results.append(result_entry)
            
            print(f"Val Loss: {results['val_loss']:.4f}, RMSE: {results['rmse']:.4f}, "
                  f"R2: {results['r2']:.4f}, Time: {end_time - start_time:.2f}s")
            
            # Update best model
            if results['val_loss'] < best_score:
                best_score = results['val_loss']
                best_params = params.copy()
                best_model = results['model']
                print(f"[BEST] NEW BEST MODEL! Val Loss: {best_score:.4f}")
        
        # Save best hyperparameters and results
        self.best_params[day_prefix] = best_params
        self.best_score = best_score
        
        # Save tuning results
        results_path = f'{day_prefix}hyperparameter_tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.tuning_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING COMPLETE FOR {day_prefix}")
        print(f"{'='*60}")
        print(f"Best hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"Best validation loss: {best_score:.4f}")
        print(f"Results saved to: {results_path}")
        print(f"{'='*60}")
        
        return best_model, best_params

def train_optimized_lstm(X_train, y_train, X_val, y_val, input_size, 
                        day_prefix, tune_hyperparameters=True):

    if tune_hyperparameters:
        # Perform hyperparameter tuning
        tuner = LSTMHyperparameterTuner()
        best_model, best_params = tuner.tune_hyperparameters(
            X_train, y_train, X_val, y_val, input_size, day_prefix
        )
        
        # Save the best model
        model_path = f'{day_prefix}lstm_model_optimized.pth'
        torch.save(best_model.state_dict(), model_path)
        
        # Save hyperparameters
        params_path = f'{day_prefix}best_hyperparameters.json'
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"Optimized model saved: {model_path}")
        print(f"Best hyperparameters saved: {params_path}")
        
        return best_model, best_params
    
    else:
        # Use default hyperparameters (original model)
        print(f"Training with default hyperparameters for {day_prefix}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Initialize model with default parameters
        model = EnhancedLSTM(input_size=input_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = f'{day_prefix}lstm_model_default.pth'
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return model, None

if __name__ == "__main__":
    print("="*80)
    print("ENHANCED LSTM TRAINING WITH HYPERPARAMETER TUNING")
    print("="*80)
    print("Framework Compliance:")
    print("[OK] Test set remains untouched during training and validation")
    print("[OK] Hyperparameter tuning uses only train/val splits")
    print("[OK] Time series split methodology maintained")
    print("[OK] No data leakage prevention implemented")
    print("="*80)
    
    # Configuration
    TUNE_HYPERPARAMETERS = True  # Set to False for default training
    
    for prefix in DAY_PREFIXES:
        npz_path = f'{prefix}lstm_preprocessed_data.npz'
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}, skipping.")
            continue

        # Load preprocessed data
        data = np.load(npz_path, allow_pickle=True)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        # NOTE: X_test and y_test are NOT used during training/tuning
        
        input_size = X_train.shape[2]  # Number of features

        print(f"\nProcessing {prefix}:")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        print(f"Input features: {input_size}")
        print(f"Framework compliance: Test data not accessed")

        # Train with hyperparameter tuning or default parameters
        model, best_params = train_optimized_lstm(
            X_train, y_train, X_val, y_val, input_size, 
            prefix, tune_hyperparameters=TUNE_HYPERPARAMETERS
        )

        if best_params:
            print(f"Model for {prefix} trained with optimized hyperparameters")
        else:
            print(f"Model for {prefix} trained with default hyperparameters")
