"""
Complete CapsNet Training, Testing, and Hyperparameter Tuning
Combined all training-related functionality into one file
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import traceback
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import Optuna for hyperparameter tuning (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Hyperparameter tuning will be disabled.")

# Import model
from models.capsnet_model import create_capsnet_feature_extractor


def custom_collate_fn(batch):
    """Optimized custom collate function"""
    images = []
    pm25_values = []
    metadata_list = []
    
    for item in batch:
        try:
            img, pm25, metadata = item
            images.append(img)
            pm25_values.append(pm25)
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Error in collate_fn for item: {e}")
            continue
    
    if not images:
        return torch.empty(0, 3, 256, 256), torch.empty(0, 1), []
    
    # Stack tensors
    images = torch.stack(images, 0)
    pm25_values = torch.stack(pm25_values, 0)
    
    return images, pm25_values, metadata_list


class AirQualityDataset(Dataset):
    """
    Air Quality Dataset that correctly follows the data flow
    """
    def __init__(self, learning_df, patch_metadata_df, day_folder, split_type='learning', transform=None):
        self.learning_df = learning_df.reset_index(drop=True)
        self.patch_metadata_df = patch_metadata_df.reset_index(drop=True)
        self.day_folder = day_folder
        self.split_type = split_type
        self.transform = transform
        
        # Filter patch metadata for this specific day
        self.day_patches = self.patch_metadata_df[
            self.patch_metadata_df['day'] == day_folder
        ].reset_index(drop=True)
        
        print(f"📊 Dataset initialization for {day_folder} ({split_type}):")
        print(f"   Input learning data: {len(self.learning_df)} entries")
        print(f"   Available patches for day: {len(self.day_patches)} patches")
        
        # Create data mapping
        self.data_mapping = self._create_data_mapping()
        
        print(f"   Final dataset size: {len(self.data_mapping)} samples")
        if len(self.learning_df) > 0:
            print(f"   Expansion factor: {len(self.data_mapping) / len(self.learning_df):.1f}x")
    
    def _create_data_mapping(self):
        """Create mapping between learning data entries and patch files"""
        mapping = []
        images_with_patches = 0
        images_without_patches = 0
        
        print(f"   📊 Creating data mapping...")
        print(f"   Processing {len(self.learning_df)} learning entries...")
        
        # Pre-group patches by image filename for faster lookup
        patches_by_image = self.day_patches.groupby('image_filename')
        
        # Add progress bar for mapping creation
        progress_bar = tqdm(
            self.learning_df.iterrows(), 
            total=len(self.learning_df),
            desc="   Mapping data",
            leave=False,
            ncols=100
        )
        
        for idx, (_, learning_row) in enumerate(progress_bar):
            image_filename = learning_row['image_filename']
            
            # Find all patches for this image using pre-grouped data
            if image_filename in patches_by_image.groups:
                image_patches = patches_by_image.get_group(image_filename)
                images_with_patches += 1
                
                # Create one entry per patch (each patch represents one augmented version)
                for _, patch_row in image_patches.iterrows():
                    mapping.append({
                        # From learning data
                        'image_filename': image_filename,
                        'timestamp': learning_row['timestamp'],
                        'pm2.5': float(learning_row['pm2.5']),
                        'pm10': float(learning_row.get('pm10', 0)),
                        'temperature': float(learning_row.get('temperature', 0)),
                        'humidity': float(learning_row.get('humidity', 0)),
                        'location': learning_row.get('location', 'unknown'),
                        
                        # From patch metadata
                        'patch_idx': int(patch_row['patch_idx']),
                        'augmentations': patch_row['augmentations'],
                        'npy_path': patch_row['npy_path']
                    })
            else:
                images_without_patches += 1
        
        # Update progress bar with current stats every 10 items
        if (idx + 1) % 10 == 0:
            progress_bar.set_postfix({
                'mapped': len(mapping),
                'with_patches': images_with_patches,
                'without': images_without_patches,
                'avg_patches_per_img': f"{len(mapping)/max(images_with_patches, 1):.1f}"
            })
    
        progress_bar.close()
        
        print(f"   ✅ Mapping completed:")
        print(f"      Total samples: {len(mapping)}")
        print(f"      Images with patches: {images_with_patches}")
        print(f"      Images without patches: {images_without_patches}")
        if images_with_patches > 0:
            avg_patches = len(mapping) / images_with_patches
            print(f"      Average patches per image: {avg_patches:.1f}")
            print(f"      This means each image has ~{avg_patches:.0f} augmented versions")
    
        return mapping
    
    def __len__(self):
        return len(self.data_mapping)
    
    def __getitem__(self, idx):
        data_entry = self.data_mapping[idx]
        
        # Load preprocessed .npy file
        npy_path = data_entry['npy_path']
        if not os.path.isabs(npy_path):
            npy_path = os.path.join('dataset', 'e_preprocessed_img', npy_path)
        
        # Load the preprocessed image
        try:
            image_data = np.load(npy_path)
            
            # Ensure correct format: (C, H, W)
            if len(image_data.shape) == 3:
                if image_data.shape[0] == 3:  # Already CHW
                    image = torch.FloatTensor(image_data)
                elif image_data.shape[2] == 3:  # HWC -> CHW
                    image = torch.FloatTensor(image_data).permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected image shape: {image_data.shape}")
            else:
                raise ValueError(f"Expected 3D image, got shape: {image_data.shape}")
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            # Ensure correct size (256x256)
            if image.shape[1] != 256 or image.shape[2] != 256:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)
            
        except Exception as e:
            print(f"❌ Error loading {npy_path}: {e}")
            # Return dummy data as fallback
            image = torch.zeros(3, 256, 256, dtype=torch.float32)
        
        # Get PM2.5 target
        pm25_val = data_entry['pm2.5']
        if pd.isna(pm25_val) or pm25_val <= 0:
            pm25_val = 1.0  # Fallback value
        
        pm25 = torch.FloatTensor([pm25_val])
        
        return image, pm25, data_entry


class CapsNetTrainer:
    """
    Complete CapsNet Trainer with training, testing, and hyperparameter tuning
    """
    
    def __init__(self, input_size=256, feature_dim=128, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        print(f"🚀 CapsNet Trainer")
        print(f"   Device: {self.device}")
        print(f"   Input size: {input_size}x{input_size}")
        print(f"   Feature dimension: {feature_dim}")
        
        # Initialize model
        self.feature_extractor = None
        self.temp_regressor = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def create_model(self, **model_params):
        """Create CapsNet feature extractor model"""
        self.feature_extractor = create_capsnet_feature_extractor(
            input_channels=3,
            input_size=self.input_size,
            feature_dim=self.feature_dim,
            **model_params
        ).to(self.device)
        
        # Temporary regression head for training
        dropout_rate = model_params.get('dropout_rate', 0.3)
        self.temp_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        return self.feature_extractor
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4, optimizer_type='adam'):
        """Setup optimizer and scheduler"""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay, momentum=0.9
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )
    
    def load_day_data(self, day_folder):
        """Load pre-split data from d_data_split"""
        print(f"📂 Loading data for {day_folder}...")
        
        # Load learning data (80% split)
        learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
        if not os.path.exists(learning_path):
            raise FileNotFoundError(f"Learning data not found: {learning_path}")
        
        print(f"   📄 Loading learning data from {learning_path}")
        learning_df = pd.read_csv(learning_path)
        print(f"   ✅ Learning data loaded: {len(learning_df)} entries")
        
        # Load patch metadata
        patch_metadata_path = "dataset/e_preprocessed_img/patch_metadata.csv"
        if not os.path.exists(patch_metadata_path):
            raise FileNotFoundError(f"Patch metadata not found: {patch_metadata_path}")
        
        print(f"   📄 Loading patch metadata from {patch_metadata_path}")
        patch_metadata_df = pd.read_csv(patch_metadata_path)
        day_patches = patch_metadata_df[patch_metadata_df['day'] == day_folder]
        print(f"   ✅ Patches for {day_folder}: {len(day_patches)} patches")
        
        # Clean data with progress
        print(f"   🧹 Cleaning data...")
        original_count = len(learning_df)
        learning_df = learning_df.dropna(subset=['pm2.5'])
        after_dropna = len(learning_df)
        learning_df = learning_df[learning_df['pm2.5'] > 0]
        after_positive = len(learning_df)
        learning_df = learning_df[learning_df['pm2.5'] < 500]
        final_count = len(learning_df)
        
        print(f"   ✅ Data cleaning complete:")
        print(f"      Original: {original_count} entries")
        print(f"      After dropna: {after_dropna} entries (removed {original_count - after_dropna})")
        print(f"      After >0 filter: {after_positive} entries (removed {after_dropna - after_positive})")
        print(f"      Final: {final_count} entries (removed {after_positive - final_count})")
        print(f"      PM2.5 range: {learning_df['pm2.5'].min():.2f} - {learning_df['pm2.5'].max():.2f}")
        
        return learning_df, patch_metadata_df
    
    def prepare_data(self, day_folder, test_size=0.2):
        """Prepare training and validation data"""
        print(f"🔄 Preparing data for {day_folder}...")
        
        learning_df, patch_metadata_df = self.load_day_data(day_folder)
        
        # Check if separate test file exists
        test_path = f"dataset/d_data_split/{day_folder}/test.csv"
        if os.path.exists(test_path):
            print(f"   📂 Using pre-split test data from {test_path}")
            # Use pre-split test data
            test_df = pd.read_csv(test_path)
            test_df = test_df.dropna(subset=['pm2.5'])
            test_df = test_df[test_df['pm2.5'] > 0]
            test_df = test_df[test_df['pm2.5'] < 500]
            
            print(f"   🔄 Creating training dataset...")
            train_dataset = AirQualityDataset(learning_df, patch_metadata_df, day_folder, 'learning')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(test_df, patch_metadata_df, day_folder, 'test')
        else:
            print(f"   📂 Splitting learning data ({test_size*100:.0f}% for validation)")
            # Split learning data
            train_df, val_df = train_test_split(learning_df, test_size=test_size, random_state=42)
            
            print(f"   🔄 Creating training dataset...")
            train_dataset = AirQualityDataset(train_df, patch_metadata_df, day_folder, 'train')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(val_df, patch_metadata_df, day_folder, 'val')
        
        print(f"✅ Data preparation complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset, learning_df, patch_metadata_df
    
    def train_epoch(self, dataloader, gradient_clip_norm=1.0):
        """Training epoch"""
        self.feature_extractor.train()
        self.temp_regressor.train()
        
        total_loss = 0
        predictions = []
        targets = []
        successful_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_images, batch_targets, batch_metadata in progress_bar:
            try:
                batch_images = batch_images.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                if batch_images.size(0) == 0:
                    continue
                
                self.optimizer.zero_grad()
                
                # Extract features and predict
                features = self.feature_extractor(batch_images)
                pred = self.temp_regressor(features)
                loss = self.criterion(pred.squeeze(), batch_targets.squeeze())
                
                if torch.isnan(loss):
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                    max_norm=gradient_clip_norm
                )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predictions.extend(pred.squeeze().cpu().detach().numpy())
                targets.extend(batch_targets.squeeze().cpu().detach().numpy())
                successful_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"❌ Error in training batch: {e}")
                continue
        
        if successful_batches == 0:
            raise RuntimeError("No successful training batches!")
        
        # Calculate metrics
        avg_loss = total_loss / successful_batches
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def validate_epoch(self, dataloader):
        """Validation epoch"""
        self.feature_extractor.eval()
        self.temp_regressor.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        successful_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_images, batch_targets, batch_metadata in progress_bar:
                try:
                    batch_images = batch_images.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    if batch_images.size(0) == 0:
                        continue
                    
                    features = self.feature_extractor(batch_images)
                    pred = self.temp_regressor(features)
                    loss = self.criterion(pred.squeeze(), batch_targets.squeeze())
                    
                    if torch.isnan(loss):
                        continue
                    
                    total_loss += loss.item()
                    predictions.extend(pred.squeeze().cpu().numpy())
                    targets.extend(batch_targets.squeeze().cpu().numpy())
                    successful_batches += 1
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    continue
        
        if successful_batches == 0:
            raise RuntimeError("No successful validation batches!")
        
        avg_loss = total_loss / successful_batches
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train(self, train_dataset, val_dataset, epochs=30, batch_size=8, **training_params):
        """Main training loop"""
        print(f"🚀 Starting CapsNet training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=custom_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=custom_collate_fn
        )
        
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = training_params.get('early_stopping_patience', 10)
        gradient_clip_norm = training_params.get('gradient_clip_norm', 1.0)
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            try:
                # Training
                train_loss, train_metrics = self.train_epoch(train_loader, gradient_clip_norm)
                
                # Validation
                val_loss, val_metrics = self.validate_epoch(val_loader)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Save metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_metrics.append(train_metrics)
                self.val_metrics.append(val_metrics)
                
                # Print results
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
                print(f"Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_capsnet_feature_extractor.pth', epoch, val_loss)
                    print("✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs")
                    
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping at epoch {epoch+1}")
                        break
                        
            except Exception as e:
                print(f"❌ Error in epoch {epoch+1}: {e}")
                traceback.print_exc()
                continue
        
        print("\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
    
    def extract_features(self, dataset, batch_size=16):
        """Extract features using trained model"""
        print(f"🔍 Extracting features from {len(dataset)} samples...")
        
        self.feature_extractor.eval()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=custom_collate_fn
        )
        
        all_features = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_images, batch_targets, batch_metadata in tqdm(dataloader, desc="Extracting features"):
                try:
                    batch_images = batch_images.to(self.device)
                    
                    if batch_images.size(0) == 0:
                        continue
                    
                    features = self.feature_extractor(batch_images)
                    all_features.extend(features.cpu().numpy())
                    all_metadata.extend(batch_metadata)
                    
                except Exception as e:
                    print(f"❌ Error extracting features: {e}")
                    continue
        
        print(f"✅ Extracted {len(all_features)} feature vectors")
        return all_features, all_metadata
    
    def save_model(self, filepath, epoch, val_loss):
        """Save trained model"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.feature_extractor.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Model loaded from {filepath}")
        return checkpoint
    
    def plot_training_history(self, save_path='capsnet_training_history.png'):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RMSE
        train_rmse = [m['rmse'] for m in self.train_metrics]
        val_rmse = [m['rmse'] for m in self.val_metrics]
        axes[0, 1].plot(train_rmse, label='Train RMSE', color='blue')
        axes[0, 1].plot(val_rmse, label='Val RMSE', color='red')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAE
        train_mae = [m['mae'] for m in self.train_metrics]
        val_mae = [m['mae'] for m in self.val_metrics]
        axes[1, 0].plot(train_mae, label='Train MAE', color='blue')
        axes[1, 0].plot(val_mae, label='Val MAE', color='red')
        axes[1, 0].set_title('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R²
        train_r2 = [m['r2'] for m in self.train_metrics]
        val_r2 = [m['r2'] for m in self.val_metrics]
        axes[1, 1].plot(train_r2, label='Train R²', color='blue')
        axes[1, 1].plot(val_r2, label='Val R²', color='red')
        axes[1, 1].set_title('R² Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved as '{save_path}'")
        plt.show()
    
    # TESTING FUNCTIONALITY
    def quick_test(self, day_folder, max_samples=50, max_epochs=3):
        """Quick test of the complete pipeline"""
        print(f"\n🧪 Quick Test for {day_folder}")
        print("=" * 50)
        
        try:
            # Test data loading
            print("1. Testing data loading...")
            train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
            
            if max_samples and len(train_dataset) > max_samples:
                # Create smaller dataset for testing
                indices = torch.randperm(len(train_dataset))[:max_samples]
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            print(f"   ✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
            
            # Test model creation
            print("2. Testing model creation...")
            self.create_model()
            self.setup_training()
            print("   ✅ Model created successfully")
            
            # Test training loop
            print("3. Testing training loop...")
            best_loss = self.train(train_dataset, val_dataset, epochs=max_epochs, batch_size=4)
            print(f"   ✅ Training completed, best loss: {best_loss:.4f}")
            
            # Test feature extraction
            print("4. Testing feature extraction...")
            features, metadata = self.extract_features(train_dataset, batch_size=4)
            print(f"   ✅ Features extracted: {len(features)} samples")
            
            print("\n🎯 Quick test PASSED! Pipeline is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick test FAILED: {e}")
            traceback.print_exc()
            return False
    
    # HYPERPARAMETER TUNING (if Optuna is available)
    def hyperparameter_tune(self, day_folder, n_trials=50, max_epochs=25):
        """Run hyperparameter tuning using Optuna"""
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available. Cannot run hyperparameter tuning.")
            return None
        
        print(f"🔍 Starting hyperparameter tuning for {day_folder}")
        print(f"   Trials: {n_trials}")
        print(f"   Max epochs per trial: {max_epochs}")
        
        # Load data once
        learning_df, patch_metadata_df = self.load_day_data(day_folder)
        
        # Limit data for faster tuning
        if len(learning_df) > 1000:
            learning_df = learning_df.sample(n=1000, random_state=42)
        
        def objective(trial):
            """Optuna objective function"""
            try:
                # Suggest hyperparameters
                params = {
                    'feature_dim': trial.suggest_categorical('feature_dim', [64, 128, 256]),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
                    'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
                    'gradient_clip_norm': trial.suggest_float('gradient_clip_norm', 0.5, 2.0),
                    'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 15),
                }
                
                # Create datasets
                full_dataset = AirQualityDataset(learning_df, patch_metadata_df, day_folder, 'learning')
                
                # Simple train/val split for tuning
                dataset_size = len(full_dataset)
                train_size = int(0.8 * dataset_size)
                val_size = dataset_size - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )
                
                # Create trainer for this trial
                trial_trainer = CapsNetTrainer(
                    input_size=self.input_size,
                    feature_dim=params['feature_dim'],
                    device=self.device.type
                )
                
                # Create model with trial parameters
                trial_trainer.create_model(dropout_rate=params['dropout_rate'])
                trial_trainer.setup_training(
                    learning_rate=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    optimizer_type=params['optimizer_type']
                )
                
                # Train with trial parameters
                best_val_loss = trial_trainer.train(
                    train_dataset, val_dataset,
                    epochs=max_epochs,
                    batch_size=params['batch_size'],
                    gradient_clip_norm=params['gradient_clip_norm'],
                    early_stopping_patience=params['early_stopping_patience']
                )
                
                return best_val_loss
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n🎯 Hyperparameter tuning completed!")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best parameters:")
        for key, value in best_params.items():
            print(f"     {key}: {value}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'day_folder': day_folder,
            'n_trials': n_trials
        }
        
        results_path = f"hyperparameter_results_{day_folder}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: {results_path}")
        
        return best_params
    
    def train_with_best_params(self, day_folder, best_params, epochs=50):
        """Train final model with best hyperparameters"""
        print(f"🏆 Training final model with best parameters...")
        
        # Update trainer with best parameters
        self.feature_dim = best_params.get('feature_dim', self.feature_dim)
        
        # Prepare data
        train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
        
        # Create model with best parameters
        self.create_model(dropout_rate=best_params.get('dropout_rate', 0.3))
        self.setup_training(
            learning_rate=best_params.get('learning_rate', 0.001),
            weight_decay=best_params.get('weight_decay', 1e-4),
            optimizer_type=best_params.get('optimizer_type', 'adam')
        )
        
        # Train with best parameters
        best_val_loss = self.train(
            train_dataset, val_dataset,
            epochs=epochs,
            batch_size=best_params.get('batch_size', 8),
            gradient_clip_norm=best_params.get('gradient_clip_norm', 1.0),
            early_stopping_patience=best_params.get('early_stopping_patience', 10)
        )
        
        # Save final model
        final_model_path = f"best_capsnet_tuned_{day_folder}.pth"
        self.save_model(final_model_path, epochs, best_val_loss)
        
        print(f"🎉 Final model saved to: {final_model_path}")
        return final_model_path
