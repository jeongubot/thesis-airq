"""
CNN Baseline Trainer for comparison with CapsNet
Reuses the same training infrastructure but with CNN models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Import the CNN baseline models
from ..models.cnn_baseline import create_cnn_baseline, CNNEnsembleFeatureExtractor, count_parameters

# Import the dataset and utilities from CapsNet trainer
from .capsnet_trainer import AirQualityDataset, custom_collate_fn


class CNNBaselineTrainer:
    """
    CNN Baseline Trainer for fair comparison with CapsNet
    Uses the same data pipeline and training procedures
    """
    
    def __init__(self, 
                 backbone='resnet50',
                 feature_dim=128, 
                 device='cuda',
                 pretrained=True,
                 freeze_backbone=False,
                 multi_scale=False):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.multi_scale = multi_scale
        
        print(f"🏗️ CNN Baseline Trainer")
        print(f"   Backbone: {backbone}")
        print(f"   Device: {self.device}")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Frozen backbone: {freeze_backbone}")
        print(f"   Multi-scale: {multi_scale}")
        
        # Initialize model components
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
        """Create CNN baseline feature extractor"""
        
        self.feature_extractor = create_cnn_baseline(
            backbone=self.backbone,
            feature_dim=self.feature_dim,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone,
            multi_scale=self.multi_scale,
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
        
        # Count parameters
        total_params, trainable_params = count_parameters(self.feature_extractor)
        regressor_params = sum(p.numel() for p in self.temp_regressor.parameters())
        
        print(f"   Feature extractor parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"   Regression head parameters: {regressor_params:,}")
        print(f"   Total parameters: {total_params + regressor_params:,}")
        
        return self.feature_extractor
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4, optimizer_type='adam'):
        """Setup optimizer and scheduler"""
        
        # Different learning rates for pretrained vs new layers
        if self.pretrained and not self.freeze_backbone:
            # Lower learning rate for pretrained backbone
            backbone_lr = learning_rate * 0.1
            head_lr = learning_rate
            
            params = [
                {'params': self.feature_extractor.backbone.parameters(), 'lr': backbone_lr},
                {'params': self.feature_extractor.feature_head.parameters(), 'lr': head_lr},
                {'params': self.temp_regressor.parameters(), 'lr': head_lr}
            ]
            
            print(f"   Using different learning rates: backbone={backbone_lr:.6f}, head={head_lr:.6f}")
        else:
            params = list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters())
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
    
    def load_day_data(self, day_folder):
        """Load data (reuse from CapsNet trainer)"""
        print(f"📂 Loading data for {day_folder}...")
        
        # Load learning data
        learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
        if not os.path.exists(learning_path):
            raise FileNotFoundError(f"Learning data not found: {learning_path}")
        
        learning_df = pd.read_csv(learning_path)
        print(f"   Learning data: {len(learning_df)} entries")
        
        # Load patch metadata
        patch_metadata_path = "dataset/e_preprocessed_img/patch_metadata.csv"
        if not os.path.exists(patch_metadata_path):
            raise FileNotFoundError(f"Patch metadata not found: {patch_metadata_path}")
        
        patch_metadata_df = pd.read_csv(patch_metadata_path)
        day_patches = patch_metadata_df[patch_metadata_df['day'] == day_folder]
        print(f"   Patches for {day_folder}: {len(day_patches)} patches")
        
        # Clean data
        original_count = len(learning_df)
        learning_df = learning_df.dropna(subset=['pm2.5'])
        learning_df = learning_df[learning_df['pm2.5'] > 0]
        learning_df = learning_df[learning_df['pm2.5'] < 500]
        
        print(f"   After cleaning: {len(learning_df)} valid entries (removed {original_count - len(learning_df)})")
        print(f"   PM2.5 range: {learning_df['pm2.5'].min():.2f} - {learning_df['pm2.5'].max():.2f}")
        
        return learning_df, patch_metadata_df
    
    def prepare_data(self, day_folder, test_size=0.2):
        """Prepare training and validation data"""
        print(f"🔄 Preparing data for {day_folder}...")
        
        learning_df, patch_metadata_df = self.load_day_data(day_folder)
        
        # Check if separate test file exists
        test_path = f"dataset/d_data_split/{day_folder}/test.csv"
        if os.path.exists(test_path):
            # Use pre-split test data
            test_df = pd.read_csv(test_path)
            test_df = test_df.dropna(subset=['pm2.5'])
            test_df = test_df[test_df['pm2.5'] > 0]
            test_df = test_df[test_df['pm2.5'] < 500]
            
            train_dataset = AirQualityDataset(learning_df, patch_metadata_df, day_folder, 'learning')
            val_dataset = AirQualityDataset(test_df, patch_metadata_df, day_folder, 'test')
        else:
            # Split learning data
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(learning_df, test_size=test_size, random_state=42)
            
            train_dataset = AirQualityDataset(train_df, patch_metadata_df, day_folder, 'train')
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
        
        progress_bar = tqdm(dataloader, desc=f"Training ({self.backbone})", leave=False)
        
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
        
        progress_bar = tqdm(dataloader, desc=f"Validation ({self.backbone})", leave=False)
        
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
        print(f"🚀 Starting CNN baseline training...")
        print(f"   Backbone: {self.backbone}")
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
                    self.save_model(f'best_cnn_{self.backbone}_feature_extractor.pth', epoch, val_loss)
                    print("✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs")
                    
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping at epoch {epoch+1}")
                        break
                        
            except Exception as e:
                print(f"❌ Error in epoch {epoch+1}: {e}")
                continue
        
        print(f"\n🎉 CNN baseline training completed!")
        print(f"   Backbone: {self.backbone}")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
    
    def extract_features(self, dataset, batch_size=16):
        """Extract features using trained CNN"""
        print(f"🔍 Extracting CNN features from {len(dataset)} samples...")
        
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
            for batch_images, batch_targets, batch_metadata in tqdm(dataloader, desc="Extracting CNN features"):
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
        
        print(f"✅ Extracted {len(all_features)} CNN feature vectors")
        return all_features, all_metadata
    
    def save_model(self, filepath, epoch, val_loss):
        """Save trained model"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.feature_extractor.state_dict(),
            'backbone': self.backbone,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'feature_dim': self.feature_dim,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'multi_scale': self.multi_scale
        }, filepath)
        print(f"💾 CNN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 CNN model loaded from {filepath}")
        return checkpoint
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        if save_path is None:
            save_path = f'cnn_{self.backbone}_training_history.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title(f'CNN {self.backbone} - Training Loss')
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
        print(f"📊 CNN training history saved as '{save_path}'")
        plt.show()
    
    def quick_test(self, day_folder, max_samples=50, max_epochs=3):
        """Quick test of CNN baseline pipeline"""
        print(f"\n🧪 CNN Baseline Quick Test for {day_folder}")
        print(f"   Backbone: {self.backbone}")
        print("=" * 50)
        
        try:
            # Test data loading
            print("1. Testing data loading...")
            train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
            
            if max_samples and len(train_dataset) > max_samples:
                indices = torch.randperm(len(train_dataset))[:max_samples]
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            print(f"   ✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
            
            # Test model creation
            print("2. Testing model creation...")
            self.create_model()
            self.setup_training()
            print("   ✅ CNN model created successfully")
            
            # Test training loop
            print("3. Testing training loop...")
            best_loss = self.train(train_dataset, val_dataset, epochs=max_epochs, batch_size=4)
            print(f"   ✅ Training completed, best loss: {best_loss:.4f}")
            
            # Test feature extraction
            print("4. Testing feature extraction...")
            features, metadata = self.extract_features(train_dataset, batch_size=4)
            print(f"   ✅ Features extracted: {len(features)} samples")
            
            print(f"\n🎯 CNN baseline quick test PASSED! ({self.backbone})")
            return True
            
        except Exception as e:
            print(f"\n❌ CNN baseline quick test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def compare_cnn_backbones(day_folder, max_samples=100, epochs=5):
    """
    Compare different CNN backbones on the same data
    """
    print(f"🏆 CNN Backbone Comparison on {day_folder}")
    print("=" * 60)
    
    backbones = ['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']
    results = {}
    
    for backbone in backbones:
        print(f"\n🔍 Testing {backbone}...")
        
        try:
            trainer = CNNBaselineTrainer(
                backbone=backbone,
                feature_dim=128,
                device='cuda',
                pretrained=True,
                freeze_backbone=False
            )
            
            # Quick test
            success = trainer.quick_test(day_folder, max_samples, epochs)
            
            if success:
                results[backbone] = {
                    'success': True,
                    'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
                    'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
                    'final_val_rmse': trainer.val_metrics[-1]['rmse'] if trainer.val_metrics else None,
                    'final_val_r2': trainer.val_metrics[-1]['r2'] if trainer.val_metrics else None
                }
            else:
                results[backbone] = {'success': False}
                
        except Exception as e:
            print(f"❌ {backbone} failed: {e}")
            results[backbone] = {'success': False, 'error': str(e)}
    
    # Print comparison
    print(f"\n📊 CNN Backbone Comparison Results:")
    print("-" * 60)
    print(f"{'Backbone':<15} {'Success':<8} {'Val Loss':<10} {'Val RMSE':<10} {'Val R²':<8}")
    print("-" * 60)
    
    for backbone, result in results.items():
        if result['success']:
            val_loss = result.get('final_val_loss', 'N/A')
            val_rmse = result.get('final_val_rmse', 'N/A')
            val_r2 = result.get('final_val_r2', 'N/A')
            
            val_loss_str = f"{val_loss:.4f}" if val_loss != 'N/A' else 'N/A'
            val_rmse_str = f"{val_rmse:.4f}" if val_rmse != 'N/A' else 'N/A'
            val_r2_str = f"{val_r2:.4f}" if val_r2 != 'N/A' else 'N/A'
            
            print(f"{backbone:<15} {'✅':<8} {val_loss_str:<10} {val_rmse_str:<10} {val_r2_str:<8}")
        else:
            print(f"{backbone:<15} {'❌':<8} {'Failed':<10} {'Failed':<10} {'Failed':<8}")
    
    # Save results
    results_path = f"cnn_backbone_comparison_{day_folder}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_path}")
    return results


if __name__ == "__main__":
    # Run CNN backbone comparison
    compare_cnn_backbones('7_24_data', max_samples=50, epochs=3)
