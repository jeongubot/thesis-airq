#!/usr/bin/env python3
"""CapsNet Feature Extractor - Main Runner Script
Handles training, testing, and feature extraction for CapsNet models"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Also add the parent directory to handle different import scenarios
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    # Try multiple import approaches
    try:
        from training.capsnet_trainer import CapsNetTrainer
    except ImportError:
        from src.training.capsnet_trainer import CapsNetTrainer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Try to add more paths
    possible_paths = [
        os.path.join(os.getcwd(), 'src'),
        os.path.join(os.getcwd(), 'src', 'training'),
        os.path.join(os.getcwd(), 'src', 'models'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
    
    try:
        from training.capsnet_trainer import CapsNetTrainer
    except ImportError:
        try:
            from training.capsnet_trainer import CapsNetTrainer
        except ImportError:
            print("❌ Could not import CapsNetTrainer. Please check your file structure.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='CapsNet Feature Extractor')
    parser.add_argument('--mode', choices=['train', 'test', 'extract', 'tune', 'trunk', 'trunk_optimized'],
                        required=True, help='Operation mode')
    parser.add_argument('--day', required=True, help='Day folder to process')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    # Test mode specific arguments
    parser.add_argument('--test_samples', type=int, default=50, help='Max samples for testing')
    parser.add_argument('--test_epochs', type=int, default=3, help='Max epochs for testing')
    
    # Tuning specific arguments
    parser.add_argument('--n_trials', type=int, default=50, help='Number of tuning trials')
    parser.add_argument('--tune_epochs', type=int, default=25, help='Max epochs per trial')
    
    # Trunk training specific arguments
    parser.add_argument('--trunk_size', type=int, default=10000, help='Samples per trunk for memory management')
    parser.add_argument('--epochs_per_trunk', type=int, default=5, help='Epochs per trunk')
    parser.add_argument('--use_trunk', action='store_true', help='Use trunk-based training for large datasets')
    
    # OPTIMIZATION arguments
    parser.add_argument('--max_patches_per_image', type=int, default=50, help='Max patches per image for optimization')
    parser.add_argument('--optimized_trunk_size', type=int, default=50000, help='Larger trunk size for optimization')
    parser.add_argument('--optimized_batch_size', type=int, default=16, help='Larger batch size for optimization')
    
    # Output organization
    parser.add_argument('--output_dir', default='outputs', help='Base output directory')
    parser.add_argument('--model_type', default='capsnet', help='Model type for organization')
    
    args = parser.parse_args()
    
    print("🚀 CapsNet Feature Extractor")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Day: {args.day}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    if args.use_trunk or args.mode == 'trunk':
        print(f"Trunk size: {args.trunk_size:,}")
        print(f"Epochs per trunk: {args.epochs_per_trunk}")
    print("=" * 50)
    
    # Initialize trainer with organized outputs
    trainer = CapsNetTrainer(
        input_size=256,
        feature_dim=args.feature_dim,
        device=args.device,
        model_type=args.model_type
    )
    
    try:
        if args.mode == 'test':
            print("\n🧪 Running Quick Test")
            print("-" * 30)
            
            if args.use_trunk:
                print("Using trunk-based testing...")
                success = trainer.quick_test_trunk(
                    day_folder=args.day,
                    max_samples=args.test_samples,
                    max_epochs=args.test_epochs,
                    trunk_size=min(args.trunk_size, 1000)  # Smaller trunk for testing
                )
            else:
                success = trainer.quick_test(
                    day_folder=args.day,
                    max_samples=args.test_samples,
                    max_epochs=args.test_epochs
                )
            
            if success:
                print("\n✅ Test completed successfully!")
            else:
                print("\n❌ Test failed!")
                sys.exit(1)
        
        elif args.mode == 'train':
            print("\n🏋️ Training CapsNet")
            print("-" * 30)
            
            # Create model
            trainer.create_model()
            trainer.setup_training(learning_rate=args.learning_rate)
            
            if args.use_trunk:
                print("Using trunk-based training for large dataset...")
                
                # Prepare trunk data
                trunk_train_dataset, val_dataset, _, _ = trainer.prepare_trunk_data(
                    args.day, trunk_size=args.trunk_size
                )
                
                print(f"Dataset info:")
                print(f"   Total samples: {trunk_train_dataset.total_samples:,}")
                print(f"   Trunks: {trunk_train_dataset.get_trunk_count()}")
                print(f"   Validation samples: {len(val_dataset)}")
                
                # Confirm before starting (for large datasets)
                if trunk_train_dataset.total_samples > 100000:
                    response = input(f"\nThis will process {trunk_train_dataset.total_samples:,} samples. Continue? (y/N): ")
                    if response.lower() != 'y':
                        print("Training cancelled.")
                        sys.exit(0)
                
                # Train with trunks - simplified for testing
                print("Starting simplified trunk training...")
                best_loss = float('inf')
                
                # Create data loaders
                from torch.utils.data import DataLoader
                from training.capsnet_trainer import custom_collate_fn
                
                train_loader = DataLoader(
                    trunk_train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=custom_collate_fn,
                    drop_last=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=custom_collate_fn
                )
                
                # Simple training loop for testing
                for epoch in range(args.epochs_per_trunk):
                    print(f"Epoch {epoch+1}/{args.epochs_per_trunk}")
                    train_loss, train_metrics = trainer.train_epoch(train_loader, 1.0)
                    val_loss, val_metrics = trainer.validate_epoch(val_loader)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                    
                    print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"   Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
                
            else:
                print("Using standard training...")
                
                # Prepare data
                train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
                
                # Train
                best_loss = trainer.train(
                    train_dataset, val_dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    day_folder=args.day
                )
            
            # Plot training history
            trainer.plot_training_history(day_folder=args.day)
            
            print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
        
        elif args.mode == 'extract':
            print("\n🔍 Extracting Features")
            print("-" * 30)
            
            # Load trained model
            model_path = trainer.output_manager.get_path('models', 'best', f'best_capsnet_{args.day}.pth', args.day)
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    'best_capsnet_feature_extractor.pth',
                    trainer.output_manager.get_path('models', 'final', f'final_capsnet_{args.day}.pth', args.day)
                ]
                
                model_path = None
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
                
                if model_path is None:
                    print(f"❌ No trained model found. Please train a model first using --mode train")
                    sys.exit(1)
            
            # Create and load model
            trainer.create_model()
            trainer.load_model(model_path)
            
            # Prepare data
            train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
            
            # Extract features
            print("Extracting training features...")
            train_features, train_metadata = trainer.extract_features(
                train_dataset, day_folder=args.day, split_name='train'
            )
            
            print("Extracting validation features...")
            val_features, val_metadata = trainer.extract_features(
                val_dataset, day_folder=args.day, split_name='val'
            )
            
            print(f"\n✅ Features extracted and saved!")
            print(f"Training: {len(train_features)} samples")
            print(f"Validation: {len(val_features)} samples")
    
    except Exception as e:
        print(f"\n❌ Error in {args.mode} mode: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
