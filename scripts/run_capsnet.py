#!/usr/bin/env python3
"""
CapsNet Feature Extractor - Main Runner Script
Handles training, testing, and feature extraction for CapsNet models
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from training.capsnet_trainer import CapsNetTrainer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='CapsNet Feature Extractor')
    parser.add_argument('--mode', choices=['train', 'test', 'extract', 'tune'], 
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
    
    args = parser.parse_args()
    
    print("🚀 CapsNet Feature Extractor")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Day: {args.day}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = CapsNetTrainer(
        input_size=256,
        feature_dim=args.feature_dim,
        device=args.device
    )
    
    try:
        if args.mode == 'test':
            print("\n🧪 Running Quick Test")
            print("-" * 30)
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
            
            # Prepare data
            train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
            
            # Create model
            trainer.create_model()
            trainer.setup_training(learning_rate=args.learning_rate)
            
            # Train
            best_loss = trainer.train(
                train_dataset, val_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Plot training history
            trainer.plot_training_history(f'capsnet_training_{args.day}.png')
            
            print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
            
        elif args.mode == 'extract':
            print("\n🔍 Extracting Features")
            print("-" * 30)
            
            # Load trained model
            model_path = 'best_capsnet_feature_extractor.pth'
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                print("Please train a model first using --mode train")
                sys.exit(1)
            
            # Prepare data
            train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
            
            # Create and load model
            trainer.create_model()
            trainer.load_model(model_path)
            
            # Extract features
            print("Extracting training features...")
            train_features, train_metadata = trainer.extract_features(train_dataset)
            
            print("Extracting validation features...")
            val_features, val_metadata = trainer.extract_features(val_dataset)
            
            # Save features
            import numpy as np
            import pandas as pd
            
            # Save training features
            np.save(f'capsnet_train_features_{args.day}.npy', train_features)
            pd.DataFrame(train_metadata).to_csv(f'capsnet_train_metadata_{args.day}.csv', index=False)
            
            # Save validation features
            np.save(f'capsnet_val_features_{args.day}.npy', val_features)
            pd.DataFrame(val_metadata).to_csv(f'capsnet_val_metadata_{args.day}.csv', index=False)
            
            print(f"\n✅ Features extracted and saved!")
            print(f"Training: {len(train_features)} samples")
            print(f"Validation: {len(val_features)} samples")
            
        elif args.mode == 'tune':
            print("\n🔧 Hyperparameter Tuning")
            print("-" * 30)
            
            best_params = trainer.hyperparameter_tune(
                day_folder=args.day,
                n_trials=args.n_trials,
                max_epochs=args.tune_epochs
            )
            
            if best_params:
                print("\n🏆 Training final model with best parameters...")
                final_model_path = trainer.train_with_best_params(
                    args.day, best_params, epochs=args.epochs
                )
                print(f"✅ Final model saved: {final_model_path}")
            else:
                print("❌ Hyperparameter tuning failed!")
                sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error in {args.mode} mode: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
