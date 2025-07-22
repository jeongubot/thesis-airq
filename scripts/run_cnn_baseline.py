"""
Script to run CNN baseline training and comparison with CapsNet
"""
import sys
import os
from pathlib import Path
import argparse
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from training.cnn_baseline_trainer import CNNBaselineTrainer, compare_cnn_backbones

class Config:
    """Configuration for CNN baseline operations"""
    DAY_FOLDERS = ['7_24_data', '10_19_data', '11_10_data']
    BACKBONES = ['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16', 'mobilenet_v2']
    FEATURE_DIM = 128
    DEVICE = 'cuda'
    EPOCHS = 30
    BATCH_SIZE = 8

def main():
    parser = argparse.ArgumentParser(description='CNN Baseline for CapsNet Comparison')
    parser.add_argument('--mode', 
                        choices=['test', 'train', 'extract', 'compare', 'compare_all'],
                        default='test', 
                        help='Operation mode')
    parser.add_argument('--backbone', 
                        choices=Config.BACKBONES,
                        default='resnet50', 
                        help='CNN backbone architecture')
    parser.add_argument('--day', 
                        choices=Config.DAY_FOLDERS,
                        default=Config.DAY_FOLDERS[0], 
                        help='Day folder to process')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--feature_dim', type=int, default=Config.FEATURE_DIM,
                        help='Feature dimension')
    parser.add_argument('--device', type=str, default=Config.DEVICE,
                        help='Device to use (cuda/cpu)')
    
    # Model specific options
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone parameters')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Use multi-scale feature extraction')
    
    # Testing specific
    parser.add_argument('--test_samples', type=int, default=50,
                        help='Max samples for quick test')
    parser.add_argument('--test_epochs', type=int, default=3,
                        help='Epochs for quick test')
    
    # Feature extraction specific
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model for feature extraction')
    parser.add_argument('--split', choices=['learning', 'test'], default='learning',
                        help='Data split for feature extraction')
    
    # Comparison specific
    parser.add_argument('--compare_samples', type=int, default=100,
                        help='Samples for backbone comparison')
    parser.add_argument('--compare_epochs', type=int, default=5,
                        help='Epochs for backbone comparison')
    
    args = parser.parse_args()
    
    print("🏗️ CNN Baseline for CapsNet Comparison")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Backbone: {args.backbone}")
    print(f"Day: {args.day}")
    print(f"Device: {args.device}")
    print(f"Feature dimensions: {args.feature_dim}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Multi-scale: {args.multi_scale}")
    print("=" * 60)
    
    if args.mode == 'test':
        print(f"\n🧪 Running CNN Baseline Quick Test")
        
        trainer = CNNBaselineTrainer(
            backbone=args.backbone,
            feature_dim=args.feature_dim,
            device=args.device,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            multi_scale=args.multi_scale
        )
        
        success = trainer.quick_test(
            day_folder=args.day,
            max_samples=args.test_samples,
            max_epochs=args.test_epochs
        )
        
        if success:
            print(f"\n🎯 CNN baseline test PASSED! ✅")
        else:
            print(f"\n❌ CNN baseline test FAILED!")
    
    elif args.mode == 'train':
        print(f"\n🚀 Training CNN Baseline")
        
        trainer = CNNBaselineTrainer(
            backbone=args.backbone,
            feature_dim=args.feature_dim,
            device=args.device,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            multi_scale=args.multi_scale
        )
        
        # Prepare data
        train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
        
        # Create model
        trainer.create_model()
        trainer.setup_training()
        
        # Train
        best_loss = trainer.train(
            train_dataset, val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Plot results
        trainer.plot_training_history()
        
        print(f"\n✅ CNN baseline training completed!")
        print(f"   Backbone: {args.backbone}")
        print(f"   Best loss: {best_loss:.4f}")
    
    elif args.mode == 'extract':
        print(f"\n🔍 Extracting CNN Features")
        
        # Determine model path
        if args.model_path is None:
            args.model_path = f'best_cnn_{args.backbone}_feature_extractor.pth'
        
        if not os.path.exists(args.model_path):
            print(f"❌ Model not found: {args.model_path}")
            print(f"Please train a model first using --mode train")
            return
        
        trainer = CNNBaselineTrainer(
            backbone=args.backbone,
            feature_dim=args.feature_dim,
            device=args.device,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            multi_scale=args.multi_scale
        )
        
        # Create model and load weights
        trainer.create_model()
        trainer.load_model(args.model_path)
        
        # Prepare data
        if args.split == 'learning':
            train_dataset, _, _, _ = trainer.prepare_data(args.day)
            dataset = train_dataset
        else:
            _, val_dataset, _, _ = trainer.prepare_data(args.day)
            dataset = val_dataset
        
        # Extract features
        features, metadata = trainer.extract_features(dataset)
        
        # Save features
        import pandas as pd
        features_df = pd.DataFrame(
            features, 
            columns=[f'cnn_{args.backbone}_feature_{i}' for i in range(args.feature_dim)]
        )
        
        # Add metadata
        if metadata:
            for key in ['image_filename', 'timestamp', 'pm2.5', 'location']:
                if key in metadata[0]:
                    features_df[key] = [meta.get(key) for meta in metadata]
        
        output_path = f"cnn_{args.backbone}_features_{args.day}_{args.split}.csv"
        features_df.to_csv(output_path, index=False)
        
        print(f"✅ CNN features saved to: {output_path}")
        print(f"   Samples: {len(features_df)}")
        print(f"   Features: {args.feature_dim}")
        print(f"   Backbone: {args.backbone}")
    
    elif args.mode == 'compare':
        print(f"\n🏆 Comparing CNN Backbones")
        
        results = compare_cnn_backbones(
            day_folder=args.day,
            max_samples=args.compare_samples,
            epochs=args.compare_epochs
        )
        
        print(f"\n✅ CNN backbone comparison completed!")
        print(f"Results saved to: cnn_backbone_comparison_{args.day}.json")
    
    elif args.mode == 'compare_all':
        print(f"\n🌍 Comparing CNN Backbones on All Days")
        
        all_results = {}
        
        for day in Config.DAY_FOLDERS:
            print(f"\n--- Processing {day} ---")
            
            day_results = compare_cnn_backbones(
                day_folder=day,
                max_samples=args.compare_samples,
                epochs=args.compare_epochs
            )
            
            all_results[day] = day_results
        
        # Save global results
        global_results_path = "cnn_backbone_comparison_all_days.json"
        with open(global_results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 Global CNN Backbone Comparison Summary:")
        print("-" * 60)
        
        for day, day_results in all_results.items():
            print(f"\n{day}:")
            successful_backbones = [b for b, r in day_results.items() if r.get('success', False)]
            print(f"   Successful backbones: {successful_backbones}")
            
            if successful_backbones:
                # Find best performing backbone for this day
                best_backbone = min(
                    successful_backbones,
                    key=lambda b: day_results[b].get('final_val_loss', float('inf'))
                )
                best_loss = day_results[best_backbone].get('final_val_loss', 'N/A')
                print(f"   Best backbone: {best_backbone} (Val Loss: {best_loss:.4f})")
        
        print(f"\n📁 Global results saved to: {global_results_path}")
    
    print(f"\n🎉 CNN baseline operation completed!")

if __name__ == "__main__":
    main()
