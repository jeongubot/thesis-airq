"""
Direct comparison between CapsNet and CNN baselines
"""
import sys
import os
from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from training.capsnet_trainer import CapsNetTrainer
from training.cnn_baseline_trainer import CNNBaselineTrainer

def run_model_comparison(day_folder, epochs=10, max_samples=200):
    """
    Run direct comparison between CapsNet and CNN baselines
    """
    print(f"⚔️ CapsNet vs CNN Baseline Comparison")
    print(f"   Day: {day_folder}")
    print(f"   Epochs: {epochs}")
    print(f"   Max samples: {max_samples}")
    print("=" * 60)
    
    results = {}
    
    # Test CapsNet
    print(f"\n🔮 Testing CapsNet...")
    try:
        capsnet_trainer = CapsNetTrainer(
            input_size=256,
            feature_dim=128,
            device='cuda'
        )
        
        # Prepare data
        train_dataset, val_dataset, _, _ = capsnet_trainer.prepare_data(day_folder)
        
        # Limit samples for fair comparison
        if max_samples and len(train_dataset) > max_samples:
            indices = torch.randperm(len(train_dataset))[:max_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        # Create and train model
        capsnet_trainer.create_model()
        capsnet_trainer.setup_training()
        
        capsnet_best_loss = capsnet_trainer.train(
            train_dataset, val_dataset,
            epochs=epochs,
            batch_size=8
        )
        
        results['capsnet'] = {
            'success': True,
            'best_val_loss': capsnet_best_loss,
            'final_train_loss': capsnet_trainer.train_losses[-1],
            'final_val_loss': capsnet_trainer.val_losses[-1],
            'final_val_rmse': capsnet_trainer.val_metrics[-1]['rmse'],
            'final_val_r2': capsnet_trainer.val_metrics[-1]['r2'],
            'training_history': {
                'train_losses': capsnet_trainer.train_losses,
                'val_losses': capsnet_trainer.val_losses,
                'val_rmse': [m['rmse'] for m in capsnet_trainer.val_metrics],
                'val_r2': [m['r2'] for m in capsnet_trainer.val_metrics]
            }
        }
        
        print(f"✅ CapsNet completed - Val Loss: {capsnet_best_loss:.4f}")
        
    except Exception as e:
        print(f"❌ CapsNet failed: {e}")
        results['capsnet'] = {'success': False, 'error': str(e)}
    
    # Test CNN baselines
    cnn_backbones = ['resnet18', 'resnet50', 'efficientnet_b0']
    
    for backbone in cnn_backbones:
        print(f"\n🏗️ Testing CNN {backbone}...")
        
        try:
            cnn_trainer = CNNBaselineTrainer(
                backbone=backbone,
                feature_dim=128,
                device='cuda',
                pretrained=True,
                freeze_backbone=False
            )
            
            # Use same data as CapsNet
            train_dataset, val_dataset, _, _ = cnn_trainer.prepare_data(day_folder)
            
            # Limit samples for fair comparison
            if max_samples and len(train_dataset) > max_samples:
                indices = torch.randperm(len(train_dataset))[:max_samples]
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            # Create and train model
            cnn_trainer.create_model()
            cnn_trainer.setup_training()
            
            cnn_best_loss = cnn_trainer.train(
                train_dataset, val_dataset,
                epochs=epochs,
                batch_size=8
            )
            
            results[f'cnn_{backbone}'] = {
                'success': True,
                'backbone': backbone,
                'best_val_loss': cnn_best_loss,
                'final_train_loss': cnn_trainer.train_losses[-1],
                'final_val_loss': cnn_trainer.val_losses[-1],
                'final_val_rmse': cnn_trainer.val_metrics[-1]['rmse'],
                'final_val_r2': cnn_trainer.val_metrics[-1]['r2'],
                'training_history': {
                    'train_losses': cnn_trainer.train_losses,
                    'val_losses': cnn_trainer.val_losses,
                    'val_rmse': [m['rmse'] for m in cnn_trainer.val_metrics],
                    'val_r2': [m['r2'] for m in cnn_trainer.val_metrics]
                }
            }
            
            print(f"✅ CNN {backbone} completed - Val Loss: {cnn_best_loss:.4f}")
            
        except Exception as e:
            print(f"❌ CNN {backbone} failed: {e}")
            results[f'cnn_{backbone}'] = {'success': False, 'error': str(e)}
    
    return results

def create_comparison_plots(results, day_folder):
    """
    Create comprehensive comparison plots
    """
    print(f"\n📊 Creating comparison plots...")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if len(successful_results) < 2:
        print("⚠️ Not enough successful results for comparison plots")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CapsNet vs CNN Baseline Comparison - {day_folder}', fontsize=16)
    
    # 1. Final Performance Comparison (Bar Chart)
    models = list(successful_results.keys())
    val_losses = [successful_results[m]['final_val_loss'] for m in models]
    val_rmse = [successful_results[m]['final_val_rmse'] for m in models]
    val_r2 = [successful_results[m]['final_val_r2'] for m in models]
    
    # Validation Loss
    axes[0, 0].bar(models, val_losses, color=['red' if 'capsnet' in m else 'blue' for m in models])
    axes[0, 0].set_title('Final Validation Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[0, 1].bar(models, val_rmse, color=['red' if 'capsnet' in m else 'blue' for m in models])
    axes[0, 1].set_title('Final Validation RMSE')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[0, 2].bar(models, val_r2, color=['red' if 'capsnet' in m else 'blue' for m in models])
    axes[0, 2].set_title('Final Validation R²')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 2. Training History Comparison
    # Validation Loss over epochs
    for model, result in successful_results.items():
        history = result['training_history']
        color = 'red' if 'capsnet' in model else 'blue'
        linestyle = '-' if 'capsnet' in model else '--'
        axes[1, 0].plot(history['val_losses'], label=model, color=color, linestyle=linestyle)
    
    axes[1, 0].set_title('Validation Loss Over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSE over epochs
    for model, result in successful_results.items():
        history = result['training_history']
        color = 'red' if 'capsnet' in model else 'blue'
        linestyle = '-' if 'capsnet' in model else '--'
        axes[1, 1].plot(history['val_rmse'], label=model, color=color, linestyle=linestyle)
    
    axes[1, 1].set_title('Validation RMSE Over Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # R² over epochs
    for model, result in successful_results.items():
        history = result['training_history']
        color = 'red' if 'capsnet' in model else 'blue'
        linestyle = '-' if 'capsnet' in model else '--'
        axes[1, 2].plot(history['val_r2'], label=model, color=color, linestyle=linestyle)
    
    axes[1, 2].set_title('Validation R² Over Epochs')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('R²')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'capsnet_vs_cnn_comparison_{day_folder}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_path}")
    plt.show()
    
    return plot_path

def create_summary_table(results, day_folder):
    """
    Create summary table of results
    """
    print(f"\n📋 Creating summary table...")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("⚠️ No successful results for summary table")
        return None
    
    # Create summary data
    summary_data = []
    
    for model, result in successful_results.items():
        model_type = 'CapsNet' if 'capsnet' in model else 'CNN'
        backbone = result.get('backbone', 'CapsNet') if model_type == 'CNN' else 'CapsNet'
        
        summary_data.append({
            'Model': model,
            'Type': model_type,
            'Backbone': backbone,
            'Final Val Loss': f"{result['final_val_loss']:.4f}",
            'Final Val RMSE': f"{result['final_val_rmse']:.4f}",
            'Final Val R²': f"{result['final_val_r2']:.4f}",
            'Best Val Loss': f"{result['best_val_loss']:.4f}"
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by best validation loss
    summary_df = summary_df.sort_values('Best Val Loss')
    
    # Print table
    print(f"\n📊 Model Comparison Summary - {day_folder}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    csv_path = f'model_comparison_summary_{day_folder}.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n📁 Summary table saved to: {csv_path}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='CapsNet vs CNN Baseline Comparison')
    parser.add_argument('--day', 
                        choices=['7_24_data', '10_19_data', '11_10_data'],
                        default='7_24_data', 
                        help='Day folder to process')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Maximum samples for comparison')
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models')
    
    args = parser.parse_args()
    
    print("⚔️ CapsNet vs CNN Baseline Comparison")
    print("=" * 60)
    print(f"Day: {args.day}")
    print(f"Epochs: {args.epochs}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 60)
    
    # Run comparison
    results = run_model_comparison(
        day_folder=args.day,
        epochs=args.epochs,
        max_samples=args.max_samples
    )
    
    # Save results
    results_path = f'capsnet_vs_cnn_results_{args.day}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Raw results saved to: {results_path}")
    
    # Create comparison plots
    plot_path = create_comparison_plots(results, args.day)
    
    # Create summary table
    summary_df = create_summary_table(results, args.day)
    
    # Print final summary
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        print(f"\n🏆 Final Results Summary:")
        print("-" * 40)
        
        # Find best model
        best_model = min(successful_results.keys(), 
                        key=lambda x: successful_results[x]['best_val_loss'])
        best_loss = successful_results[best_model]['best_val_loss']
        
        print(f"🥇 Best Model: {best_model}")
        print(f"   Best Val Loss: {best_loss:.4f}")
        print(f"   Final Val RMSE: {successful_results[best_model]['final_val_rmse']:.4f}")
        print(f"   Final Val R²: {successful_results[best_model]['final_val_r2']:.4f}")
        
        # Compare CapsNet vs best CNN
        if 'capsnet' in successful_results:
            capsnet_loss = successful_results['capsnet']['best_val_loss']
            cnn_results = {k: v for k, v in successful_results.items() if 'cnn_' in k}
            
            if cnn_results:
                best_cnn = min(cnn_results.keys(), key=lambda x: cnn_results[x]['best_val_loss'])
                best_cnn_loss = cnn_results[best_cnn]['best_val_loss']
                
                print(f"\n📊 CapsNet vs Best CNN:")
                print(f"   CapsNet Loss: {capsnet_loss:.4f}")
                print(f"   Best CNN ({best_cnn}): {best_cnn_loss:.4f}")
                
                if capsnet_loss < best_cnn_loss:
                    improvement = ((best_cnn_loss - capsnet_loss) / best_cnn_loss) * 100
                    print(f"   🎯 CapsNet is {improvement:.1f}% better!")
                else:
                    degradation = ((capsnet_loss - best_cnn_loss) / best_cnn_loss) * 100
                    print(f"   📉 CapsNet is {degradation:.1f}% worse")
    
    print(f"\n🎉 Comparison completed!")
    print(f"📁 Check these files:")
    print(f"   - Results: {results_path}")
    if 'plot_path' in locals():
        print(f"   - Plots: {plot_path}")
    if summary_df is not None:
        print(f"   - Summary: model_comparison_summary_{args.day}.csv")

if __name__ == "__main__":
    main()
