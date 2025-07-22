#!/bin/bash

# Complete CapsNet Feature Extraction Pipeline
# Run this script to execute the entire pipeline for all days

echo "🚀 Starting Complete CapsNet Feature Extraction Pipeline"
echo "========================================================"

# Configuration
DAYS=("7_24_data" "10_19_data" "11_10_data")
BACKBONES=("resnet18" "resnet50" "efficientnet_b0")
EPOCHS=30
BATCH_SIZE=8
TEST_SAMPLES=50
TEST_EPOCHS=3

# Step 1: Test all pipelines
echo ""
echo "📋 Step 1: Testing All Pipelines"
echo "================================"

echo "Testing CapsNet on all days..."
for day in "${DAYS[@]}"; do
    echo "  Testing CapsNet on $day"
    python scripts/run_capsnet.py --mode test --day $day --test_samples $TEST_SAMPLES --test_epochs $TEST_EPOCHS
    
    if [ $? -ne 0 ]; then
        echo "❌ CapsNet test failed for $day"
        exit 1
    fi
done

echo "Testing CNN baselines on all days..."
for day in "${DAYS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        echo "  Testing CNN $backbone on $day"
        python scripts/run_cnn_baseline.py --mode test --backbone $backbone --day $day --test_samples $TEST_SAMPLES --test_epochs $TEST_EPOCHS
        
        if [ $? -ne 0 ]; then
            echo "❌ CNN $backbone test failed for $day"
            exit 1
        fi
    done
done

echo "✅ All pipeline tests passed!"

# Step 2: Train models
echo ""
echo "📋 Step 2: Training Models"
echo "=========================="

echo "Training CapsNet on all days..."
for day in "${DAYS[@]}"; do
    echo "  Training CapsNet on $day"
    python scripts/run_capsnet.py --mode train --day $day --epochs $EPOCHS --batch_size $BATCH_SIZE
    
    if [ $? -ne 0 ]; then
        echo "❌ CapsNet training failed for $day"
        exit 1
    fi
done

echo "Training best CNN backbone (ResNet50) on all days..."
for day in "${DAYS[@]}"; do
    echo "  Training CNN ResNet50 on $day"
    python scripts/run_cnn_baseline.py --mode train --backbone resnet50 --day $day --epochs $EPOCHS --batch_size $BATCH_SIZE
    
    if [ $? -ne 0 ]; then
        echo "❌ CNN ResNet50 training failed for $day"
        exit 1
    fi
done

echo "✅ All model training completed!"

# Step 3: Extract features
echo ""
echo "📋 Step 3: Extracting Features"
echo "==============================="

echo "Extracting CapsNet features..."
for day in "${DAYS[@]}"; do
    for split in "learning" "test"; do
        echo "  Extracting CapsNet features for $day ($split)"
        python scripts/run_capsnet.py --mode extract --day $day --split $split
        
        if [ $? -ne 0 ]; then
            echo "❌ CapsNet feature extraction failed for $day ($split)"
            exit 1
        fi
    done
done

echo "Extracting CNN features..."
for day in "${DAYS[@]}"; do
    for split in "learning" "test"; do
        echo "  Extracting CNN ResNet50 features for $day ($split)"
        python scripts/run_cnn_baseline.py --mode extract --backbone resnet50 --day $day --split $split
        
        if [ $? -ne 0 ]; then
            echo "❌ CNN feature extraction failed for $day ($split)"
            exit 1
        fi
    done
done

echo "✅ All feature extraction completed!"

# Step 4: Compare models
echo ""
echo "📋 Step 4: Model Comparison"
echo "==========================="

echo "Comparing CapsNet vs CNN on all days..."
for day in "${DAYS[@]}"; do
    echo "  Comparing models on $day"
    python scripts/compare_capsnet_vs_cnn.py --day $day --epochs 10 --max_samples 200
    
    if [ $? -ne 0 ]; then
        echo "⚠️ Model comparison failed for $day (continuing...)"
    fi
done

echo "✅ Model comparison completed!"

# Step 5: Integration with LSTM and LightGBM
echo ""
echo "📋 Step 5: Integration with LSTM and LightGBM"
echo "=============================================="

echo "Running feature integration and LightGBM training..."
python integration_guide.py

if [ $? -ne 0 ]; then
    echo "⚠️ Integration step failed (this is expected if LSTM features are not available)"
    echo "   Please run the integration manually after preparing LSTM features"
fi

# Step 6: Summary
echo ""
echo "📋 Step 6: Pipeline Summary"
echo "==========================="

echo "🎉 Complete CapsNet Feature Extraction Pipeline Finished!"
echo ""
echo "📁 Generated Files:"
echo "   Models:"
echo "     - best_capsnet_feature_extractor.pth"
echo "     - best_cnn_resnet50_feature_extractor.pth"
echo ""
echo "   Features:"
for day in "${DAYS[@]}"; do
    for split in "learning" "test"; do
        echo "     - capsnet_features_${day}_${split}.csv"
        echo "     - cnn_resnet50_features_${day}_${split}.csv"
    done
done
echo ""
echo "   Comparisons:"
for day in "${DAYS[@]}"; do
    echo "     - capsnet_vs_cnn_results_${day}.json"
    echo "     - capsnet_vs_cnn_comparison_${day}.png"
    echo "     - model_comparison_summary_${day}.csv"
done
echo ""
echo "   Integration (if LSTM features available):"
echo "     - integrated_features/ directory"
echo "     - lightgbm_results/ directory"
echo ""
echo "🚀 Next Steps:"
echo "   1. Review the generated feature files"
echo "   2. Integrate CapsNet features with your LSTM pipeline"
echo "   3. Train LightGBM with combined features"
echo "   4. Evaluate final model performance"
echo ""
echo "📖 See README.md for detailed integration instructions"
