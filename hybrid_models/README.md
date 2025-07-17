# HYBRID_MODELS NOT FINAL CODE GUYS , PINAVISUALIZE KO LANG YUNG OVERALL CODES 
# Individual Model Evaluators

This directory contains individual model evaluators that allow you to run each of the three hybrid models separately, providing you with the **choice** to run all models together (using the original `nested_cv_pipeline.py`) or run each model independently.

## Available Models

1. **CapsNet-LSTM-LightGBM**: Hybrid model combining CapsNet spatial features, LSTM temporal features, and LightGBM regression
2. **CNN-LSTM-LightGBM**: Hybrid model combining CNN spatial features, LSTM temporal features, and LightGBM regression  
3. **CapsNet-LSTM**: Hybrid model combining CapsNet spatial features and LSTM temporal features (without LightGBM)

## Files Structure

```
model/
├── nested_cv_pipeline.py              # Original nested CV pipeline (all models together)
├── individual_model_evaluators.py     # Base classes for individual model evaluation
├── run_capsnet_lstm_lightgbm.py      # Run CapsNet-LSTM-LightGBM only
├── run_cnn_lstm_lightgbm.py          # Run CNN-LSTM-LightGBM only
├── run_capsnet_lstm.py               # Run CapsNet-LSTM only
├── compare_models.py                  # Compare results from all models
├── README_individual_models.md        # This file
└── results/                           # Directory for saved results
    ├── capsnet_lstm_lightgbm_results.json
    ├── cnn_lstm_lightgbm_results.json
    ├── capsnet_lstm_results.json
    └── model_comparison.csv
```

## Usage Options

### Option 1: Run All Models Together (Original Approach)
```bash
python nested_cv_pipeline.py
```
This runs all three models in sequence using the original nested cross-validation pipeline.

### Option 2: Run Individual Models Separately

#### Run CapsNet-LSTM-LightGBM only:
```bash
python run_capsnet_lstm_lightgbm.py
```

#### Run CNN-LSTM-LightGBM only:
```bash
python run_cnn_lstm_lightgbm.py
```

#### Run CapsNet-LSTM only:
```bash
python run_capsnet_lstm.py
```

#### Run all models with interactive selection:
```bash
python individual_model_evaluators.py
```

### Option 3: Compare Results
After running individual models, compare their performance:
```bash
python compare_models.py
```

## Methodology

Each model follows the same evaluation methodology:

### 5-Fold Cross-Validation
- **Outer Loop**: 5 folds using TimeSeriesSplit for final evaluation
- **Inner Loop**: 3 folds for hyperparameter tuning
- **Output**: 5 performance scores per model for statistical comparison

### Evaluation Process
1. **Data Loading**: Load temporal data for each day (7_24, 10_19, 11_10)
2. **Time Series Split**: Create 5 folds maintaining temporal order
3. **For Each Fold**:
   - **Hyperparameter Tuning**: Inner 3-fold CV to find optimal parameters
   - **Model Training**: Train with optimal parameters
   - **Prediction**: Generate predictions for test set
   - **Evaluation**: Calculate RMSE, MAE, and R² scores
4. **Statistical Analysis**: Generate summary statistics across all folds

### Output Format
Each model generates:
- **Individual fold results**: Parameters, predictions, and metrics for each fold
- **Summary statistics**: Mean and standard deviation across folds
- **JSON file**: Complete results saved for later analysis
- **Console output**: Real-time progress and results

## Statistical Comparison

The comparison script provides:
- **Performance Table**: RMSE and R² scores for all models and days
- **Best Model Analysis**: Overall best performing model identification
- **Statistical Tests**: Paired t-tests between models
- **Effect Size Analysis**: Cohen's d for practical significance
- **CSV Export**: Results exported for further analysis

## Examples

### Running Individual Models

```bash
# Run only CapsNet-LSTM-LightGBM
python run_capsnet_lstm_lightgbm.py

# Expected output:
# ================================================================================
# CapsNet-LSTM-LightGBM MODEL EVALUATION
# ================================================================================
# 
# ================================================================================
# EVALUATING CapsNet-LSTM-LightGBM FOR 7_24
# ================================================================================
# Cross-validation folds: 5
# 
# --- FOLD 1/5 ---
# Train size: 1000, Test size: 200
#   Hyperparameter tuning...
#   Optimal parameters: {'lstm': {...}, 'lightgbm': {...}}
#   Training and predicting...
#   Fold 1 results:
#     RMSE: 15.2341
#     MAE: 12.1234
#     R²: 0.7845
# ...
```

### Comparing Results

```bash
python compare_models.py

# Expected output:
# ================================================================================
# MODEL COMPARISON ANALYSIS
# ================================================================================
# 
# ========================================================================================================================
# COMPREHENSIVE MODEL COMPARISON TABLE
# ========================================================================================================================
# Model                     | Day      | RMSE Mean    | RMSE Std     | R² Mean      | R² Std       | Success
# ------------------------------------------------------------------------------------------------------------------------
# CapsNet-LSTM-LightGBM     | 7_24     | 15.2341      | 1.2345       | 0.7845       | 0.0234       | 5/5
# CapsNet-LSTM-LightGBM     | 10_19    | 16.1234      | 1.3456       | 0.7654       | 0.0345       | 5/5
# ...
```

## Key Features

### Individual Model Benefits
- **Focused Evaluation**: Evaluate specific models of interest
- **Faster Execution**: Run only the models you need
- **Detailed Analysis**: Deep dive into specific model performance
- **Resource Efficiency**: Use computational resources more efficiently

### Statistical Rigor
- **Proper Cross-Validation**: TimeSeriesSplit respects temporal order
- **Hyperparameter Tuning**: Nested CV prevents overfitting
- **Statistical Testing**: Paired t-tests for significance
- **Effect Size Analysis**: Practical significance assessment

### Reproducibility
- **Saved Results**: All results saved in JSON format
- **Parameter Tracking**: Optimal parameters saved for each fold
- **Seed Control**: (Can be added) Random seed control for reproducibility
- **Detailed Logs**: Complete execution logs for debugging

## Notes

1. **Spatial Features**: Currently using placeholder spatial features. Replace with actual CapsNet/CNN implementations.
2. **Hyperparameter Spaces**: Optimized for reasonable execution time. Expand as needed.
3. **Memory Usage**: Models store all results in memory. Consider disk-based storage for large datasets.
4. **Parallel Execution**: Models can be run in parallel on different machines/cores.

## Future Enhancements

- [ ] Implement actual CapsNet spatial feature extraction
- [ ] Implement actual CNN spatial feature extraction
- [ ] Add parallel processing for faster execution
- [ ] Add more sophisticated hyperparameter optimization (e.g., Bayesian optimization)
- [ ] Add model interpretability features
- [ ] Add early stopping for LSTM training
- [ ] Add cross-day evaluation (train on one day, test on another)

This design gives you the flexibility to run models individually while maintaining the option to use the original nested CV pipeline for comprehensive evaluation.
