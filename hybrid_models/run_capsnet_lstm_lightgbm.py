"""
CapsNet-LSTM-LightGBM Individual Model Evaluator
Run this script to evaluate only the CapsNet-LSTM-LightGBM model
"""

from hybrid_models.individual_model_evaluators import CapsNetLSTMLightGBMEvaluator


def main():
    """
    Run CapsNet-LSTM-LightGBM evaluation
    """
    print("="*100)
    print("CapsNet-LSTM-LightGBM MODEL EVALUATION")
    print("="*100)
    print("This script will evaluate the CapsNet-LSTM-LightGBM model using 5-fold cross-validation")
    print("across all available days: 7_24, 10_19, 11_10")
    print("="*100)
    
    # Initialize evaluator
    evaluator = CapsNetLSTMLightGBMEvaluator(
        days=['7_24', '10_19', '11_10'],
        cv_folds=5
    )
    
    # Run evaluation
    evaluator.run_complete_evaluation()
    
    print("\n" + "="*100)
    print("CapsNet-LSTM-LightGBM EVALUATION COMPLETE")
    print("="*100)
    print("Results have been saved to: results/capsnet_lstm_lightgbm_results.json")
    print("="*100)


if __name__ == "__main__":
    main()
