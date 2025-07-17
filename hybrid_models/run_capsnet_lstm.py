"""
CapsNet-LSTM Individual Model Evaluator
Run this script to evaluate only the CapsNet-LSTM model
"""

from hybrid_models.individual_model_evaluators import CapsNetLSTMEvaluator


def main():
    """
    Run CapsNet-LSTM evaluation
    """
    print("="*100)
    print("CapsNet-LSTM MODEL EVALUATION")
    print("="*100)
    print("This script will evaluate the CapsNet-LSTM model using 5-fold cross-validation")
    print("across all available days: 7_24, 10_19, 11_10")
    print("="*100)
    
    # Initialize evaluator
    evaluator = CapsNetLSTMEvaluator(
        days=['7_24', '10_19', '11_10'],
        cv_folds=5
    )
    
    # Run evaluation
    evaluator.run_complete_evaluation()
    
    print("\n" + "="*100)
    print("CapsNet-LSTM EVALUATION COMPLETE")
    print("="*100)
    print("Results have been saved to: results/capsnet_lstm_results.json")
    print("="*100)


if __name__ == "__main__":
    main()
