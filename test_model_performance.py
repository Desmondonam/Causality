import pytest
import sys
sys.path.insert(0, './src')
from causal_ml_analysis import BreastCancerCausalML

def test_model_training():
    """Test if models train successfully"""
    analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
    analyzer.load_and_prepare_data()
    analyzer.split_and_scale_data()
    analyzer.perform_feature_selection()
    analyzer.train_models()
    
    assert len(analyzer.models) > 0, "No models were trained"
    assert len(analyzer.results) > 0, "No results were generated"
    print("✓ Model training test passed")

def test_model_accuracy():
    """Test if models achieve minimum accuracy threshold"""
    analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
    analyzer.load_and_prepare_data()
    analyzer.split_and_scale_data()
    analyzer.perform_feature_selection()
    analyzer.train_models()
    
    min_accuracy = 0.90  # 90% minimum accuracy
    for model_name, results in analyzer.results.items():
        assert results['accuracy'] >= min_accuracy, \
            f"{model_name} accuracy {results['accuracy']:.2f} below threshold {min_accuracy}"
    print(f"✓ All models achieved >{min_accuracy*100}% accuracy")

def test_model_roc_auc():
    """Test if models achieve minimum ROC-AUC threshold"""
    analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
    analyzer.load_and_prepare_data()
    analyzer.split_and_scale_data()
    analyzer.perform_feature_selection()
    analyzer.train_models()
    
    min_roc_auc = 0.95  # 95% minimum ROC-AUC
    for model_name, results in analyzer.results.items():
        assert results['roc_auc'] >= min_roc_auc, \
            f"{model_name} ROC-AUC {results['roc_auc']:.2f} below threshold {min_roc_auc}"
    print(f"✓ All models achieved >{min_roc_auc*100}% ROC-AUC")

if __name__ == "__main__":
    test_model_training()
    test_model_accuracy()
    test_model_roc_auc()
    print("\n✓ All model performance tests passed!")