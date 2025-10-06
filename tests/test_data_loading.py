import pytest
import pandas as pd
import sys
sys.path.insert(0, './src')
from causal_ml_analysis import BreastCancerCausalML

def test_data_loading():
    """Test if data loads correctly"""
    try:
        analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
        analyzer.load_and_prepare_data()
        assert analyzer.df is not None
        assert len(analyzer.df) > 0
        assert 'diagnosis' in analyzer.df.columns
        print("✓ Data loading test passed")
    except Exception as e:
        pytest.fail(f"Data loading failed: {str(e)}")

def test_data_shape():
    """Test if data has correct shape"""
    analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
    analyzer.load_and_prepare_data()
    assert analyzer.X.shape[1] > 0, "Features should exist"
    assert len(analyzer.y) == len(analyzer.X), "Target and features length mismatch"
    print("✓ Data shape test passed")

def test_target_encoding():
    """Test if target variable is properly encoded"""
    analyzer = BreastCancerCausalML('data/breast_cancer_data.csv')
    analyzer.load_and_prepare_data()
    assert set(analyzer.y.unique()).issubset({0, 1}), "Target should be binary (0, 1)"
    print("✓ Target encoding test passed")

if __name__ == "__main__":
    test_data_loading()
    test_data_shape()
    test_target_encoding()
    print("\n✓ All data tests passed!")