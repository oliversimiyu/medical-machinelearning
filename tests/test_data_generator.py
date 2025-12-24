"""
Tests for data generation module
"""

import pytest
import pandas as pd
from src.data.data_generator import generate_synthetic_data


def test_data_generation_shape():
    """Test that generated data has correct shape"""
    n_samples = 1000
    X, y = generate_synthetic_data(n_samples=n_samples)
    
    assert len(X) == n_samples
    assert len(y) == n_samples
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_fraud_ratio():
    """Test that fraud ratio is approximately correct"""
    fraud_ratio = 0.1
    X, y = generate_synthetic_data(n_samples=10000, fraud_ratio=fraud_ratio)
    
    actual_ratio = y.sum() / len(y)
    assert abs(actual_ratio - fraud_ratio) < 0.02  # Within 2%


def test_feature_columns():
    """Test that all expected features are present"""
    X, y = generate_synthetic_data(n_samples=100)
    
    expected_features = [
        'claim_amount', 'patient_age', 'num_procedures',
        'hospital_stay_days', 'num_previous_claims', 'provider_claim_count',
        'diagnosis_complexity', 'treatment_cost_ratio', 
        'claim_processing_time', 'geographic_risk_score'
    ]
    
    for feature in expected_features:
        assert feature in X.columns


def test_value_ranges():
    """Test that feature values are within expected ranges"""
    X, y = generate_synthetic_data(n_samples=1000)
    
    assert X['claim_amount'].min() >= 100
    assert X['claim_amount'].max() <= 50000
    assert X['patient_age'].min() >= 0
    assert X['patient_age'].max() <= 100
    assert X['num_procedures'].min() >= 1
    assert X['hospital_stay_days'].min() >= 0


def test_reproducibility():
    """Test that random state produces reproducible results"""
    X1, y1 = generate_synthetic_data(n_samples=100, random_state=42)
    X2, y2 = generate_synthetic_data(n_samples=100, random_state=42)
    
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
