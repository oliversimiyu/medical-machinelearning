"""
Synthetic data generation for medical insurance fraud detection
"""

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=10000, fraud_ratio=0.05, random_state=42):
    """
    Generate synthetic medical insurance fraud dataset
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    fraud_ratio : float
        Proportion of fraudulent claims
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels (0=Legitimate, 1=Fraudulent)
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Legitimate claims (normal distributions)
    legitimate_data = {
        'claim_amount': np.random.gamma(2, 1000, n_legitimate),
        'patient_age': np.random.normal(45, 15, n_legitimate),
        'num_procedures': np.random.poisson(2, n_legitimate),
        'hospital_stay_days': np.random.poisson(3, n_legitimate),
        'num_previous_claims': np.random.poisson(1.5, n_legitimate),
        'provider_claim_count': np.random.poisson(50, n_legitimate),
        'diagnosis_complexity': np.random.uniform(0, 1, n_legitimate),
        'treatment_cost_ratio': np.random.normal(1.0, 0.2, n_legitimate),
        'claim_processing_time': np.random.normal(15, 5, n_legitimate),
        'geographic_risk_score': np.random.uniform(0, 0.5, n_legitimate),
    }
    
    # Fraudulent claims (different distributions - suspicious patterns)
    fraud_data = {
        'claim_amount': np.random.gamma(3, 2000, n_fraud),
        'patient_age': np.random.normal(50, 20, n_fraud),
        'num_procedures': np.random.poisson(4, n_fraud),
        'hospital_stay_days': np.random.poisson(5, n_fraud),
        'num_previous_claims': np.random.poisson(3, n_fraud),
        'provider_claim_count': np.random.poisson(100, n_fraud),
        'diagnosis_complexity': np.random.uniform(0.3, 1, n_fraud),
        'treatment_cost_ratio': np.random.normal(1.5, 0.4, n_fraud),
        'claim_processing_time': np.random.normal(10, 3, n_fraud),
        'geographic_risk_score': np.random.uniform(0.4, 1, n_fraud),
    }
    
    # Combine datasets
    X_legit = pd.DataFrame(legitimate_data)
    X_fraud = pd.DataFrame(fraud_data)
    X = pd.concat([X_legit, X_fraud], ignore_index=True)
    
    # Create target variable
    y = pd.Series([0] * n_legitimate + [1] * n_fraud, name='is_fraud')
    
    # Shuffle data
    shuffle_idx = np.random.permutation(n_samples)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    
    # Clip values to realistic ranges
    X['claim_amount'] = X['claim_amount'].clip(lower=100, upper=50000)
    X['patient_age'] = X['patient_age'].clip(lower=0, upper=100)
    X['num_procedures'] = X['num_procedures'].clip(lower=1, upper=20)
    X['hospital_stay_days'] = X['hospital_stay_days'].clip(lower=0, upper=30)
    X['num_previous_claims'] = X['num_previous_claims'].clip(lower=0, upper=20)
    X['provider_claim_count'] = X['provider_claim_count'].clip(lower=1, upper=500)
    X['claim_processing_time'] = X['claim_processing_time'].clip(lower=1, upper=60)
    
    return X, y


if __name__ == "__main__":
    # Test data generation
    X, y = generate_synthetic_data(n_samples=1000)
    print(f"Generated {len(X)} samples")
    print(f"Fraud ratio: {y.sum() / len(y) * 100:.2f}%")
    print(f"\nFeatures:\n{X.head()}")
