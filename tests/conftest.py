"""
Pytest configuration and fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

@pytest.fixture
def sample_data():
    """Create sample customer churn data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'phone_service': np.random.choice([0, 1], n_samples),
        'multiple_lines': np.random.choice([0, 1], n_samples),
        'internet_service': np.random.choice(['No', 'DSL', 'Fiber optic'], n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        'data': {
            'raw_path': 'data/raw',
            'processed_path': 'data/processed',
            'validation': {
                'max_missing_percentage': 0.1,
                'min_rows': 50,
                'required_columns': ['customer_id', 'churn']
            }
        },
        'model': {
            'algorithm': 'random_forest',
            'hyperparameters': {
                'n_estimators': 10,
                'random_state': 42
            }
        }
    }
