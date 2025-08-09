"""
Unit tests for data validation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidator

def test_schema_validation(sample_data, config):
    """Test schema validation."""
    validator = DataValidator(config)
    
    # Test valid data
    is_valid, errors = validator.validate_schema(sample_data)
    assert is_valid == True
    assert len(errors) == 0
    
    # Test missing required column
    invalid_data = sample_data.drop('churn', axis=1)
    is_valid, errors = validator.validate_schema(invalid_data)
    assert is_valid == False
    assert len(errors) > 0

def test_data_quality_validation(sample_data, config):
    """Test data quality validation."""
    validator = DataValidator(config)
    
    is_valid, metrics = validator.validate_data_quality(sample_data)
    assert is_valid == True
    assert 'max_missing_percentage' in metrics
    assert 'duplicate_percentage' in metrics

def test_target_distribution_validation(sample_data, config):
    """Test target distribution validation."""
    validator = DataValidator(config)
    
    is_valid, metrics = validator.validate_target_distribution(sample_data, 'churn')
    assert is_valid == True
    assert 'unique_values' in metrics
    assert 'value_counts' in metrics
