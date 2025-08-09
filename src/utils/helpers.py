"""
Helper utilities for the MLOps pipeline.
"""
import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def ensure_dir(directory: Union[str, Path]) -> None:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to file.
    
    Args:
        model: Model object
        filepath: Path to save model
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model from file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save object
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_prob is not None:
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['log_loss'] = log_loss(y_true, y_prob)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def split_data(
    df: pd.DataFrame, 
    target_col: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        train_size: Training set size
        val_size: Validation set size
        test_size: Test set size
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Validate sizes
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_feature_names(features: List[str], prefix: str = "") -> List[str]:
    """
    Create feature names with optional prefix.
    
    Args:
        features: List of feature names
        prefix: Prefix to add to feature names
        
    Returns:
        List of feature names with prefix
    """
    if prefix:
        return [f"{prefix}_{feature}" for feature in features]
    return features


def validate_data_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate dataframe schema.
    
    Args:
        df: Input dataframe
        required_columns: List of required columns
        
    Returns:
        True if schema is valid
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get model information.
    
    Args:
        model: Model object
        
    Returns:
        Dictionary with model information
    """
    info = {
        'model_type': type(model).__name__,
        'model_module': type(model).__module__,
        'timestamp': get_timestamp()
    }
    
    # Try to get model parameters
    if hasattr(model, 'get_params'):
        info['parameters'] = model.get_params()
    
    return info
