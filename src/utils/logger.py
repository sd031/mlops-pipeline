"""
Logging utilities for the MLOps pipeline.
"""
import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml
from pythonjsonlogger import jsonlogger


def setup_logging(
    config_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
        default_level: Default logging level
        env_key: Environment variable key for config path
    """
    path = os.getenv(env_key, config_path)
    
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        
        # Ensure log directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found: {path}. Using basic config.")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MLOpsLogger:
    """Custom logger for MLOps operations."""
    
    def __init__(self, name: str, level: Optional[int] = None):
        """
        Initialize MLOps logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        if level:
            self.logger.setLevel(level)
    
    def log_experiment(self, experiment_name: str, run_id: str, metrics: dict):
        """Log experiment information."""
        self.logger.info(
            "Experiment logged",
            extra={
                "experiment_name": experiment_name,
                "run_id": run_id,
                "metrics": metrics,
                "event_type": "experiment"
            }
        )
    
    def log_model_training(self, model_name: str, training_time: float, metrics: dict):
        """Log model training information."""
        self.logger.info(
            "Model training completed",
            extra={
                "model_name": model_name,
                "training_time": training_time,
                "metrics": metrics,
                "event_type": "training"
            }
        )
    
    def log_prediction(self, model_version: str, prediction_count: int, latency: float):
        """Log prediction information."""
        self.logger.info(
            "Predictions made",
            extra={
                "model_version": model_version,
                "prediction_count": prediction_count,
                "latency": latency,
                "event_type": "prediction"
            }
        )
    
    def log_data_drift(self, drift_score: float, threshold: float, features: list):
        """Log data drift detection."""
        self.logger.warning(
            "Data drift detected",
            extra={
                "drift_score": drift_score,
                "threshold": threshold,
                "affected_features": features,
                "event_type": "drift"
            }
        )
    
    def log_model_performance(self, model_version: str, metrics: dict, degradation: bool):
        """Log model performance monitoring."""
        level = logging.WARNING if degradation else logging.INFO
        message = "Model performance degradation detected" if degradation else "Model performance monitored"
        
        self.logger.log(
            level,
            message,
            extra={
                "model_version": model_version,
                "metrics": metrics,
                "degradation": degradation,
                "event_type": "performance"
            }
        )


# Initialize logging on import
setup_logging()
