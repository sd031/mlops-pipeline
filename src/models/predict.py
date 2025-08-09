"""
Model prediction module for the MLOps pipeline.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from datetime import datetime
import joblib

from src.utils.logger import get_logger
from src.utils.helpers import load_model, get_model_info
from src.api.schemas import ModelInfo

logger = get_logger(__name__)


class ModelPredictor:
    """Model predictor class for making predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self.model_path = None
        self.feature_names = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and preprocessor."""
        try:
            # Get model path from config
            model_config = self.config.get('api', {})
            model_version = model_config.get('model_version', 'latest')
            
            if model_version == 'latest':
                # Find the latest model
                models_dir = 'models/artifacts'
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                    if model_files:
                        # Sort by modification time and get the latest
                        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                        model_file = model_files[0]
                        self.model_path = os.path.join(models_dir, model_file)
                        self.model_version = model_file.replace('.pkl', '')
            else:
                self.model_path = f'models/artifacts/{model_version}.pkl'
                self.model_version = model_version
            
            if self.model_path and os.path.exists(self.model_path):
                # Load model
                model_data = joblib.load(self.model_path)
                
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.preprocessor = model_data.get('preprocessor')
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    self.model = model_data
                
                logger.info(f"Model loaded successfully: {self.model_path}")
            else:
                logger.warning("No trained model found. Please train a model first.")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Preprocess data if preprocessor is available
            if self.preprocessor is not None:
                processed_data = self.preprocessor.transform(data)
            else:
                processed_data = self._preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                if probabilities.shape[1] > 1:
                    probabilities = probabilities[:, 1]  # Get probability of positive class
                else:
                    probabilities = probabilities[:, 0]
            else:
                # For models without predict_proba, use decision function or predictions
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(processed_data)
                    probabilities = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
                else:
                    probabilities = predictions.astype(float)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Basic data preprocessing if no preprocessor is available.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed data array
        """
        # Remove customer_id if present
        if 'customer_id' in data.columns:
            data = data.drop('customer_id', axis=1)
        
        # Handle categorical variables
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col == 'gender':
                data[col] = data[col].map({'Male': 1, 'Female': 0})
            elif col == 'internet_service':
                data[col] = data[col].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
            elif col == 'contract':
                data[col] = data[col].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            elif col == 'payment_method':
                # Simple encoding for payment method
                payment_mapping = {
                    'Electronic check': 0,
                    'Mailed check': 1,
                    'Bank transfer (automatic)': 2,
                    'Credit card (automatic)': 3
                }
                data[col] = data[col].map(payment_mapping)
        
        # Fill missing values with median for numerical columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
        
        return data.values
    
    def get_model_info(self) -> ModelInfo:
        """
        Get model information.
        
        Returns:
            ModelInfo object
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Get model info
            info = get_model_info(self.model)
            
            # Load metrics if available
            metrics = {}
            metrics_path = f'models/artifacts/{self.model_version}_metrics.json'
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            
            return ModelInfo(
                name="churn_prediction_model",
                version=self.model_version,
                algorithm=info['model_type'],
                training_date=datetime.fromisoformat(info['timestamp']) if 'timestamp' in info else datetime.now(),
                metrics=metrics,
                features=self.feature_names or []
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def reload_model(self):
        """Reload the model (useful for model updates)."""
        logger.info("Reloading model...")
        self._load_model()
        logger.info("Model reloaded successfully")


def main():
    """Main function for prediction CLI."""
    import argparse
    from src.utils.helpers import load_config
    
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", help="Output predictions file")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data = pd.read_csv(args.input)
    
    # Initialize predictor
    predictor = ModelPredictor(config)
    
    # Make predictions
    predictions, probabilities = predictor.predict(data)
    
    # Create results DataFrame
    results = data.copy()
    results['prediction'] = predictions
    results['probability'] = probabilities
    
    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
    else:
        print(results)
    
    logger.info(f"Made predictions for {len(data)} samples")


if __name__ == "__main__":
    main()
