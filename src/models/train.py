"""
Model training module for the MLOps pipeline.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn

from src.utils.logger import get_logger, MLOpsLogger
from src.utils.helpers import load_config, ensure_dir, calculate_metrics, get_timestamp
from src.utils.database import db_manager

logger = get_logger(__name__)
mlops_logger = MLOpsLogger(__name__)


class ModelTrainer:
    """Model trainer class for training ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        
        # Set up MLflow
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        experiment_name = self.training_config.get('experiment_name', 'churn_prediction')
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        target_col = 'churn'
        X = df.drop(columns=[target_col, 'customer_id'])
        y = df[target_col]
        
        # Split data
        test_size = self.config.get('preprocessing', {}).get('test_size', 0.15)
        random_state = self.config.get('preprocessing', {}).get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def create_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """
        Create data preprocessor.
        
        Args:
            X_train: Training features
            
        Returns:
            Fitted preprocessor
        """
        # Identify categorical and numerical columns
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def train_model(self, data_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            data_path: Path to training data
            model_name: Name of the model algorithm to use
            
        Returns:
            Dictionary with training results
        """
        with mlflow.start_run(run_name=self.training_config.get('run_name', 'training_run')):
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(data_path)
                
                # Create preprocessor
                preprocessor = self.create_preprocessor(X_train)
                
                # Get model algorithm
                algorithm = model_name or self.model_config.get('algorithm', 'random_forest')
                model = self._create_model(algorithm)
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                # Log parameters
                mlflow.log_params(self.model_config.get('hyperparameters', {}))
                mlflow.log_param('algorithm', algorithm)
                mlflow.log_param('train_size', len(X_train))
                mlflow.log_param('test_size', len(X_test))
                
                # Train model
                logger.info(f"Training {algorithm} model...")
                start_time = datetime.now()
                pipeline.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)
                y_train_proba = pipeline.predict_proba(X_train)[:, 1]
                y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
                test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
                
                # Log metrics
                for metric, value in train_metrics.items():
                    mlflow.log_metric(f'train_{metric}', value)
                
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f'test_{metric}', value)
                
                mlflow.log_metric('training_time', training_time)
                
                # Cross-validation
                cv_folds = self.model_config.get('cv_folds', 5)
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                mlflow.log_metric('cv_roc_auc_mean', cv_mean)
                mlflow.log_metric('cv_roc_auc_std', cv_std)
                
                # Save model
                timestamp = get_timestamp()
                model_filename = f"{algorithm}_{timestamp}.pkl"
                model_path = os.path.join('models/artifacts', model_filename)
                ensure_dir('models/artifacts')
                
                # Save model with preprocessor
                model_data = {
                    'model': pipeline.named_steps['classifier'],
                    'preprocessor': pipeline.named_steps['preprocessor'],
                    'feature_names': list(X_train.columns),
                    'algorithm': algorithm,
                    'training_time': training_time,
                    'timestamp': timestamp,
                    'metrics': test_metrics
                }
                
                joblib.dump(model_data, model_path)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(pipeline, "model")
                
                # Save metrics separately
                metrics_path = os.path.join('models/artifacts', f"{algorithm}_{timestamp}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'cv_metrics': {
                            'mean': cv_mean,
                            'std': cv_std
                        },
                        'training_time': training_time
                    }, f, indent=2)
                
                # Log to database
                db_manager.log_model_metrics(
                    model_name=algorithm,
                    model_version=timestamp,
                    metrics=test_metrics,
                    experiment_id=mlflow.active_run().info.experiment_id,
                    run_id=mlflow.active_run().info.run_id
                )
                
                # Log training completion
                mlops_logger.log_model_training(
                    model_name=algorithm,
                    training_time=training_time,
                    metrics=test_metrics
                )
                
                logger.info(f"Model training completed successfully")
                logger.info(f"Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")
                logger.info(f"Model saved to: {model_path}")
                
                return {
                    'model_path': model_path,
                    'metrics_path': metrics_path,
                    'algorithm': algorithm,
                    'training_time': training_time,
                    'test_metrics': test_metrics,
                    'cv_score': cv_mean,
                    'mlflow_run_id': mlflow.active_run().info.run_id
                }
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                mlflow.log_param('error', str(e))
                raise
    
    def _create_model(self, algorithm: str):
        """
        Create model instance based on algorithm name.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Model instance
        """
        hyperparameters = self.model_config.get('hyperparameters', {})
        
        if algorithm == 'random_forest':
            return RandomForestClassifier(**hyperparameters)
        elif algorithm == 'gradient_boosting':
            return GradientBoostingClassifier(**hyperparameters)
        elif algorithm == 'logistic_regression':
            return LogisticRegression(**hyperparameters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def hyperparameter_tuning(self, data_path: str, algorithm: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            data_path: Path to training data
            algorithm: Algorithm to tune
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting hyperparameter tuning for {algorithm}")
        
        with mlflow.start_run(run_name=f'hyperparameter_tuning_{algorithm}'):
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(data_path)
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(X_train)
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            }
            
            # Create model
            model = self._create_model(algorithm)
            
            # Perform grid search
            grid_search = GridSearchCV(
                model,
                param_grids[algorithm],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_processed, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions with best model
            y_pred = best_model.predict(X_test_processed)
            y_proba = best_model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            
            # Log results
            mlflow.log_params(grid_search.best_params_)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            mlflow.log_metric('best_cv_score', grid_search.best_score_)
            
            logger.info(f"Hyperparameter tuning completed")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_metrics': metrics,
                'best_model': best_model
            }


def main():
    """Main function for model training CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--data", default="data/processed/validated_data.csv", help="Training data path")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--model", help="Model algorithm to use")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    if args.tune:
        # Hyperparameter tuning
        results = trainer.hyperparameter_tuning(args.data, args.model or 'random_forest')
        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV score: {results['best_score']:.4f}")
    else:
        # Regular training
        results = trainer.train_model(args.data, args.model)
        print(f"Model trained successfully: {results['model_path']}")
        print(f"Test ROC-AUC: {results['test_metrics'].get('roc_auc', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
