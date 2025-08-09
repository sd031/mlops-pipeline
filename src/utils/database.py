"""
Database utilities for the MLOps pipeline.
"""
import os
from typing import Optional
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class ModelMetrics(Base):
    """Model metrics table."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    model_version = Column(String, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)


class DataQuality(Base):
    """Data quality metrics table."""
    __tablename__ = "data_quality"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    data_version = Column(String, index=True)


class ModelPredictions(Base):
    """Model predictions table."""
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    model_version = Column(String, index=True)
    prediction_id = Column(String, index=True)
    input_data = Column(Text)  # JSON string
    prediction = Column(Float)
    probability = Column(Float)
    timestamp = Column(DateTime, default=func.now())


class DriftMetrics(Base):
    """Data drift metrics table."""
    __tablename__ = "drift_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, index=True)
    drift_score = Column(Float)
    drift_detected = Column(Boolean)
    threshold = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    reference_period = Column(String)
    current_period = Column(String)


class DatabaseManager:
    """Database manager for MLOps pipeline."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./mlops.db")
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_db(self):
        """Initialize database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session."""
        session.close()
    
    def log_model_metrics(self, model_name: str, model_version: str, metrics: dict, 
                         experiment_id: str = None, run_id: str = None):
        """
        Log model metrics to database.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            metrics: Dictionary of metrics
            experiment_id: Experiment ID
            run_id: Run ID
        """
        session = self.get_session()
        try:
            for metric_name, metric_value in metrics.items():
                metric_record = ModelMetrics(
                    model_name=model_name,
                    model_version=model_version,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    experiment_id=experiment_id,
                    run_id=run_id
                )
                session.add(metric_record)
            session.commit()
            logger.info(f"Logged metrics for model {model_name} v{model_version}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging model metrics: {e}")
            raise
        finally:
            self.close_session(session)
    
    def log_data_quality(self, dataset_name: str, metrics: dict, data_version: str = None):
        """
        Log data quality metrics.
        
        Args:
            dataset_name: Name of the dataset
            metrics: Dictionary of quality metrics
            data_version: Version of the data
        """
        session = self.get_session()
        try:
            for metric_name, metric_value in metrics.items():
                quality_record = DataQuality(
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    data_version=data_version
                )
                session.add(quality_record)
            session.commit()
            logger.info(f"Logged data quality metrics for {dataset_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging data quality metrics: {e}")
            raise
        finally:
            self.close_session(session)
    
    def log_prediction(self, model_name: str, model_version: str, prediction_id: str,
                      input_data: str, prediction: float, probability: float = None):
        """
        Log model prediction.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            prediction_id: Unique prediction ID
            input_data: Input data as JSON string
            prediction: Prediction value
            probability: Prediction probability
        """
        session = self.get_session()
        try:
            prediction_record = ModelPredictions(
                model_name=model_name,
                model_version=model_version,
                prediction_id=prediction_id,
                input_data=input_data,
                prediction=prediction,
                probability=probability
            )
            session.add(prediction_record)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging prediction: {e}")
            raise
        finally:
            self.close_session(session)
    
    def log_drift_metrics(self, feature_name: str, drift_score: float, drift_detected: bool,
                         threshold: float, reference_period: str, current_period: str):
        """
        Log drift detection metrics.
        
        Args:
            feature_name: Name of the feature
            drift_score: Drift score
            drift_detected: Whether drift was detected
            threshold: Drift threshold
            reference_period: Reference period
            current_period: Current period
        """
        session = self.get_session()
        try:
            drift_record = DriftMetrics(
                feature_name=feature_name,
                drift_score=drift_score,
                drift_detected=drift_detected,
                threshold=threshold,
                reference_period=reference_period,
                current_period=current_period
            )
            session.add(drift_record)
            session.commit()
            logger.info(f"Logged drift metrics for feature {feature_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging drift metrics: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_model_metrics(self, model_name: str, model_version: str = None) -> pd.DataFrame:
        """
        Get model metrics from database.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (optional)
            
        Returns:
            DataFrame with model metrics
        """
        session = self.get_session()
        try:
            query = session.query(ModelMetrics).filter(ModelMetrics.model_name == model_name)
            if model_version:
                query = query.filter(ModelMetrics.model_version == model_version)
            
            results = query.all()
            data = [{
                'model_name': r.model_name,
                'model_version': r.model_version,
                'metric_name': r.metric_name,
                'metric_value': r.metric_value,
                'timestamp': r.timestamp,
                'experiment_id': r.experiment_id,
                'run_id': r.run_id
            } for r in results]
            
            return pd.DataFrame(data)
        finally:
            self.close_session(session)
    
    def get_recent_predictions(self, model_name: str, limit: int = 100) -> pd.DataFrame:
        """
        Get recent predictions from database.
        
        Args:
            model_name: Name of the model
            limit: Number of recent predictions to retrieve
            
        Returns:
            DataFrame with recent predictions
        """
        session = self.get_session()
        try:
            results = session.query(ModelPredictions)\
                .filter(ModelPredictions.model_name == model_name)\
                .order_by(ModelPredictions.timestamp.desc())\
                .limit(limit).all()
            
            data = [{
                'prediction_id': r.prediction_id,
                'model_version': r.model_version,
                'prediction': r.prediction,
                'probability': r.probability,
                'timestamp': r.timestamp
            } for r in results]
            
            return pd.DataFrame(data)
        finally:
            self.close_session(session)


# Global database manager instance
db_manager = DatabaseManager()


def init_db():
    """Initialize database."""
    db_manager.init_db()


def get_db_session():
    """Get database session."""
    return db_manager.get_session()
