"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for model predictions."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure: int = Field(..., ge=0, le=72, description="Customer tenure in months")
    phone_service: int = Field(..., ge=0, le=1, description="Phone service (0/1)")
    multiple_lines: int = Field(..., ge=0, le=1, description="Multiple lines (0/1)")
    internet_service: str = Field(..., description="Internet service type")
    online_security: int = Field(..., ge=0, le=1, description="Online security (0/1)")
    online_backup: int = Field(..., ge=0, le=1, description="Online backup (0/1)")
    device_protection: int = Field(..., ge=0, le=1, description="Device protection (0/1)")
    tech_support: int = Field(..., ge=0, le=1, description="Tech support (0/1)")
    streaming_tv: int = Field(..., ge=0, le=1, description="Streaming TV (0/1)")
    streaming_movies: int = Field(..., ge=0, le=1, description="Streaming movies (0/1)")
    contract: str = Field(..., description="Contract type")
    paperless_billing: int = Field(..., ge=0, le=1, description="Paperless billing (0/1)")
    payment_method: str = Field(..., description="Payment method")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001234",
                "gender": "Female",
                "age": 45,
                "tenure": 24,
                "phone_service": 1,
                "multiple_lines": 0,
                "internet_service": "Fiber optic",
                "online_security": 1,
                "online_backup": 0,
                "device_protection": 1,
                "tech_support": 1,
                "streaming_tv": 1,
                "streaming_movies": 0,
                "contract": "One year",
                "paperless_billing": 1,
                "payment_method": "Credit card (automatic)",
                "monthly_charges": 75.50,
                "total_charges": 1812.00
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "CUST_001234",
                        "gender": "Female",
                        "age": 45,
                        "tenure": 24,
                        "phone_service": 1,
                        "multiple_lines": 0,
                        "internet_service": "Fiber optic",
                        "online_security": 1,
                        "online_backup": 0,
                        "device_protection": 1,
                        "tech_support": 1,
                        "streaming_tv": 1,
                        "streaming_movies": 0,
                        "contract": "One year",
                        "paperless_billing": 1,
                        "payment_method": "Credit card (automatic)",
                        "monthly_charges": 75.50,
                        "total_charges": 1812.00
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for model predictions."""
    
    customer_id: str = Field(..., description="Customer identifier")
    prediction: int = Field(..., description="Churn prediction (0/1)")
    probability: float = Field(..., ge=0, le=1, description="Churn probability")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001234",
                "prediction": 0,
                "probability": 0.23,
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_id: str = Field(..., description="Batch identifier")
    total_predictions: int = Field(..., description="Total number of predictions")
    processing_time: float = Field(..., description="Processing time in seconds")


class ModelInfo(BaseModel):
    """Model information schema."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="Algorithm used")
    training_date: datetime = Field(..., description="Training date")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    features: List[str] = Field(..., description="Model features")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "churn_prediction_model",
                "version": "1.0.0",
                "algorithm": "RandomForestClassifier",
                "training_date": "2024-01-15T08:00:00",
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78,
                    "f1_score": 0.80,
                    "roc_auc": 0.88
                },
                "features": ["tenure", "monthly_charges", "total_charges"]
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model status")
    database_status: str = Field(..., description="Database status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "model_status": "loaded",
                "database_status": "connected"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "The requested model version is not available",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    
    model_version: str = Field(..., description="Model version")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    period: str = Field(..., description="Metrics period")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class DriftResponse(BaseModel):
    """Data drift response schema."""
    
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Drift score")
    threshold: float = Field(..., description="Drift threshold")
    affected_features: List[str] = Field(..., description="Features with drift")
    timestamp: datetime = Field(..., description="Detection timestamp")
