"""
FastAPI application for MLOps model serving.
"""
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.api.schemas import (
    PredictionRequest, BatchPredictionRequest, PredictionResponse, 
    BatchPredictionResponse, ModelInfo, HealthResponse, ErrorResponse,
    MetricsResponse, DriftResponse
)
from src.utils.logger import get_logger, MLOpsLogger
from src.utils.helpers import load_config, load_model
from src.utils.database import db_manager
from src.models.predict import ModelPredictor
from src.monitoring.performance import PerformanceMonitor
from src.monitoring.drift import DriftDetector

logger = get_logger(__name__)
mlops_logger = MLOpsLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Churn Prediction API",
    description="Production-ready API for customer churn prediction with monitoring and drift detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
model_predictor = None
performance_monitor = None
drift_detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global config, model_predictor, performance_monitor, drift_detector
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        logger.info("Configuration loaded successfully")
        
        # Initialize database
        db_manager.init_db()
        logger.info("Database initialized")
        
        # Initialize model predictor
        model_predictor = ModelPredictor(config)
        logger.info("Model predictor initialized")
        
        # Initialize monitoring components
        performance_monitor = PerformanceMonitor(config)
        drift_detector = DriftDetector(config)
        logger.info("Monitoring components initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API shutdown initiated")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check model status
        model_status = "loaded" if model_predictor and model_predictor.model else "not_loaded"
        
        # Check database status
        try:
            db_session = db_manager.get_session()
            db_session.execute("SELECT 1")
            db_manager.close_session(db_session)
            database_status = "connected"
        except Exception:
            database_status = "disconnected"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            model_status=model_status,
            database_status=database_status
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a single prediction."""
    start_time = time.time()
    
    try:
        if not model_predictor or not model_predictor.model:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction, probability = model_predictor.predict(input_data)
        
        # Create response
        response = PredictionResponse(
            customer_id=request.customer_id,
            prediction=int(prediction[0]),
            probability=float(probability[0]),
            model_version=model_predictor.model_version,
            timestamp=datetime.now()
        )
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            request.customer_id,
            input_data.to_json(),
            response.prediction,
            response.probability,
            time.time() - start_time
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Make batch predictions."""
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        if not model_predictor or not model_predictor.model:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Convert requests to DataFrame
        input_data = pd.DataFrame([req.dict() for req in request.predictions])
        
        # Make predictions
        predictions, probabilities = model_predictor.predict(input_data)
        
        # Create responses
        responses = []
        for i, req in enumerate(request.predictions):
            responses.append(PredictionResponse(
                customer_id=req.customer_id,
                prediction=int(predictions[i]),
                probability=float(probabilities[i]),
                model_version=model_predictor.model_version,
                timestamp=datetime.now()
            ))
        
        processing_time = time.time() - start_time
        
        # Log batch prediction in background
        background_tasks.add_task(
            log_batch_prediction,
            batch_id,
            len(request.predictions),
            processing_time
        )
        
        return BatchPredictionResponse(
            predictions=responses,
            batch_id=batch_id,
            total_predictions=len(responses),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information."""
    try:
        if not model_predictor or not model_predictor.model:
            raise HTTPException(status_code=503, detail="Model not available")
        
        return model_predictor.get_model_info()
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(period: str = "24h"):
    """Get model performance metrics."""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        metrics = performance_monitor.get_metrics(period)
        
        return MetricsResponse(
            model_version=model_predictor.model_version if model_predictor else "unknown",
            metrics=metrics,
            period=period,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/drift", response_model=DriftResponse)
async def check_drift():
    """Check for data drift."""
    try:
        if not drift_detector:
            raise HTTPException(status_code=503, detail="Drift detector not available")
        
        drift_result = drift_detector.detect_drift()
        
        return DriftResponse(
            drift_detected=drift_result['drift_detected'],
            drift_score=drift_result['drift_score'],
            threshold=drift_result['threshold'],
            affected_features=drift_result['affected_features'],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Failed to check drift: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check drift: {str(e)}")


async def log_prediction(customer_id: str, input_data: str, prediction: int, 
                        probability: float, latency: float):
    """Log prediction to database."""
    try:
        # Log to database
        db_manager.log_prediction(
            model_name="churn_prediction",
            model_version=model_predictor.model_version,
            prediction_id=f"{customer_id}_{int(time.time())}",
            input_data=input_data,
            prediction=float(prediction),
            probability=probability
        )
        
        # Log to MLOps logger
        mlops_logger.log_prediction(
            model_version=model_predictor.model_version,
            prediction_count=1,
            latency=latency
        )
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


async def log_batch_prediction(batch_id: str, prediction_count: int, processing_time: float):
    """Log batch prediction to database."""
    try:
        mlops_logger.log_prediction(
            model_version=model_predictor.model_version,
            prediction_count=prediction_count,
            latency=processing_time
        )
        
        logger.info(f"Batch prediction logged: {batch_id}, count: {prediction_count}, time: {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")


def main():
    """Main function to run the API server."""
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
