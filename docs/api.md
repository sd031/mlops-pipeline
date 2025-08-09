# API Documentation

## Overview

The MLOps Churn Prediction API provides endpoints for making predictions, monitoring model performance, and detecting data drift.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement proper authentication and authorization.

## Endpoints

### Health Check

**GET** `/health`

Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "model_status": "loaded",
  "database_status": "connected"
}
```

### Single Prediction

**POST** `/predict`

Make a prediction for a single customer.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "customer_id": "CUST_001234",
  "prediction": 0,
  "probability": 0.23,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Batch Predictions

**POST** `/predict/batch`

Make predictions for multiple customers.

### Model Information

**GET** `/model/info`

Get information about the current model.

### Performance Metrics

**GET** `/metrics`

Get model performance metrics.

### Drift Detection

**GET** `/drift`

Check for data drift.

## Interactive Documentation

Visit `/docs` for interactive API documentation with Swagger UI.
Visit `/redoc` for alternative documentation with ReDoc.
