# End-to-End MLOps Pipeline Demo

A comprehensive demonstration of a production-ready MLOps pipeline implementing best practices for machine learning operations, from data ingestion to model deployment and monitoring.

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow for a **Customer Churn Prediction** use case, showcasing:

- **Data Pipeline**: Automated data ingestion, validation, and preprocessing
- **Model Development**: Experiment tracking, model training, and evaluation
- **Model Deployment**: Containerized model serving with REST API
- **Monitoring**: Model performance monitoring and drift detection
- **CI/CD**: Automated testing and deployment pipelines
- **Infrastructure**: Docker containerization and orchestration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Pipeline  │───▶│   Feature Store │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│ Model Training  │───▶│ Model Registry  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Alerting     │    │   Experiments   │    │ Model Serving   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
mlops_project/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── docker-compose.yml        # Multi-service orchestration
├── Makefile                  # Common commands
│
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── config.yaml          # Main configuration
│   ├── logging.yaml         # Logging configuration
│   └── model_config.yaml    # Model-specific configs
│
├── data/                    # Data directory
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── features/           # Feature store
│   └── external/           # External data sources
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── ingestion.py   # Data ingestion
│   │   ├── validation.py  # Data validation
│   │   └── preprocessing.py # Data preprocessing
│   │
│   ├── features/          # Feature engineering
│   │   ├── __init__.py
│   │   ├── engineering.py # Feature creation
│   │   └── selection.py   # Feature selection
│   │
│   ├── models/            # Model development
│   │   ├── __init__.py
│   │   ├── train.py      # Model training
│   │   ├── evaluate.py   # Model evaluation
│   │   ├── predict.py    # Prediction logic
│   │   └── registry.py   # Model registry
│   │
│   ├── api/              # API service
│   │   ├── __init__.py
│   │   ├── app.py        # FastAPI application
│   │   ├── routes.py     # API routes
│   │   └── schemas.py    # Pydantic schemas
│   │
│   ├── monitoring/       # Monitoring components
│   │   ├── __init__.py
│   │   ├── drift.py      # Data/model drift detection
│   │   ├── performance.py # Performance monitoring
│   │   └── alerts.py     # Alerting system
│   │
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── logger.py     # Logging utilities
│       ├── database.py   # Database connections
│       └── helpers.py    # Helper functions
│
├── tests/                # Test suite
│   ├── __init__.py
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── conftest.py      # Pytest configuration
│
├── notebooks/           # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_analysis.ipynb
│
├── scripts/             # Utility scripts
│   ├── setup_environment.sh
│   ├── run_pipeline.py
│   ├── deploy_model.py
│   └── generate_data.py
│
├── docker/              # Docker configurations
│   ├── Dockerfile.api   # API service
│   ├── Dockerfile.training # Training service
│   └── Dockerfile.monitoring # Monitoring service
│
├── infrastructure/      # Infrastructure as Code
│   ├── terraform/       # Terraform configs
│   └── kubernetes/      # K8s manifests
│
├── models/              # Trained models
│   ├── artifacts/       # Model artifacts
│   ├── experiments/     # Experiment tracking
│   └── registry/        # Model registry
│
├── logs/                # Application logs
├── metrics/             # Model metrics
└── docs/                # Documentation
    ├── api.md          # API documentation
    ├── deployment.md   # Deployment guide
    └── monitoring.md   # Monitoring guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Git

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd mlops_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. **Generate sample data**:
```bash
python scripts/generate_data.py
```

4. **Run the complete pipeline**:
```bash
make run-pipeline
```

5. **Start services**:
```bash
docker-compose up -d
```

## 🔧 Usage

### Data Pipeline

```bash
# Ingest new data
python -m src.data.ingestion --source data/raw/customers.csv

# Validate data quality
python -m src.data.validation --input data/raw/customers.csv

# Preprocess data
python -m src.data.preprocessing --input data/raw/customers.csv --output data/processed/
```

### Model Training

```bash
# Train model with experiment tracking
python -m src.models.train --config config/model_config.yaml --experiment churn_prediction_v1

# Evaluate model
python -m src.models.evaluate --model-path models/artifacts/latest_model.pkl

# Register model
python -m src.models.registry --model-path models/artifacts/latest_model.pkl --version 1.0.0
```

### Model Serving

```bash
# Start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Make predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "12345", "features": {...}}'
```

### Monitoring

```bash
# Check model performance
python -m src.monitoring.performance --model-version 1.0.0

# Detect data drift
python -m src.monitoring.drift --reference-data data/processed/train.csv --current-data data/processed/current.csv
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📊 Key Features

### 1. **Data Pipeline**
- Automated data ingestion from multiple sources
- Data quality validation and profiling
- Feature engineering and selection
- Data versioning and lineage tracking

### 2. **Model Development**
- Experiment tracking with MLflow
- Automated hyperparameter tuning
- Model versioning and registry
- A/B testing framework

### 3. **Model Deployment**
- REST API with FastAPI
- Batch prediction capabilities
- Model serving with Docker
- Blue-green deployment strategy

### 4. **Monitoring & Observability**
- Real-time model performance monitoring
- Data drift detection
- Model drift detection
- Automated alerting system
- Comprehensive logging

### 5. **CI/CD Pipeline**
- Automated testing on code changes
- Model validation pipeline
- Automated deployment to staging/production
- Rollback mechanisms

## 🛠️ Technologies Used

- **Languages**: Python 3.8+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Database**: PostgreSQL, SQLite
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Testing**: pytest, coverage
- **CI/CD**: GitHub Actions
- **Infrastructure**: Terraform (optional)

## 📈 Metrics & KPIs

The pipeline tracks various metrics:

- **Data Quality**: Completeness, consistency, validity
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Operational**: Latency, throughput, error rates
- **Business**: Model impact on business metrics

## 🔍 Monitoring Dashboard

Access the monitoring dashboard at:
- **Grafana**: http://localhost:3000
- **MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

## 🚨 Alerting

The system provides alerts for:
- Data quality issues
- Model performance degradation
- System failures
- Resource utilization

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring Guide](docs/monitoring.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:
- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the example notebooks in `notebooks/`

---

**Happy MLOps! 🚀**
