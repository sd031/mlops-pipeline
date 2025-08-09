.PHONY: help install setup test lint format clean run-pipeline docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install dependencies"
	@echo "  setup            Setup project environment"
	@echo "  test             Run tests"
	@echo "  lint             Run linting"
	@echo "  format           Format code"
	@echo "  clean            Clean temporary files"
	@echo "  run-pipeline     Run complete ML pipeline"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start services with Docker Compose"
	@echo "  docker-down      Stop services"
	@echo "  generate-data    Generate sample data"
	@echo "  train-model      Train model"
	@echo "  serve-model      Serve model API"
	@echo "  monitor          Start monitoring"

# Installation and setup
install:
	pip install -r requirements.txt

setup: install
	python -m pip install -e .
	mkdir -p data/raw data/processed data/features data/external
	mkdir -p models/artifacts models/experiments models/registry
	mkdir -p logs metrics
	cp .env.example .env

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# ML Pipeline
generate-data:
	python scripts/generate_data.py

train-model:
	python -m src.models.train --config config/model_config.yaml

evaluate-model:
	python -m src.models.evaluate

serve-model:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-pipeline: generate-data
	python scripts/run_pipeline.py

# Docker operations
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Monitoring
monitor:
	python -m src.monitoring.performance &
	python -m src.monitoring.drift &

# Development
jupyter:
	jupyter notebook notebooks/

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Database operations
db-init:
	python -c "from src.utils.database import init_db; init_db()"

db-migrate:
	python -c "from src.utils.database import migrate_db; migrate_db()"
