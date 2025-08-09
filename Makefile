.PHONY: help venv install setup test lint format clean run-pipeline docker-build docker-up docker-down

# Virtual environment settings
VENV_NAME = venv
VENV_PATH = ./$(VENV_NAME)
PYTHON = $(VENV_PATH)/bin/python
PIP = $(VENV_PATH)/bin/pip

# Default target
help:
	@echo "Available commands:"
	@echo "  venv             Create virtual environment"
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

# Virtual environment creation
venv:
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	@echo "Virtual environment created. Activate with: source $(VENV_NAME)/bin/activate"

# Installation and setup
install: venv
	$(PIP) install -r requirements.txt

setup: install
	$(PYTHON) -m pip install -e .
	mkdir -p data/raw data/processed data/features data/external
	mkdir -p models/artifacts models/experiments models/registry
	mkdir -p logs metrics
	cp .env.example .env

# Testing
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# Code quality
lint:
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m black --check src/ tests/

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

clean-venv:
	rm -rf $(VENV_NAME)

clean-all: clean clean-venv

# ML Pipeline
generate-data:
	$(PYTHON) scripts/generate_data.py

train-model:
	$(PYTHON) -m src.models.train --config config/model_config.yaml

evaluate-model:
	$(PYTHON) -m src.models.evaluate

serve-model:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-pipeline: generate-data
	$(PYTHON) scripts/run_pipeline.py

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
	$(PYTHON) -m src.monitoring.performance &
	$(PYTHON) -m src.monitoring.drift &

# Development
jupyter:
	$(PYTHON) -m jupyter notebook notebooks/

mlflow-ui:
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000

# Database operations
db-init:
	$(PYTHON) -c "from src.utils.database import init_db; init_db()"

db-migrate:
	$(PYTHON) -c "from src.utils.database import migrate_db; migrate_db()"
