# Makefile for AstroML - Modern ML for Astronomy
# Simplifies common development and deployment tasks

.PHONY: help install install-dev test lint format docker-build docker-up api web docs clean

help:  ## Show this help message
	@echo "AstroML - Modern Machine Learning for Astronomy"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install dependencies
	poetry install

install-dev:  ## Install with dev dependencies
	poetry install --with dev

# Code Quality
lint:  ## Run linters (Ruff)
	poetry run ruff check src/ tests/

lint-fix:  ## Run linters with auto-fix
	poetry run ruff check --fix src/ tests/

format:  ## Format code with Black
	poetry run black src/ tests/

format-check:  ## Check code formatting
	poetry run black --check src/ tests/

typecheck:  ## Run type checking with MyPy
	poetry run mypy src/

pre-commit:  ## Run all pre-commit hooks
	poetry run pre-commit run --all-files

# Testing
test:  ## Run tests
	poetry run pytest tests/ -v

test-cov:  ## Run tests with coverage
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-watch:  ## Run tests in watch mode
	poetry run ptw tests/ -- -v

# Docker
docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start all services
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## Show logs
	docker-compose logs -f

docker-rebuild:  ## Rebuild and restart services
	docker-compose down && docker-compose build && docker-compose up -d

# Development Servers
api:  ## Run FastAPI development server
	poetry run uvicorn src.astroml.api.main:app --reload --host 0.0.0.0 --port 8000

web:  ## Run Streamlit web interface
	poetry run streamlit run src/astroml/web/streamlit_app.py

jupyter:  ## Start Jupyter Lab
	poetry run jupyter lab

# Training
train:  ## Train models
	poetry run python scripts/train.py

evaluate:  ## Evaluate models
	poetry run python scripts/evaluate.py

mlflow:  ## Start MLflow UI
	poetry run mlflow ui --host 0.0.0.0 --port 5000

# Documentation
docs-serve:  ## Serve documentation locally
	poetry run mkdocs serve

docs-build:  ## Build documentation
	poetry run mkdocs build

docs-deploy:  ## Deploy documentation to GitHub Pages
	poetry run mkdocs gh-deploy

# Deployment
export-requirements:  ## Export requirements.txt from Poetry
	poetry export -f requirements.txt --output requirements.txt --without-hashes

build:  ## Build Python package
	poetry build

publish:  ## Publish to PyPI
	poetry publish

# Cleanup
clean:  ## Clean temporary files
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data:  ## Clean data files (careful!)
	rm -rf data/processed/* data/results/*

clean-models:  ## Clean model checkpoints (careful!)
	rm -rf models/checkpoints/*

# Setup
setup:  ## Initial project setup
	poetry install --with dev
	poetry run pre-commit install
	mkdir -p data/{raw,processed,results} models/{pretrained,checkpoints,onnx}
	@echo "✅ Setup complete! Run 'make help' to see available commands"

# Git
git-setup:  ## Setup git hooks
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg

# All-in-one commands
dev:  ## Start development environment
	@echo "Starting development environment..."
	docker-compose up -d redis mlflow
	make api

all: clean install lint test  ## Run full CI pipeline locally
