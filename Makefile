.PHONY: install data fixtures train tune eval serve serve-prod test lint format mlflow-ui clean

# Monorepo: the Python backend lives in backend/ (its own uv project). Recipes cd into
# it; scripts/ stays at the repo root and imports the installed creditlens package.

install:           ## Create/sync backend/.venv with all deps (incl dev)
	cd backend && uv sync --extra dev

data:              ## Download Home Credit Default Risk data from Kaggle into data/raw/
	cd backend && uv run python ../scripts/download_data.py

fixtures:          ## Build tiny synthetic samples for tests/CI
	cd backend && uv run python ../scripts/make_fixtures.py

train:             ## Train + calibrate + save all 6 models (logs to MLflow)
	cd backend && uv run python -m creditlens.pipeline

tune:              ## GridSearch tuning (proves registry hyperparameters; --sample for speed)
	cd backend && uv run python ../scripts/tune.py

serve:             ## Launch FastAPI app + frontend at http://localhost:8000 (dev, reload)
	cd backend && uv run uvicorn creditlens.serve.api:app --reload --port 8000

serve-prod:        ## Launch with multiple workers (stateless -> scales horizontally)
	cd backend && uv run uvicorn creditlens.serve.api:app --host 0.0.0.0 --port 8000 --workers 4

test:              ## Run test suite
	cd backend && uv run pytest

lint:              ## Lint with ruff (backend package + root scripts)
	cd backend && uv run ruff check . ../scripts

format:            ## Auto-format with ruff
	cd backend && uv run ruff format . ../scripts
	cd backend && uv run ruff check --fix . ../scripts

mlflow-ui:         ## Open the MLflow experiment tracking UI (http://localhost:5000)
	cd backend && uv run mlflow ui

clean:             ## Remove caches and processed artifacts
	rm -rf backend/.pytest_cache backend/.ruff_cache **/__pycache__ data/processed/* models/*
