.PHONY: install data fixtures train eval serve serve-prod test lint format mlflow-ui clean

# All targets run inside the uv-managed .venv via `uv run` (single source of truth).

install:           ## Create/sync .venv with all deps (incl dev)
	uv sync --extra dev

data:              ## Download Home Credit Default Risk data from Kaggle into data/raw/
	uv run python scripts/download_data.py

fixtures:          ## Build tiny synthetic samples for tests/CI
	uv run python scripts/make_fixtures.py

train:             ## Train + calibrate + save all 6 models (logs to MLflow)
	uv run python -m creditlens.pipeline

serve:             ## Launch FastAPI app + frontend at http://localhost:8000 (dev, reload)
	uv run uvicorn creditlens.serve.api:app --reload --port 8000

serve-prod:        ## Launch with multiple workers (stateless -> scales horizontally)
	uv run uvicorn creditlens.serve.api:app --host 0.0.0.0 --port 8000 --workers 4

test:              ## Run test suite
	uv run pytest

lint:              ## Lint with ruff
	uv run ruff check .

format:            ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

mlflow-ui:         ## Open the MLflow experiment tracking UI (http://localhost:5000)
	uv run mlflow ui

clean:             ## Remove caches and processed artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/processed/* models/*
