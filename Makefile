.PHONY: install data fixtures train eval serve test lint format clean

install:           ## Install package + dev deps
	pip install -e ".[dev]"

data:              ## Download Home Credit Default Risk data from Kaggle into data/raw/
	python scripts/download_data.py

fixtures:          ## Build tiny synthetic sample for tests/CI
	python scripts/make_fixtures.py

train:             ## Run full pipeline: FE -> CV-train all models -> eval -> save best
	python -m creditlens.pipeline

eval:              ## Regenerate evaluation report (metrics table + plots)
	python -m creditlens.evaluation.report

serve:             ## Launch FastAPI app + frontend at http://localhost:8000
	uvicorn creditlens.serve.api:app --reload --port 8000

test:              ## Run test suite
	pytest

lint:              ## Lint with ruff
	ruff check .

format:            ## Auto-format with ruff
	ruff format .
	ruff check --fix .

clean:             ## Remove caches and processed artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/processed/* models/*
