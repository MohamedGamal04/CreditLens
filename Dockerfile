# CreditLens — FastAPI + trained models, served by uvicorn.
#
# Uses uv + uv.lock so the container installs the EXACT same dependency versions
# the models were trained with (.venv, Python 3.12) — joblib/sklearn pickles load
# cleanly. Models are baked in (Kaggle data can't be redistributed): run
# `.venv/bin/python -m creditlens.pipeline` (make train) before building.
FROM python:3.12-slim

# uv binary (fast, reads uv.lock for reproducible installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# libgomp1: OpenMP runtime for LightGBM / XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Install runtime deps only (exact lock, no dev: no mlflow/pytest in the image).
COPY pyproject.toml uv.lock README.md ./
COPY creditlens ./creditlens
RUN uv sync --frozen --no-dev --no-editable

# Trained model artifacts only — this is a BACKEND-ONLY image (API). The frontend
# deploys separately to Vercel. With no app/ dir, api.py skips its static mount
# (guarded by APP_DIR.exists()), so the container serves pure API.
COPY models ./models

ENV PORT=8000
EXPOSE 8000

# Liveness probe (slim image has no curl, so use python). 40s start period covers model load.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT','8000')}/health\").status==200 else 1)"

# Honor an injected $PORT (HF Spaces / Render / Cloud Run) with a sane default.
CMD ["sh", "-c", ".venv/bin/uvicorn creditlens.serve.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
