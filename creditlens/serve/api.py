"""FastAPI app: serve the calibrated models + the CreditLens frontend.

All 6 models are loaded at startup (the UI lets the user pick one). ``/predict`` derives
the 15-feature contract from a raw applicant payload, scores it with the chosen model, and
returns a calibrated PD + risk band + decision. ``/models`` exposes the per-model metrics.

Run: ``uvicorn creditlens.serve.api:app --reload --port 8000`` (or ``make serve``).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from creditlens.config import APP_DIR, MODELS_DIR
from creditlens.data.features import MODEL_FEATURES, applicant_to_features
from creditlens.serve.schema import PredictRequest, PredictResponse

MODELS: dict = {}
METADATA: dict = {}
ALIAS = {"stack": "stacking"}  # frontend key -> saved artifact name


def _load() -> None:
    """Load metadata + every saved model into memory."""
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        METADATA.update(json.loads(meta_path.read_text()))
    for path in MODELS_DIR.glob("*.joblib"):
        MODELS[path.stem] = joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    MODELS.clear()
    METADATA.clear()


app = FastAPI(title="CreditLens", version="0.1.0", lifespan=lifespan)


def _band(pd_: float, low: float, high: float) -> str:
    return "low" if pd_ < low else ("med" if pd_ < high else "high")


_DECISION = {"low": "approve", "med": "review", "high": "decline"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": sorted(MODELS)}


@app.get("/models")
def models() -> dict:
    """Per-model metrics for the UI selector (AUC/KS/ECE)."""
    if not METADATA:
        raise HTTPException(503, "No metadata. Run `make train` first.")
    return METADATA


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    key = ALIAS.get(req.model, req.model)
    model = MODELS.get(key)
    if model is None:
        raise HTTPException(
            503 if not MODELS else 400,
            f"Model '{req.model}' unavailable. Loaded: {sorted(MODELS)} (run `make train`).",
        )

    feats = applicant_to_features(req.applicant.model_dump())
    row = pd.DataFrame([feats])[MODEL_FEATURES]
    proba = float(model.predict_proba(row)[:, 1][0])
    band = _band(proba, req.low, req.high)

    return PredictResponse(
        probability=proba,
        band=band,
        decision=_DECISION[band],
        model=key,
        features=feats,
    )


# Serve the frontend (app/CreditLens.html + app/src/*) at the root.
if APP_DIR.exists():

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(APP_DIR / "CreditLens.html")

    app.mount("/", StaticFiles(directory=APP_DIR), name="app")
