"""FastAPI app: serve the calibrated models + the CreditLens frontend.

All 6 models are loaded at startup (the UI lets the user pick one). ``/predict`` derives
the 15-feature contract from a raw applicant payload, scores it with the chosen model, and
returns a calibrated PD + risk band + decision. ``/models`` exposes the per-model metrics.

Run: ``uvicorn creditlens.serve.api:app --reload --port 8000`` (or ``make serve``).
"""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from creditlens.config import APP_DIR, MODELS_DIR
from creditlens.data.features import (
    APPLICANT_COLUMNS,
    MODEL_FEATURES,
    applicant_to_features,
    applicants_frame_to_features,
)
from creditlens.evaluation.metrics import summarize
from creditlens.serve.schema import PredictRequest, PredictResponse

BATCH_ROW_CAP = 1000  # max scored rows returned to the UI

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


@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    model: str = "lgbm",
    low: float = 0.06,
    high: float = 0.15,
) -> dict:
    """Score an uploaded CSV of applicants. Required columns: see APPLICANT_COLUMNS.

    Optional: ``SK_ID_CURR`` / ``name`` (display), ``TARGET`` (enables AUC/KS + realised
    defaults). Returns per-row PD/band/decision + a portfolio summary.
    """
    key = ALIAS.get(model, model)
    m = MODELS.get(key)
    if m is None:
        raise HTTPException(503 if not MODELS else 400,
                            f"Model '{model}' unavailable. Loaded: {sorted(MODELS)}.")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    missing = [c for c in APPLICANT_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"CSV missing required columns: {missing}")

    proba = m.predict_proba(applicants_frame_to_features(df))[:, 1]
    bands = [_band(float(p), low, high) for p in proba]

    has_target = "TARGET" in df.columns and df["TARGET"].nunique() > 1
    ids = df["SK_ID_CURR"] if "SK_ID_CURR" in df.columns else df.index
    names = df["name"] if "name" in df.columns else None

    rows = []
    for i in range(len(df)):
        rows.append({
            "id": int(ids.iloc[i]) if hasattr(ids, "iloc") else int(ids[i]),
            "name": str(names.iloc[i]) if names is not None else f"APP-{int(ids.iloc[i]) if hasattr(ids,'iloc') else i}",
            "credit": float(df["amt_credit"].iloc[i]),
            "ext2": float(df["ext_source_2"].iloc[i]),
            "prob": float(proba[i]),
            "band": bands[i],
            "decision": _DECISION[bands[i]],
            "defaulted": int(df["TARGET"].iloc[i]) if has_target else None,
            "applicant": {c: float(df[c].iloc[i]) for c in APPLICANT_COLUMNS},
        })

    summary = {
        "n": len(df),
        "avg_pd": float(proba.mean()),
        "exposure": float(df["amt_credit"].sum()),
        "bands": {b: bands.count(b) for b in ("low", "med", "high")},
    }
    if has_target:
        summary.update({k: round(v, 4) for k, v in summarize(df["TARGET"], proba).items()})

    return {"model": key, "summary": summary, "rows": rows[:BATCH_ROW_CAP]}


# Serve the frontend (app/CreditLens.html + app/src/*) at the root.
if APP_DIR.exists():

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(APP_DIR / "CreditLens.html")

    app.mount("/", StaticFiles(directory=APP_DIR), name="app")
