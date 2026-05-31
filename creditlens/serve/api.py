"""FastAPI app: serve the calibrated models + the CreditLens frontend.

All 6 models are loaded once at startup (the UI lets the user pick one). ``/predict``
scores a single applicant; ``/batch_predict`` scores an uploaded CSV; ``/models`` exposes
per-model metrics.

Scalability / concurrency:
- Routes are ``async``; the **blocking, CPU-bound** model inference runs in a threadpool
  (``run_in_threadpool``) so a slow prediction never blocks the event loop.
- The app is **stateless** (models are read-only, loaded per worker) → scale horizontally:
  ``uvicorn creditlens.serve.api:app --workers N`` (or ``make serve-prod``).
- Inputs are bounded: upload size (``MAX_UPLOAD_BYTES``) and row count (``MAX_BATCH_ROWS``);
  only ``BATCH_ROW_CAP`` rows are returned to the UI (summary is computed over all rows).

Run: ``uvicorn creditlens.serve.api:app --reload --port 8000`` (or ``make serve``).
"""

from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from creditlens.config import APP_DIR, MODELS_DIR
from creditlens.data.features import (
    APPLICANT_COLUMNS,
    MODEL_FEATURES,
    applicant_to_features,
    applicants_frame_to_features,
)
from creditlens.evaluation.drift import drift_report
from creditlens.evaluation.explain import explain_one
from creditlens.evaluation.metrics import summarize
from creditlens.serve.schema import (
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("creditlens.api")

BATCH_ROW_CAP = 1000  # max scored rows returned to the UI
MAX_BATCH_ROWS = 100_000  # reject larger uploads (protect the worker)
MAX_UPLOAD_BYTES = 32 * 1024 * 1024  # 32 MB

DEFAULT_LOW = 0.06   # fallback band cuts when metadata has none
DEFAULT_HIGH = 0.15

MODELS: dict = {}
METADATA: dict = {}
REFERENCE: dict = {}  # training feature distribution for drift (PSI)
ALIAS = {"stack": "stacking"}  # frontend key -> saved artifact name


def _load() -> None:
    """Load metadata + every saved model into memory (once per worker)."""
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        METADATA.update(json.loads(meta_path.read_text()))
        bands = METADATA.get("bands")
        if bands:
            global DEFAULT_LOW, DEFAULT_HIGH
            DEFAULT_LOW, DEFAULT_HIGH = bands["low"], bands["high"]
    ref_path = MODELS_DIR / "reference.json"
    if ref_path.exists():
        REFERENCE.update(json.loads(ref_path.read_text()))
    for path in MODELS_DIR.glob("*.joblib"):
        MODELS[path.stem] = joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    if MODELS:
        log.info("startup: loaded %d models %s", len(MODELS), sorted(MODELS))
    else:
        log.warning("startup: no models found in %s — run `make train`", MODELS_DIR)
    yield
    MODELS.clear()
    METADATA.clear()


app = FastAPI(title="CreditLens", version="0.1.0", lifespan=lifespan)


# --- helpers -----------------------------------------------------------------
def _band(pd_: float, low: float, high: float) -> str:
    return "low" if pd_ < low else ("med" if pd_ < high else "high")


_DECISION = {"low": "approve", "med": "review", "high": "decline"}


def _require_model(name: str):
    """Resolve a model key (with 'stack' alias) or raise 400/503."""
    key = ALIAS.get(name, name)
    model = MODELS.get(key)
    if model is None:
        log.warning("model '%s' unavailable (loaded: %s)", name, sorted(MODELS))
        raise HTTPException(
            503 if not MODELS else 400,
            f"Model '{name}' unavailable. Loaded: {sorted(MODELS)} (run `make train`).",
        )
    return model, key


def _predict_one(model, feats: dict) -> float:
    """CPU-bound: score one applicant. Runs in a threadpool."""
    row = pd.DataFrame([feats])[MODEL_FEATURES]
    return float(model.predict_proba(row)[:, 1][0])


def _score_batch(model, df: pd.DataFrame, low: float, high: float) -> dict:
    """CPU-bound: vectorized scoring of a whole frame. Runs in a threadpool."""
    feats_frame = applicants_frame_to_features(df)
    proba = model.predict_proba(feats_frame)[:, 1]
    bands = np.where(proba < low, "low", np.where(proba < high, "med", "high"))

    has_target = "TARGET" in df.columns and df["TARGET"].nunique() > 1
    has_id = "SK_ID_CURR" in df.columns
    ids = df["SK_ID_CURR"].astype(int).tolist() if has_id else list(range(len(df)))
    names = df["name"].astype(str).tolist() if "name" in df.columns else [f"APP-{i}" for i in ids]
    credit = df["amt_credit"].astype(float).tolist()
    ext2 = df["ext_source_2"].astype(float).tolist()
    targets = df["TARGET"].astype(int).tolist() if has_target else None
    applicants = df[APPLICANT_COLUMNS].astype(float).to_dict("records")

    cap = min(len(df), BATCH_ROW_CAP)
    rows = [
        {
            "id": ids[i], "name": names[i], "credit": credit[i], "ext2": ext2[i],
            "prob": float(proba[i]), "band": bands[i], "decision": _DECISION[bands[i]],
            "defaulted": targets[i] if has_target else None,
            "applicant": applicants[i],
        }
        for i in range(cap)
    ]

    summary = {
        "n": int(len(df)),
        "avg_pd": float(proba.mean()),
        "exposure": float(df["amt_credit"].sum()),
        "bands": {b: int((bands == b).sum()) for b in ("low", "med", "high")},
    }
    if has_target:
        summary.update({k: round(v, 4) for k, v in summarize(df["TARGET"], proba).items()})
    if REFERENCE:
        summary["drift"] = drift_report(REFERENCE, feats_frame, MODEL_FEATURES)
    return {"summary": summary, "rows": rows}


# --- routes ------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models_loaded": sorted(MODELS)}


@app.get("/models")
async def models() -> dict:
    """Per-model metrics for the UI selector (AUC/KS/ECE)."""
    if not METADATA:
        raise HTTPException(503, "No metadata. Run `make train` first.")
    return METADATA


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    model, key = _require_model(req.model)
    feats = applicant_to_features(req.applicant.model_dump())
    proba = await run_in_threadpool(_predict_one, model, feats)
    low = req.low if req.low is not None else DEFAULT_LOW
    high = req.high if req.high is not None else DEFAULT_HIGH
    band = _band(proba, low, high)
    log.info("predict model=%s pd=%.4f band=%s (%.1fms)",
             key, proba, band, (time.perf_counter() - t0) * 1000)
    return PredictResponse(
        probability=proba, band=band, decision=_DECISION[band], model=key, features=feats,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    t0 = time.perf_counter()
    model, key = _require_model(req.model)
    feats = applicant_to_features(req.applicant.model_dump())
    proba = await run_in_threadpool(_predict_one, model, feats)
    contribs = await run_in_threadpool(explain_one, model, key, feats, req.top_n)
    log.info("explain model=%s available=%s (%.1fms)",
             key, contribs is not None, (time.perf_counter() - t0) * 1000)
    return ExplainResponse(
        model=key, probability=proba,
        available=contribs is not None, contributions=contribs or [],
    )


@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    model: str = "lgbm",
    low: float | None = None,
    high: float | None = None,
) -> dict:
    """Score an uploaded CSV of applicants. Required columns: see APPLICANT_COLUMNS.

    Optional: ``SK_ID_CURR`` / ``name`` (display), ``TARGET`` (enables AUC/KS + realised
    defaults). Returns per-row PD/band/decision + a portfolio summary.
    """
    low = low if low is not None else DEFAULT_LOW
    high = high if high is not None else DEFAULT_HIGH
    t0 = time.perf_counter()
    model_obj, key = _require_model(model)

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        log.warning("batch upload rejected: %d bytes > limit", len(raw))
        raise HTTPException(413, f"File too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        log.warning("batch CSV parse error: %s", exc)
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    missing = [c for c in APPLICANT_COLUMNS if c not in df.columns]
    if missing:
        log.warning("batch CSV missing columns: %s", missing)
        raise HTTPException(400, f"CSV missing required columns: {missing}")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df)} > {MAX_BATCH_ROWS}).")

    result = await run_in_threadpool(_score_batch, model_obj, df, low, high)
    result["model"] = key
    log.info("batch_predict model=%s rows=%d bands=%s (%.0fms)",
             key, result["summary"]["n"], result["summary"]["bands"],
             (time.perf_counter() - t0) * 1000)
    return result


# Serve the frontend (app/CreditLens.html + app/src/*) at the root.
if APP_DIR.exists():

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(APP_DIR / "CreditLens.html")

    app.mount("/", StaticFiles(directory=APP_DIR), name="app")
