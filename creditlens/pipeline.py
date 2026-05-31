"""End-to-end training pipeline: train, calibrate, and save all 6 models.

The frontend lets the user pick the model, so we persist **every** model (calibrated),
not just the best — each as ``models/<name>.joblib`` — plus ``models/metadata.json`` with
per-model metrics (the selector shows AUC) and the feature contract.

Split: 70% train / 15% calibrate / 15% test (stratified). Each model is fit on train,
isotonic-calibrated on the calibration set (PD must be trustworthy for credit decisions),
and scored on the held-out test set.

Run: ``python -m creditlens.pipeline``  (or ``make train``).
"""

from __future__ import annotations

import json
import logging
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from creditlens.config import COST_RATIO, MODELS_DIR, RANDOM_SEED, TARGET
from creditlens.data.features import MODEL_FEATURES, load_or_build_model_matrix
from creditlens.evaluation.metrics import summarize
from creditlens.evaluation.thresholds import decision_bands
from creditlens.models.registry import BASE_MODELS, make_pipeline, make_stacking

# MLflow is a dev/training-time dependency. Guarded so the pipeline still runs
# (and the serving image, which never imports it) does not require it.
try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW = True
except ImportError:  # pragma: no cover
    _MLFLOW = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("creditlens.train")

ALL_MODELS = BASE_MODELS + ["stacking"]
MLFLOW_EXPERIMENT = "creditlens"


def _build(name: str):
    return make_stacking() if name == "stacking" else make_pipeline(name)


def _loggable_params(est) -> dict:
    """Scalar hyperparameters of the classifier step, for mlflow.log_params."""
    has_clf = hasattr(est, "named_steps") and "clf" in est.named_steps
    clf = est.named_steps["clf"] if has_clf else est
    return {
        k: v for k, v in clf.get_params().items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }


def train_all(*, calibrate: bool = True, track: bool = True) -> dict:
    """Train, calibrate, and save every model; return the metadata payload.

    If ``track`` and MLflow is installed, each model is logged as an MLflow run
    (params + metrics + the calibrated model artifact) under the ``creditlens`` experiment.
    """
    mat = load_or_build_model_matrix()
    X, y = mat[MODEL_FEATURES], mat[TARGET]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED)
    log.info("train %s | calib %s | test %s", X_tr.shape, X_cal.shape, X_te.shape)

    use_mlflow = track and _MLFLOW
    if use_mlflow:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        log.info("mlflow tracking -> experiment '%s'", MLFLOW_EXPERIMENT)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    te_proba: dict = {}

    for name in ALL_MODELS:
        t0 = time.time()
        est = _build(name).fit(X_tr, y_tr)
        raw = summarize(y_te, est.predict_proba(X_te)[:, 1])

        if calibrate:
            # FrozenEstimator => isotonic fit on the calib set without refitting the model.
            model = CalibratedClassifierCV(
                FrozenEstimator(est), method="isotonic"
            ).fit(X_cal, y_cal)
            cal = summarize(y_te, model.predict_proba(X_te)[:, 1])
        else:
            model, cal = est, raw

        te_proba[name] = model.predict_proba(X_te)[:, 1]
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)  # compress (RF is large)
        meta[name] = {
            "auc": round(cal["auc"], 4),
            "gini": round(cal["gini"], 4),
            "ks": round(cal["ks"], 4),
            "ece": round(cal["ece"], 4),
            "ece_uncalibrated": round(raw["ece"], 4),
            "auc_uncalibrated": round(raw["auc"], 4),
            "calibrated": calibrate,
        }
        log.info("%-9s AUC=%.4f KS=%.4f ECE %.4f->%.4f (%.0fs)",
                 name, cal["auc"], cal["ks"], raw["ece"], cal["ece"], time.time() - t0)

        if use_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("model", name)
                mlflow.set_tag("calibrated", calibrate)
                mlflow.log_params(_loggable_params(est))
                mlflow.log_metrics({
                    "auc": cal["auc"], "ks": cal["ks"], "gini": cal["gini"], "ece": cal["ece"],
                    "auc_uncalibrated": raw["auc"], "ece_uncalibrated": raw["ece"],
                })
                mlflow.sklearn.log_model(model, name="model")

    payload = {
        "features": MODEL_FEATURES,
        "target": TARGET,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "seed": RANDOM_SEED,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best": max(meta, key=lambda k: meta[k]["auc"]),
        "models": meta,
    }
    low, high = decision_bands(y_te, te_proba[payload["best"]], COST_RATIO)
    payload["bands"] = {"low": low, "high": high, "cost_ratio": COST_RATIO}
    log.info("decision bands (cost_ratio=%.1f): low=%.4f high=%.4f", COST_RATIO, low, high)
    (MODELS_DIR / "metadata.json").write_text(json.dumps(payload, indent=2))
    log.info("best by AUC: %s | saved %d models -> %s",
             payload["best"], len(ALL_MODELS), MODELS_DIR)
    return payload


if __name__ == "__main__":
    train_all()
