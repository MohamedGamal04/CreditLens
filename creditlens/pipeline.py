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
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from creditlens.config import MODELS_DIR, RANDOM_SEED, TARGET
from creditlens.data.features import MODEL_FEATURES, load_or_build_model_matrix
from creditlens.evaluation.metrics import summarize
from creditlens.models.registry import BASE_MODELS, make_pipeline, make_stacking

ALL_MODELS = BASE_MODELS + ["stacking"]


def _build(name: str):
    return make_stacking() if name == "stacking" else make_pipeline(name)


def train_all(*, calibrate: bool = True) -> dict:
    """Train, calibrate, and save every model; return the metadata payload."""
    mat = load_or_build_model_matrix()
    X, y = mat[MODEL_FEATURES], mat[TARGET]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED)
    print(f"train {X_tr.shape} | calib {X_cal.shape} | test {X_te.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

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
        print(f"{name:9s} AUC={cal['auc']:.4f} KS={cal['ks']:.4f} "
              f"ECE {raw['ece']:.4f}->{cal['ece']:.4f}  ({time.time() - t0:.0f}s)")

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
    (MODELS_DIR / "metadata.json").write_text(json.dumps(payload, indent=2))
    print(f"\nbest by AUC: {payload['best']}  | saved {len(ALL_MODELS)} models -> {MODELS_DIR}")
    return payload


if __name__ == "__main__":
    train_all()
