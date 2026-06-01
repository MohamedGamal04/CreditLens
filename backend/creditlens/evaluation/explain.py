"""Per-feature attributions for a single applicant.

TreeSHAP runs on the model's raw booster (xgb/lgbm/catboost/rf); LogisticRegression
uses coefficient x standardised value. The calibrated wrapper is a monotone
post-transform, so contribution directions are unchanged — these explain the model's
RAW score, not the calibrated PD, and need not sum to it.

The stacking ensemble has no tractable single-tree explanation -> returns None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from creditlens.data.features import MODEL_FEATURES


def _unwrap(model):
    """Underlying estimator/pipeline from a (possibly isotonic-calibrated) model."""
    if hasattr(model, "calibrated_classifiers_"):
        est = model.calibrated_classifiers_[0].estimator
        return getattr(est, "estimator", est)  # FrozenEstimator -> wrapped estimator
    return model


def explain_one(model, model_key: str, feats: dict, top_n: int = 8) -> list[dict] | None:
    """Top-N signed feature contributions for one applicant, or None if unavailable."""
    est = _unwrap(model)
    named = getattr(est, "named_steps", {})
    if "clf" not in named:  # e.g. a bare StackingClassifier
        return None
    clf = named["clf"]
    if clf.__class__.__name__ == "StackingClassifier":
        return None

    row = pd.DataFrame([feats])[MODEL_FEATURES]
    Xt = np.asarray(est[:-1].transform(row))  # apply impute (+scale) like the model

    if isinstance(clf, LogisticRegression):
        contrib = clf.coef_[0] * Xt[0]
    else:
        import shap

        sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
        contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]  # rf -> (n,feat,classes)

    out = [
        {"feature": f, "contribution": float(c), "value": float(v)}
        for f, c, v in zip(MODEL_FEATURES, contrib, Xt[0], strict=True)
    ]
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out[:top_n]
