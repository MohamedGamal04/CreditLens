"""Tuning script: GridSearch returns best params for a model."""

from creditlens.config import TARGET
from creditlens.data.features import MODEL_FEATURES
from scripts.tune import tune


def test_tune_logreg_returns_best(matrix):
    X, y = matrix[MODEL_FEATURES], matrix[TARGET]
    out = tune("logreg", X, y, cv=2)
    assert "best_params" in out and "best_auc" in out
    assert "clf__C" in out["best_params"]          # logreg grid tunes C
    assert 0.0 <= out["best_auc"] <= 1.0
