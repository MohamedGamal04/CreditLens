"""Model registry: every pipeline builds; smoke-train produces valid probabilities."""

import pytest

from creditlens.config import TARGET
from creditlens.data.features import MODEL_FEATURES
from creditlens.models.registry import BASE_MODELS, make_pipeline, make_stacking


@pytest.mark.parametrize("name", BASE_MODELS)
def test_pipeline_builds(name):
    pipe = make_pipeline(name)
    assert pipe.steps[-1][0] == "clf"


def test_stacking_builds():
    assert make_stacking() is not None


@pytest.mark.parametrize("name", ["logreg", "lgbm"])
def test_smoke_train_predicts_proba_in_unit_interval(matrix, name):
    X, y = matrix[MODEL_FEATURES], matrix[TARGET]
    pipe = make_pipeline(name).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert len(proba) == len(X)


def test_train_all_writes_bands(tmp_path, monkeypatch, matrix):
    import creditlens.pipeline as pipeline

    monkeypatch.setattr(pipeline, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ALL_MODELS", ["logreg"])  # one fast model
    monkeypatch.setattr(pipeline, "load_or_build_model_matrix", lambda: matrix)

    payload = pipeline.train_all(track=False)
    bands = payload["bands"]
    # Valid range + written into the payload. Strict low<high separation is unit-tested
    # in test_thresholds on realistic data; tiny synthetic fixtures can collapse the band.
    assert 0.0 <= bands["low"] <= bands["high"] <= 1.0
    assert bands["cost_ratio"] == 5.0
