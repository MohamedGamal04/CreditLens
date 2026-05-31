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
