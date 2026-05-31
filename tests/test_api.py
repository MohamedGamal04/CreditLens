"""API tests: health, predict, batch_predict, models, validation errors.

A tiny model is injected into the app's registry so no `make train` artifacts are needed.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditlens.config import ID_COL, TARGET
from creditlens.data.features import MODEL_FEATURES
from creditlens.models.registry import make_pipeline
from creditlens.serve import api

VALID_APPLICANT = {
    "amt_income": 162000, "amt_credit": 640000, "amt_annuity": 31500, "age": 36,
    "emp_years": 3.5, "region_rating": 2, "cnt_children": 1,
    "ext_source_1": 0.42, "ext_source_2": 0.48, "ext_source_3": 0.43,
    "bureau_active": 2, "bureau_dpd": 0, "bureau_debt": 165000,
    "prev_approval": 0.58, "prev_refused": 1, "prev_count": 4,
}


@pytest.fixture
def client(matrix):
    """TestClient with one tiny fitted model injected (no lifespan/_load)."""
    model = make_pipeline("logreg").fit(matrix[MODEL_FEATURES], matrix[TARGET])
    api.MODELS.clear()
    api.MODELS["logreg"] = model
    api.METADATA.clear()
    api.METADATA.update({"best": "logreg", "models": {"logreg": {"auc": 0.7}}})
    yield TestClient(api.app)
    api.MODELS.clear()
    api.METADATA.clear()


def _batch_csv(matrix, app_df) -> bytes:
    m = matrix.merge(
        app_df[[ID_COL, "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]], on=ID_COL)
    df = pd.DataFrame({
        "amt_income": m["AMT_INCOME_TOTAL"], "amt_credit": m["AMT_CREDIT"],
        "amt_annuity": m["AMT_ANNUITY"], "age": m["age"], "emp_years": m["emp_years"],
        "region_rating": m["region_rating"], "cnt_children": m["cnt_children"],
        "ext_source_1": m["ext_source_1"], "ext_source_2": m["ext_source_2"],
        "ext_source_3": m["ext_source_3"], "bureau_active": m["bureau_active"],
        "bureau_dpd": m["bureau_dpd"], "bureau_debt": m["bureau_debt"],
        "prev_approval": m["prev_approval"], "prev_refused": m["prev_refused"],
        "prev_count": m["prev_count"], "SK_ID_CURR": m[ID_COL], "TARGET": m[TARGET],
    })
    return df.to_csv(index=False).encode()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "logreg" in r.json()["models_loaded"]


def test_models(client):
    assert client.get("/models").json()["best"] == "logreg"


def test_predict_ok(client):
    r = client.post("/predict", json={"applicant": VALID_APPLICANT, "model": "logreg"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "decline"}


def test_predict_invalid_input_422(client):
    bad = dict(VALID_APPLICANT, ext_source_1=5.0)  # out of [0,1]
    r = client.post("/predict", json={"applicant": bad, "model": "logreg"})
    assert r.status_code == 422


def test_predict_unknown_model_400(client):
    r = client.post("/predict", json={"applicant": VALID_APPLICANT, "model": "xgb"})
    assert r.status_code == 400


def test_batch_predict_ok(client, matrix, app_df):
    csv = _batch_csv(matrix, app_df)
    r = client.post("/batch_predict?model=logreg",
                    files={"file": ("a.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n"] == len(matrix)
    assert sum(body["summary"]["bands"].values()) == len(matrix)
    assert "auc" in body["summary"]  # TARGET present


def test_batch_predict_missing_columns_400(client):
    bad = b"a,b\n1,2\n"
    r = client.post("/batch_predict?model=logreg",
                    files={"file": ("a.csv", io.BytesIO(bad), "text/csv")})
    assert r.status_code == 400
