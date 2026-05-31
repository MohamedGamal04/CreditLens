"""Per-applicant attribution: SHAP for trees, coefficients for logreg, None for stacking."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from creditlens.config import TARGET
from creditlens.data.features import MODEL_FEATURES
from creditlens.evaluation.explain import explain_one
from creditlens.models.registry import make_pipeline, make_stacking


def _feats(matrix):
    return {f: float(matrix[f].iloc[0]) for f in MODEL_FEATURES}


def test_explain_logreg_ranked_contributions(matrix):
    m = make_pipeline("logreg").fit(matrix[MODEL_FEATURES], matrix[TARGET])
    out = explain_one(m, "logreg", _feats(matrix), top_n=5)
    assert len(out) == 5
    assert set(out[0]) == {"feature", "contribution", "value"}
    assert abs(out[0]["contribution"]) >= abs(out[-1]["contribution"])  # sorted desc


def test_explain_tree_contributions(matrix):
    m = make_pipeline("lgbm").fit(matrix[MODEL_FEATURES], matrix[TARGET])
    out = explain_one(m, "lgbm", _feats(matrix))
    assert len(out) == 8  # default top_n
    assert all(d["feature"] in MODEL_FEATURES for d in out)


def test_explain_stacking_unavailable(matrix):
    m = make_stacking().fit(matrix[MODEL_FEATURES], matrix[TARGET])
    assert explain_one(m, "stacking", _feats(matrix)) is None


def test_explain_through_calibrated_wrapper(matrix):
    X, y = matrix[MODEL_FEATURES], matrix[TARGET]
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    base = make_pipeline("lgbm").fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X_cal, y_cal)
    out = explain_one(cal, "lgbm", _feats(matrix))
    assert len(out) == 8
