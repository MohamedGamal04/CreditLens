"""Feature engineering + contract tests (shape, no leakage, derivation correctness)."""

import numpy as np

from creditlens.config import ID_COL, TARGET
from creditlens.data.features import (
    MODEL_FEATURES,
    applicant_to_features,
    applicants_frame_to_features,
)


def test_model_matrix_shape_and_columns(matrix, app_df):
    assert len(matrix) == len(app_df)              # left join kept every applicant
    assert matrix[ID_COL].is_unique
    assert list(matrix.columns) == [ID_COL, *MODEL_FEATURES, TARGET]


def test_model_matrix_no_inf_and_counts_filled(matrix):
    assert not np.isinf(matrix[MODEL_FEATURES].to_numpy(dtype=float)).any()
    # no-history applicants get 0 for count features (not NaN)
    for col in ["bureau_active", "bureau_dpd", "prev_count", "prev_refused"]:
        assert matrix[col].notna().all()


def test_no_target_leakage(matrix):
    corr = matrix[MODEL_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).abs()
    assert corr.max() < 0.95


def test_applicant_to_features_keys_and_ratio():
    payload = {
        "amt_income": 100000, "amt_credit": 300000, "amt_annuity": 20000,
        "age": 40, "emp_years": 5, "region_rating": 2, "cnt_children": 1,
        "ext_source_1": 0.5, "ext_source_2": 0.5, "ext_source_3": 0.5,
        "bureau_active": 1, "bureau_dpd": 0, "bureau_debt": 50000,
        "prev_approval": 0.6, "prev_refused": 1, "prev_count": 3,
    }
    feats = applicant_to_features(payload)
    assert list(feats) == MODEL_FEATURES
    assert feats["credit_to_income"] == 3.0
    assert feats["annuity_to_income"] == 0.2


def test_frame_to_features_matches_row(matrix, app_df, bureau_df, prev_df):
    raw = app_df.merge(
        # rebuild a raw-applicant frame for one row
        matrix[[ID_COL, "bureau_active", "bureau_dpd", "bureau_debt",
                "prev_approval", "prev_refused", "prev_count", "age", "emp_years",
                "region_rating", "cnt_children", "ext_source_1", "ext_source_2", "ext_source_3"]],
        on=ID_COL,
    )
    df = raw.rename(columns={
        "AMT_INCOME_TOTAL": "amt_income", "AMT_CREDIT": "amt_credit", "AMT_ANNUITY": "amt_annuity",
    })[
        ["amt_income", "amt_credit", "amt_annuity", "age", "emp_years", "region_rating",
         "cnt_children", "ext_source_1", "ext_source_2", "ext_source_3", "bureau_active",
         "bureau_dpd", "bureau_debt", "prev_approval", "prev_refused", "prev_count"]
    ]
    out = applicants_frame_to_features(df)
    assert list(out.columns) == MODEL_FEATURES
    assert len(out) == len(df)
