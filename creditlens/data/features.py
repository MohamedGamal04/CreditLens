"""Feature engineering: turn the three raw tables into one model-ready frame.

One row per ``SK_ID_CURR``. Three steps:
  1. ``add_application_features``  — affordability ratios + decoded time features.
  2. ``aggregate_bureau``          — collapse bureau (many rows/applicant) to one row.
  3. ``aggregate_previous``        — collapse previous_application to one row.
``build_features`` orchestrates them and left-joins onto the application frame so
every applicant is kept (no-history applicants get NaN, imputed later in the model
pipeline).

Leakage note: nothing here uses ``TARGET``. Aggregations are within a single
applicant's own history, so they are safe to compute on the full frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlens.config import ID_COL, PROCESSED_DIR, TARGET


def _clean_inf(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace +/-inf (from divide-by-zero) with NaN in the given columns."""
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df


def add_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Add affordability ratios and decoded age/employment features.

    Keeps ``SK_ID_CURR`` and (if present) ``TARGET`` untouched.
    """
    df = app.copy()
    ratios = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO"]

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    # DAYS_EMPLOYED already has the 365243 anomaly mapped to NaN by the loader.
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    df["EMPLOYED_TO_AGE"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    return _clean_inf(df, ratios)


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Collapse the bureau table to one row per applicant (``BUREAU_`` prefix)."""
    b = bureau.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = b.groupby(ID_COL).agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_AMT_CREDIT_SUM_sum=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_mean=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_DEBT_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_DAY_OVERDUE_sum=("CREDIT_DAY_OVERDUE", "sum"),
        BUREAU_DAY_OVERDUE_max=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_mean=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_min=("DAYS_CREDIT", "min"),
    )
    # Share of the applicant's bureau credits that are currently active.
    agg["BUREAU_ACTIVE_RATE"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_COUNT"]
    return agg.reset_index()


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Collapse previous_application to one row per applicant (``PREV_`` prefix)."""
    p = prev.copy()
    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    agg = p.groupby(ID_COL).agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_RATE=("IS_APPROVED", "mean"),
        PREV_AMT_APPLICATION_mean=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_max=("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_mean=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_max=("AMT_CREDIT", "max"),
        PREV_CNT_PAYMENT_mean=("CNT_PAYMENT", "mean"),
    )
    return agg.reset_index()


def build_features(
    app: pd.DataFrame, bureau: pd.DataFrame, prev: pd.DataFrame
) -> pd.DataFrame:
    """Full feature frame: application features + bureau/previous aggregates, joined.

    Returns one row per applicant. Row count equals ``len(app)`` (left join).
    """
    df = add_application_features(app)
    df = df.merge(aggregate_bureau(bureau), on=ID_COL, how="left")
    df = df.merge(aggregate_previous(prev), on=ID_COL, how="left")

    # Applicants with no bureau / previous history: count features are truly 0,
    # not "unknown". Fill those; leave amount/mean features as NaN for the imputer.
    for col in ["BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "PREV_COUNT"]:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 15-feature MODEL CONTRACT
# ---------------------------------------------------------------------------
# These are the exact inputs the web frontend collects (see app/src/model.js).
# The served models train and predict on THIS set so the form -> API -> model
# path is fully consistent. ``build_features`` above stays as the richer EDA /
# exploration frame; ``build_model_matrix`` below is the production contract.
MODEL_FEATURES = [
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "credit_to_income",
    "annuity_to_income",
    "age",
    "emp_years",
    "region_rating",
    "cnt_children",
    "bureau_dpd",
    "bureau_active",
    "bureau_debt",
    "prev_approval",
    "prev_refused",
    "prev_count",
]


def applicant_to_features(a: dict) -> dict:
    """Map a raw frontend applicant payload to the 15-feature contract (one row).

    The form already collects ``age`` and ``emp_years`` in years and the bureau /
    previous aggregates directly, so we only derive the two affordability ratios.
    Mirrors ``deriveInputs`` in app/src/model.js and ``build_model_matrix``.
    """
    income = max(float(a["amt_income"]), 1.0)  # guard divide-by-zero
    return {
        "ext_source_1": a["ext_source_1"],
        "ext_source_2": a["ext_source_2"],
        "ext_source_3": a["ext_source_3"],
        "credit_to_income": float(a["amt_credit"]) / income,
        "annuity_to_income": float(a["amt_annuity"]) / income,
        "age": a["age"],
        "emp_years": a["emp_years"],
        "region_rating": a["region_rating"],
        "cnt_children": a["cnt_children"],
        "bureau_dpd": a["bureau_dpd"],
        "bureau_active": a["bureau_active"],
        "bureau_debt": a["bureau_debt"],
        "prev_approval": a["prev_approval"],
        "prev_refused": a["prev_refused"],
        "prev_count": a["prev_count"],
    }


# Raw applicant columns a batch CSV must provide (same as the web form).
APPLICANT_COLUMNS = [
    "amt_income", "amt_credit", "amt_annuity", "age", "emp_years",
    "region_rating", "cnt_children", "ext_source_1", "ext_source_2", "ext_source_3",
    "bureau_active", "bureau_dpd", "bureau_debt", "prev_approval", "prev_refused", "prev_count",
]


def applicants_frame_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized raw-applicant frame -> 15-feature contract (for batch scoring).

    Same mapping as ``applicant_to_features`` but over a whole DataFrame.
    """
    income = df["amt_income"].clip(lower=1.0)
    out = pd.DataFrame(index=df.index)
    out["ext_source_1"] = df["ext_source_1"]
    out["ext_source_2"] = df["ext_source_2"]
    out["ext_source_3"] = df["ext_source_3"]
    out["credit_to_income"] = df["amt_credit"] / income
    out["annuity_to_income"] = df["amt_annuity"] / income
    out["age"] = df["age"]
    out["emp_years"] = df["emp_years"]
    out["region_rating"] = df["region_rating"]
    out["cnt_children"] = df["cnt_children"]
    out["bureau_dpd"] = df["bureau_dpd"]
    out["bureau_active"] = df["bureau_active"]
    out["bureau_debt"] = df["bureau_debt"]
    out["prev_approval"] = df["prev_approval"]
    out["prev_refused"] = df["prev_refused"]
    out["prev_count"] = df["prev_count"]
    return out[MODEL_FEATURES]


def _bureau_contract(bureau: pd.DataFrame) -> pd.DataFrame:
    """Bureau aggregates for the contract: active count, DPD count, total debt."""
    b = bureau.copy()
    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["_dpd"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)  # count of past-due tradelines
    agg = b.groupby(ID_COL).agg(
        bureau_active=("_active", "sum"),
        bureau_dpd=("_dpd", "sum"),
        bureau_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
    )
    return agg.reset_index()


def _previous_contract(prev: pd.DataFrame) -> pd.DataFrame:
    """Previous-application aggregates: count, approval rate, refusal count."""
    p = prev.copy()
    p["_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    agg = p.groupby(ID_COL).agg(
        prev_count=("SK_ID_PREV", "count"),
        prev_approval=("_approved", "mean"),
        prev_refused=("_refused", "sum"),
    )
    return agg.reset_index()


def build_model_matrix(
    app: pd.DataFrame,
    bureau: pd.DataFrame,
    prev: pd.DataFrame,
    *,
    with_target: bool = True,
) -> pd.DataFrame:
    """Build the 15-feature contract frame (plus ``SK_ID_CURR`` and optional TARGET).

    Columns are exactly ``MODEL_FEATURES`` — the same inputs the frontend form
    collects — so a request from the UI maps one-to-one onto the model's features.
    No-history applicants get 0 for count features; ``bureau_debt`` / ``prev_approval``
    stay NaN for the pipeline's imputer.
    """
    a = app
    income = a["AMT_INCOME_TOTAL"].replace(0, np.nan)

    out = pd.DataFrame({ID_COL: a[ID_COL].to_numpy()})
    out["ext_source_1"] = a["EXT_SOURCE_1"].to_numpy()
    out["ext_source_2"] = a["EXT_SOURCE_2"].to_numpy()
    out["ext_source_3"] = a["EXT_SOURCE_3"].to_numpy()
    out["credit_to_income"] = (a["AMT_CREDIT"] / income).to_numpy()
    out["annuity_to_income"] = (a["AMT_ANNUITY"] / income).to_numpy()
    out["age"] = (-a["DAYS_BIRTH"] / 365).to_numpy()
    out["emp_years"] = (-a["DAYS_EMPLOYED"] / 365).to_numpy()
    out["region_rating"] = a["REGION_RATING_CLIENT"].to_numpy()
    out["cnt_children"] = a["CNT_CHILDREN"].to_numpy()

    out = out.merge(_bureau_contract(bureau), on=ID_COL, how="left")
    out = out.merge(_previous_contract(prev), on=ID_COL, how="left")

    for col in ["bureau_active", "bureau_dpd", "prev_count", "prev_refused"]:
        out[col] = out[col].fillna(0)

    out = _clean_inf(out, ["credit_to_income", "annuity_to_income"])

    if with_target and TARGET in a.columns:
        out[TARGET] = a[TARGET].to_numpy()

    # Stable column order: id, the 15 features, then target.
    cols = [ID_COL, *MODEL_FEATURES] + ([TARGET] if with_target and TARGET in out.columns else [])
    return out[cols]


def load_or_build_model_matrix(*, use_cache: bool = True, cache_name: str = "model_matrix.pkl") -> pd.DataFrame:
    """Return the 15-feature contract frame, caching it to ``data/processed/``.

    Aggregating the 1.7M-row side tables every run is wasteful; this caches the
    result (gitignored) so modeling notebooks load it instantly. Delete the cache
    file or pass ``use_cache=False`` to force a rebuild after changing the features.
    """
    from creditlens.data.load import (
        load_application,
        load_bureau,
        load_previous_application,
    )

    cache = PROCESSED_DIR / cache_name
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    mat = build_model_matrix(load_application(), load_bureau(), load_previous_application())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mat.to_pickle(cache)
    return mat
