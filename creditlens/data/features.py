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

from creditlens.config import ID_COL


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
