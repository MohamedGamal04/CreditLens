"""Lightweight schema / sanity validation for the raw application table.

Catches the failure modes that silently corrupt a model: missing key columns,
a target that is not binary, duplicate IDs, or impossible value ranges. Kept
dependency-free (plain pandas) so it runs anywhere, including CI on fixtures.

Raises ``DataValidationError`` on the first hard failure; returns a list of
soft warnings (e.g. unexpectedly high missingness) for the caller to log.
"""

from __future__ import annotations

import pandas as pd

from creditlens.config import ID_COL, TARGET


class DataValidationError(ValueError):
    """Raised when the application table violates a hard schema expectation."""


# Columns we rely on downstream; absence is a hard error.
REQUIRED_COLUMNS = (
    ID_COL,
    TARGET,
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
)

# (column, min, max) physically plausible ranges; violations are hard errors.
RANGE_CHECKS = (
    ("AMT_INCOME_TOTAL", 0, None),
    ("AMT_CREDIT", 0, None),
    ("DAYS_BIRTH", None, 0),  # stored as negative days from application
)


def validate_application(df: pd.DataFrame, *, require_target: bool = True) -> list[str]:
    """Validate the application frame. Returns soft warnings; raises on hard errors."""
    warnings: list[str] = []

    required = REQUIRED_COLUMNS if require_target else tuple(c for c in REQUIRED_COLUMNS if c != TARGET)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if df[ID_COL].duplicated().any():
        raise DataValidationError(f"Duplicate {ID_COL} values found")

    if require_target:
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad:
            raise DataValidationError(f"{TARGET} must be binary 0/1; found {bad}")

    for col, lo, hi in RANGE_CHECKS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if lo is not None and (s < lo).any():
            raise DataValidationError(f"{col} has values below {lo}")
        if hi is not None and (s > hi).any():
            raise DataValidationError(f"{col} has values above {hi}")

    # Soft signal: columns that are almost entirely missing.
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.6]
    if high_missing:
        warnings.append(f"High missingness (>60%): {high_missing}")

    return warnings
