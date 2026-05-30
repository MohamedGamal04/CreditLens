"""Load raw Home Credit CSVs into pandas DataFrames.

Thin wrappers around ``pd.read_csv`` that centralize file locations (via
``config``) and replace Home Credit's sentinel anomalies with proper NaN so
downstream code does not have to remember them.

Notable dataset quirk: ``DAYS_EMPLOYED == 365243`` is a placeholder for
"not employed / pensioner" (~18% of rows). Left as-is it poisons any model;
we map it to NaN here.
"""

from __future__ import annotations

import pandas as pd

from creditlens import config

DAYS_EMPLOYED_ANOMALY = 365243


def load_application(path=None) -> pd.DataFrame:
    """Load the main applicant table (one row per loan application)."""
    df = pd.read_csv(path or config.APPLICATION_TRAIN)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, pd.NA)
    return df


def load_bureau(path=None) -> pd.DataFrame:
    """Load prior credits reported by other institutions to the credit bureau.

    Many rows per ``SK_ID_CURR`` — must be aggregated before joining.
    """
    return pd.read_csv(path or config.BUREAU)


def load_previous_application(path=None) -> pd.DataFrame:
    """Load the applicant's previous Home Credit applications.

    Many rows per ``SK_ID_CURR`` — must be aggregated before joining.
    """
    return pd.read_csv(path or config.PREVIOUS_APPLICATION)
