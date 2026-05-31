"""Generate tiny synthetic samples mimicking the Home Credit schema.

Used by tests/CI so we never download the multi-GB Kaggle data. Exposes generator
functions (imported directly by tests) and a ``main`` that writes CSVs to
``data/processed/`` for the CI ``fixtures`` step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlens.config import PROCESSED_DIR

N_ROWS = 300
SEED = 42

EDU = ["Lower secondary", "Secondary / secondary special", "Higher education"]


def make_application_sample(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "TARGET": rng.integers(0, 2, n),
            "AMT_INCOME_TOTAL": rng.uniform(30_000, 300_000, n),
            "AMT_CREDIT": rng.uniform(50_000, 1_000_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 60_000, n),
            "AMT_GOODS_PRICE": rng.uniform(50_000, 900_000, n),
            "DAYS_BIRTH": rng.integers(-25000, -7000, n),
            "DAYS_EMPLOYED": rng.integers(-15000, 0, n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n),
            "CNT_CHILDREN": rng.integers(0, 4, n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "NAME_EDUCATION_TYPE": rng.choice(EDU, n),
        }
    )


def make_bureau_sample(ids, seed: int = 1) -> pd.DataFrame:
    """0–4 prior bureau credits per applicant (some applicants have none)."""
    rng = np.random.default_rng(seed)
    rows = []
    bid = 200000
    for cur in ids:
        for _ in range(int(rng.integers(0, 5))):
            rows.append(
                {
                    "SK_ID_CURR": int(cur),
                    "SK_ID_BUREAU": bid,
                    "CREDIT_ACTIVE": rng.choice(["Active", "Closed"]),
                    "CREDIT_DAY_OVERDUE": int(rng.integers(0, 3)),
                    "AMT_CREDIT_SUM": float(rng.uniform(10_000, 500_000)),
                    "AMT_CREDIT_SUM_DEBT": float(rng.uniform(0, 300_000)),
                    "DAYS_CREDIT": int(rng.integers(-3000, -10)),
                }
            )
            bid += 1
    return pd.DataFrame(rows)


def make_previous_sample(ids, seed: int = 2) -> pd.DataFrame:
    """0–4 prior Home Credit applications per applicant."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 300000
    for cur in ids:
        for _ in range(int(rng.integers(0, 5))):
            rows.append(
                {
                    "SK_ID_PREV": pid,
                    "SK_ID_CURR": int(cur),
                    "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"]),
                    "AMT_APPLICATION": float(rng.uniform(10_000, 800_000)),
                    "AMT_CREDIT": float(rng.uniform(10_000, 800_000)),
                    "CNT_PAYMENT": float(rng.integers(6, 60)),
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app = make_application_sample()
    bureau = make_bureau_sample(app["SK_ID_CURR"])
    prev = make_previous_sample(app["SK_ID_CURR"])
    for name, df in [("application", app), ("bureau", bureau), ("previous_application", prev)]:
        out = PROCESSED_DIR / f"sample_{name}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {len(df):>5} rows -> {out}")


if __name__ == "__main__":
    main()
