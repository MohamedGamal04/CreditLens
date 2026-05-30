"""Generate a tiny synthetic sample mimicking the Home Credit schema.

Used by tests/CI so we never download the multi-GB Kaggle data in CI.
This is a Phase 0 placeholder: it currently writes a minimal application-like
frame. It will be expanded with bureau / previous_application columns as the
feature engineering schema is finalized (Phase 2/6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlens.config import PROCESSED_DIR

N_ROWS = 200
SEED = 42


def make_application_sample(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "TARGET": rng.integers(0, 2, n),
            "AMT_INCOME_TOTAL": rng.uniform(30_000, 300_000, n),
            "AMT_CREDIT": rng.uniform(50_000, 1_000_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 60_000, n),
            "DAYS_BIRTH": rng.integers(-25000, -7000, n),
            "DAYS_EMPLOYED": rng.integers(-15000, 0, n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        }
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = make_application_sample()
    out = PROCESSED_DIR / "sample_application.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
