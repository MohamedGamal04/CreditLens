"""Generate app/sample_applications.csv — a batch to test /batch_predict.

Builds the raw applicant columns the endpoint expects from the real data, plus
SK_ID_CURR + TARGET. Gitignored (Kaggle license) — regenerate with this script.

Usage: python scripts/make_sample_batch.py [n]
"""

from __future__ import annotations

import sys

import pandas as pd

from creditlens.config import APP_DIR
from creditlens.data.features import load_or_build_model_matrix
from creditlens.data.load import load_application


def main(n: int = 240) -> None:
    mat = load_or_build_model_matrix()  # 15 features + SK_ID_CURR + TARGET
    app = load_application()[["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]]
    df = mat.merge(app, on="SK_ID_CURR")

    out = pd.DataFrame({
        "SK_ID_CURR": df["SK_ID_CURR"],
        "amt_income": df["AMT_INCOME_TOTAL"],
        "amt_credit": df["AMT_CREDIT"],
        "amt_annuity": df["AMT_ANNUITY"],
        "age": df["age"].round(1),
        "emp_years": df["emp_years"].round(1),
        "region_rating": df["region_rating"],
        "cnt_children": df["cnt_children"],
        "ext_source_1": df["ext_source_1"].round(4),
        "ext_source_2": df["ext_source_2"].round(4),
        "ext_source_3": df["ext_source_3"].round(4),
        "bureau_active": df["bureau_active"].astype(int),
        "bureau_dpd": df["bureau_dpd"].astype(int),
        "bureau_debt": df["bureau_debt"].round(0),
        "prev_approval": df["prev_approval"].round(3),
        "prev_refused": df["prev_refused"].astype(int),
        "prev_count": df["prev_count"].astype(int),
        "TARGET": df["TARGET"],
    }).sample(n=n, random_state=7).reset_index(drop=True)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    path = APP_DIR / "sample_applications.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows -> {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 240)
