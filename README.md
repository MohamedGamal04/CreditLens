# CreditLens

Credit default risk modeling on the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/) dataset.

CreditLens trains and compares several models to predict an applicant's **probability of default (PD)**,
evaluates them with credit-realistic metrics (ROC AUC, KS, calibration, lift), calibrates the best model,
and serves it through a FastAPI backend with a small web frontend.

> **Status:** under active construction (portfolio MVP, built in phases). See
> `.claude/plans/` for the implementation plan.

## Data scope

- `application_train.csv` (main table)
- Aggregated features from `bureau.csv` and `previous_application.csv`

(The full competition has 7 tables; CreditLens uses these three for a strong, manageable feature set.)

## Models

LogisticRegression (baseline) · RandomForest · XGBoost · LightGBM · CatBoost · Stacking ensemble.

> Honest note: gradient-boosted trees dominate this dataset. The multi-model comparison is partly
> pedagogical — it demonstrates breadth and a fair evaluation harness, not a real selection dilemma.

## Quickstart

```bash
make install        # install package + dev deps (use a virtualenv)
make data           # download Kaggle CSVs (needs ~/.kaggle/kaggle.json + accepted comp rules)
make train          # FE -> CV-train all models -> evaluate -> save best calibrated model
make eval           # regenerate metrics table + plots
make serve          # FastAPI + frontend at http://localhost:8000
make test           # run the test suite
```

## Project layout

```
creditlens/      # python package: data, models, evaluation, serving
notebooks/       # teaching/EDA notebooks (refactored into the package)
scripts/         # data download + test-fixture generation
tests/           # pytest suite (runs on tiny synthetic fixtures in CI)
app/             # CreditLens.html frontend
```

## Metrics

Beyond the competition's ROC AUC, CreditLens reports **KS statistic**, **calibration / reliability**
(PD must be calibrated for credit decisions), and **gains/lift** by decile.
