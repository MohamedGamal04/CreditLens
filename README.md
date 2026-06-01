# CreditLens

Credit default-risk modeling on the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/) dataset — an end-to-end ML project: data → features → 6 tuned & calibrated models → FastAPI serving → an interactive risk-assessment web app.

> Predicts an applicant's **probability of default (PD)**, calibrated and explainable, with a model the user can choose at scoring time.

![CI](https://github.com/MohamedGamal04/CreditLens/actions/workflows/ci.yml/badge.svg)

## What it does

- **6 models compared** — LogisticRegression (baseline) · RandomForest · XGBoost · LightGBM · CatBoost · Stacking ensemble.
- **Credit-realistic evaluation** — ROC AUC **plus** KS, probability **calibration** (isotonic), and lift by decile. A 0.74-AUC model with bad PDs is useless for lending; we calibrate.
- **FastAPI backend** — `/predict` (single applicant), `/batch_predict` (CSV upload), `/explain` (attributions), async, scales horizontally.
- **Web app** — score one applicant live, switch models, or upload a portfolio CSV.
- **Explainable** — `/explain` returns real SHAP (trees) / coefficient (logreg) per-feature attributions, shown in the app as adverse-action reasons (the stacking ensemble reports "unavailable").
- **Cost-based decisions** — approve/review/decline bands derived from a 5:1 false-negative:false-positive cost ratio, written into `metadata.json` — not hand-picked.
- **Drift monitoring** — per-feature PSI on every `/batch_predict` upload vs the training reference distribution.
- **Fairness audited** — disparate-impact (80% rule) + equal-opportunity checks across gender, age, and region.

## Results (held-out test, calibrated)

| Model | ROC AUC | KS | ECE (raw → calibrated) |
|---|---|---|---|
| **Stacking** | **0.7432** | 0.368 | 0.004 → 0.002 |
| LightGBM | 0.7429 | 0.368 | 0.331 → **0.002** |
| XGBoost | 0.7428 | 0.366 | 0.342 → 0.002 |
| CatBoost | 0.7414 | 0.365 | 0.343 → 0.001 |
| RandomForest | 0.7350 | 0.355 | 0.277 → 0.003 |
| LogisticRegression | 0.7276 | 0.340 | 0.353 → 0.002 |

- **Calibration matters most:** raw boosters were badly over-confident (ECE ≈ 0.34); isotonic calibration cuts it to ≈ 0.002 — that's what makes the served PD trustworthy.
- **Lift:** the riskiest score decile catches **~3.3×** the base default rate.
- **Honest note on AUC ≈ 0.74:** the served models use a **15-feature contract** the web form can actually supply (see below). That's ~0.02–0.03 below a full 146-feature model — a deliberate trade for an end-to-end, form-servable product. Gradient-boosted trees dominate; the multi-model comparison is partly pedagogical.

## Fairness

The served model is audited for disparate impact on the held-out test set across gender, age band, and region rating (notebook [`05_fairness.ipynb`](notebooks/05_fairness.ipynb)):

| Group | Disparate impact (min/max approval) | 80% rule |
|---|---|---|
| Gender | 0.96 | pass |
| Age band | 0.83 | pass (marginal) |
| Region rating | 0.84 | pass (marginal) |

No group **fails** the 80% rule, but **age and region carry real equal-opportunity gaps**: among true non-defaulters, under-30 applicants are approved ~14 pts less than 60+ (84% vs 98%). The model uses age/region as predictive proxies, so *qualified* applicants in higher-risk cohorts are disadvantaged — disclosed here rather than hidden; in production this warrants group-aware thresholds or dropping these proxies.

## The 15-feature contract

The web form (and `/predict`) collect exactly the 15 features the models train on, so request → model is consistent end-to-end:

```
ext_source_1/2/3 · credit_to_income · annuity_to_income · age · emp_years
region_rating · cnt_children · bureau_dpd · bureau_active · bureau_debt
prev_approval · prev_refused · prev_count
```

`EXT_SOURCE_1/2/3` (anonymized external credit scores) are the **strongest predictors** but a real applicant can't type them — in production they'd come from a bureau API. The demo exposes them as sliders.

## Architecture

```
Kaggle CSVs ──> creditlens.data (load, validate, features)
                     │  build_model_matrix  → 15-feature contract
                     ▼
        creditlens.models.registry (6 leakage-safe sklearn Pipelines)
                     │  GridSearch tuning + StratifiedKFold CV
                     ▼
        creditlens.pipeline  (train on full data → isotonic calibrate → save)
                     │  models/*.joblib + metadata.json
                     ▼
        backend/creditlens.serve.api (FastAPI)  ──>  frontend/CreditLens.html (React)
           /predict · /explain · /batch_predict · /models · /health
```

## Quickstart

```bash
make install        # package + dev deps (use a virtualenv / uv)
make data           # download Kaggle CSVs (needs ~/.kaggle/kaggle.json + accepted comp rules)
make train          # FE → tune → train 6 models on full data → calibrate → save to models/
make serve          # FastAPI + web app at http://localhost:8000
make test           # pytest (runs on synthetic fixtures, no real data)
make lint           # ruff
```

Then open **http://localhost:8000**: adjust an applicant and watch the calibrated PD/decision update, switch models, or go to **Portfolio** and upload a CSV.

`make serve-prod` runs multiple uvicorn workers (the app is stateless → scales horizontally).

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Score one applicant → calibrated PD + band + decision |
| `/explain` | POST | Per-feature attributions (SHAP for trees, coefficients for logreg; N/A for stacking) |
| `/batch_predict` | POST | Upload a CSV → per-row scores + portfolio summary (AUC/KS if `TARGET` present) + per-feature drift PSI |
| `/models` | GET | Per-model metrics (drives the UI selector) |
| `/health` | GET | Liveness + loaded models |

**Batch CSV columns** (required): `amt_income, amt_credit, amt_annuity, age, emp_years, region_rating, cnt_children, ext_source_1, ext_source_2, ext_source_3, bureau_active, bureau_dpd, bureau_debt, prev_approval, prev_refused, prev_count`. Optional: `SK_ID_CURR`, `name`, `TARGET` (enables AUC/KS + realised-default shading). Generate a sample: `python scripts/make_sample_batch.py`.

## Notebooks

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | Target imbalance, missingness, EXT_SOURCE signal, ratios, side-table cardinality, ydata-profiling |
| `02_features.ipynb` | Feature engineering + leakage/sanity checks |
| `03_modeling.ipynb` | GridSearch tuning, CV leaderboard, permutation feature importance (plotly) |
| `04_evaluation.ipynb` | Calibration, reliability/ROC/lift curves, cost-based thresholds |
| `05_fairness.ipynb` | Disparate-impact + equal-opportunity audit by gender/age/region |
| `06_monitoring.ipynb` | Drift monitoring (PSI) demo on a shifted batch |
| `exp_pca.ipynb` | PCA experiment (negative result — kept out of production) |

## Project layout

```
backend/         FastAPI + ML (deploys to HF Spaces)
  creditlens/    data · models · evaluation · serve · pipeline.py · config.py
  tests/         pytest suite (synthetic fixtures; runs in CI without Kaggle data)
  Dockerfile · pyproject.toml · uv.lock · conftest.py
frontend/        React app (deploys to Vercel): CreditLens.html + src/ + vercel.json
notebooks/       EDA → features → modeling → evaluation → fairness → monitoring + experiments
scripts/         download_data · make_fixtures · make_sample_batch · tune
models/ · data/  trained artifacts + Kaggle CSVs (gitignored)
```

## Limitations / honest notes

- **AUC ceiling ≈ 0.74** from the 15-feature form contract (full features would score higher but aren't form-servable).
- **Portfolio & Model-card pages** use synthetic data for the demo; the Applicant + Explanation pages and the Portfolio CSV upload hit the real backend.
- **Plotly charts** render in VSCode/Jupyter but not static GitHub previews.
- Kaggle competition data is **not redistributed** (gitignored); reviewers run `make data` with their own token.

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · FastAPI · uvicorn · pydantic · React (CDN + Babel) · plotly · pytest · ruff · GitHub Actions.
