"""Model registry — 6 leakage-safe pipelines + tuning grids for the 15-feature contract.

Each model is wrapped in a sklearn ``Pipeline`` that does its own preprocessing
(median impute; standard-scale only for the linear model), so every transform is
fit on training folds only — no leakage. Imbalance (8.07% positives, ~11.4:1) is
handled per family: ``class_weight='balanced'`` for sklearn estimators,
``scale_pos_weight`` for the gradient-boosted ones.

Use ``make_pipeline(name)`` for a ready estimator, ``PARAM_GRIDS[name]`` for its
GridSearch grid, and ``make_stacking()`` for the meta-ensemble.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from creditlens.config import RANDOM_SEED

# neg/pos ratio at 8.07% positives — passed to the boosters for imbalance.
SCALE_POS_WEIGHT = 11.4

BASE_MODELS = ["logreg", "rf", "xgb", "lgbm", "catboost"]


def _pre(scale: bool = False) -> list:
    """Preprocessing steps: median impute always; standard-scale only when asked."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_pipeline(name: str) -> Pipeline:
    """Return a fresh preprocessing + classifier pipeline for ``name`` in BASE_MODELS."""
    name = name.lower()
    if name == "logreg":
        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
        )
        return Pipeline(_pre(scale=True) + [("clf", clf)])
    if name == "rf":
        # tuned: n_estimators=400, max_depth=15, min_samples_leaf=20
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=15, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
        )
        return Pipeline(_pre() + [("clf", clf)])
    if name == "xgb":
        # tuned: max_depth=3, learning_rate=0.03, n_estimators=600
        clf = XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SCALE_POS_WEIGHT,
            eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED,
        )
        return Pipeline(_pre() + [("clf", clf)])
    if name == "lgbm":
        # tuned: num_leaves=31, learning_rate=0.03, n_estimators=400
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED, verbose=-1,
        )
        return Pipeline(_pre() + [("clf", clf)])
    if name == "catboost":
        # tuned: depth=6, learning_rate=0.03
        clf = CatBoostClassifier(
            iterations=400, learning_rate=0.03, depth=6,
            scale_pos_weight=SCALE_POS_WEIGHT, verbose=0, random_seed=RANDOM_SEED,
        )
        return Pipeline(_pre() + [("clf", clf)])
    raise ValueError(f"unknown model: {name!r} (expected one of {BASE_MODELS})")


# Small GridSearch grids (keys target the 'clf' pipeline step). Kept deliberately
# compact so tuning all models is feasible; widen on a subsample if exploring.
PARAM_GRIDS = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [15, None],
        "clf__min_samples_leaf": [1, 20],
    },
    "xgb": {
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.03, 0.1],
        "clf__n_estimators": [300, 600],
    },
    "lgbm": {
        "clf__num_leaves": [31, 63],
        "clf__learning_rate": [0.03, 0.1],
        "clf__n_estimators": [400, 800],
    },
    "catboost": {
        "clf__depth": [4, 6],
        "clf__learning_rate": [0.03, 0.1],
    },
}


def make_stacking() -> StackingClassifier:
    """Stacking ensemble: the 3 boosters as base learners, LogReg meta-learner.

    ``stack_method='predict_proba'`` + internal CV gives out-of-fold base predictions
    to the meta-model, so the blend does not leak.
    """
    estimators = [(n, make_pipeline(n)) for n in ["lgbm", "xgb", "catboost"]]
    final = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    return StackingClassifier(
        estimators=estimators, final_estimator=final,
        stack_method="predict_proba", cv=3, n_jobs=-1,
    )
