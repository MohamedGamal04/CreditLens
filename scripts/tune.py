"""GridSearch tuning for the base models — proves the hyperparameters baked into
``registry.make_pipeline``. Reads ``PARAM_GRIDS`` and tunes with ``N_FOLDS``-fold
roc_auc CV.

Run: ``python scripts/tune.py [--sample N] [--model NAME]``  (or ``make tune``).
The full data over every grid is slow; ``--sample`` (default 50k) keeps it feasible.
"""

from __future__ import annotations

import argparse

from sklearn.model_selection import GridSearchCV

from creditlens.config import N_FOLDS, RANDOM_SEED, TARGET
from creditlens.data.features import MODEL_FEATURES, load_or_build_model_matrix
from creditlens.models.registry import BASE_MODELS, PARAM_GRIDS, make_pipeline


def tune(name: str, X, y, cv: int = N_FOLDS) -> dict:
    """GridSearch ``name`` over its PARAM_GRIDS; return best params + CV AUC."""
    gs = GridSearchCV(make_pipeline(name), PARAM_GRIDS[name],
                      scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    return {"best_params": gs.best_params_, "best_auc": round(float(gs.best_score_), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50_000, help="row subsample (0 = full)")
    ap.add_argument("--model", choices=BASE_MODELS, help="tune one model (default: all)")
    args = ap.parse_args()

    mat = load_or_build_model_matrix()
    if args.sample and len(mat) > args.sample:
        mat = mat.sample(args.sample, random_state=RANDOM_SEED)
    X, y = mat[MODEL_FEATURES], mat[TARGET]

    for name in ([args.model] if args.model else BASE_MODELS):
        r = tune(name, X, y)
        print(f"{name:9s} AUC={r['best_auc']}  {r['best_params']}")


if __name__ == "__main__":
    main()
