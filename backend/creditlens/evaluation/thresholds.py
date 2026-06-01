"""Cost-based decision thresholds.

Turn held-out predictions into operating cutoffs:
- ``optimal_threshold`` — the approve/decline cut that minimises expected
  misclassification cost, where a false negative (approving a defaulter) costs
  ``cost_ratio``x a false positive (declining a good applicant).
- ``decision_bands`` — (low, high) cuts for the three-band UI: below ``low``
  auto-approve (cohort risk <= portfolio base rate), above ``high`` decline
  (cost-optimal), in between review.
- ``strategy_table`` — per-threshold approval-rate / bad-rate diagnostics for
  the evaluation notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_threshold(y_true, proba, cost_ratio: float) -> float:
    """Decline cut minimising ``cost_ratio * FN + FP`` over candidate cutoffs."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    candidates = np.unique(p)
    best_t, best_cost = float(candidates[0]), None
    for t in candidates:
        decline = p >= t
        fn = int(((y == 1) & ~decline).sum())
        fp = int(((y == 0) & decline).sum())
        cost = cost_ratio * fn + fp
        if best_cost is None or cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def decision_bands(y_true, proba, cost_ratio: float) -> tuple[float, float]:
    """Return (low, high) band cuts. ``high`` is cost-optimal; ``low`` is the
    largest cut whose below-cohort defaults at or under the portfolio base rate."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    high = optimal_threshold(y, p, cost_ratio)

    order = np.argsort(p)
    ys, ps = y[order], p[order]
    cohort_bad_rate = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    base = float(y.mean())

    eligible = ps[(cohort_bad_rate <= base) & (ps < high)]
    low = float(eligible.max()) if eligible.size else min(high / 2, float(ps.min()))
    low = min(low, high)
    return round(low, 4), round(high, 4)


def strategy_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Per-threshold approval-rate and realised bad-rate (decline if proba >= t)."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    thresholds = np.linspace(p.min(), p.max(), n_bins)
    rows = []
    for t in thresholds:
        approve = p < t
        n_app = int(approve.sum())
        rows.append({
            "threshold": float(t),
            "approval_rate": n_app / len(p),
            "bad_rate": float(y[approve].mean()) if n_app else 0.0,
        })
    return pd.DataFrame(rows)
