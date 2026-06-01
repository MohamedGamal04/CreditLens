"""Credit-realistic evaluation metrics.

Beyond ROC AUC we report:
- **KS** — max separation between the cumulative good/bad distributions (industry-standard
  discrimination metric for scorecards).
- **ECE** — expected calibration error: are predicted PDs trustworthy? (a 0.10 prediction
  should default ~10% of the time). Credit decisions need calibrated probabilities.
- **Lift table** — by score decile, how many more defaulters than the base rate each band catches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ks_statistic(y_true, proba) -> float:
    """Kolmogorov–Smirnov: max gap between cumulative bad-rate and good-rate curves."""
    order = np.argsort(proba)
    y = np.asarray(y_true)[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return 0.0
    return float(np.max(np.abs(np.cumsum(y) / pos - np.cumsum(1 - y) / neg)))


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """ECE: weighted mean gap between predicted confidence and observed frequency per bin."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += abs(p[m].mean() - y[m].mean()) * m.sum() / len(p)
    return float(ece)


def lift_table(y_true, proba, n_deciles: int = 10) -> pd.DataFrame:
    """Per-score-decile default rate and lift vs the population base rate (decile 0 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = np.minimum((df.index * n_deciles // len(df)).astype(int), n_deciles - 1)
    base = df["y"].mean()
    g = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"), rate=("y", "mean"))
    g["lift"] = g["rate"] / base
    return g


def summarize(y_true, proba) -> dict:
    """Headline metric dict for a set of predictions."""
    auc = float(roc_auc_score(y_true, proba))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }
