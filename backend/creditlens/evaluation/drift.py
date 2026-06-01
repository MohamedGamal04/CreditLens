"""Population Stability Index (PSI) drift monitoring.

A reference distribution is captured at train time (quantile bin edges + per-bin
proportions + a NaN bucket, all JSON-serialisable). PSI compares a new batch to it:
    PSI = sum_bins (new_pct - ref_pct) * ln(new_pct / ref_pct)
Convention: <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant (flag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def _edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Finite, deduped quantile cut points (>=2 needed for any bin)."""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(values, q))


def build_reference(df: pd.DataFrame, features: list[str], n_bins: int = 10) -> dict:
    """Per-feature reference: bin edges, per-bin proportions, NaN fraction."""
    ref: dict = {}
    for f in features:
        s = df[f]
        n = len(s)
        nan_pct = float(s.isna().sum() / n) if n else 0.0
        x = s.dropna().to_numpy(dtype=float)
        edges = _edges(x, n_bins) if x.size else np.array([])
        if edges.size < 2:
            ref[f] = {"edges": [], "ref_pct": [], "nan_pct": nan_pct}
            continue
        counts, _ = np.histogram(x, bins=edges)
        ref[f] = {
            "edges": edges.tolist(),
            "ref_pct": (counts / n).tolist(),  # fraction of ALL rows (sums to 1-nan_pct)
            "nan_pct": nan_pct,
        }
    return ref


def population_stability_index(ref_entry: dict, values) -> float:
    """PSI of ``values`` against one feature's reference entry."""
    edges = np.asarray(ref_entry["edges"], dtype=float)
    if edges.size < 2:
        return 0.0
    s = pd.Series(values)
    n = len(s)
    if n == 0:
        return 0.0
    nan_new = float(s.isna().sum() / n)
    x = s.dropna().to_numpy(dtype=float)
    x = np.clip(x, edges[0], edges[-1])  # out-of-range -> end bins
    new_counts, _ = np.histogram(x, bins=edges)

    ref_all = np.append(np.asarray(ref_entry["ref_pct"], dtype=float), ref_entry["nan_pct"])
    new_all = np.append(new_counts / n, nan_new)
    ref_all = np.clip(ref_all, _EPS, None)
    new_all = np.clip(new_all, _EPS, None)
    return float(np.sum((new_all - ref_all) * np.log(new_all / ref_all)))


def drift_report(reference: dict, df: pd.DataFrame, features: list[str],
                 threshold: float = 0.2) -> dict:
    """Per-feature PSI + the list of features above ``threshold``."""
    per = {
        f: round(population_stability_index(reference[f], df[f]), 4)
        for f in features if f in reference
    }
    return {
        "per_feature": per,
        "flagged": [f for f, v in per.items() if v > threshold],
        "max_psi": max(per.values()) if per else 0.0,
    }
