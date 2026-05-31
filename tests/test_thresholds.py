"""Cost-based threshold math on small, known inputs."""

import numpy as np

from creditlens.evaluation.thresholds import (
    decision_bands,
    optimal_threshold,
    strategy_table,
)


def test_optimal_threshold_separable_has_zero_cost():
    # perfectly ordered: a cut between the classes misclassifies nobody
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    t = optimal_threshold(y, p, cost_ratio=5.0)
    assert 0.3 < t <= 0.7


def test_higher_cost_ratio_lowers_threshold():
    # when missing a default is costlier, the model should decline more (lower cut)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)  # higher score -> more defaults
    strict = optimal_threshold(y, p, cost_ratio=10.0)
    loose = optimal_threshold(y, p, cost_ratio=1.0)
    assert strict <= loose


def test_decision_bands_ordered_and_in_range():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    low, high = decision_bands(y, p, cost_ratio=5.0)
    assert 0.0 <= low < high <= 1.0


def test_strategy_table_shape_and_monotone_approval():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < p).astype(int)
    tbl = strategy_table(y, p, n_bins=10)
    assert len(tbl) == 10
    assert {"threshold", "approval_rate", "bad_rate"} <= set(tbl.columns)
    # raising the cut approves more people
    assert tbl["approval_rate"].iloc[-1] >= tbl["approval_rate"].iloc[0]
