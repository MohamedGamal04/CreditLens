"""Metric correctness on small, known inputs."""

import numpy as np

from creditlens.evaluation.metrics import (
    expected_calibration_error,
    ks_statistic,
    lift_table,
    summarize,
)


def test_ks_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]  # perfectly ordered
    assert ks_statistic(y, proba) == 1.0


def test_ks_single_class_is_zero():
    assert ks_statistic([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0


def test_ece_perfectly_calibrated_is_low():
    # 100 samples at p=0.0 (none default) + 100 at p=1.0 (all default)
    y = [0] * 100 + [1] * 100
    p = [0.0] * 100 + [1.0] * 100
    assert expected_calibration_error(y, p) < 1e-9


def test_lift_table_top_decile_highest():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)  # higher score -> more likely default
    lt = lift_table(y, p, n_deciles=10)
    assert len(lt) == 10
    assert lt["lift"].iloc[0] > lt["lift"].iloc[-1]   # riskiest decile lifts most


def test_summarize_keys_and_bounds():
    y = [0, 1, 0, 1]
    p = [0.2, 0.8, 0.3, 0.6]
    s = summarize(y, p)
    assert set(s) == {"auc", "gini", "ks", "ece"}
    assert 0.0 <= s["auc"] <= 1.0
