"""PSI drift: ~0 for identical data, large under a distribution shift."""

import numpy as np
import pandas as pd

from creditlens.evaluation.drift import (
    build_reference,
    drift_report,
    population_stability_index,
)


def test_psi_zero_for_identical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.normal(size=5000)})
    ref = build_reference(df, ["f"])
    assert population_stability_index(ref["f"], df["f"]) < 0.01


def test_psi_high_for_shift():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"f": rng.normal(0, 1, 5000)}), ["f"])
    shifted = pd.Series(rng.normal(3, 1, 5000))
    assert population_stability_index(ref["f"], shifted) > 0.25


def test_psi_handles_nan_bucket():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"f": rng.normal(0, 1, 5000)}), ["f"])
    # a batch that is 50% NaN vs ~0% NaN reference -> drift
    half_nan = pd.Series(list(rng.normal(0, 1, 2500)) + [np.nan] * 2500)
    assert population_stability_index(ref["f"], half_nan) > 0.2


def test_drift_report_flags_only_shifted():
    rng = np.random.default_rng(1)
    ref_df = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(0, 1, 4000)})
    new = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(4, 1, 4000)})
    ref = build_reference(ref_df, ["a", "b"])
    rep = drift_report(ref, new, ["a", "b"])
    assert "b" in rep["flagged"] and "a" not in rep["flagged"]
    assert rep["max_psi"] == rep["per_feature"]["b"]
