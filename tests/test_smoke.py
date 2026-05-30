"""Phase 0 smoke tests: package imports and the synthetic fixture builds."""

import creditlens
from scripts.make_fixtures import make_application_sample


def test_package_imports():
    assert creditlens.__version__


def test_fixture_has_target_and_id():
    df = make_application_sample(n=50)
    assert len(df) == 50
    assert {"SK_ID_CURR", "TARGET"}.issubset(df.columns)
    assert df["TARGET"].isin([0, 1]).all()
