"""Data validation tests."""

import pytest

from creditlens.data.validate import DataValidationError, validate_application


def test_valid_application_passes(app_df):
    warnings = validate_application(app_df)
    assert isinstance(warnings, list)


def test_negative_income_raises(app_df):
    bad = app_df.copy()
    bad.loc[0, "AMT_INCOME_TOTAL"] = -1
    with pytest.raises(DataValidationError):
        validate_application(bad)


def test_non_binary_target_raises(app_df):
    bad = app_df.copy()
    bad.loc[0, "TARGET"] = 2
    with pytest.raises(DataValidationError):
        validate_application(bad)


def test_duplicate_id_raises(app_df):
    bad = app_df.copy()
    bad.loc[1, "SK_ID_CURR"] = bad.loc[0, "SK_ID_CURR"]
    with pytest.raises(DataValidationError):
        validate_application(bad)


def test_missing_target_ok_when_not_required(app_df):
    no_target = app_df.drop(columns=["TARGET"])
    assert isinstance(validate_application(no_target, require_target=False), list)
