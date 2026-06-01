"""Shared test fixtures. Adds the repo root to sys.path so ``scripts`` (one level up
from backend/) is importable for the synthetic-fixture builders."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from scripts.make_fixtures import (  # noqa: E402
    make_application_sample,
    make_bureau_sample,
    make_previous_sample,
)


@pytest.fixture
def app_df():
    return make_application_sample(n=200, seed=42)


@pytest.fixture
def bureau_df(app_df):
    return make_bureau_sample(app_df["SK_ID_CURR"], seed=1)


@pytest.fixture
def prev_df(app_df):
    return make_previous_sample(app_df["SK_ID_CURR"], seed=2)


@pytest.fixture
def matrix(app_df, bureau_df, prev_df):
    """The 15-feature model matrix built from the synthetic samples."""
    from creditlens.data.features import build_model_matrix

    return build_model_matrix(app_df, bureau_df, prev_df)
