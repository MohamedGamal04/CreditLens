"""Central configuration: paths, seeds, CV and target settings.

Single source of truth so notebooks and the package agree on locations and
hyperparameters. Import from here rather than hard-coding paths.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"

# Raw Kaggle files we use (data scope: application + bureau + previous_application)
APPLICATION_TRAIN = RAW_DIR / "application_train.csv"
BUREAU = RAW_DIR / "bureau.csv"
PREVIOUS_APPLICATION = RAW_DIR / "previous_application.csv"

# --- Modeling ----------------------------------------------------------------
TARGET = "TARGET"          # 1 = default / payment difficulty, 0 = repaid
ID_COL = "SK_ID_CURR"
RANDOM_SEED = 42
N_FOLDS = 5

# Kaggle competition slug for the download script
KAGGLE_COMPETITION = "home-credit-default-risk"
