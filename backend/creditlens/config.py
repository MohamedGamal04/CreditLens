"""Central configuration: paths, seeds, CV and target settings.

Single source of truth so notebooks and the package agree on locations and
hyperparameters. Import from here rather than hard-coding paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# backend/creditlens/config.py -> parents[1] = backend/, its parent = repo root.
# data/, models/, frontend/ live at the repo root locally; in the backend-only
# Docker image they are relocated, so each can be overridden with an env var.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]   # backend/
REPO_ROOT = PACKAGE_ROOT.parent                       # repo root

DATA_DIR = Path(os.environ.get("CL_DATA_DIR") or REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("CL_MODELS_DIR") or REPO_ROOT / "models")
APP_DIR = Path(os.environ.get("CL_APP_DIR") or REPO_ROOT / "frontend")

# Raw Kaggle files we use (data scope: application + bureau + previous_application)
APPLICATION_TRAIN = RAW_DIR / "application_train.csv"
BUREAU = RAW_DIR / "bureau.csv"
PREVIOUS_APPLICATION = RAW_DIR / "previous_application.csv"

# --- Modeling ----------------------------------------------------------------
TARGET = "TARGET"          # 1 = default / payment difficulty, 0 = repaid
ID_COL = "SK_ID_CURR"
RANDOM_SEED = 42
N_FOLDS = 5

# Decision-threshold economics: cost of a false negative (approve a defaulter)
# relative to a false positive (decline a good applicant). Drives the band cuts.
COST_RATIO = 5.0

# Kaggle competition slug for the download script
KAGGLE_COMPETITION = "home-credit-default-risk"
