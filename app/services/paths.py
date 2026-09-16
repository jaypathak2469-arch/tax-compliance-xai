"""Path resolution for the application.

Every path is derived from this file's own location, so the project can live
anywhere (including a folder whose name contains a space). Nothing here is
hardcoded to an absolute machine path.

Layout assumed::

    <project root>/
        app/services/paths.py   <- this file
        code/
        data/
        models/
        results/
"""
from __future__ import annotations

import sys
from pathlib import Path

# app/services/paths.py -> app/services -> app -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Existing artifacts. These are read-only as far as the application is
# concerned: nothing in app/ ever writes to any of these paths.
# ---------------------------------------------------------------------------

CONFIG_YAML = CODE_DIR / "config.yaml"

# Phase 5 loads the preprocessor from data/processed. run_phase2.py writes it to
# models/. The shipped project only has the data/processed copy, so that is the
# primary and models/ is a fallback (see integration note B).
PREPROCESSOR_PRIMARY = PROCESSED_DIR / "preprocessor.joblib"
PREPROCESSOR_FALLBACK = MODELS_DIR / "preprocessor.joblib"

SCALER_STATS = PROCESSED_DIR / "scaler_stats.csv"
MODEL_CHECKPOINT = MODELS_DIR / "mlp_final_selected.pt"

TRAIN_RAW = PROCESSED_DIR / "train_raw.csv"
VAL_RAW = PROCESSED_DIR / "val_raw.csv"
TEST_RAW = PROCESSED_DIR / "test_raw.csv"

INTEGRATED_RAW = RAW_DIR / "integrated_tax_financial_profiles.csv"
TRANSACTIONS_RAW = RAW_DIR / "financial_transactions_synthetic.csv"
DATA_DICTIONARY = RAW_DIR / "data_dictionary.csv"

CF_RESULTS = METRICS_DIR / "counterfactual_results.csv"
CF_SUMMARY = METRICS_DIR / "counterfactual_summary.csv"
CF_REPRESENTATIVE = METRICS_DIR / "representative_counterfactuals.csv"
NEGATIVE_TESTS = METRICS_DIR / "negative_validation_tests.csv"
PHASE5_METRICS = METRICS_DIR / "phase5_metrics.json"

# Application-owned files.
REPORTED_METRICS = APP_DIR / "config" / "reported_metrics.json"
STYLESHEET = APP_DIR / "styles" / "main.css"


def resolve_preprocessor() -> Path | None:
    """Return whichever preprocessor artifact exists, preferring the Phase 5 one."""
    for candidate in (PREPROCESSOR_PRIMARY, PREPROCESSOR_FALLBACK):
        if candidate.exists():
            return candidate
    return None


def ensure_code_on_path() -> bool:
    """Put ``code/`` on sys.path so the existing flat imports resolve.

    ``code/counterfactual.py`` does ``from constraints import ...`` rather than a
    package-relative import, so the directory itself must be importable. Called
    once, lazily, by the modules that import the Phase 5 code.

    Returns True if the directory exists and is now importable.
    """
    if not CODE_DIR.is_dir():
        return False
    entry = str(CODE_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return True
