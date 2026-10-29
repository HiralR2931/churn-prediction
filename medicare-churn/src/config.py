"""Central configuration for the Medicare County-Level Churn Prediction project.

All tunable parameters (paths, churn definition, feature lists, model
hyper-parameters) live here so the rest of the pipeline stays declarative.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RAW_CSV = DATA_DIR / "medicare_enrollment_2013_2025.csv"

for _d in (OUTPUT_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data conventions
# --------------------------------------------------------------------------- #
# CMS suppresses beneficiary counts between 1 and 10 with an asterisk to protect
# privacy. We replace those cells with the mid-point of the suppressed range so
# ratios remain numerically stable, and keep a boolean flag marking imputation.
SUPPRESSION_TOKEN = "*"
SUPPRESSION_MIDPOINT = 5.0

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
# "Year" rows are annual roll-ups we exclude from the monthly panel.
ANNUAL_MONTH_TOKEN = "Year"

# --------------------------------------------------------------------------- #
# Churn definition  (matches the "Dec 2025 - Feb 2026" project window)
# --------------------------------------------------------------------------- #
# A county "churns" when its total beneficiary count over a forward 3-month
# window (December -> February) declines by more than CHURN_THRESHOLD.
BASE_MONTH = "December"          # snapshot month features are computed from
TARGET_MONTH = "February"       # month the outcome is measured in (year + 1)
CHURN_THRESHOLD = 0.005         # >0.5% net enrollment loss counts as churn

# Latest snapshot we predict forward from (no ground truth yet -> the ask).
PREDICT_FROM_YEAR = 2025        # Dec 2025 -> Feb 2026
# Most recent snapshot with a known outcome, held out for honest evaluation.
TEST_YEAR = 2024                # Dec 2024 -> Feb 2025

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
# Engineered composition ratios (the headline features from the project brief).
RATIO_FEATURES = [
    "aged_ratio",
    "disabled_ratio",
    "dual_ratio",
    "full_dual_ratio",
    "ma_ratio",
    "esrd_ratio",
    "rx_ratio",
    "lis_ratio",
    "female_ratio",
    "white_ratio",
    "black_ratio",
    "hispanic_ratio",
    "api_ratio",
    "age_85_plus_ratio",
]
# Dynamics: how the county was already trending going into the snapshot.
TREND_FEATURES = [
    "momentum_3m",        # Sep -> Dec pct change in total benes
    "prior_window_change",  # last year's Dec -> Feb pct change
    "log_tot_benes",      # county size (scale matters for stability)
]
FEATURE_COLUMNS = RATIO_FEATURES + TREND_FEATURES

# --------------------------------------------------------------------------- #
# Random Forest hyper-parameters
# --------------------------------------------------------------------------- #
RF_PARAMS = dict(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=25,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Medicare Advantage switching (the main analysis)
# --------------------------------------------------------------------------- #
# Plan choices made in the Oct-Dec enrollment period take effect on January 1,
# so the Dec -> Feb window is where Original Medicare -> MA switching shows up.
# The outcome is the change in a county's MA share across that window, in
# percentage points.
SITE_DIR = PROJECT_ROOT / "site"
DOCS_DIR = PROJECT_ROOT / "docs"

# Counties below this size swing by several points on a handful of people, so
# they are left out of the model and the county statistics (kept on the map).
MA_MIN_BENES = 1000
MA_FIRST_YEAR = 2015            # needs two prior windows for the lag features
MA_BACKTEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
MA_FORECAST_YEAR = 2025         # Dec 2025 -> Feb 2026
# Weight on the county's own last window vs its state's last window.
MA_BLEND_WEIGHT = 0.5
MA_DEMOGRAPHICS = [
    "dual_ratio", "lis_ratio", "disabled_ratio", "age_85_plus_ratio",
    "black_ratio", "hispanic_ratio", "log_tot_benes",
]
MA_GBM_FEATURES = MA_DEMOGRAPHICS + [
    "ma_ratio", "prev_rel_shift", "prev2_rel_shift", "ma_trend_12m", "state_prev_rel_shift",
]
GBM_PARAMS = dict(
    max_iter=400,
    learning_rate=0.05,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)
