"""Central configuration for the Netflix churn project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
SITE_DIR = PROJECT_ROOT / "site"
RAW_CSV = DATA_DIR / "netflix_customer_churn.csv"

for _d in (REPORT_DIR, DOCS_DIR, SITE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGET = "churned"
CATEGORICAL = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
NUMERIC = ["age", "watch_hours", "last_login_days", "monthly_fee", "number_of_profiles",
           "avg_watch_time_per_day"]
FEATURES = CATEGORICAL + NUMERIC

# --------------------------------------------------------------------------- #
# The six conditions of the churn scorecard. Cut-points are where churn jumps
# in the data (see analysis.cliff_curves): watch hours at 5 and 20, days since
# last login past 30. Each entry is (name, label, test on the raw frame).
# --------------------------------------------------------------------------- #
CONDITIONS = [
    ("watch_lt5", "Under 5 watch hours", lambda d: d["watch_hours"] < 5),
    ("watch_5_20", "5 to under 20 watch hours", lambda d: (d["watch_hours"] >= 5) & (d["watch_hours"] < 20)),
    ("login_gt30", "Last login over 30 days ago", lambda d: d["last_login_days"] > 30),
    ("basic_plan", "Basic plan", lambda d: d["subscription_type"] == "Basic"),
    ("profiles_lt4", "Fewer than 4 profiles", lambda d: d["number_of_profiles"] < 4),
    ("pay_crypto_gift", "Pays by crypto or gift card",
     lambda d: d["payment_method"].isin(["Crypto", "Gift Card"])),
]
# Integer points searched per condition when fitting the scorecard.
POINT_GRID = {
    "watch_lt5": range(4, 11),
    "watch_5_20": range(1, 6),
    "login_gt30": range(2, 7),
    "basic_plan": range(1, 4),
    "profiles_lt4": range(1, 3),
    "pay_crypto_gift": range(1, 3),
}

CV_SPLITS = 5
CV_REPEATS = 3
RANDOM_STATE = 42

# Test accuracy reported by the original notebook (80/20 split, label-encoded,
# unscaled, with the avg_watch_time_per_day "/10" change). Kept for comparison.
NOTEBOOK_ACCURACY = {
    "Gradient boosting": 0.991, "Random forest": 0.970, "Decision tree": 0.969,
    "Logistic regression": 0.884, "KNN (k=5)": 0.878, "Naive Bayes": 0.831,
}
