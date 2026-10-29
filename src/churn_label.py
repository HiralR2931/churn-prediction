"""Build the county-level churn label and the modelling table.

Churn window (matches the project title, "Dec 2025 - Feb 2026"):
    base month   = December of year Y      -> feature snapshot
    target month = February of year Y + 1  -> outcome measured here

A county is labelled ``churn = 1`` when total beneficiaries decline by more than
``CHURN_THRESHOLD`` across that forward window.

The resulting table has one row per (county, base-year) with:
    * the engineered features observed at December Y, and
    * the churn label observed at February Y+1  (NaN for the forecast year).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def build_modeling_table(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the (county, base-year) modelling table with the churn label."""
    base = C.MONTH_ORDER[C.BASE_MONTH]
    target = C.MONTH_ORDER[C.TARGET_MONTH]

    # December snapshots carry the features.
    dec = feat_df[feat_df["month_num"] == base].copy()
    dec["base_year"] = dec["year"]

    # February totals (the outcome) belong to the *following* calendar year.
    feb = feat_df[feat_df["month_num"] == target][["fips", "year", "TOT_BENES"]].copy()
    feb = feb.rename(columns={"TOT_BENES": "target_benes"})
    feb["base_year"] = feb["year"] - 1
    feb = feb.drop(columns="year")

    tbl = dec.merge(feb, on=["fips", "base_year"], how="left")

    # Forward change and label.
    tbl["forward_change"] = (tbl["target_benes"] - tbl["TOT_BENES"]) / tbl["TOT_BENES"]
    tbl["churn"] = np.where(
        tbl["target_benes"].isna(),
        np.nan,
        (tbl["forward_change"] < -C.CHURN_THRESHOLD).astype(float),
    )

    keep = (["fips", "state", "state_name", "county", "base_year", "date",
             "TOT_BENES", "target_benes", "forward_change", "churn"]
            + C.FEATURE_COLUMNS)
    return tbl[keep].reset_index(drop=True)


def train_test_forecast_split(tbl: pd.DataFrame):
    """Temporal split.

    * train    : base years with a known outcome, excluding the held-out test year
    * test     : the most recent year with ground truth (``TEST_YEAR``)
    * forecast : the latest snapshot with no outcome yet (``PREDICT_FROM_YEAR``)
    """
    labelled = tbl.dropna(subset=["churn"])
    train = labelled[labelled["base_year"] < C.TEST_YEAR].copy()
    test = labelled[labelled["base_year"] == C.TEST_YEAR].copy()
    forecast = tbl[tbl["base_year"] == C.PREDICT_FROM_YEAR].copy()
    return train, test, forecast


if __name__ == "__main__":
    from .data_cleaning import load_clean
    from .feature_engineering import engineer

    clean_df, _ = load_clean()
    tbl = build_modeling_table(engineer(clean_df))
    tr, te, fc = train_test_forecast_split(tbl)
    print(f"train={len(tr)}  test={len(te)}  forecast={len(fc)}")
    print("train churn rate:", round(tr["churn"].mean(), 3))
    print("test  churn rate:", round(te["churn"].mean(), 3))
