"""Feature engineering for the churn model.

Turns the clean monthly panel into county composition ratios and short-term
enrollment dynamics. The three headline ratios from the project brief —
``aged_ratio``, ``disabled_ratio`` and ``dual_ratio`` — are produced here along
with a broader demographic/coverage mix and trend momentum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise ratio guarded against divide-by-zero."""
    return (num / den.replace(0, np.nan)).fillna(0.0)


def add_composition_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add demographic / coverage composition ratios (share of TOT_BENES)."""
    df = df.copy()
    tot = df["TOT_BENES"]

    # --- Headline ratios from the project brief ---------------------------- #
    df["aged_ratio"] = _safe_ratio(df["AGED_TOT_BENES"], tot)
    df["disabled_ratio"] = _safe_ratio(df["DSBLD_TOT_BENES"], tot)
    df["dual_ratio"] = _safe_ratio(df["DUAL_TOT_BENES"], tot)

    # --- Supporting coverage / eligibility ratios -------------------------- #
    df["full_dual_ratio"] = _safe_ratio(df["FULL_DUAL_TOT_BENES"], tot)
    df["ma_ratio"] = _safe_ratio(df["MA_AND_OTH_BENES"], tot)
    esrd = df["AGED_ESRD_BENES"].fillna(0) + df["DSBLD_ESRD_AND_ESRD_ONLY_BENES"].fillna(0)
    df["esrd_ratio"] = _safe_ratio(esrd, tot)
    df["rx_ratio"] = _safe_ratio(df["PRSCRPTN_DRUG_TOT_BENES"], tot)
    lis = (df["PRSCRPTN_DRUG_FULL_LIS_BENES"].fillna(0)
           + df["PRSCRPTN_DRUG_PARTIAL_LIS_BENES"].fillna(0))
    df["lis_ratio"] = _safe_ratio(lis, tot)

    # --- Demographic mix --------------------------------------------------- #
    df["female_ratio"] = _safe_ratio(df["FEMALE_TOT_BENES"], tot)
    df["white_ratio"] = _safe_ratio(df["WHITE_TOT_BENES"], tot)
    df["black_ratio"] = _safe_ratio(df["BLACK_TOT_BENES"], tot)
    df["hispanic_ratio"] = _safe_ratio(df["HSPNC_TOT_BENES"], tot)
    df["api_ratio"] = _safe_ratio(df["API_TOT_BENES"], tot)
    age_85p = (df["AGE_85_TO_89_BENES"].fillna(0) + df["AGE_90_TO_94_BENES"].fillna(0)
               + df["AGE_GT_94_BENES"].fillna(0))
    df["age_85_plus_ratio"] = _safe_ratio(age_85p, tot)

    df["log_tot_benes"] = np.log1p(tot)
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-term enrollment dynamics per county.

    * ``momentum_3m`` – pct change in total benes over the prior 3 months.
    * ``prior_window_change`` – the same Dec->Feb pct change one year earlier
      (the persistence of churn is a strong signal).
    """
    df = df.sort_values(["fips", "date"]).copy()
    g = df.groupby("fips", sort=False)["TOT_BENES"]
    df["momentum_3m"] = g.pct_change(periods=3).fillna(0.0)
    df["yoy_change"] = g.pct_change(periods=12).fillna(0.0)

    # Dec (Y-1) -> Feb (Y) change, keyed to the calendar year it ends in. A row
    # sees the latest window that has finished: Feb Y onwards sees year Y's,
    # January still sees the year before.
    dec = df.loc[df["month_num"] == 12, ["fips", "year", "TOT_BENES"]]
    feb = df.loc[df["month_num"] == 2, ["fips", "year", "TOT_BENES"]]
    win = feb.merge(dec.assign(year=dec["year"] + 1), on=["fips", "year"],
                    suffixes=("_feb", "_dec"))
    win["window_change"] = win["TOT_BENES_feb"] / win["TOT_BENES_dec"] - 1
    df["_window_year"] = df["year"] - (df["month_num"] < 2).astype(int)
    df = df.merge(win[["fips", "year", "window_change"]].rename(columns={"year": "_window_year"}),
                  on=["fips", "_window_year"], how="left")
    df["prior_window_change"] = df.pop("window_change").fillna(0.0)
    return df.drop(columns="_window_year")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: composition ratios + trend dynamics."""
    df = add_composition_ratios(df)
    df = add_trend_features(df)
    # Clip pathological ratios from suppression imputation into [0, 1].
    for col in C.RATIO_FEATURES:
        df[col] = df[col].clip(lower=0.0, upper=1.0)
    return df


if __name__ == "__main__":
    from .data_cleaning import load_clean
    clean_df, _ = load_clean()
    feats = engineer(clean_df)
    print(feats[["fips", "date"] + C.FEATURE_COLUMNS].describe().T)
