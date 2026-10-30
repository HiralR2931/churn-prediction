"""Tests for the Medicare pipeline on tiny made-up samples (no CMS download needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.churn_label import build_modeling_table, train_test_forecast_split
from src.data_cleaning import clean
from src.feature_engineering import add_trend_features
from src.ma_switching import build_switch_table

MONTHS = list(C.MONTH_ORDER)


def _raw_row(year, month, fips, tot, om, ma, geo="County", aged="10"):
    return {"YEAR": year, "MONTH": month, "BENE_GEO_LVL": geo, "BENE_STATE_ABRVTN": "VT",
            "BENE_STATE_DESC": "Vermont", "BENE_COUNTY_DESC": "Test", "BENE_FIPS_CD": fips,
            "TOT_BENES": tot, "ORGNL_MDCR_BENES": om, "MA_AND_OTH_BENES": ma, "AGED_TOT_BENES": aged}


# --------------------------------------------------------------------------- #
# Cleaning: suppressed cells, row filters, audit flag
# --------------------------------------------------------------------------- #
@pytest.fixture
def cleaned():
    raw = pd.DataFrame([
        _raw_row(2024, "January", "50001", "100", "60", "40", aged="*"),   # one suppressed cell
        _raw_row(2024, "February", "50001", "100", "60", "40"),            # nothing suppressed
        _raw_row(2024, "Year", "50001", "100", "60", "40"),                # annual roll-up
        _raw_row(2024, "January", "50999", "30", "20", "10"),              # statewide "Unknown"
        _raw_row(2024, "January", "     ", "900", "500", "400", geo="State"),
        _raw_row(2024, "January", "50003", "*", "*", "*"),                 # tiny county: 1-10 people
        _raw_row(2024, "January", "5X00A", "100", "60", "40"),             # malformed FIPS
    ])
    return clean(raw)


def test_suppressed_cells_are_imputed_at_midpoint(cleaned):
    df, report = cleaned
    jan = df[(df["fips"] == "50001") & (df["month_num"] == 1)].iloc[0]
    assert jan["AGED_TOT_BENES"] == C.SUPPRESSION_MIDPOINT
    assert report["suppressed_cells_imputed"] == 1 + 3
    tiny = df[df["fips"] == "50003"].iloc[0]
    assert tiny["TOT_BENES"] == C.SUPPRESSION_MIDPOINT    # kept, not dropped


def test_suppression_flag_is_per_row(cleaned):
    df, _ = cleaned
    flags = df[df["fips"] == "50001"].set_index("month_num")["had_suppression"]
    assert bool(flags[1]) is True
    assert bool(flags[2]) is False


def test_only_real_county_months_survive(cleaned):
    df, report = cleaned
    assert sorted(df["fips"].unique()) == ["50001", "50003"]   # no 50999, state row or 5X00A
    assert report["rows_dropped_unknown_county"] == 1
    assert report["rows_dropped_invalid"] == 1
    assert set(df["month_num"]) == {1, 2}                       # annual "Year" row gone


# --------------------------------------------------------------------------- #
# Feature engineering: last finished Dec -> Feb window
# --------------------------------------------------------------------------- #
def _monthly(fips, start_year, totals):
    rows = []
    for i, tot in enumerate(totals):
        year, m = start_year + i // 12, i % 12 + 1
        rows.append({"fips": fips, "year": year, "month_num": m,
                     "date": pd.Timestamp(year=year, month=m, day=1), "TOT_BENES": float(tot)})
    return pd.DataFrame(rows)


def test_prior_window_change_is_last_dec_to_feb():
    # Dec 2022 = 1000, Feb 2023 = 1100; every other month 1000.
    totals = [1000.0] * 24
    totals[13] = 1100.0                       # Feb 2023
    out = add_trend_features(_monthly("50001", 2022, totals)).set_index("date")["prior_window_change"]
    assert out[pd.Timestamp("2023-12-01")] == pytest.approx(0.10)   # sees Dec22 -> Feb23
    assert out[pd.Timestamp("2023-02-01")] == pytest.approx(0.10)   # its own window
    assert out[pd.Timestamp("2023-01-01")] == 0.0                   # Dec21 -> Feb22 unknown


# --------------------------------------------------------------------------- #
# Switching measure and lag alignment
# --------------------------------------------------------------------------- #
def _feat(fips, state, year, month, tot, ma_share):
    row = {"fips": fips, "state": state, "state_name": state, "county": fips, "year": year,
           "month_num": month, "TOT_BENES": tot, "ORGNL_MDCR_BENES": tot * (1 - ma_share),
           "MA_AND_OTH_BENES": tot * ma_share, "ma_ratio": ma_share}
    row.update({c: 0.1 for c in C.MA_DEMOGRAPHICS})
    return row


def test_switching_is_feb_minus_dec_share_in_points():
    feat = pd.DataFrame([_feat("1", "AA", 2023, 12, 5000, 0.40), _feat("1", "AA", 2024, 2, 5000, 0.43)])
    t = build_switch_table(feat)
    assert t.loc[t["base_year"] == 2023, "shift_pp"].iloc[0] == pytest.approx(3.0)


def test_lags_join_on_year_not_row_position():
    rows = []
    for y, share in [(2018, 0.20), (2020, 0.30), (2021, 0.32)]:           # 2019 is missing
        rows += [_feat("1", "AA", y, 12, 5000, share), _feat("1", "AA", y + 1, 2, 5000, share + 0.01)]
    t = build_switch_table(pd.DataFrame(rows)).set_index("base_year")
    assert np.isnan(t.loc[2020, "prev_rel_shift"])                        # not 2018's value
    assert not np.isnan(t.loc[2021, "prev_rel_shift"])


def test_small_counties_are_not_eligible():
    feat = pd.DataFrame([_feat("1", "AA", 2023, 12, C.MA_MIN_BENES - 1, 0.4),
                         _feat("1", "AA", 2024, 2, C.MA_MIN_BENES - 1, 0.4)])
    assert not build_switch_table(feat)["eligible"].any()


# --------------------------------------------------------------------------- #
# Net-decline label and time-based split
# --------------------------------------------------------------------------- #
def test_decline_label_and_temporal_split():
    rows = []
    years = [C.TEST_YEAR - 1, C.TEST_YEAR, C.PREDICT_FROM_YEAR]
    for y in years:
        rows.append({"fips": "1", "state": "AA", "state_name": "A", "county": "c", "year": y,
                     "month_num": 12, "date": pd.Timestamp(year=y, month=12, day=1), "TOT_BENES": 1000.0})
        if y != C.PREDICT_FROM_YEAR:   # no February yet for the forecast year
            feb = 990.0 if y == C.TEST_YEAR else 999.0        # -1.0% vs -0.1%
            rows.append({"fips": "1", "state": "AA", "state_name": "A", "county": "c", "year": y + 1,
                         "month_num": 2, "date": pd.Timestamp(year=y + 1, month=2, day=1), "TOT_BENES": feb})
    feat = pd.DataFrame(rows)
    for col in C.FEATURE_COLUMNS:
        feat[col] = 0.0
    tbl = build_modeling_table(feat).set_index("base_year")
    assert tbl.loc[C.TEST_YEAR, "churn"] == 1.0             # fell more than 0.5%
    assert tbl.loc[C.TEST_YEAR - 1, "churn"] == 0.0         # fell less than 0.5%
    assert np.isnan(tbl.loc[C.PREDICT_FROM_YEAR, "churn"])  # outcome not known yet

    train, test, fc = train_test_forecast_split(tbl.reset_index())
    assert (train["base_year"] < C.TEST_YEAR).all()
    assert (test["base_year"] == C.TEST_YEAR).all()
    assert (fc["base_year"] == C.PREDICT_FROM_YEAR).all()
