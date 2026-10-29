"""Original Medicare -> Medicare Advantage switching, county by county.

The CMS file has no beneficiary-level transitions, but plan choices made during
the Oct-Dec enrollment period take effect on January 1. A county's change in MA
share between December of year Y and February of Y+1 therefore measures how
much of its Original Medicare population moved to MA that season.

    shift_pp      = MA share (Feb Y+1) - MA share (Dec Y), in percentage points
    rel_shift_pp  = shift_pp - the median county's shift_pp for that window

The overall level swings with the insurance market (2020 peak, 2024 stall) and
cannot be read from county data, so forecasting works on ``rel_shift_pp``: which
counties switch more or less than the median county.

Outputs (``outputs/reports/``):
    ma_national.csv        MA share (CMS national rows) and window shift per year
    ma_saturation.csv      shift by current MA share (the adoption curve)
    ma_states.csv          state median shift, latest two windows
    ma_correlates.csv      who switches: Spearman rho with county mix, per season
    ma_backtest.csv        forecasting methods compared, year by year
    ma_forecast_dec2025_feb2026.csv
    ma_summary.json        headline numbers used by the README and site
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C

_DEC, _FEB = C.MONTH_ORDER["December"], C.MONTH_ORDER["February"]


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def build_switch_table(feat_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (county, base year): Dec snapshot, Feb outcome, lag features."""
    snap_cols = (["fips", "state", "state_name", "county", "year", "TOT_BENES",
                  "ORGNL_MDCR_BENES", "MA_AND_OTH_BENES", "ma_ratio"]
                 + C.MA_DEMOGRAPHICS)
    dec = feat_df.loc[feat_df["month_num"] == _DEC, snap_cols].rename(
        columns={"year": "base_year"})
    feb = feat_df.loc[feat_df["month_num"] == _FEB, ["fips", "year", "ma_ratio"]].rename(
        columns={"ma_ratio": "ma_ratio_feb"})
    feb["base_year"] = feb.pop("year") - 1

    t = dec.merge(feb, on=["fips", "base_year"], how="left")
    t = t.sort_values(["fips", "base_year"]).reset_index(drop=True)
    t["shift_pp"] = (t["ma_ratio_feb"] - t["ma_ratio"]) * 100
    t["eligible"] = t["TOT_BENES"] >= C.MA_MIN_BENES

    nat_median = t[t["eligible"]].groupby("base_year")["shift_pp"].median()
    t["rel_shift_pp"] = t["shift_pp"] - t["base_year"].map(nat_median)

    # Lags join on the year, not the row before, so a county with a missing
    # season gets NaN rather than an older season.
    def lag(col: str, years: int) -> pd.Series:
        prev = t[["fips", "base_year", col]].assign(base_year=t["base_year"] + years)
        return t[["fips", "base_year"]].merge(prev, on=["fips", "base_year"], how="left")[col]

    t["prev_rel_shift"] = lag("rel_shift_pp", 1)
    t["prev2_rel_shift"] = lag("rel_shift_pp", 2)
    t["ma_trend_12m"] = (t["ma_ratio"] - lag("ma_ratio", 1)) * 100
    # The state's last window: insurer plan offerings are set market by market.
    t["state_prev_rel_shift"] = (t[t["eligible"]]
                                 .groupby(["state", "base_year"])["prev_rel_shift"]
                                 .transform("median"))
    return t


# --------------------------------------------------------------------------- #
# Descriptive findings
# --------------------------------------------------------------------------- #
def national_series(national_df: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """National MA share each December plus the size of each switching window.

    Shares and totals come from CMS's national rows; the county percentiles
    come from the county table.
    """
    nat = national_df.set_index(["year", "month_num"])
    dec = nat.xs(_DEC, level="month_num")
    feb = nat.xs(_FEB, level="month_num")
    out = pd.DataFrame({"year": dec.index})
    out["ma_share_dec"] = (dec["MA_AND_OTH_BENES"] / dec["TOT_BENES"] * 100).to_numpy()
    out["ma_millions_dec"] = (dec["MA_AND_OTH_BENES"] / 1e6).to_numpy()
    out["om_millions_dec"] = (dec["ORGNL_MDCR_BENES"] / 1e6).to_numpy()

    feb_next = feb.reindex(out["year"] + 1)
    out["national_shift_pp"] = (
        feb_next["MA_AND_OTH_BENES"].to_numpy() / feb_next["TOT_BENES"].to_numpy() * 100
        - out["ma_share_dec"])
    elig = t[t["eligible"]].dropna(subset=["shift_pp"])
    stats = elig.groupby("base_year")["shift_pp"].agg(
        county_median_shift_pp="median",
        county_p10_shift_pp=lambda s: s.quantile(0.10),
        county_p90_shift_pp=lambda s: s.quantile(0.90),
        share_counties_ma_fell=lambda s: (s < 0).mean(),
    )
    return out.merge(stats, left_on="year", right_index=True, how="left")


def saturation_curve(t: pd.DataFrame) -> pd.DataFrame:
    """Relative switching by the county's MA share going into the window."""
    d = t[t["eligible"] & (t["base_year"] >= C.MA_FIRST_YEAR)].dropna(subset=["rel_shift_pp"])
    edges = np.arange(0, 0.80, 0.10)
    d = d.assign(band=pd.cut(d["ma_ratio"], np.append(edges, 1.0), right=False))
    out = d.groupby("band", observed=True)["rel_shift_pp"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        n="size").reset_index()
    out["ma_share_from"] = [b.left * 100 for b in out["band"]]
    out["ma_share_to"] = [b.right * 100 for b in out["band"]]
    return out.drop(columns="band")


def state_table(t: pd.DataFrame, year: int) -> pd.DataFrame:
    """Median county shift per state for ``year`` and the window before it."""
    d = t[t["eligible"] & t["base_year"].isin([year - 1, year])]
    p = (d.pivot_table(index=["state", "state_name"], columns="base_year",
                       values="shift_pp", aggfunc="median")
         .rename(columns={year - 1: "shift_prev_pp", year: "shift_pp"}))
    p["counties"] = d[d["base_year"] == year].groupby(["state", "state_name"]).size()
    p["change_pp"] = p["shift_pp"] - p["shift_prev_pp"]
    return p.reset_index().sort_values("shift_pp").reset_index(drop=True)


def correlates(t: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho between county mix and relative switching, season by season.

    Each county appears once per season, so rho is computed within each season
    and summarised across seasons (median, range, how often positive) instead
    of pooling repeated county-years into one test.
    """
    d = t[t["eligible"] & (t["base_year"] >= C.MA_FIRST_YEAR)].dropna(subset=["rel_shift_pp"])
    rows = []
    for col in C.MA_DEMOGRAPHICS + ["ma_ratio"]:
        per_year = d.groupby("base_year").apply(
            lambda g: spearmanr(g[col], g["rel_shift_pp"]).statistic, include_groups=False)
        rows.append({"feature": col, "rho": per_year.median(),
                     "rho_min": per_year.min(), "rho_max": per_year.max(),
                     "seasons_positive": int((per_year > 0).sum()), "seasons": len(per_year)})
    return pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def _blend(d: pd.DataFrame) -> pd.Series:
    """Half the county's own last window, half its state's last window."""
    own = d["prev_rel_shift"].fillna(d["state_prev_rel_shift"])
    state = d["state_prev_rel_shift"].fillna(own)
    return (C.MA_BLEND_WEIGHT * own + (1 - C.MA_BLEND_WEIGHT) * state).fillna(0.0)


def _fit_gbm(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**C.GBM_PARAMS).fit(
        train[C.MA_GBM_FEATURES], train["rel_shift_pp"])


def _scores(pred: np.ndarray, y: np.ndarray) -> dict:
    """Rank agreement plus how many of the true top/bottom 10% get flagged."""
    k = max(1, int(len(y) * 0.10))
    top_true, top_pred = set(np.argsort(-y)[:k]), set(np.argsort(-pred)[:k])
    bot_true, bot_pred = set(np.argsort(y)[:k]), set(np.argsort(pred)[:k])
    return {
        "spearman": float(spearmanr(pred, y).statistic),
        "top10_hit": len(top_true & top_pred) / k,
        "bottom10_hit": len(bot_true & bot_pred) / k,
        "mae_pp": float(np.mean(np.abs(pred - y))),
    }


def backtest(t: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward: for each year, train on earlier years only and score it."""
    model_rows = t[t["eligible"] & (t["base_year"] >= C.MA_FIRST_YEAR)]
    rows = []
    for year in C.MA_BACKTEST_YEARS:
        test = model_rows[model_rows["base_year"] == year].dropna(subset=["rel_shift_pp"])
        train = model_rows[model_rows["base_year"] < year].dropna(subset=["rel_shift_pp"])
        y = test["rel_shift_pp"].to_numpy()
        preds = {
            "County's last window": test["prev_rel_shift"].fillna(0).to_numpy(),
            "State's last window": test["state_prev_rel_shift"].fillna(0).to_numpy(),
            "Blend (county + state)": _blend(test).to_numpy(),
            "Gradient boosting": _fit_gbm(train).predict(test[C.MA_GBM_FEATURES]),
        }
        for method, pred in preds.items():
            rows.append({"year": year, "method": method, "n": len(y), **_scores(pred, y)})
    return pd.DataFrame(rows)


def forecast(t: pd.DataFrame) -> pd.DataFrame:
    """Score Dec 2025 -> Feb 2026 with the blend: who will lead or lag the nation."""
    fc = t[t["eligible"] & (t["base_year"] == C.MA_FORECAST_YEAR)].copy()
    fc["predicted_rel_shift_pp"] = _blend(fc)
    fc["outlook"] = np.select(
        [fc["predicted_rel_shift_pp"] >= 0.5, fc["predicted_rel_shift_pp"] <= -0.5],
        ["Ahead of the median county", "Behind the median county"], "Near the median county")
    keep = ["fips", "state", "state_name", "county", "TOT_BENES", "ma_ratio",
            "prev_rel_shift", "state_prev_rel_shift", "predicted_rel_shift_pp", "outlook"]
    return (fc[keep].sort_values("predicted_rel_shift_pp", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all(feat_df: pd.DataFrame, national_df: pd.DataFrame) -> dict:
    """Build every MA-switching output, write the reports, return them."""
    t = build_switch_table(feat_df)
    nat = national_series(national_df, t)
    latest = int(t.dropna(subset=["shift_pp"])["base_year"].max())
    res = {
        "table": t,
        "national": nat,
        "saturation": saturation_curve(t),
        "states": state_table(t, latest),
        "correlates": correlates(t),
        "backtest": backtest(t),
        "forecast": forecast(t),
    }
    res["summary"] = summarise(res, latest)

    res["national"].to_csv(C.REPORT_DIR / "ma_national.csv", index=False)
    res["saturation"].to_csv(C.REPORT_DIR / "ma_saturation.csv", index=False)
    res["states"].to_csv(C.REPORT_DIR / "ma_states.csv", index=False)
    res["correlates"].to_csv(C.REPORT_DIR / "ma_correlates.csv", index=False)
    res["backtest"].to_csv(C.REPORT_DIR / "ma_backtest.csv", index=False)
    res["forecast"].to_csv(C.REPORT_DIR / "ma_forecast_dec2025_feb2026.csv", index=False)
    (C.REPORT_DIR / "ma_summary.json").write_text(json.dumps(res["summary"], indent=2))
    return res


def summarise(res: dict, latest: int) -> dict:
    """Headline numbers, computed once so the README and site can't drift."""
    nat = res["national"].set_index("year")
    t = res["table"]
    elig = t[t["eligible"]].dropna(subset=["shift_pp"])
    cur = elig[elig["base_year"] == latest].set_index("fips")["shift_pp"]
    prev = elig[elig["base_year"] == latest - 1].set_index("fips")["shift_pp"]
    both = cur.index.intersection(prev.index)

    peak_year = int(nat["county_median_shift_pp"].idxmax())
    bt = res["backtest"]
    mean_rho = bt.groupby("method")["spearman"].mean().sort_values(ascending=False)
    latest_bt = bt[bt["year"] == latest].set_index("method")["spearman"]
    sat = res["saturation"]
    peak_band = sat.loc[sat["median"].idxmax()]
    states = res["states"]
    fell = states[states["shift_pp"] < 0]

    return {
        "first_year": int(nat.index.min()),
        "last_year": int(nat.index.max()),
        "ma_share_first": round(float(nat["ma_share_dec"].iloc[0]), 1),
        "ma_share_last": round(float(nat["ma_share_dec"].iloc[-1]), 1),
        "ma_majority_year": int(nat.index[nat["ma_millions_dec"] > nat["om_millions_dec"]][0]),
        "ma_millions_last": round(float(nat["ma_millions_dec"].iloc[-1]), 1),
        "om_millions_last": round(float(nat["om_millions_dec"].iloc[-1]), 1),
        "latest_window": f"Dec {latest} -> Feb {latest + 1}",
        "peak_window_year": peak_year,
        "peak_county_median_shift_pp": round(float(nat.loc[peak_year, "county_median_shift_pp"]), 2),
        "latest_county_median_shift_pp": round(float(nat.loc[latest, "county_median_shift_pp"]), 2),
        "latest_national_shift_pp": round(float(nat.loc[latest, "national_shift_pp"]), 2),
        "share_counties_slowed": round(float((cur[both] < prev[both]).mean()), 3),
        "counties_compared": int(len(both)),
        "share_counties_ma_fell": round(float((cur < 0).mean()), 3),
        "counties_modelled": int(len(cur)),
        "saturation_peak_band": f"{peak_band['ma_share_from']:.0f}-{peak_band['ma_share_to']:.0f}%",
        "saturation_peak_rel_pp": round(float(peak_band["median"]), 2),
        "saturation_top_band_rel_pp": round(float(sat["median"].iloc[-1]), 2),
        "states_ma_fell": fell[["state", "state_name", "shift_pp"]].round(2).to_dict("records"),
        "backtest_mean_spearman": mean_rho.round(3).to_dict(),
        "backtest_latest_spearman": latest_bt.round(3).to_dict(),
        "best_method": mean_rho.index[0],
        "forecast_counties": int(len(res["forecast"])),
        "forecast_outlook_counts": res["forecast"]["outlook"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    from .data_cleaning import load_clean, load_national
    from .feature_engineering import engineer

    clean_df, _ = load_clean()
    out = run_all(engineer(clean_df), load_national())
    print(json.dumps(out["summary"], indent=2))
    print(out["backtest"].pivot(index="year", columns="method", values="spearman").round(3))
