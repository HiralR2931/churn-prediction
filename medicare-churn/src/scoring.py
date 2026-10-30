"""Score the Dec 2025 -> Feb 2026 outlook against what CMS later published.

The outlook was made from data ending December 2025. The outcome month comes
from a later CMS release (``fetch_scoring_data``), and December is refetched
from that same release so the two months are comparable. CMS revises past
months, so the revision itself is measured too.

Outputs (``outputs/reports/``):
    scored_counties.csv       per county: predicted vs actual relative switching
    scored_methods.csv        each method's accuracy on the scored season
    scored_outlook.csv        what happened to each outlook category
    scored_states.csv         state median switching, scored season vs the one before
    scored_summary.json       headline numbers for the README and site
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .data_cleaning import clean, load_raw
from .ma_switching import _fit_gbm, _scores

_DEC, _FEB = C.MONTH_ORDER["December"], C.MONTH_ORDER["February"]


def actual_outcome() -> tuple[pd.DataFrame, dict]:
    """County switching for the scored season, from the newer CMS release."""
    raw = load_raw(C.SCORING_CSV)
    county, _ = clean(raw)
    dec = county[county["month_num"] == _DEC].set_index("fips")
    feb = county[county["month_num"] == _FEB].set_index("fips")
    idx = dec.index.intersection(feb.index)
    share = lambda d: d.loc[idx, "MA_AND_OTH_BENES"] / d.loc[idx, "TOT_BENES"] * 100
    out = pd.DataFrame({
        "state": dec.loc[idx, "state"], "county": dec.loc[idx, "county"],
        "benes_dec": dec.loc[idx, "TOT_BENES"],
        "ma_share_dec": share(dec), "ma_share_feb": share(feb),
    })
    out["actual_shift_pp"] = out["ma_share_feb"] - out["ma_share_dec"]
    out["eligible"] = out["benes_dec"] >= C.MA_MIN_BENES
    med = out.loc[out["eligible"], "actual_shift_pp"].median()
    out["actual_rel_pp"] = out["actual_shift_pp"] - med

    nat = raw[raw["BENE_GEO_LVL"] == "National"].copy()
    for col in ("TOT_BENES", "MA_AND_OTH_BENES"):
        nat[col] = pd.to_numeric(nat[col])
    nat_share = (nat.set_index("MONTH")["MA_AND_OTH_BENES"] / nat.set_index("MONTH")["TOT_BENES"] * 100)
    national = {
        "ma_share_dec_revised": float(nat_share["December"]),
        "ma_share_feb": float(nat_share["February"]),
        "national_shift_pp": float(nat_share["February"] - nat_share["December"]),
        "county_median_shift_pp": float(med),
    }
    return out.reset_index(), national


def revisions(t: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """How much CMS changed December between the two releases."""
    old = t[t["base_year"] == C.MA_FORECAST_YEAR].set_index("fips")
    new = actual.set_index("fips")
    idx = old.index.intersection(new.index)
    tot = new.loc[idx, "benes_dec"] / old.loc[idx, "TOT_BENES"] - 1
    ma = new.loc[idx, "ma_share_dec"] - old.loc[idx, "ma_ratio"] * 100
    return {"counties_compared": int(len(idx)),
            "share_counties_total_revised": float((tot.abs() > 0).mean()),
            "median_total_revision_pct": float(tot.median() * 100),
            "median_ma_share_revision_pp": float(ma.median())}


def score_methods(t: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every backtest method, as it would have predicted with data to Dec 2025."""
    rows = t[t["eligible"] & (t["base_year"] == C.MA_FORECAST_YEAR)].set_index("fips")
    train = t[t["eligible"] & (t["base_year"] >= C.MA_FIRST_YEAR)].dropna(subset=["rel_shift_pp"])
    gbm = pd.Series(_fit_gbm(train).predict(rows[C.MA_GBM_FEATURES]), index=rows.index)

    act = actual[actual["eligible"]].set_index("fips")
    f = forecast.set_index("fips")
    j = f.join(act[["actual_shift_pp", "actual_rel_pp", "ma_share_dec", "ma_share_feb"]], how="inner")
    j["gbm_rel_pp"] = gbm.reindex(j.index)
    y = j["actual_rel_pp"].to_numpy()
    preds = {
        "Blend (county + state)": j["predicted_rel_shift_pp"],
        "County's last window": j["prev_rel_shift"].fillna(0),
        "State's last window": j["state_prev_rel_shift"].fillna(0),
        "Gradient boosting": j["gbm_rel_pp"],
    }
    # Post-hoc check: the same scores without the states where the median
    # county's MA share fell this season (chosen after seeing the outcome).
    st_med = act.groupby("state")["actual_shift_pp"].median()
    reversed_states = sorted(st_med[st_med < 0].index)
    keep = ~j["state"].isin(reversed_states).to_numpy()
    out = []
    for name, p in preds.items():
        s_all = _scores(p.to_numpy(), y)
        s_kept = _scores(p.to_numpy()[keep], y[keep])
        out.append({"method": name, "n": len(y), **s_all,
                    "spearman_excl_reversed_states": s_kept["spearman"]})
    j = j.reset_index()
    j.attrs["reversed_states"] = reversed_states
    return pd.DataFrame(out), j


def outlook_check(j: pd.DataFrame) -> pd.DataFrame:
    g = j.groupby("outlook")
    out = g.agg(counties=("fips", "size"), predicted_median=("predicted_rel_shift_pp", "median"),
                actual_median=("actual_rel_pp", "median"),
                share_above_median=("actual_rel_pp", lambda s: (s > 0).mean()),
                share_ma_fell=("actual_shift_pp", lambda s: (s < 0).mean())).reset_index()
    order = {"Ahead of the median county": 0, "Near the median county": 1, "Behind the median county": 2}
    return out.sort_values("outlook", key=lambda s: s.map(order)).reset_index(drop=True)


def state_check(t: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    prev = (t[t["eligible"] & (t["base_year"] == C.MA_FORECAST_YEAR - 1)]
            .groupby(["state", "state_name"])["shift_pp"].median().rename("shift_prev_pp"))
    act = actual[actual["eligible"]]
    cur = act.groupby("state").agg(shift_pp=("actual_shift_pp", "median"), counties=("fips", "size"))
    out = prev.reset_index().merge(cur.reset_index(), on="state", how="inner")
    out["change_pp"] = out["shift_pp"] - out["shift_prev_pp"]
    return out.sort_values("shift_pp").reset_index(drop=True)


def run_all(ma: dict) -> dict | None:
    """Score the outlook if the outcome data has been fetched; else return None."""
    if not C.SCORING_CSV.exists():
        return None
    t, forecast = ma["table"], ma["forecast"]
    actual, national = actual_outcome()
    methods, joined = score_methods(t, forecast, actual)
    states = state_check(t, actual)
    outlook = outlook_check(joined)
    blend = methods.set_index("method").loc["Blend (county + state)"]
    backtest_latest = ma["summary"]["backtest_latest_spearman"]["Blend (county + state)"]
    elig = actual[actual["eligible"]]
    summary = {
        "season": f"Dec {C.MA_FORECAST_YEAR} -> Feb {C.MA_FORECAST_YEAR + 1}",
        "cms_release": C.CMS_API_RELEASE,
        **{k: round(v, 3) for k, v in national.items()},
        "counties_scored": int(len(joined)),
        "share_counties_ma_fell": round(float((elig["actual_shift_pp"] < 0).mean()), 3),
        "reversed_states": [{"state": r.state, "state_name": r.state_name, "shift_pp": round(r.shift_pp, 2)}
                            for r in states[states["shift_pp"] < 0].itertuples()],
        "blend_spearman": round(float(blend["spearman"]), 3),
        "blend_spearman_excl_reversed_states": round(float(blend["spearman_excl_reversed_states"]), 3),
        "blend_top10_hit": round(float(blend["top10_hit"]), 3),
        "blend_bottom10_hit": round(float(blend["bottom10_hit"]), 3),
        "backtest_latest_blend_spearman": backtest_latest,
        "best_method_scored": methods.sort_values("spearman", ascending=False).iloc[0]["method"],
        "revisions": {k: round(v, 3) for k, v in revisions(t, actual).items()},
    }
    joined[["fips", "county", "state", "TOT_BENES", "predicted_rel_shift_pp", "outlook",
            "actual_shift_pp", "actual_rel_pp", "gbm_rel_pp"]].to_csv(
        C.REPORT_DIR / "scored_counties.csv", index=False)
    methods.to_csv(C.REPORT_DIR / "scored_methods.csv", index=False)
    outlook.to_csv(C.REPORT_DIR / "scored_outlook.csv", index=False)
    states.to_csv(C.REPORT_DIR / "scored_states.csv", index=False)
    (C.REPORT_DIR / "scored_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "methods": methods, "counties": joined, "outlook": outlook,
            "states": states, "actual": actual}


if __name__ == "__main__":
    from .data_cleaning import load_national
    from .feature_engineering import engineer
    from .data_cleaning import load_clean
    from .ma_switching import run_all as run_ma

    clean_df, _ = load_clean()
    res = run_all(run_ma(engineer(clean_df), load_national()))
    print(json.dumps(res["summary"], indent=2))
    print(res["methods"].round(3).to_string())
    print(res["outlook"].round(3).to_string())
    print(res["states"].head(8).round(2).to_string())
