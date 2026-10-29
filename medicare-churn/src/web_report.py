"""Data file for the interactive report in ``site/`` (served by GitHub Pages).

``site/index.html`` is hand-written and stays fixed; this module writes
``site/data.js`` (``window.REPORT = {...}``) so the page works from GitHub Pages
and straight from disk. Everything on the page comes from this one file.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

MAP_FIRST_YEAR = 2016


def _records(df: pd.DataFrame, digits: int = 3) -> list[dict]:
    return json.loads(df.round(digits).to_json(orient="records"))


def _county_map(t: pd.DataFrame) -> dict:
    """Per-county switching history, one value per window, for the map."""
    d = t[t["base_year"] >= MAP_FIRST_YEAR]
    years = sorted(d.dropna(subset=["shift_pp"])["base_year"].unique().tolist())
    wide = d.pivot_table(index="fips", columns="base_year", values="shift_pp").reindex(columns=years)
    # Each season uses that December's size, as the statistics do.
    elig = (d.pivot_table(index="fips", columns="base_year", values="eligible", aggfunc="max")
            .reindex(index=wide.index, columns=years))
    names = (t.sort_values("base_year").groupby("fips").last()
             .apply(lambda r: f"{r['county']}, {r['state']}", axis=1))
    counties = {}
    for fips, row in wide.iterrows():
        counties[fips] = {
            "n": names.get(fips, fips),
            # null: no data; "s": under the size cutoff that season; else the shift
            "s": [None if pd.isna(v) else ("s" if not elig.loc[fips, y] else round(float(v), 2))
                  for y, v in zip(years, row.to_numpy())],
        }
    return {"years": [int(y) for y in years], "counties": counties}


def _decline_section(labelled: pd.DataFrame, metrics: dict) -> dict:
    """The original net-decline model, with the county-size caveat made visible."""
    d = labelled.assign(size=pd.qcut(labelled["TOT_BENES"], 5,
                                     labels=["Smallest fifth", "2nd", "3rd", "4th", "Largest fifth"]))
    by_size = (d.groupby("size", observed=True)
               .agg(churn_rate=("churn", "mean"), median_benes=("TOT_BENES", "median"))
               .reset_index().rename(columns={"size": "group"}))
    by_size["group"] = by_size["group"].astype(str)
    keep = ["roc_auc", "pr_auc", "positive_rate", "n_test"]
    return {"metrics": {k: metrics[k] for k in keep}, "by_size": _records(by_size)}


def build_site(ma: dict, labelled: pd.DataFrame, decline_metrics: dict) -> list:
    """Write ``site/data.js``. Returns written paths."""
    C.SITE_DIR.mkdir(exist_ok=True)
    fc = ma["forecast"][["fips", "county", "state", "TOT_BENES", "ma_ratio",
                         "predicted_rel_shift_pp", "outlook"]].copy()
    fc["ma_ratio"] *= 100
    payload = {
        "summary": ma["summary"],
        "national": _records(ma["national"]),
        "saturation": _records(ma["saturation"]),
        "correlates": _records(ma["correlates"]),
        "states": _records(ma["states"]),
        "backtest": _records(ma["backtest"]),
        "forecast": _records(fc.rename(columns={"TOT_BENES": "benes",
                                                "ma_ratio": "ma_share",
                                                "predicted_rel_shift_pp": "pred"}), 2),
        "map": _county_map(ma["table"]),
        "min_benes": C.MA_MIN_BENES,
        "decline": _decline_section(labelled, decline_metrics),
    }
    path = C.SITE_DIR / "data.js"
    path.write_text("window.REPORT = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    return [path]
