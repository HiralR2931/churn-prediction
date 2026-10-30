"""Fetch the two months needed to score the Dec 2025 -> Feb 2026 forecast.

Pulls December 2025 and February 2026 rows (county and national) from the CMS
data API, so both months come from the same release, and writes them in the
raw CSV layout that ``data_cleaning.clean`` expects.

    python -m src.fetch_scoring_data
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from . import config as C

PAGE = 5000


def _get(params: dict) -> list[dict]:
    url = (f"https://data.cms.gov/data-api/v1/dataset/{C.CMS_API_DATASET}/data?"
           + urllib.parse.urlencode(params))
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.load(resp)


def fetch(year: int, month: str, level: str) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        page = _get({"filter[YEAR]": year, "filter[MONTH]": month,
                     "filter[BENE_GEO_LVL]": level, "size": PAGE, "offset": offset})
        rows += page
        if len(page) < PAGE:
            return pd.DataFrame(rows)
        offset += PAGE


def main() -> None:
    year = C.MA_FORECAST_YEAR
    parts = [fetch(y, m, lvl) for y, m in [(year, "December"), (year + 1, "February")]
             for lvl in ("County", "National")]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(C.SCORING_CSV, index=False)
    counts = df.groupby(["YEAR", "MONTH", "BENE_GEO_LVL"]).size().to_dict()
    print(f"wrote {C.SCORING_CSV.relative_to(C.PROJECT_ROOT)} ({len(df):,} rows, "
          f"CMS release {C.CMS_API_RELEASE}): {counts}")


if __name__ == "__main__":
    main()
