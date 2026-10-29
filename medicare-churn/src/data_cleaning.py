"""Load, clean, and validate the raw CMS Medicare monthly enrollment extract.

Responsibilities
----------------
1. Read the raw CSV with correct dtypes (FIPS codes keep leading zeros).
2. Resolve inconsistent records: strip suppressed ("*") cells, coerce numeric
   columns, drop malformed rows, and normalise the month/date fields.
3. Emit a clean county-level monthly panel plus a data-quality report.

The logic mirrors the Databricks/PySpark cleaning step but runs on pandas so the
project is fully reproducible on a laptop.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

# Raw count columns that may contain suppressed ("*") values.
_ALWAYS_STRING = {"BENE_STATE_ABRVTN", "BENE_STATE_DESC", "BENE_COUNTY_DESC",
                  "BENE_FIPS_CD", "BENE_GEO_LVL", "MONTH"}


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, keeping identifier columns as strings."""
    df = pd.read_csv(
        path,
        dtype={col: "string" for col in _ALWAYS_STRING},
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _ALWAYS_STRING and c not in {"YEAR"}]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean county-level monthly panel, data-quality report)."""
    report: dict = {"raw_rows": len(df)}

    # --- 1. Keep only county-level monthly records -------------------------- #
    df = df[df["BENE_GEO_LVL"] == "County"].copy()
    df = df[df["MONTH"] != C.ANNUAL_MONTH_TOKEN].copy()  # drop annual roll-ups
    report["county_monthly_rows"] = len(df)

    # --- 2. Resolve suppressed cells --------------------------------------- #
    num_cols = _numeric_columns(df)
    suppressed_cells = 0
    row_suppressed = pd.Series(False, index=df.index)
    for col in num_cols:
        s = df[col].astype("string").str.strip()
        mask = (s == C.SUPPRESSION_TOKEN).fillna(False)
        suppressed_cells += int(mask.sum())
        row_suppressed |= mask
        s = s.mask(mask, str(C.SUPPRESSION_MIDPOINT))
        df[col] = pd.to_numeric(s, errors="coerce")
    report["suppressed_cells_imputed"] = suppressed_cells

    # Flag rows that carried at least one suppressed count (kept for auditing).
    df["had_suppression"] = row_suppressed
    report["rows_with_suppression"] = int(row_suppressed.sum())

    # --- 3. Drop structurally broken rows ---------------------------------- #
    before = len(df)
    df = df.dropna(subset=["TOT_BENES"])
    df = df[df["TOT_BENES"] > 0]
    # A valid county FIPS is 5 digits.
    df = df[df["BENE_FIPS_CD"].str.fullmatch(r"\d{5}").fillna(False)]
    report["rows_dropped_invalid"] = before - len(df)

    # Each state also has an "Unknown" bucket (FIPS ending 999) for people
    # without a county; it is not a place, so it is left out of county analysis.
    unknown = df["BENE_FIPS_CD"].str.endswith("999")
    report["rows_dropped_unknown_county"] = int(unknown.sum())
    df = df[~unknown]

    # --- 4. Consistency check: components should not exceed the total ------- #
    # (data-integrity guard rather than a hard filter; small overshoots from
    #  suppression imputation are tolerated up to 2%).
    tol = 1.02
    inconsistent = (df["ORGNL_MDCR_BENES"] + df["MA_AND_OTH_BENES"]) > df["TOT_BENES"] * tol
    report["rows_flagged_inconsistent"] = int(inconsistent.sum())
    df["flag_inconsistent_split"] = inconsistent

    # --- 5. Normalise time fields ------------------------------------------ #
    df["month_num"] = df["MONTH"].map(C.MONTH_ORDER)
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["month_num"], day=1)
    )
    df = df.rename(columns={
        "YEAR": "year",
        "BENE_STATE_ABRVTN": "state",
        "BENE_STATE_DESC": "state_name",
        "BENE_COUNTY_DESC": "county",
        "BENE_FIPS_CD": "fips",
    })

    df = df.sort_values(["fips", "date"]).reset_index(drop=True)
    report["clean_rows"] = len(df)
    report["counties"] = df["fips"].nunique()
    report["date_min"] = str(df["date"].min().date())
    report["date_max"] = str(df["date"].max().date())
    return df, report


def load_clean(path=C.RAW_CSV) -> tuple[pd.DataFrame, dict]:
    """Convenience: load raw then clean in one call."""
    return clean(load_raw(path))


def load_national(path=C.RAW_CSV) -> pd.DataFrame:
    """CMS's own national monthly totals.

    County rows leave out roughly half a million people with no county on
    file (mostly in Original Medicare), so national figures come from the
    ``National`` rows rather than a sum of counties.
    """
    cols = ["YEAR", "MONTH", "BENE_GEO_LVL", "TOT_BENES", "ORGNL_MDCR_BENES", "MA_AND_OTH_BENES"]
    df = pd.read_csv(path, usecols=cols, dtype={"MONTH": "string", "BENE_GEO_LVL": "string"})
    df = df[(df["BENE_GEO_LVL"] == "National") & df["MONTH"].isin(list(C.MONTH_ORDER))].copy()
    df["month_num"] = df["MONTH"].map(C.MONTH_ORDER).astype(int)
    for col in cols[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (df.rename(columns={"YEAR": "year"})
            [["year", "month_num"] + cols[3:]].reset_index(drop=True))


if __name__ == "__main__":
    _, rep = load_clean()
    for k, v in rep.items():
        print(f"{k:28s}: {v}")
