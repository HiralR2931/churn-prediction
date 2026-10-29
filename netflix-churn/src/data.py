"""Load the Kaggle Netflix churn CSV and check what the columns really contain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load(path=C.RAW_CSV) -> tuple[pd.DataFrame, dict]:
    """Return (data, data-quality report). The data is used as published."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in C.CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()

    implied = df["watch_hours"] / (df["last_login_days"] + 1)
    fee_per_plan = df.groupby("subscription_type")["monthly_fee"].nunique()
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_customer_ids": int(df["customer_id"].duplicated().sum()),
        "churn_rate": round(float(df[C.TARGET].mean()), 4),
        "churned": int(df[C.TARGET].sum()),
        # avg_watch_time_per_day is derived, not measured.
        "avg_watch_equals_watch_over_login_plus_1": round(
            float((np.abs(implied - df["avg_watch_time_per_day"]) <= 0.005 + 1e-9).mean()), 4),
        # The original notebook divided values above 10 by 10 as "entry errors".
        "rows_changed_by_notebook_div10": int((df["avg_watch_time_per_day"] > 10).sum()),
        "max_last_login_days_of_those_rows": int(
            df.loc[df["avg_watch_time_per_day"] > 10, "last_login_days"].max()),
        # monthly_fee is fixed by plan, so it repeats subscription_type.
        "monthly_fee_fixed_by_plan": bool((fee_per_plan == 1).all()),
    }
    return df, report


def condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    """0/1 column for each scorecard condition."""
    return pd.DataFrame({name: test(df).astype(int) for name, _, test in C.CONDITIONS},
                        index=df.index)


if __name__ == "__main__":
    _, rep = load()
    for k, v in rep.items():
        print(f"{k:44s}: {v}")
