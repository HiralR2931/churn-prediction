"""Statistical comparison of churn across demographic / coverage segments.

Splits counties into high vs low exposure on each engineered ratio (median
split) and tests whether historical churn rates differ, using a two-proportion
z-test. This is the analysis that "strengthens confidence in the model's key
churn drivers" — it corroborates the Random Forest importances with a simple,
transparent statistical test.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

# Segments defined by a county's exposure to each ratio (high = above median).
_SEGMENT_RATIOS = {
    "High dual-eligible share": "dual_ratio",
    "High disabled share": "disabled_ratio",
    "High aged share": "aged_ratio",
    "High Medicare Advantage share": "ma_ratio",
    "High ESRD share": "esrd_ratio",
    "High low-income-subsidy share": "lis_ratio",
    "Large county (benes)": "log_tot_benes",
}


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval for a proportion."""
    if n == 0:
        return 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return margin  # symmetric-ish approximation for plotting


def segment_churn_stats(labelled: pd.DataFrame) -> pd.DataFrame:
    """Churn rate + 95% CI for the overall population and each segment."""
    rows = []
    y = labelled["churn"].astype(int)
    n, k = len(y), int(y.sum())
    p = k / n
    rows.append({"segment": "All counties", "n": n, "churn_rate": p,
                 "ci95": _wilson_halfwidth(p, n)})
    for name, ratio in _SEGMENT_RATIOS.items():
        med = labelled[ratio].median()
        sub = labelled[labelled[ratio] > med]
        ys = sub["churn"].astype(int)
        ns, ps = len(ys), (ys.mean() if len(ys) else 0.0)
        rows.append({"segment": name, "n": ns, "churn_rate": ps,
                     "ci95": _wilson_halfwidth(ps, ns)})
    return pd.DataFrame(rows)


def two_proportion_tests(labelled: pd.DataFrame) -> pd.DataFrame:
    """Two-proportion z-test: high- vs low-exposure counties per ratio."""
    rows = []
    for name, ratio in _SEGMENT_RATIOS.items():
        med = labelled[ratio].median()
        high = labelled[labelled[ratio] > med]["churn"].astype(int)
        low = labelled[labelled[ratio] <= med]["churn"].astype(int)
        n1, n2 = len(high), len(low)
        p1, p2 = high.mean(), low.mean()
        p_pool = (high.sum() + low.sum()) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else 0.0
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({
            "segment": name, "ratio": ratio,
            "churn_high": p1, "churn_low": p2, "difference": p1 - p2,
            "z_stat": z, "p_value": pval,
            "significant_0.01": pval < 0.01,
        })
    out = pd.DataFrame(rows).sort_values("z_stat", key=lambda s: s.abs(),
                                         ascending=False)
    return out.reset_index(drop=True)


def correlation_with_churn(labelled: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlation of each feature with the churn outcome."""
    rows = []
    y = labelled["churn"].astype(int)
    for feat in C.FEATURE_COLUMNS:
        r, p = stats.pointbiserialr(y, labelled[feat])
        rows.append({"feature": feat, "correlation": r, "p_value": p})
    return (pd.DataFrame(rows)
            .sort_values("correlation", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


def run_all(labelled: pd.DataFrame) -> dict:
    """Run every analysis and persist CSVs. Returns the frames."""
    seg = segment_churn_stats(labelled)
    tests = two_proportion_tests(labelled)
    corr = correlation_with_churn(labelled)
    seg.to_csv(C.REPORT_DIR / "segment_churn_stats.csv", index=False)
    tests.to_csv(C.REPORT_DIR / "segment_ztests.csv", index=False)
    corr.to_csv(C.REPORT_DIR / "feature_churn_correlation.csv", index=False)
    return {"segment_stats": seg, "ztests": tests, "correlation": corr}


if __name__ == "__main__":
    from .churn_label import build_modeling_table, train_test_forecast_split
    from .data_cleaning import load_clean
    from .feature_engineering import engineer

    clean_df, _ = load_clean()
    tbl = build_modeling_table(engineer(clean_df))
    tr, te, _ = train_test_forecast_split(tbl)
    res = run_all(pd.concat([tr, te]))
    print(res["ztests"].to_string(index=False))
