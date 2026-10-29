"""What drives churn in this dataset, and the scorecard that reproduces it.

Outputs (``outputs/reports/``):
    feature_effects.csv     churn rate per value of every categorical feature
    feature_tests.csv       one test per feature: does churn differ at all?
    cliff_watch_hours.csv   churn rate by watch hours (where the cut-points are)
    cliff_last_login.csv    churn rate by days since last login
    interaction_grid.csv    churn rate by watch hours x days since login
    scorecard.json          fitted points and cut-off
    scorecard_points.csv    churn rate at each total score
    risk_tiers.csv          customers and churn rate per risk tier
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from . import config as C
from .data import condition_flags


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson interval for a proportion k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def _rates(df: pd.DataFrame, by: str | pd.Series) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[C.TARGET].agg(churned="sum", n="size")
    g["churn_rate"] = g["churned"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson(g["churned"], g["n"])
    return g.reset_index()


# --------------------------------------------------------------------------- #
# What matters and what doesn't
# --------------------------------------------------------------------------- #
def feature_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square for categorical features, point-biserial for continuous ones."""
    rows = []
    for col in C.CATEGORICAL + ["number_of_profiles"]:
        tab = pd.crosstab(df[col], df[C.TARGET])
        chi2, p, _, _ = chi2_contingency(tab)
        cramers_v = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
        rows.append({"feature": col, "test": "chi-square", "effect": cramers_v, "p_value": p})
    for col in ["age", "watch_hours", "last_login_days", "avg_watch_time_per_day"]:
        r, p = pointbiserialr(df[C.TARGET], df[col])
        rows.append({"feature": col, "test": "point-biserial r", "effect": abs(r), "p_value": p})
    out = pd.DataFrame(rows)
    out["matters"] = out["p_value"] < 0.01
    return out.sort_values("effect", ascending=False).reset_index(drop=True)


def feature_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate and 95% CI for every value of every categorical feature."""
    parts = []
    for col in C.CATEGORICAL + ["number_of_profiles"]:
        r = _rates(df, col).rename(columns={col: "value"})
        r.insert(0, "feature", col)
        r["value"] = r["value"].astype(str)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def cliff_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Churn rate along watch hours (1-hour bins) and days since login (3-day bins).

    Login bins start at day 1 so that one ends at day 30 and the next starts at
    31, the scorecard's cut-point; day 0 gets a bin of its own.
    """
    w_edges = list(range(0, 31)) + [np.inf]
    w = _rates(df, pd.cut(df["watch_hours"], w_edges, right=False).rename("bin"))
    w["from"] = [b.left for b in w["bin"]]
    w["to"] = [b.right if np.isfinite(b.right) else None for b in w["bin"]]
    l_edges = [0] + list(range(1, 62, 3))
    lg = _rates(df, pd.cut(df["last_login_days"], l_edges, right=False).rename("bin"))
    lg["from"] = [b.left for b in lg["bin"]]
    lg["to"] = [b.right - 1 for b in lg["bin"]]
    return w.drop(columns="bin"), lg.drop(columns="bin")


def interaction_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate for each (days since login band, watch hours band) cell."""
    lb = pd.cut(df["last_login_days"], [0, 11, 21, 31, 41, 51, 61], right=False,
                labels=["0–10", "11–20", "21–30", "31–40", "41–50", "51–60"])
    wb = pd.cut(df["watch_hours"], [0, 5, 10, 20, 30, np.inf], right=False,
                labels=["<5", "5–10", "10–20", "20–30", "30+"])
    g = df.groupby([lb.rename("login_band"), wb.rename("watch_band")], observed=False)[C.TARGET]
    out = g.agg(churn_rate="mean", n="size").reset_index()
    out["login_band"] = out["login_band"].astype(str)
    out["watch_band"] = out["watch_band"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def fit_scorecard(flags: pd.DataFrame, y: pd.Series) -> dict:
    """Search integer points per condition and a cut-off that maximise accuracy."""
    names = list(C.POINT_GRID)
    F = flags[names].to_numpy()
    yv = y.to_numpy()
    best = None
    for pts in itertools.product(*(C.POINT_GRID[n] for n in names)):
        score = F @ np.array(pts)
        for cut in range(1, int(sum(max(C.POINT_GRID[n]) for n in names)) + 1):
            acc = ((score >= cut) == yv).mean()
            # ties: prefer the smaller total, i.e. the simpler card
            key = (acc, -sum(pts))
            if best is None or key > best[0]:
                best = (key, dict(zip(names, map(int, pts))), cut)
    (acc, _), points, cut = best
    return {"points": points, "cutoff": int(cut), "train_accuracy": float(acc)}


def score(flags: pd.DataFrame, card: dict) -> pd.Series:
    return sum(flags[n] * p for n, p in card["points"].items())


def points_table(df: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    return _rates(df.assign(points=s), "points")


def risk_tiers(df: pd.DataFrame, s: pd.Series, card: dict) -> pd.DataFrame:
    """Four tiers by total points, anchored on the cut-off."""
    cut = card["cutoff"]
    bins = [-1, cut - 3, cut - 1, cut + 1, np.inf]
    labels = ["Low", "Watch", "High", "Very high"]
    tier = pd.cut(s, bins, labels=labels)
    out = _rates(df.assign(tier=tier), "tier")
    out["tier"] = out["tier"].astype(str)
    out["points_from"] = [max(0, int(b) + 1) for b in bins[:-1]]
    out["points_to"] = [int(b) if np.isfinite(b) else None for b in bins[1:]]
    out["share_of_customers"] = out["n"] / out["n"].sum()
    out["share_of_churners"] = out["churned"] / out["churned"].sum()
    return out


def condition_lift(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Churn with vs. without each condition, and how common it is."""
    rows = []
    for name, label, _ in C.CONDITIONS:
        on = flags[name] == 1
        rows.append({"condition": name, "label": label, "share_customers": on.mean(),
                     "churn_if_true": df.loc[on, C.TARGET].mean(),
                     "churn_if_false": df.loc[~on, C.TARGET].mean()})
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame) -> dict:
    flags = condition_flags(df)
    # Fitted on all rows for display; its held-out accuracy comes from the
    # model benchmark (models.py), which refits it inside every fold.
    card = fit_scorecard(flags, df[C.TARGET])
    s = score(flags, card)
    card["max_points"] = int(sum(v for k, v in card["points"].items() if k != "watch_5_20"))
    w, lg = cliff_curves(df)
    res = {
        "tests": feature_tests(df),
        "effects": feature_effects(df),
        "cliff_watch": w,
        "cliff_login": lg,
        "grid": interaction_grid(df),
        "card": card,
        "points": points_table(df, s),
        "tiers": risk_tiers(df, s, card),
        "lift": condition_lift(df, flags),
        "score": s,
    }
    res["tests"].to_csv(C.REPORT_DIR / "feature_tests.csv", index=False)
    res["effects"].to_csv(C.REPORT_DIR / "feature_effects.csv", index=False)
    w.to_csv(C.REPORT_DIR / "cliff_watch_hours.csv", index=False)
    lg.to_csv(C.REPORT_DIR / "cliff_last_login.csv", index=False)
    res["grid"].to_csv(C.REPORT_DIR / "interaction_grid.csv", index=False)
    res["points"].to_csv(C.REPORT_DIR / "scorecard_points.csv", index=False)
    res["tiers"].to_csv(C.REPORT_DIR / "risk_tiers.csv", index=False)
    res["lift"].to_csv(C.REPORT_DIR / "condition_lift.csv", index=False)
    (C.REPORT_DIR / "scorecard.json").write_text(json.dumps(card, indent=2))
    return res


if __name__ == "__main__":
    from .data import load

    df, _ = load()
    out = run_all(df)
    print(json.dumps(out["card"], indent=2))
    print(out["tests"].round(4).to_string())
    print(out["points"].round(3).to_string())
    print(out["tiers"].round(3).to_string())
