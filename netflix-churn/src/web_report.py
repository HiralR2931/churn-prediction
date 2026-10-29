"""Data file for the interactive report (``site/data.js``, ``window.REPORT``)."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def _records(df: pd.DataFrame, digits: int = 4) -> list[dict]:
    return json.loads(df.round(digits).to_json(orient="records"))


def summarise(df: pd.DataFrame, quality: dict, res: dict, bench: pd.DataFrame,
              errors: pd.DataFrame) -> dict:
    """Headline numbers, computed once for the README and the site."""
    b = bench.set_index("model")
    card, pts, tiers = res["card"], res["points"], res["tiers"].set_index("tier")
    cut = card["cutoff"]
    below = pts[pts["points"] <= cut - 3]
    top = pts[pts["points"] >= cut + 2]
    tests = res["tests"]
    border = errors[(errors["points"] >= cut - 2) & (errors["points"] <= cut + 2)]
    return {
        "customers": quality["rows"],
        "churn_rate": quality["churn_rate"],
        "features": len(C.FEATURES),
        "features_no_effect": tests.loc[~tests["matters"], "feature"].tolist(),
        "monthly_fee_fixed_by_plan": quality["monthly_fee_fixed_by_plan"],
        "avg_watch_identity_share": quality["avg_watch_equals_watch_over_login_plus_1"],
        "rows_changed_by_notebook_div10": quality["rows_changed_by_notebook_div10"],
        "cutoff": cut,
        "points": card["points"],
        "max_points": card["max_points"],
        "low_points_to": cut - 3,
        "low_customers": int(below["n"].sum()),
        "low_churned": int(below["churned"].sum()),
        "top_points_from": cut + 2,
        "top_customers": int(top["n"].sum()),
        "top_churned": int(top["churned"].sum()),
        "scorecard_accuracy": round(float(b.loc["Six-condition scorecard", "accuracy"]), 4),
        "best_model": bench.iloc[0]["model"],
        "best_accuracy": round(float(bench.iloc[0]["accuracy"]), 4),
        "drivers_only_accuracy": round(float(b.loc["Gradient boosting, 5 drivers only", "accuracy"]), 4),
        "logistic_accuracy": round(float(b.loc["Logistic regression", "accuracy"]), 4),
        "gb_errors_total": int(errors["gb_errors"].sum()),
        "gb_errors_near_cutoff": int(border["gb_errors"].sum()),
        "near_cutoff_range": [cut - 2, cut + 2],
        "high_tiers_share_of_churners": round(float(
            tiers.loc[["High", "Very high"], "share_of_churners"].sum()), 4),
        "high_tiers_share_of_customers": round(float(
            tiers.loc[["High", "Very high"], "share_of_customers"].sum()), 4),
    }


def build_site(summary: dict, res: dict, bench: pd.DataFrame, errors: pd.DataFrame) -> list:
    labels = {n: l for n, l, _ in C.CONDITIONS}
    payload = {
        "summary": summary,
        "conditions": [{"name": n, "label": labels[n], "points": p} for n, p in res["card"]["points"].items()],
        "cliff_watch": _records(res["cliff_watch"]),
        "cliff_login": _records(res["cliff_login"]),
        "grid": _records(res["grid"]),
        "effects": _records(res["effects"]),
        "tests": json.loads(res["tests"].to_json(orient="records")),  # keep tiny p-values
        "points": _records(res["points"]),
        "tiers": _records(res["tiers"]),
        "lift": _records(res["lift"]),
        "benchmark": _records(bench, 6),
        "errors": _records(errors),
    }
    path = C.SITE_DIR / "data.js"
    path.write_text("window.REPORT = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    (C.REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return [path]
