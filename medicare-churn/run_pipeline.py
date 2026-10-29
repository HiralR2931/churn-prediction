"""End-to-end orchestrator for the Medicare County-Level Churn Prediction project.

Runs the full pipeline:
    clean -> engineer features -> label churn -> train/evaluate Random Forest
    -> forecast Dec 2025→Feb 2026 -> statistical analysis -> Plotly dashboards

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src import config as C
from src.churn_label import build_modeling_table, train_test_forecast_split
from src.data_cleaning import load_clean
from src.dashboards import build_dashboard
from src.feature_engineering import engineer
from src.model import (
    evaluate,
    feature_importance,
    forecast,
    save_artifacts,
    train_model,
)
from src.stats_analysis import run_all as run_stats


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> None:
    t0 = time.time()

    _hr("1. CLEAN & VALIDATE")
    clean_df, report = load_clean()
    for k, v in report.items():
        print(f"  {k:28s}: {v}")

    _hr("2. FEATURE ENGINEERING")
    feat_df = engineer(clean_df)
    print(f"  engineered {len(C.FEATURE_COLUMNS)} features on {len(feat_df):,} county-months")

    _hr("3. CHURN LABELLING & SPLIT")
    tbl = build_modeling_table(feat_df)
    train, test, fc = train_test_forecast_split(tbl)
    print(f"  train rows   : {len(train):,}  (churn rate {train['churn'].mean():.1%})")
    print(f"  test  rows   : {len(test):,}  (churn rate {test['churn'].mean():.1%})")
    print(f"  forecast rows: {len(fc):,}  (Dec {C.PREDICT_FROM_YEAR} → Feb {C.PREDICT_FROM_YEAR + 1})")

    _hr("4. RANDOM FOREST — TRAIN & EVALUATE")
    clf = train_model(train)
    metrics = evaluate(clf, test)
    print(f"  ROC-AUC : {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC  : {metrics['pr_auc']:.3f}")
    print(f"  Brier   : {metrics['brier']:.3f}")
    print(f"  confusion matrix (rows=actual): {metrics['confusion_matrix']}")
    importance = feature_importance(clf, test)
    print("  top drivers:")
    for _, r in importance.head(5).iterrows():
        print(f"    {r['feature']:22s} {r['permutation_importance']:.4f}")

    _hr("5. FORECAST  Dec 2025 → Feb 2026")
    forecast_df = forecast(clf, fc)
    n_risk = int(forecast_df["predicted_churn"].sum())
    print(f"  counties scored     : {len(forecast_df):,}")
    print(f"  predicted to churn  : {n_risk:,} ({n_risk / len(forecast_df):.1%})")
    print("  highest-risk counties:")
    for _, r in forecast_df.head(5).iterrows():
        print(f"    {r['county']}, {r['state']:2s}  p={r['churn_probability']:.2f}")
    save_artifacts(clf, metrics, importance, forecast_df)

    _hr("6. STATISTICAL ANALYSIS — DEMOGRAPHIC SEGMENTS")
    labelled = pd.concat([train, test])
    stats_res = run_stats(labelled)
    sig = stats_res["ztests"]
    print("  two-proportion z-tests (high vs low exposure):")
    for _, r in sig.iterrows():
        star = "***" if r["significant_0.01"] else "   "
        print(f"    {star} {r['segment']:32s} Δ={r['difference']:+.1%}  z={r['z_stat']:+.1f}  p={r['p_value']:.1e}")

    _hr("7. PLOTLY DASHBOARDS")
    written = build_dashboard(clean_df, forecast_df, importance,
                              stats_res["segment_stats"])
    for p in written:
        print(f"  wrote {p.relative_to(C.PROJECT_ROOT)}")

    # Run summary
    summary = {
        "data_quality": report,
        "model_metrics": {k: metrics[k] for k in
                          ["n_test", "positive_rate", "roc_auc", "pr_auc", "brier"]},
        "forecast_window": f"Dec {C.PREDICT_FROM_YEAR} -> Feb {C.PREDICT_FROM_YEAR + 1}",
        "counties_scored": int(len(forecast_df)),
        "counties_predicted_churn": n_risk,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (C.REPORT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    _hr(f"DONE in {summary['runtime_seconds']}s — open outputs/figures/index.html")


if __name__ == "__main__":
    main()
