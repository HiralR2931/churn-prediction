"""Random Forest churn model: train, evaluate, explain, and forecast."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from . import config as C


def train_model(train: pd.DataFrame) -> RandomForestClassifier:
    """Fit the Random Forest on the engineered feature matrix."""
    X = train[C.FEATURE_COLUMNS].to_numpy()
    y = train["churn"].astype(int).to_numpy()
    clf = RandomForestClassifier(**C.RF_PARAMS)
    clf.fit(X, y)
    return clf


def evaluate(clf: RandomForestClassifier, test: pd.DataFrame) -> dict:
    """Score the held-out temporal test set."""
    X = test[C.FEATURE_COLUMNS].to_numpy()
    y = test["churn"].astype(int).to_numpy()
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_test": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
    }
    return metrics


def feature_importance(clf: RandomForestClassifier, test: pd.DataFrame) -> pd.DataFrame:
    """Impurity-based and permutation importance, sorted by permutation."""
    imp = pd.DataFrame({
        "feature": C.FEATURE_COLUMNS,
        "impurity_importance": clf.feature_importances_,
    })
    perm = permutation_importance(
        clf,
        test[C.FEATURE_COLUMNS].to_numpy(),
        test["churn"].astype(int).to_numpy(),
        n_repeats=10,
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    imp["permutation_importance"] = perm.importances_mean
    imp["permutation_std"] = perm.importances_std
    return imp.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def forecast(clf: RandomForestClassifier, fc: pd.DataFrame) -> pd.DataFrame:
    """Score the forecast snapshot (Dec 2025 -> predicted Feb 2026 churn)."""
    out = fc[["fips", "state", "state_name", "county", "base_year",
              "TOT_BENES"] + C.FEATURE_COLUMNS].copy()
    out["churn_probability"] = clf.predict_proba(fc[C.FEATURE_COLUMNS].to_numpy())[:, 1]
    out["predicted_churn"] = (out["churn_probability"] >= 0.5).astype(int)
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def save_artifacts(clf, metrics: dict, importance: pd.DataFrame,
                   forecast_df: pd.DataFrame) -> None:
    joblib.dump(clf, C.MODEL_DIR / "random_forest_churn.joblib")
    (C.REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.to_csv(C.REPORT_DIR / "feature_importance.csv", index=False)
    forecast_df.to_csv(C.REPORT_DIR / "forecast_dec2025_feb2026.csv", index=False)


if __name__ == "__main__":
    from .churn_label import build_modeling_table, train_test_forecast_split
    from .data_cleaning import load_clean
    from .feature_engineering import engineer

    clean_df, _ = load_clean()
    tbl = build_modeling_table(engineer(clean_df))
    tr, te, fc = train_test_forecast_split(tbl)
    clf = train_model(tr)
    print(json.dumps(evaluate(clf, te), indent=2)[:600])
