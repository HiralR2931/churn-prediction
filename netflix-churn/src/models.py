"""Benchmark: six classifiers with proper preprocessing, against the scorecard.

Every model is scored with the same repeated stratified 5-fold cross-validation.
Linear and distance-based models get one-hot categories and scaled numbers;
tree models get ordinal codes. ``outputs/reports/model_benchmark.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .analysis import fit_scorecard, score
from .data import condition_flags

DRIVERS = ["watch_hours", "last_login_days", "subscription_type", "number_of_profiles",
           "payment_method"]


class Scorecard(ClassifierMixin, BaseEstimator):
    """The six-condition points card as an sklearn estimator (refit per fold)."""

    def fit(self, X, y):
        self.card_ = fit_scorecard(condition_flags(X), pd.Series(np.asarray(y), index=X.index))
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return score(condition_flags(X), self.card_).to_numpy(float)

    def predict_proba(self, X):
        s = self.decision_function(X) - self.card_["cutoff"] + 0.5
        p = 1 / (1 + np.exp(-2 * s))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= self.card_["cutoff"]).astype(int)


def _onehot_scaled(cat, num):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler(), num)])


def _ordinal(cat, num):
    return ColumnTransformer([("cat", OrdinalEncoder(), cat), ("num", "passthrough", num)])


def candidates() -> dict:
    cat, num = C.CATEGORICAL, C.NUMERIC
    rs = C.RANDOM_STATE
    return {
        "Gradient boosting": (make_pipeline(_ordinal(cat, num), GradientBoostingClassifier(random_state=rs)), C.FEATURES),
        "Random forest": (make_pipeline(_ordinal(cat, num), RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1)), C.FEATURES),
        "Decision tree": (make_pipeline(_ordinal(cat, num), DecisionTreeClassifier(random_state=rs)), C.FEATURES),
        "Logistic regression": (make_pipeline(_onehot_scaled(cat, num), LogisticRegression(max_iter=2000)), C.FEATURES),
        "KNN (k=5)": (make_pipeline(_onehot_scaled(cat, num), KNeighborsClassifier(5)), C.FEATURES),
        "Naive Bayes": (make_pipeline(_onehot_scaled(cat, num), GaussianNB()), C.FEATURES),
        "Gradient boosting, 5 drivers only": (
            make_pipeline(_ordinal(["subscription_type", "payment_method"],
                                   ["watch_hours", "last_login_days", "number_of_profiles"]),
                          GradientBoostingClassifier(random_state=rs)), DRIVERS),
        "Six-condition scorecard": (Scorecard(), C.FEATURES),
    }


def benchmark(df: pd.DataFrame) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS,
                                 random_state=C.RANDOM_STATE)
    y = df[C.TARGET]
    rows = []
    for name, (model, cols) in candidates().items():
        r = cross_validate(model, df[cols], y, cv=cv, scoring=["accuracy", "roc_auc", "f1"])
        rows.append({
            "model": name, "features": len(cols),
            "accuracy": r["test_accuracy"].mean(), "accuracy_sd": r["test_accuracy"].std(),
            "roc_auc": r["test_roc_auc"].mean(), "f1": r["test_f1"].mean(),
            "errors_per_1000": (1 - r["test_accuracy"].mean()) * 1000,
            "notebook_accuracy": C.NOTEBOOK_ACCURACY.get(name),
        })
    out = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    out.to_csv(C.REPORT_DIR / "model_benchmark.csv", index=False)
    return out


def border_errors(df: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    """Where the best model's mistakes fall on the scorecard (out-of-fold)."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    model, cols = candidates()["Gradient boosting"]
    pred = cross_val_predict(model, df[cols], df[C.TARGET],
                             cv=StratifiedKFold(C.CV_SPLITS, shuffle=True, random_state=C.RANDOM_STATE))
    wrong = pred != df[C.TARGET].to_numpy()
    out = (pd.DataFrame({"points": s, "wrong": wrong}).groupby("points")
           .agg(customers=("wrong", "size"), gb_errors=("wrong", "sum")).reset_index())
    out.to_csv(C.REPORT_DIR / "gb_errors_by_points.csv", index=False)
    return out


if __name__ == "__main__":
    from .analysis import run_all
    from .data import load

    df, _ = load()
    print(benchmark(df).round(4).to_string())
    print(border_errors(df, run_all(df)["score"]).to_string())
