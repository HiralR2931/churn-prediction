"""Tests for the Netflix pipeline on tiny made-up samples (no Kaggle download needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config as C
from src.analysis import fit_scorecard, risk_tiers, score
from src.data import condition_flags, load
from src.models import Scorecard

CARD = {"watch_lt5": 6, "watch_5_20": 3, "login_gt30": 3,
        "basic_plan": 1, "profiles_lt4": 1, "pay_crypto_gift": 1}


def make_customers(n: int = 600, seed: int = 0, cutoff: int = 7) -> pd.DataFrame:
    """Customers whose churn label follows CARD exactly."""
    rng = np.random.default_rng(seed)
    plans = rng.choice(["Basic", "Standard", "Premium"], n)
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "age": rng.integers(18, 71, n),
        "gender": rng.choice(["Male", "Female", "Other"], n),
        "subscription_type": plans,
        "watch_hours": rng.uniform(0, 40, n).round(2),
        "last_login_days": rng.integers(0, 61, n),
        "region": rng.choice(["Asia", "Europe"], n),
        "device": rng.choice(["TV", "Mobile"], n),
        "monthly_fee": pd.Series(plans).map({"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}).to_numpy(),
        "payment_method": rng.choice(["Credit Card", "PayPal", "Crypto", "Gift Card"], n),
        "number_of_profiles": rng.integers(1, 6, n),
        "favorite_genre": rng.choice(["Drama", "Comedy"], n),
    })
    df["avg_watch_time_per_day"] = (df["watch_hours"] / (df["last_login_days"] + 1)).round(2)
    pts = sum(condition_flags(df)[k] * v for k, v in CARD.items())
    df["churned"] = (pts >= cutoff).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Conditions: exact cut-points
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hours, lt5, mid", [(4.99, 1, 0), (5.0, 0, 1), (19.99, 0, 1), (20.0, 0, 0)])
def test_watch_hour_cut_points(hours, lt5, mid):
    f = condition_flags(pd.DataFrame({"watch_hours": [hours], "last_login_days": [0],
                                      "subscription_type": ["Premium"], "number_of_profiles": [5],
                                      "payment_method": ["PayPal"]})).iloc[0]
    assert (f["watch_lt5"], f["watch_5_20"]) == (lt5, mid)


@pytest.mark.parametrize("days, flag", [(30, 0), (31, 1)])
def test_login_cut_point(days, flag):
    f = condition_flags(pd.DataFrame({"watch_hours": [30.0], "last_login_days": [days],
                                      "subscription_type": ["Premium"], "number_of_profiles": [5],
                                      "payment_method": ["PayPal"]})).iloc[0]
    assert f["login_gt30"] == flag


def test_watch_conditions_never_stack():
    f = condition_flags(make_customers())
    assert not ((f["watch_lt5"] == 1) & (f["watch_5_20"] == 1)).any()


# --------------------------------------------------------------------------- #
# Scorecard: fitting, scoring, tiers
# --------------------------------------------------------------------------- #
def test_fit_recovers_a_perfect_rule():
    df = make_customers()
    card = fit_scorecard(condition_flags(df), df["churned"])
    assert card["train_accuracy"] == 1.0
    pred = (score(condition_flags(df), card) >= card["cutoff"]).astype(int)
    assert (pred == df["churned"]).all()


def test_score_adds_points_for_met_conditions():
    one = pd.DataFrame({"watch_hours": [3.0], "last_login_days": [40], "subscription_type": ["Basic"],
                        "number_of_profiles": [2], "payment_method": ["Crypto"]})
    assert score(condition_flags(one), {"points": CARD, "cutoff": 7}).iloc[0] == 6 + 3 + 1 + 1 + 1


def test_risk_tiers_cover_everyone_once():
    df = make_customers()
    card = {"points": CARD, "cutoff": 7}
    tiers = risk_tiers(df, score(condition_flags(df), card), card)
    assert list(tiers["tier"]) == ["Low", "Watch", "High", "Very high"]
    assert tiers["n"].sum() == len(df)
    assert tiers["share_of_churners"].sum() == pytest.approx(1.0)
    assert tiers.loc[tiers["tier"] == "Low", "churn_rate"].iloc[0] == 0.0     # 0-4 points never churn here


def test_scorecard_estimator_cross_validates():
    df = make_customers(400, seed=1)
    acc = cross_val_score(Scorecard(), df[C.FEATURES], df["churned"],
                          cv=StratifiedKFold(4, shuffle=True, random_state=0))
    assert acc.min() == 1.0


# --------------------------------------------------------------------------- #
# Data checks
# --------------------------------------------------------------------------- #
def test_data_checks_spot_derived_column_and_fee(tmp_path):
    df = make_customers(200)
    df.loc[0, ["watch_hours", "last_login_days"]] = [33.0, 0]
    df.loc[0, "avg_watch_time_per_day"] = 33.0          # a "> 10" value that is correct
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    _, report = load(path)
    assert report["avg_watch_equals_watch_over_login_plus_1"] == 1.0
    assert report["rows_changed_by_notebook_div10"] >= 1
    assert report["monthly_fee_fixed_by_plan"] is True
