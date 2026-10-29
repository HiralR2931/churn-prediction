"""Static PNG charts for the README, in the same style as the Medicare project."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.titlelocation": "left", "axes.titlesize": 14, "axes.titlepad": 26,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
})

NAMES = {"subscription_type": "Plan", "payment_method": "Payment", "number_of_profiles": "Profiles",
         "gender": "Gender", "region": "Region", "device": "Device", "favorite_genre": "Genre"}


def _clean(ax, grid_axis="y") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def _pct(ax, axis="y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def chart_cliffs(w: pd.DataFrame, lg: pd.DataFrame, path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.25})
    x = w["from"] + 0.5
    a.fill_between(x, w["ci_low"], w["ci_high"], color=BLUE, alpha=0.10, lw=0, step="mid")
    a.step(x, w["churn_rate"], where="mid", color=BLUE, lw=2.2)
    for cut in (5, 20):
        a.axvline(cut, color=INK2, lw=1)
        a.text(cut + 0.4, 0.97, f"{cut} h", color=INK2, fontsize=10, va="top")
    a.set_xlim(0, 31)
    a.set_xticks([0, 5, 10, 15, 20, 25, 30], ["0", "5", "10", "15", "20", "25", "30+"])
    a.set_ylim(0, 1)
    _pct(a)
    a.set_xlabel("Watch hours")
    a.set_ylabel("Share who churned")
    a.set_title("Churn falls off two cliffs in watch hours")
    _subtitle(a, "1-hour bins with 95% intervals, 5,000 customers")
    _clean(a)

    x = (lg["from"] + lg["to"] + 1) / 2
    b.fill_between(x, lg["ci_low"], lg["ci_high"], color=ORANGE, alpha=0.12, lw=0, step="mid")
    b.step(x, lg["churn_rate"], where="mid", color=ORANGE, lw=2.2)
    b.axvline(30.5, color=INK2, lw=1)
    b.text(31.2, 0.97, "30 days", color=INK2, fontsize=10, va="top")
    b.set_xlim(0, 61)
    b.set_ylim(0, 1)
    _pct(b)
    b.set_xlabel("Days since last login")
    b.set_title("…and jumps once a login is a month old")
    _subtitle(b, "3-day bins with 95% intervals")
    _clean(b)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_scorecard(points: pd.DataFrame, card: dict, path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1.25]})
    a.axis("off")
    a.set_title("The six-condition scorecard")
    _subtitle(a, f"Add the points; {card['cutoff']} or more predicts churn")
    labels = {n: l for n, l, _ in C.CONDITIONS}
    y = 0.86
    for name, pts in card["points"].items():
        a.text(0.0, y, labels[name], fontsize=12, va="center", transform=a.transAxes)
        a.text(0.86, y, f"+{pts}", fontsize=12, va="center", ha="right", transform=a.transAxes,
               fontweight="bold")
        a.plot([0, 0.88], [y - 0.07, y - 0.07], color=GRID, lw=1, transform=a.transAxes)
        y -= 0.14
    a.text(0.0, y + 0.01, "Watch-hour conditions don't stack; use one of the two.", fontsize=9.5,
           color=INK2, transform=a.transAxes, va="center")

    colors = [ORANGE if p >= card["cutoff"] else BLUE for p in points["points"]]
    b.bar(points["points"], points["churn_rate"], width=0.6, color=colors)
    for _, r in points.iterrows():
        b.text(r["points"], r["churn_rate"] + 0.02, f"{r['churn_rate']:.0%}", ha="center", fontsize=9, color=INK)
    b.axvline(card["cutoff"] - 0.5, color=INK2, lw=1)
    b.text(card["cutoff"] - 0.4, 1.1, "predict churn →", fontsize=10, color=INK2)
    b.set_xticks(points["points"])
    b.set_ylim(0, 1.15)
    _pct(b)
    b.set_xlabel("Total points")
    b.set_ylabel("Share who churned")
    b.set_title("Churn by total points")
    _subtitle(b, "All 5,000 customers")
    _clean(b)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_what_matters(effects: pd.DataFrame, tests: pd.DataFrame, overall: float, path) -> None:
    order = ["subscription_type", "payment_method", "number_of_profiles",
             "gender", "region", "device", "favorite_genre"]
    p = tests.set_index("feature")["p_value"]
    rows, labels, groups = [], [], []
    for f in order:
        d = effects[effects["feature"] == f].sort_values("churn_rate")
        for _, r in d.iterrows():
            rows.append(r)
            labels.append(r["value"])
            groups.append(f)
        rows.append(None)
        labels.append("")
        groups.append(None)
    rows, labels, groups = rows[:-1], labels[:-1], groups[:-1]
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    ys = np.arange(len(rows))[::-1]
    for y, r, g in zip(ys, rows, groups):
        if r is None:
            continue
        color = BLUE if p[g] < 0.01 else MUTED
        ax.hlines(y, r["ci_low"], r["ci_high"], color=color, lw=2, alpha=0.6)
        ax.scatter(r["churn_rate"], y, s=42, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.axvline(overall, color=INK2, lw=1)
    ax.text(overall + 0.005, ys[0] + 1.2, f"all customers {overall:.0%}", fontsize=9.5, color=INK2)
    ax.set_yticks(ys, labels, fontsize=9.5)
    starts = {}
    for y, g in zip(ys, groups):
        if g and g not in starts:
            starts[g] = y
    for g, y in starts.items():
        tag = f"{NAMES[g]}  ·  p = {p[g]:.1g}" if p[g] < 0.01 else f"{NAMES[g]}  ·  no effect (p = {p[g]:.2f})"
        ax.text(0.205, y + 0.55, tag, fontsize=10, fontweight="bold", color=INK if p[g] < 0.01 else INK2)
    ax.set_xlim(0.2, 0.7)
    _pct(ax, "x")
    ax.set_xlabel("Share who churned (95% interval)")
    ax.set_title("Three features matter; four don't")
    _subtitle(ax, "Blue: churn differs between values (chi-square p < 0.01) · Grey: it doesn't")
    _clean(ax, grid_axis="x")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_models(bench: pd.DataFrame, path) -> None:
    d = bench.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(d))
    colors = [ORANGE if m == "Six-condition scorecard" else BLUE for m in d["model"]]
    ax.errorbar(d["accuracy"], y, xerr=d["accuracy_sd"], fmt="none", ecolor=INK2, lw=1, capsize=3, zorder=2)
    ax.scatter(d["accuracy"], y, s=70, color=colors, edgecolor=SURFACE, lw=2, zorder=3,
               label="Cross-validated (this pipeline)")
    nb = d["notebook_accuracy"]
    # one row-third below its model so it never hides the cross-validated dot
    ax.scatter(nb, y - 0.32, s=26, marker="D", facecolor=SURFACE, edgecolor=INK, lw=1.2, zorder=4,
               label="Original notebook, one split")
    for yi, (acc, sd) in enumerate(zip(d["accuracy"], d["accuracy_sd"])):
        ax.text(acc + sd + 0.003, yi, f"{acc:.1%}", va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(y, d["model"])
    ax.set_ylim(-0.7, len(d) - 0.5)
    ax.set_xlim(0.8, 1.02)
    ax.set_xticks(np.arange(0.80, 1.001, 0.05))
    _pct(ax, "x")
    ax.set_xlabel("Accuracy, repeated 5-fold cross-validation (± 1 sd)")
    ax.set_title("A six-line scorecard gets within 3.5 points of the best model")
    _subtitle(ax, "Same folds for every model; orange is the scorecard")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    _clean(ax, grid_axis="x")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_charts(res: dict, bench: pd.DataFrame, overall: float) -> list:
    out = {
        "01_cliffs.png": lambda p: chart_cliffs(res["cliff_watch"], res["cliff_login"], p),
        "02_scorecard.png": lambda p: chart_scorecard(res["points"], res["card"], p),
        "03_what_matters.png": lambda p: chart_what_matters(res["effects"], res["tests"], overall, p),
        "04_models.png": lambda p: chart_models(bench, p),
    }
    written = []
    for name, draw in out.items():
        path = C.DOCS_DIR / name
        draw(path)
        written.append(path)
    return written
