"""Static PNG charts for the README (GitHub shows images, not HTML).

Every chart is built from the ``ma_switching`` outputs so the README numbers and
the pictures come from the same run. Colors are the validated reference palette:
blue = Medicare Advantage / the chosen method, orange = Original Medicare.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT, RED = "#9ec5f4", "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlelocation": "left",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 26,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
})


def _clean(ax, grid_axis="y") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def chart_tipping_point(nat: pd.DataFrame, path) -> None:
    """MA vs Original Medicare enrollment, and the size of each switching window."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.38})

    a.plot(nat["year"], nat["ma_millions_dec"], color=BLUE, lw=2.2, solid_capstyle="round")
    a.plot(nat["year"], nat["om_millions_dec"], color=ORANGE, lw=2.2, solid_capstyle="round")
    last = nat.iloc[-1]
    for col, color, name, dy in [("ma_millions_dec", BLUE, "Medicare Advantage", 9),
                                 ("om_millions_dec", ORANGE, "Original Medicare", -9)]:
        a.scatter(last["year"], last[col], s=46, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        a.annotate(f"{name} {last[col]:.1f}M", (last["year"], last[col]),
                   xytext=(8, dy), textcoords="offset points", va="center", fontsize=10, color=INK)
    cross = nat[nat["ma_millions_dec"] > nat["om_millions_dec"]].iloc[0]
    a.axvline(cross["year"], color=BASE, lw=1)
    a.text(cross["year"] - 0.2, nat["om_millions_dec"].max() * 1.005,
           f"MA passes Original\nMedicare, Dec {int(cross['year'])}", ha="right", va="top",
           fontsize=9.5, color=INK2)
    a.set_xlim(nat["year"].min() - 0.3, nat["year"].max() + 4.6)
    a.set_xticks(range(int(nat["year"].min()), int(nat["year"].max()) + 1, 2))
    a.spines["bottom"].set_bounds(nat["year"].min() - 0.3, nat["year"].max() + 0.3)
    a.set_ylim(10, 42)
    a.set_ylabel("Beneficiaries each December (millions)")
    a.set_title("Medicare Advantage became the majority")
    _subtitle(a, "U.S. totals each December, CMS Medicare Monthly Enrollment")
    _clean(a)

    w = nat.dropna(subset=["county_median_shift_pp"])
    b.bar(w["year"], w["county_median_shift_pp"], width=0.6, color=BLUE)
    for _, r in w[w["year"].isin([int(w.loc[w["county_median_shift_pp"].idxmax(), "year"]),
                                  int(w["year"].max())])].iterrows():
        b.annotate(f"+{r['county_median_shift_pp']:.1f} pp", (r["year"], r["county_median_shift_pp"]),
                   xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, color=INK)
    b.set_xticks(w["year"], [f"'{str(y)[2:]}" for y in w["year"]])
    b.set_ylabel("Median county change in MA share (pp)")
    b.set_title("…but the yearly switching wave has stalled")
    _subtitle(b, "Change in MA share from December to February, median U.S. county")
    _clean(b)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_where_and_who(sat: pd.DataFrame, corr: pd.DataFrame, path) -> None:
    """The adoption curve and the county traits linked to faster switching."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.45})

    mid = (sat["ma_share_from"] + sat["ma_share_to"].clip(upper=80)) / 2
    a.fill_between(mid, sat["q25"], sat["q75"], color=BLUE, alpha=0.10, lw=0)
    a.plot(mid, sat["median"], color=BLUE, lw=2.2, marker="o", ms=6,
           markeredgecolor=SURFACE, markeredgewidth=2)
    a.axhline(0, color=BASE, lw=1)
    a.text(mid.iloc[-1], 0.08, "median county", ha="right", va="bottom", fontsize=9.5, color=INK2)
    labels = [f"{f:.0f}–{t:.0f}%" if t < 100 else f"{f:.0f}%+"
              for f, t in zip(sat["ma_share_from"], sat["ma_share_to"])]
    a.set_xticks(mid, labels, fontsize=9.5)
    a.set_xlabel("County MA share going into the window")
    a.set_ylabel("Switching vs. the median county (pp)")
    a.set_title("Switching rises, peaks, then saturates")
    _subtitle(a, "Median county (line) and middle half (band), 2015–2024 windows")
    _clean(a)

    names = {"lis_ratio": "Low-income subsidy share", "dual_ratio": "Dual-eligible share",
             "disabled_ratio": "Disabled share", "black_ratio": "Black beneficiary share",
             "log_tot_benes": "County size", "ma_ratio": "Current MA share",
             "hispanic_ratio": "Hispanic beneficiary share", "age_85_plus_ratio": "Age 85+ share"}
    d = corr.assign(label=corr["feature"].map(names)).sort_values("rho")
    colors = [BLUE if r > 0 else RED for r in d["rho"]]
    b.barh(d["label"], d["rho"], height=0.55, color=colors)
    for y, r in enumerate(d["rho"]):
        b.text(r + (0.008 if r > 0 else -0.008), y, f"{r:+.2f}", va="center",
               ha="left" if r > 0 else "right", fontsize=9.5, color=INK)
    b.axvline(0, color=BASE, lw=1)
    b.set_xlim(-0.3, 0.4)
    b.set_xlabel("Spearman correlation with relative switching\n(median of 10 seasons, 2015–2024)")
    b.set_title("Lower-income counties switch faster")
    _subtitle(b, "Blue: switches more than average · Red: less")
    _clean(b, grid_axis="x")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_states(states: pd.DataFrame, path) -> None:
    """Dumbbell: each state's median switching, previous vs latest window."""
    d = states.dropna(subset=["shift_pp", "shift_prev_pp"]).sort_values("shift_pp")
    fig, ax = plt.subplots(figsize=(7.5, 12))
    y = range(len(d))
    ax.hlines(y, d["shift_pp"], d["shift_prev_pp"], color=BASE, lw=1.4)
    ax.scatter(d["shift_prev_pp"], y, s=40, facecolor=SURFACE, edgecolor=MUTED, lw=1.5,
               zorder=3, label="Dec 2023 → Feb 2024")
    ax.scatter(d["shift_pp"], y, s=48, color=BLUE, edgecolor=SURFACE, lw=1.5,
               zorder=4, label="Dec 2024 → Feb 2025")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(list(y), d["state_name"], fontsize=9)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("Median county change in MA share (pp)")
    ax.set_title("Switching slowed almost everywhere")
    _subtitle(ax, "Median county per state; Vermont is the only reversal")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    _clean(ax, grid_axis="x")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_backtest(bt: pd.DataFrame, path, scored: pd.DataFrame | None = None) -> None:
    """Walk-forward rank accuracy of each method, plus the season scored on new data."""
    p = bt.pivot(index="year", columns="method", values="spearman")
    order = [("Blend (county + state)", BLUE), ("County's last window", AQUA),
             ("State's last window", YELLOW), ("Gradient boosting", ORANGE)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    last = int(p.index.max())
    for name, color in order:
        lw = 3 if name.startswith("Blend") else 2
        ax.plot(p.index, p[name], color=color, lw=lw, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{name}  (backtest avg {p[name].mean():.2f})")
        if scored is not None:
            v = scored.set_index("method").loc[name, "spearman"]
            ax.plot([last, last + 1], [p.loc[last, name], v], color=color, lw=lw, ls=(0, (2, 2)))
            ax.scatter(last + 1, v, s=46, color=color, edgecolor=SURFACE, lw=2, zorder=4)
    if scored is not None:
        ax.axvspan(last + 0.5, last + 1.5, color=GRID, alpha=0.5, lw=0)
        ax.text(last + 1, 0.72, "scored on\nnew CMS data", ha="center", va="top", fontsize=9.5, color=INK2)
        ax.set_xlim(p.index.min() - 0.4, last + 1.5)
    ticks = list(p.index) + ([last + 1] if scored is not None else [])
    ax.set_xticks(ticks, [f"'{str(y)[2:]}–'{str(y + 1)[2:]}" for y in ticks])
    ax.set_ylim(0.1, 0.75)
    ax.set_ylabel("Spearman rank correlation, predicted vs actual")
    ax.set_title("Every method weakened when the forecast met real 2026 data"
                 if scored is not None else "A simple blend beats the machine-learning model")
    _subtitle(ax, "Walk-forward backtest, then the Dec 2025 → Feb 2026 forecast scored on the July 2026 release"
              if scored is not None else "Walk-forward backtest: each year forecast using earlier years only")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_reversals(states: pd.DataFrame, path, n: int = 15) -> None:
    """The states where switching reversed in the scored season."""
    d = states.sort_values("shift_pp").head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = range(len(d))
    ax.hlines(y, d["shift_pp"], d["shift_prev_pp"], color=BASE, lw=1.4)
    ax.scatter(d["shift_prev_pp"], y, s=40, facecolor=SURFACE, edgecolor=MUTED, lw=1.5, zorder=3,
               label="Dec 2024 → Feb 2025")
    ax.scatter(d["shift_pp"], y, s=48, color=RED, edgecolor=SURFACE, lw=1.5, zorder=4,
               label="Dec 2025 → Feb 2026")
    for yi, v in zip(y, d["shift_pp"]):
        ax.text(v - 0.4, yi, f"{v:+.2f}" if abs(v) < 0.1 else f"{v:+.1f}", ha="right", va="center",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(list(y), d["state_name"], fontsize=9.5)
    ax.set_xlim(min(-20, d["shift_pp"].min() - 2.5), 3)
    ax.set_xlabel("Median county change in MA share (pp)")
    ax.set_title("In 2026 the reversal spread beyond Vermont")
    _subtitle(ax, f"The {len(d)} states where the median county's MA share fell")
    ax.legend(loc="center left", frameon=False, fontsize=9.5)
    _clean(ax, grid_axis="x")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_charts(ma: dict, scored: dict | None = None) -> list:
    """Write the README charts to ``docs/``. Returns written paths."""
    C.DOCS_DIR.mkdir(exist_ok=True)
    out = {
        "01_tipping_point.png": lambda p: chart_tipping_point(ma["national"], p),
        "02_where_and_who.png": lambda p: chart_where_and_who(ma["saturation"], ma["correlates"], p),
        "03_states.png": lambda p: chart_states(ma["states"], p),
        "04_backtest.png": lambda p: chart_backtest(ma["backtest"], p,
                                                    scored["methods"] if scored else None),
    }
    if scored:
        n_rev = len(scored["summary"]["reversed_states"])
        out["05_reversals.png"] = lambda p: chart_reversals(scored["states"], p, n_rev)
    written = []
    for name, draw in out.items():
        path = C.DOCS_DIR / name
        draw(path)
        written.append(path)
    return written


if __name__ == "__main__":
    from .data_cleaning import load_clean, load_national
    from .feature_engineering import engineer
    from .ma_switching import run_all

    clean_df, _ = load_clean()
    for p in build_charts(run_all(engineer(clean_df), load_national())):
        print("wrote", p.relative_to(C.PROJECT_ROOT))
