"""Plotly dashboards surfacing county-level Medicare enrollment trends.

Produces self-contained HTML figures plus an ``index.html`` that stitches them
into one dashboard. Colors follow a CVD-validated categorical palette; the
churn-risk map uses a single-hue sequential blue ramp.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C

# CVD-validated palette (light surface).
PAL = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
    "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "surface": "#fcfcfb",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
              color=PAL["ink"], size=13),
    paper_bgcolor=PAL["surface"],
    plot_bgcolor=PAL["surface"],
    margin=dict(l=60, r=30, t=70, b=50),
    title_font_size=18,
)


def _national_monthly(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the county panel to a national monthly series."""
    cols = ["TOT_BENES", "ORGNL_MDCR_BENES", "MA_AND_OTH_BENES",
            "AGED_TOT_BENES", "DSBLD_TOT_BENES", "DUAL_TOT_BENES"]
    return clean_df.groupby("date")[cols].sum().reset_index()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_enrollment_trend(clean_df: pd.DataFrame) -> go.Figure:
    """Original Medicare vs Medicare Advantage over time — the churn story."""
    nat = _national_monthly(clean_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nat["date"], y=nat["ORGNL_MDCR_BENES"] / 1e6, name="Original Medicare",
        mode="lines", line=dict(color=PAL["blue"], width=2)))
    fig.add_trace(go.Scatter(
        x=nat["date"], y=nat["MA_AND_OTH_BENES"] / 1e6, name="Medicare Advantage",
        mode="lines", line=dict(color=PAL["orange"], width=2)))
    fig.update_layout(
        title="National enrollment: Original Medicare vs Medicare Advantage (2013–2025)",
        xaxis_title="Month", yaxis_title="Beneficiaries (millions)",
        legend=dict(orientation="h", y=1.08, x=0), **_LAYOUT)
    fig.update_xaxes(gridcolor=PAL["grid"])
    fig.update_yaxes(gridcolor=PAL["grid"])
    return fig


def fig_segment_trend(clean_df: pd.DataFrame) -> go.Figure:
    """Aged / Disabled / Dual-eligible beneficiary trends."""
    nat = _national_monthly(clean_df)
    fig = go.Figure()
    for col, name, color in [
        ("AGED_TOT_BENES", "Aged", PAL["blue"]),
        ("DSBLD_TOT_BENES", "Disabled", PAL["violet"]),
        ("DUAL_TOT_BENES", "Dual-eligible", PAL["aqua"]),
    ]:
        fig.add_trace(go.Scatter(x=nat["date"], y=nat[col] / 1e6, name=name,
                                 mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(
        title="Enrollment by beneficiary segment (2013–2025)",
        xaxis_title="Month", yaxis_title="Beneficiaries (millions)",
        legend=dict(orientation="h", y=1.08, x=0), **_LAYOUT)
    fig.update_xaxes(gridcolor=PAL["grid"])
    fig.update_yaxes(gridcolor=PAL["grid"])
    return fig


def fig_state_churn_map(forecast_df: pd.DataFrame) -> go.Figure:
    """State-level choropleth of mean predicted churn probability (offline-safe)."""
    st = (forecast_df.groupby("state")
          .apply(lambda d: pd.Series({
              "mean_prob": (d["churn_probability"] * d["TOT_BENES"]).sum() / d["TOT_BENES"].sum(),
              "counties": len(d),
          }), include_groups=False)
          .reset_index())
    fig = go.Figure(go.Choropleth(
        locations=st["state"], z=st["mean_prob"] * 100, locationmode="USA-states",
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar_title="Churn risk (%)",
        marker_line_color=PAL["surface"], marker_line_width=0.5,
        hovertemplate="%{location}<br>Enrollment-weighted churn risk: %{z:.1f}%<extra></extra>"))
    fig.update_layout(
        title="Predicted churn risk by state (Dec 2025 → Feb 2026)",
        geo=dict(scope="usa", bgcolor=PAL["surface"], lakecolor=PAL["surface"]),
        **{k: v for k, v in _LAYOUT.items() if k != "plot_bgcolor"})
    return fig


def fig_top_counties(forecast_df: pd.DataFrame, n: int = 25) -> go.Figure:
    """Highest-risk counties (min size 1,000 benes to avoid noise)."""
    d = forecast_df[forecast_df["TOT_BENES"] >= 1000].head(n).iloc[::-1]
    label = d["county"] + ", " + d["state"]
    fig = go.Figure(go.Bar(
        x=d["churn_probability"] * 100, y=label, orientation="h",
        marker_color=PAL["red"],
        hovertemplate="%{y}<br>Churn probability: %{x:.1f}%<extra></extra>"))
    fig.update_layout(
        title=f"Top {n} counties by predicted churn risk (≥1,000 benes)",
        xaxis_title="Churn probability (%)", yaxis_title="",
        height=680, **_LAYOUT)
    fig.update_xaxes(gridcolor=PAL["grid"])
    return fig


def fig_feature_importance(importance: pd.DataFrame) -> go.Figure:
    """Permutation feature importance — the model's churn drivers."""
    d = importance.sort_values("permutation_importance").tail(14)
    fig = go.Figure(go.Bar(
        x=d["permutation_importance"], y=d["feature"], orientation="h",
        error_x=dict(type="data", array=d["permutation_std"], color=PAL["muted"]),
        marker_color=PAL["blue"],
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>"))
    fig.update_layout(
        title="Churn drivers — permutation importance (held-out test)",
        xaxis_title="Mean decrease in ROC-AUC", yaxis_title="", **_LAYOUT)
    fig.update_xaxes(gridcolor=PAL["grid"])
    return fig


def fig_segment_churn_rates(segment_stats: pd.DataFrame) -> go.Figure:
    """Historical churn rate across demographic segments (bar w/ CI)."""
    d = segment_stats.sort_values("churn_rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["churn_rate"] * 100, y=d["segment"], orientation="h",
        marker_color=PAL["violet"],
        error_x=dict(type="data", array=d["ci95"] * 100, color=PAL["muted"]),
        hovertemplate="%{y}<br>Churn rate: %{x:.1f}%<extra></extra>"))
    fig.update_layout(
        title="Historical churn rate by county segment (95% CI)",
        xaxis_title="Churn rate (%)", yaxis_title="", **_LAYOUT)
    fig.update_xaxes(gridcolor=PAL["grid"])
    return fig


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_dashboard(clean_df, forecast_df, importance, segment_stats) -> list:
    """Write all figures + an index.html dashboard. Returns written paths."""
    figs = {
        "01_enrollment_trend.html": fig_enrollment_trend(clean_df),
        "02_segment_trend.html": fig_segment_trend(clean_df),
        "03_state_churn_map.html": fig_state_churn_map(forecast_df),
        "04_top_counties.html": fig_top_counties(forecast_df),
        "05_feature_importance.html": fig_feature_importance(importance),
        "06_segment_churn_rates.html": fig_segment_churn_rates(segment_stats),
    }
    written = []
    for name, fig in figs.items():
        path = C.FIG_DIR / name
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
        written.append(path)

    index = C.FIG_DIR / "index.html"
    cards = "\n".join(
        f'<section><iframe src="{name}" loading="lazy"></iframe></section>'
        for name in figs
    )
    index.write_text(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medicare County-Level Churn Dashboard</title>
<style>
  body {{ margin:0; background:#f9f9f7; color:#0b0b0b;
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
  header {{ padding:28px 32px 8px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  p.sub {{ color:#52514e; margin:0 0 8px; }}
  main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr));
          gap:20px; padding:20px 32px 40px; }}
  section {{ background:#fcfcfb; border:1px solid rgba(11,11,11,0.10);
             border-radius:12px; overflow:hidden; }}
  iframe {{ width:100%; height:520px; border:0; }}
</style></head><body>
<header>
  <h1>Medicare County-Level Churn Prediction</h1>
  <p class="sub">Enrollment trends 2013–2025 and predicted churn for the
     Dec 2025 → Feb 2026 window. Source: CMS Medicare Monthly Enrollment.</p>
</header>
<main>{cards}</main>
</body></html>""")
    written.append(index)
    return written
