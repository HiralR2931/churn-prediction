"""End-to-end pipeline for the Netflix churn project.

    load + check data -> what drives churn + scorecard -> model benchmark
    -> README charts -> interactive report data (site/data.js)

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import time

from src import config as C
from src.analysis import run_all as run_analysis
from src.charts import build_charts
from src.data import load
from src.models import benchmark, border_errors
from src.web_report import build_site, summarise


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> None:
    t0 = time.time()

    _hr("1. LOAD & CHECK")
    df, quality = load()
    for k, v in quality.items():
        print(f"  {k:44s}: {v}")

    _hr("2. WHAT DRIVES CHURN")
    res = run_analysis(df)
    for _, r in res["tests"].iterrows():
        print(f"  {'*' if r['matters'] else ' '} {r['feature']:24s} effect={r['effect']:.3f}  p={r['p_value']:.2g}")
    card = res["card"]
    print("  scorecard:", ", ".join(f"{k}+{v}" for k, v in card["points"].items()),
          f"| churn if >= {card['cutoff']}")

    _hr("3. MODEL BENCHMARK (repeated 5-fold CV)")
    bench = benchmark(df)
    for _, r in bench.iterrows():
        print(f"  {r['model']:36s} acc={r['accuracy']:.4f} ± {r['accuracy_sd']:.4f}  auc={r['roc_auc']:.4f}")
    errors = border_errors(df, res["score"])

    _hr("4. CHARTS & REPORT")
    summary = summarise(df, quality, res, bench, errors)
    for p in build_charts(res, bench, quality["churn_rate"]) + build_site(summary, res, bench, errors):
        print(f"  wrote {p.relative_to(C.PROJECT_ROOT)}")

    _hr(f"DONE in {time.time() - t0:.1f}s — open site/index.html")


if __name__ == "__main__":
    main()
