# Medicare County-Level Churn Prediction (Dec 2025 → Feb 2026)

Predicts which U.S. counties will experience a net decline in Medicare
enrollment over a forward three-month window, using CMS Medicare Monthly
Enrollment data (county level, **2013–2025**). The project cleans and validates
13 years of monthly records, engineers demographic/coverage ratios, trains a
Random Forest churn classifier, corroborates its drivers with statistical
tests, and surfaces the results in Plotly dashboards.

*Hackathon project.* Part of the [churn-prediction](../) repo.

**Stack:** Python · pandas · scikit-learn · SciPy · Plotly · PySpark (Databricks)

![Churn drivers and segment churn rates](docs/results.png)

## Results at a glance

| Metric | Value |
|---|---|
| Data span | Jan 2013 – Dec 2025 (511k clean county-months, 3,289 counties) |
| Suppressed cells resolved | ~3.2M `*` cells imputed to CMS midpoint |
| Model | Random Forest (400 trees, balanced classes) |
| Held-out test (Dec 2024 → Feb 2025) | **ROC-AUC 0.85**, PR-AUC 0.46 (base churn rate 16%) |
| Churn-class recall / precision | 0.82 / 0.36 (balanced class weights favor catching at-risk counties) |
| Forecast | 3,278 counties scored for Dec 2025 → Feb 2026; 1,221 flagged at risk |

Top churn drivers (permutation importance + point-biserial correlation): county
size, prior-window enrollment change, Medicare Advantage share, and the
prescription-drug / dual-eligible mix — all confirmed by two-proportion z-tests
at p < 0.01.

## How churn is defined

For each county at **December of year _Y_**, the model observes an enrollment
snapshot (features). The outcome is measured at **February of year _Y+1_**:

> `churn = 1` if total beneficiaries decline by more than 0.5% across the window.

- **Train:** base years 2013–2023 (known outcomes)
- **Test:** Dec 2024 → Feb 2025 (held out — temporal, no leakage)
- **Forecast:** Dec 2025 → Feb 2026 (no ground truth yet — the deliverable)

## Engineered features

Composition ratios (share of total beneficiaries) plus short-term dynamics:

- **Headline ratios:** `aged_ratio`, `disabled_ratio`, `dual_ratio`
- **Coverage/eligibility:** `ma_ratio`, `full_dual_ratio`, `esrd_ratio`, `rx_ratio`, `lis_ratio`
- **Demographics:** `female_ratio`, `white/black/hispanic/api_ratio`, `age_85_plus_ratio`
- **Dynamics:** `momentum_3m` (Sep→Dec change), `prior_window_change` (last year's Dec→Feb), `log_tot_benes`

## Project layout

```
medicare-churn/
├── run_pipeline.py              # end-to-end orchestrator
├── src/
│   ├── config.py                # paths, churn window, feature list, RF params
│   ├── data_cleaning.py         # load, resolve suppressed cells, validate
│   ├── feature_engineering.py   # composition ratios + trend momentum
│   ├── churn_label.py           # Dec→Feb labelling + temporal split
│   ├── model.py                 # Random Forest train / eval / importance / forecast
│   ├── stats_analysis.py        # segment z-tests + point-biserial correlation
│   └── dashboards.py            # Plotly figures + index.html dashboard
├── notebooks/
│   ├── medicare_churn_databricks.py          # PySpark cleaning/feature stage
│   └── medicare_churn_ALLINONE_databricks.py # whole pipeline in one notebook
├── docs/results.png             # README chart
├── data/                        # place the CMS CSV here (git-ignored)
└── outputs/                     # figures + reports (committed), model (generated)
```

## Run it

```bash
pip install -r requirements.txt
# download the CMS extract (see Data note) to data/medicare_enrollment_2013_2025.csv
python run_pipeline.py
open outputs/figures/index.html      # dashboard
```

Runs end-to-end in ~15s on a laptop. Individual stages are runnable too, e.g.
`python -m src.data_cleaning` or `python -m src.stats_analysis`.

## Outputs

- `outputs/figures/index.html` — dashboard (enrollment trends, state churn-risk
  map, top-risk counties, feature importance, segment churn rates)
- `outputs/reports/forecast_dec2025_feb2026.csv` — per-county churn probabilities
- `outputs/reports/model_metrics.json`, `feature_importance.csv`,
  `segment_ztests.csv`, `run_summary.json`
- `outputs/models/random_forest_churn.joblib` (not in the repo; created when you run the pipeline)

The figures are standalone HTML files; download one and open it in a browser to
explore it, since GitHub shows HTML as source.

## Databricks

`notebooks/medicare_churn_databricks.py` is the Databricks-source version of the
cleaning + feature-engineering stage in PySpark (suppression handling, validation
guards, windowed trend features, Delta-table output). The modelling and dashboard
steps reuse `src/` on the collected pandas frame.

## Data note

The CMS [Medicare Monthly Enrollment](https://data.cms.gov/summary-statistics-on-beneficiary-enrollment/medicare-and-medicaid-reports/medicare-monthly-enrollment)
file is public (download the full CSV; ~564k rows). Beneficiary counts of 1–10
are suppressed with `*`; this project imputes them to the range midpoint (5) and
retains a `had_suppression` flag for auditability. Counts are enrollment
snapshots, not longitudinal beneficiary tracking, so "churn" here means **net
county-level enrollment decline**, not individual disenrollment.
