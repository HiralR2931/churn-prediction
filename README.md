# Churn Prediction

[![Tests](https://github.com/HiralR2931/churn-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/HiralR2931/churn-prediction/actions/workflows/tests.yml)

Two churn projects with the same question at different scales: who is about to
leave? One follows people leaving Original Medicare for private Medicare
Advantage plans across every U.S. county. The other predicts churn for
individual Netflix subscribers.

| Project | Scale | Data | Model | Headline result |
|---|---|---|---|---|
| [The Medicare Advantage Stall](medicare-churn/) · [interactive report](https://hiralr2931.github.io/churn-prediction/) | 3,236 U.S. counties, 2013–2025 | CMS Medicare Monthly Enrollment (564k rows) | Walk-forward backtest of 4 forecasters, Random Forest, PySpark on Databricks | MA passed 50% of beneficiaries; yearly switching fell from +3.1 to +0.9 pp; the Dec 2025 → Feb 2026 forecast, scored on later CMS data, caught the direction but not the state-wide reversals (Spearman 0.20) |
| [The Six-Line Churn Rule](netflix-churn/) · [interactive report](https://hiralr2931.github.io/churn-prediction/netflix/) | 5,000 subscribers | Kaggle Netflix churn dataset (synthetic, 12 features) | 6 classifiers + a points scorecard, repeated 5-fold CV | A six-condition scorecard matches 95.5% of outcomes; gradient boosting reaches 99.1%, and only wins near the scorecard's cut-off |

![Medicare Advantage passed Original Medicare while the yearly switching wave fell](medicare-churn/docs/01_tipping_point.png)

![Netflix churn drops in steps at 5 and 20 watch hours and jumps after 30 days without a login](netflix-churn/docs/01_cliffs.png)

## What's in each folder

**[`medicare-churn/`](medicare-churn/)** is a full pipeline. It cleans 13 years of
county-level CMS data and resolves ~3.2M suppressed cells. It then measures
each county's December → February move from Original Medicare to Medicare
Advantage, and finds the adoption curve and the 2024 stall. Four forecasting
methods are compared in a walk-forward backtest, and the pipeline publishes an
interactive D3 report with a county map. The 2026 forecast is then scored
against February 2026 data from a later CMS release. An earlier net-decline Random Forest
is kept alongside, with a feature bug fixed. The cleaning and feature stage also
runs in PySpark on Databricks with Delta tables.

**[`netflix-churn/`](netflix-churn/)** is a pipeline that decodes the rule behind
the dataset's churn label. It finds the cliffs at 5 and 20 watch hours and 30 days
without a login, and shows that demographics carry no signal. It then fits a
six-condition points scorecard and benchmarks it against six classifiers on the same
repeated cross-validation, and publishes an interactive report with a scorecard
calculator. It also fixes a cleaning step in the original notebook that corrupted 57
rows. The notebook and PDF reports are kept.

## Run them

```bash
git clone https://github.com/HiralR2931/churn-prediction.git
cd churn-prediction/medicare-churn
pip install -r requirements.txt
# download the CMS CSV into data/ (link in medicare-churn/README.md)
python run_pipeline.py
```

For Netflix, download `netflix_customer_churn.csv` from
[Kaggle](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)
into `netflix-churn/data/`, then:

```bash
cd ../netflix-churn
pip install -r requirements.txt
python run_pipeline.py
```

Neither dataset is stored in this repo.

Each project has a `tests/` folder that runs on small made-up samples, with no download needed. GitHub Actions runs both suites on every push (`.github/workflows/tests.yml`).

## Stack

Python · pandas · NumPy · scikit-learn · SciPy · D3 · Plotly · Matplotlib/Seaborn · PySpark · Databricks · Jupyter

## License

[MIT](LICENSE)

**Author:** Hiral Rana · [GitHub](https://github.com/HiralR2931) · [LinkedIn](https://linkedin.com/in/ranahiral)
