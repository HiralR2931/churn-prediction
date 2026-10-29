# Churn Prediction

Two churn projects with the same question at different scales: who is about to
leave? One follows people leaving Original Medicare for private Medicare
Advantage plans across every U.S. county. The other predicts churn for
individual Netflix subscribers.

| Project | Scale | Data | Model | Headline result |
|---|---|---|---|---|
| [The Medicare Advantage Stall](medicare-churn/) · [interactive report](https://hiralr2931.github.io/churn-prediction/) | 3,236 U.S. counties, 2013–2025 | CMS Medicare Monthly Enrollment (564k rows) | Walk-forward backtest of 4 forecasters, Random Forest, PySpark on Databricks | MA passed 50% of beneficiaries; yearly switching fell from +3.1 to +0.9 pp; a county + state blend beats gradient boosting (Spearman 0.53 vs 0.47) |
| [Netflix Customer Churn](netflix-churn/) | 5,000 subscribers | Kaggle Netflix churn dataset (14 features) | 6 classifiers benchmarked | Gradient Boosting 99.1% test accuracy |

![Medicare Advantage passed Original Medicare while the yearly switching wave fell](medicare-churn/docs/01_tipping_point.png)

## What's in each folder

**[`medicare-churn/`](medicare-churn/)** is a full pipeline. It cleans 13 years of
county-level CMS data and resolves ~3.2M suppressed cells. It then measures
each county's December → February move from Original Medicare to Medicare
Advantage, and finds the adoption curve and the 2024 stall. Four forecasting
methods are compared in a walk-forward backtest, and the pipeline publishes an
interactive D3 report with a county map. An earlier net-decline Random Forest
is kept alongside, with a feature bug fixed. The cleaning and feature stage also
runs in PySpark on Databricks with Delta tables.

**[`netflix-churn/`](netflix-churn/)** is a notebook project. It covers EDA,
encoding, scaling and outlier treatment, then compares Logistic Regression,
KNN, Naive Bayes, Decision Tree, Random Forest and Gradient Boosting, with
feature importance for the churn drivers. Written PDF reports for the data and
modelling stages are included.

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
and open `netflix-churn/NetflixdataTesting.ipynb` in Jupyter or Google Colab.

Neither dataset is stored in this repo.

## Stack

Python · pandas · NumPy · scikit-learn · SciPy · D3 · Plotly · Matplotlib/Seaborn · PySpark · Databricks · Jupyter

**Author:** Hiral Rana · [GitHub](https://github.com/HiralR2931) · [LinkedIn](https://linkedin.com/in/ranahiral)
