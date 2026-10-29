# Databricks notebook source
# MAGIC %md
# MAGIC # Medicare County-Level Churn Prediction — All-in-One (Databricks)
# MAGIC
# MAGIC Self-contained notebook: **clean → engineer features → label churn →
# MAGIC Random Forest → statistical tests → Plotly dashboards.**
# MAGIC No external `src/` files needed — just upload the CSV and Run All.
# MAGIC
# MAGIC **Before you run:** edit the paths in the CONFIG cell below to match where
# MAGIC you uploaded the CSV (see the README steps).

# COMMAND ----------

# MAGIC %md ## 0. Config — EDIT THESE if your names differ

# COMMAND ----------

# Where you uploaded the data (Unity Catalog Volume).
CATALOG = "workspace"          # Free Edition default catalog
SCHEMA  = "medicare"           # you will create this
VOLUME  = "data"               # you will create this
CSV_FILE = "medicare_enrollment_2013_2025.csv"

CSV_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{CSV_FILE}"
OUT_DIR  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/outputs"

# Churn window + model params (same as the local project).
CHURN_THRESHOLD = 0.005        # >0.5% net enrollment loss = churn
PREDICT_FROM_YEAR = 2025       # Dec 2025 -> Feb 2026 forecast
TEST_YEAR = 2024               # Dec 2024 -> Feb 2025 held-out test

print("Reading from:", CSV_PATH)

# COMMAND ----------

# MAGIC %md ## 1. Create schema + volume (safe to re-run)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
dbutils.fs.mkdirs(OUT_DIR)
print("schema + volume ready")

# COMMAND ----------

# MAGIC %md ## 2. Ingest + clean (PySpark)
# MAGIC County-level monthly rows only; resolve CMS-suppressed `*` cells (counts
# MAGIC 1–10) to the range midpoint (5); cast numerics; drop malformed rows.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

raw = spark.read.option("header", True).option("inferSchema", False).csv(CSV_PATH)
print("raw rows:", raw.count())

df = raw.filter((F.col("BENE_GEO_LVL") == "County") & (F.col("MONTH") != "Year"))

ID_COLS = {"BENE_STATE_ABRVTN", "BENE_STATE_DESC", "BENE_COUNTY_DESC",
           "BENE_FIPS_CD", "BENE_GEO_LVL", "MONTH", "YEAR"}
num_cols = [c for c in df.columns if c not in ID_COLS]
for c in num_cols:
    df = df.withColumn(
        c,
        F.when(F.trim(F.col(c)) == "*", F.lit(5.0)).otherwise(F.col(c).cast("double")),
    )
df = df.withColumn("YEAR", F.col("YEAR").cast("int"))

# Validate.
df = df.filter(F.col("TOT_BENES").isNotNull() & (F.col("TOT_BENES") > 0))
df = df.filter(F.col("BENE_FIPS_CD").rlike(r"^\d{5}$"))
df = df.withColumn(
    "flag_inconsistent_split",
    (F.col("ORGNL_MDCR_BENES") + F.col("MA_AND_OTH_BENES")) > F.col("TOT_BENES") * 1.02,
)
print("clean county-monthly rows:", df.count())

# COMMAND ----------

# MAGIC %md ## 3. Time fields + feature engineering (PySpark)

# COMMAND ----------

MONTHS = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,
          "August":8,"September":9,"October":10,"November":11,"December":12}
month_map = F.create_map([F.lit(x) for kv in MONTHS.items() for x in kv])
df = (df.withColumn("month_num", month_map[F.col("MONTH")])
        .withColumn("date", F.to_date(F.concat_ws("-", F.col("YEAR"),
                                                   F.col("month_num"), F.lit(1)))))

def ratio(num, den):
    return F.when(F.col(den) > 0, F.col(num) / F.col(den)).otherwise(0.0)

esrd = F.coalesce(F.col("AGED_ESRD_BENES"), F.lit(0.0)) + \
       F.coalesce(F.col("DSBLD_ESRD_AND_ESRD_ONLY_BENES"), F.lit(0.0))
lis = F.coalesce(F.col("PRSCRPTN_DRUG_FULL_LIS_BENES"), F.lit(0.0)) + \
      F.coalesce(F.col("PRSCRPTN_DRUG_PARTIAL_LIS_BENES"), F.lit(0.0))
age85 = (F.coalesce(F.col("AGE_85_TO_89_BENES"), F.lit(0.0)) +
         F.coalesce(F.col("AGE_90_TO_94_BENES"), F.lit(0.0)) +
         F.coalesce(F.col("AGE_GT_94_BENES"), F.lit(0.0)))

df = (df
    .withColumn("aged_ratio", ratio("AGED_TOT_BENES", "TOT_BENES"))
    .withColumn("disabled_ratio", ratio("DSBLD_TOT_BENES", "TOT_BENES"))
    .withColumn("dual_ratio", ratio("DUAL_TOT_BENES", "TOT_BENES"))
    .withColumn("full_dual_ratio", ratio("FULL_DUAL_TOT_BENES", "TOT_BENES"))
    .withColumn("ma_ratio", ratio("MA_AND_OTH_BENES", "TOT_BENES"))
    .withColumn("esrd_ratio", F.when(F.col("TOT_BENES") > 0, esrd / F.col("TOT_BENES")).otherwise(0.0))
    .withColumn("rx_ratio", ratio("PRSCRPTN_DRUG_TOT_BENES", "TOT_BENES"))
    .withColumn("lis_ratio", F.when(F.col("TOT_BENES") > 0, lis / F.col("TOT_BENES")).otherwise(0.0))
    .withColumn("female_ratio", ratio("FEMALE_TOT_BENES", "TOT_BENES"))
    .withColumn("white_ratio", ratio("WHITE_TOT_BENES", "TOT_BENES"))
    .withColumn("black_ratio", ratio("BLACK_TOT_BENES", "TOT_BENES"))
    .withColumn("hispanic_ratio", ratio("HSPNC_TOT_BENES", "TOT_BENES"))
    .withColumn("api_ratio", ratio("API_TOT_BENES", "TOT_BENES"))
    .withColumn("age_85_plus_ratio", F.when(F.col("TOT_BENES") > 0, age85 / F.col("TOT_BENES")).otherwise(0.0))
    .withColumn("log_tot_benes", F.log1p("TOT_BENES")))

w = Window.partitionBy("BENE_FIPS_CD").orderBy("date")
df = (df
    .withColumn("tot_lag3", F.lag("TOT_BENES", 3).over(w))
    .withColumn("momentum_3m", F.when(F.col("tot_lag3") > 0,
                (F.col("TOT_BENES") - F.col("tot_lag3")) / F.col("tot_lag3")).otherwise(0.0)))

# Last finished Dec -> Feb window: Dec (Y-1) -> Feb (Y), seen from Feb Y onwards.
_dec = df.filter(F.col("month_num") == 12).select(
    "BENE_FIPS_CD", (F.col("YEAR") + 1).alias("_window_year"), F.col("TOT_BENES").alias("_dec"))
_win = (df.filter(F.col("month_num") == 2)
    .select("BENE_FIPS_CD", F.col("YEAR").alias("_window_year"), F.col("TOT_BENES").alias("_feb"))
    .join(_dec, ["BENE_FIPS_CD", "_window_year"])
    .withColumn("prior_window_change",
                F.when(F.col("_dec") > 0, F.col("_feb") / F.col("_dec") - 1).otherwise(0.0))
    .select("BENE_FIPS_CD", "_window_year", "prior_window_change"))
df = (df.withColumn("_window_year", F.col("YEAR") - F.when(F.col("month_num") < 2, 1).otherwise(0))
        .join(_win, ["BENE_FIPS_CD", "_window_year"], "left")
        .fillna({"prior_window_change": 0.0})
        .drop("_window_year"))

# COMMAND ----------

# MAGIC %md ## 4. Churn label (December Y → February Y+1) + save Delta table

# COMMAND ----------

dec = df.filter(F.col("month_num") == 12).withColumn("base_year", F.col("YEAR"))
feb = (df.filter(F.col("month_num") == 2)
         .select("BENE_FIPS_CD", (F.col("YEAR") - 1).alias("base_year"),
                 F.col("TOT_BENES").alias("target_benes")))
tbl = dec.join(feb, ["BENE_FIPS_CD", "base_year"], "left")
tbl = (tbl
    .withColumn("forward_change",
                (F.col("target_benes") - F.col("TOT_BENES")) / F.col("TOT_BENES"))
    .withColumn("churn",
                F.when(F.col("target_benes").isNull(), F.lit(None))
                 .otherwise((F.col("forward_change") < -CHURN_THRESHOLD).cast("int"))))

tbl.write.mode("overwrite").option("overwriteSchema", "true") \
   .saveAsTable(f"{CATALOG}.{SCHEMA}.churn_modeling_table")
print("saved table:", f"{CATALOG}.{SCHEMA}.churn_modeling_table")

# COMMAND ----------

# MAGIC %md ## 5. Random Forest — train, evaluate, forecast (pandas + scikit-learn)

# COMMAND ----------

# plotly is usually pre-installed; uncomment if the import fails.
# %pip install plotly

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

FEATURES = ["aged_ratio","disabled_ratio","dual_ratio","full_dual_ratio","ma_ratio",
            "esrd_ratio","rx_ratio","lis_ratio","female_ratio","white_ratio",
            "black_ratio","hispanic_ratio","api_ratio","age_85_plus_ratio",
            "momentum_3m","prior_window_change","log_tot_benes"]

pdf = (spark.table(f"{CATALOG}.{SCHEMA}.churn_modeling_table")
       .select("BENE_FIPS_CD","BENE_STATE_ABRVTN","BENE_STATE_DESC","BENE_COUNTY_DESC",
               "base_year","TOT_BENES","target_benes","forward_change","churn", *FEATURES)
       .toPandas())
pdf = pdf.rename(columns={"BENE_FIPS_CD":"fips","BENE_STATE_ABRVTN":"state",
                          "BENE_STATE_DESC":"state_name","BENE_COUNTY_DESC":"county"})
for col in ["aged_ratio","disabled_ratio","dual_ratio","full_dual_ratio","ma_ratio",
            "esrd_ratio","rx_ratio","lis_ratio","female_ratio","white_ratio",
            "black_ratio","hispanic_ratio","api_ratio","age_85_plus_ratio"]:
    pdf[col] = pdf[col].clip(0, 1)

labelled = pdf.dropna(subset=["churn"])
train = labelled[labelled["base_year"] < TEST_YEAR]
test  = labelled[labelled["base_year"] == TEST_YEAR]
fc    = pdf[pdf["base_year"] == PREDICT_FROM_YEAR]

clf = RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=25,
                             max_features="sqrt", class_weight="balanced",
                             n_jobs=-1, random_state=42)
clf.fit(train[FEATURES], train["churn"].astype(int))

proba = clf.predict_proba(test[FEATURES])[:, 1]
print(f"ROC-AUC : {roc_auc_score(test['churn'].astype(int), proba):.3f}")
print(f"PR-AUC  : {average_precision_score(test['churn'].astype(int), proba):.3f}")
print(f"Brier   : {brier_score_loss(test['churn'].astype(int), proba):.3f}")
print("confusion:", confusion_matrix(test['churn'].astype(int), (proba>=0.5).astype(int)).tolist())

fc = fc.copy()
fc["churn_probability"] = clf.predict_proba(fc[FEATURES])[:, 1]
fc["predicted_churn"] = (fc["churn_probability"] >= 0.5).astype(int)
fc = fc.sort_values("churn_probability", ascending=False)
print(f"\nForecast Dec {PREDICT_FROM_YEAR} -> Feb {PREDICT_FROM_YEAR+1}: "
      f"{int(fc['predicted_churn'].sum())} of {len(fc)} counties at risk")

# Feature importance (permutation on the held-out test set).
perm = permutation_importance(clf, test[FEATURES], test["churn"].astype(int),
                              n_repeats=10, random_state=42, n_jobs=-1)
importance = (pd.DataFrame({"feature": FEATURES,
                            "permutation_importance": perm.importances_mean,
                            "permutation_std": perm.importances_std})
              .sort_values("permutation_importance", ascending=False))
display(importance)

# COMMAND ----------

# MAGIC %md ## 6. Statistical analysis — demographic segments (two-proportion z-test)

# COMMAND ----------

import math
from scipy import stats

SEGMENTS = {"High dual-eligible share":"dual_ratio","High disabled share":"disabled_ratio",
            "High aged share":"aged_ratio","High Medicare Advantage share":"ma_ratio",
            "High ESRD share":"esrd_ratio","High low-income-subsidy share":"lis_ratio",
            "Large county (benes)":"log_tot_benes"}
rows = []
for name, r in SEGMENTS.items():
    med = labelled[r].median()
    high = labelled[labelled[r] > med]["churn"].astype(int)
    low  = labelled[labelled[r] <= med]["churn"].astype(int)
    p1, p2 = high.mean(), low.mean()
    pp = (high.sum() + low.sum()) / (len(high) + len(low))
    se = math.sqrt(pp*(1-pp)*(1/len(high)+1/len(low)))
    z = (p1 - p2)/se if se else 0.0
    rows.append({"segment":name,"churn_high":p1,"churn_low":p2,"difference":p1-p2,
                 "z_stat":z,"p_value":2*(1-stats.norm.cdf(abs(z)))})
ztests = pd.DataFrame(rows).sort_values("z_stat", key=lambda s: s.abs(), ascending=False)
display(ztests)

# COMMAND ----------

# MAGIC %md ## 7. Plotly dashboards (rendered inline)

# COMMAND ----------

import plotly.graph_objects as go

PAL = {"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","violet":"#4a3aa7","red":"#e34948"}
SEQ_BLUE = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]

# National monthly aggregation for trend charts.
nat = (df.groupBy("date").agg(
        F.sum("ORGNL_MDCR_BENES").alias("orig"),
        F.sum("MA_AND_OTH_BENES").alias("ma"),
        F.sum("AGED_TOT_BENES").alias("aged"),
        F.sum("DSBLD_TOT_BENES").alias("disabled"),
        F.sum("DUAL_TOT_BENES").alias("dual"))
       .orderBy("date").toPandas())

# 7a. Original Medicare vs Medicare Advantage
f1 = go.Figure()
f1.add_scatter(x=nat["date"], y=nat["orig"]/1e6, name="Original Medicare", line=dict(color=PAL["blue"], width=2))
f1.add_scatter(x=nat["date"], y=nat["ma"]/1e6, name="Medicare Advantage", line=dict(color=PAL["orange"], width=2))
f1.update_layout(title="National: Original Medicare vs Medicare Advantage (2013–2025)",
                 template="plotly_white", yaxis_title="Beneficiaries (millions)")
displayHTML(f1.to_html(include_plotlyjs="cdn", full_html=True))

# 7b. State-level churn-risk map
st = fc.groupby("state").apply(
        lambda d: pd.Series({"prob": (d["churn_probability"]*d["TOT_BENES"]).sum()/d["TOT_BENES"].sum()}),
        include_groups=False).reset_index()
f2 = go.Figure(go.Choropleth(
        locations=st["state"], z=st["prob"]*100, locationmode="USA-states",
        colorscale=[[i/(len(SEQ_BLUE)-1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar_title="Churn risk (%)"))
f2.update_layout(title=f"Predicted churn risk by state (Dec {PREDICT_FROM_YEAR} → Feb {PREDICT_FROM_YEAR+1})",
                 geo=dict(scope="usa"), template="plotly_white")
displayHTML(f2.to_html(include_plotlyjs="cdn", full_html=True))

# 7c. Top 25 counties by churn risk
top = fc[fc["TOT_BENES"] >= 1000].head(25).iloc[::-1]
f3 = go.Figure(go.Bar(x=top["churn_probability"]*100, y=top["county"]+", "+top["state"],
                      orientation="h", marker_color=PAL["red"]))
f3.update_layout(title="Top 25 counties by predicted churn risk (≥1,000 benes)",
                 xaxis_title="Churn probability (%)", height=680, template="plotly_white")
displayHTML(f3.to_html(include_plotlyjs="cdn", full_html=True))

# COMMAND ----------

# MAGIC %md ## 8. Save outputs to the Volume

# COMMAND ----------

fc.to_csv(f"{OUT_DIR}/forecast_dec{PREDICT_FROM_YEAR}_feb{PREDICT_FROM_YEAR+1}.csv", index=False)
importance.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
ztests.to_csv(f"{OUT_DIR}/segment_ztests.csv", index=False)
print("wrote CSVs to", OUT_DIR)
print("Done — forecast, importances, and z-tests saved; dashboards rendered above.")

# COMMAND ----------

# MAGIC %md ## 9. Full combined dashboard (same layout as local index.html)
# MAGIC All six figures in one card grid, rendered in a single cell.

# COMMAND ----------

# Save the forecast as a table too (so it is queryable in SQL Editor).
spark.createDataFrame(
    fc[["fips", "state", "state_name", "county", "base_year",
        "TOT_BENES", "churn_probability", "predicted_churn"]]
).write.mode("overwrite").option("overwriteSchema", "true") \
 .saveAsTable(f"{CATALOG}.{SCHEMA}.churn_forecast")

L = dict(template="plotly_white", paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
         font=dict(family='system-ui,-apple-system,"Segoe UI",sans-serif',
                   color="#0b0b0b", size=13),
         margin=dict(l=60, r=30, t=60, b=45), title_font_size=15)

# 1. Original Medicare vs Medicare Advantage
g1 = go.Figure()
g1.add_scatter(x=nat["date"], y=nat["orig"]/1e6, name="Original Medicare", line=dict(color=PAL["blue"], width=2))
g1.add_scatter(x=nat["date"], y=nat["ma"]/1e6, name="Medicare Advantage", line=dict(color=PAL["orange"], width=2))
g1.update_layout(title="Original Medicare vs Medicare Advantage (2013–2025)",
                 yaxis_title="Beneficiaries (millions)", height=420,
                 legend=dict(orientation="h", y=1.12, x=0), **L)

# 2. Enrollment by segment
g2 = go.Figure()
for col, nm, c in [("aged", "Aged", PAL["blue"]), ("disabled", "Disabled", PAL["violet"]),
                   ("dual", "Dual-eligible", PAL["aqua"])]:
    g2.add_scatter(x=nat["date"], y=nat[col]/1e6, name=nm, line=dict(color=c, width=2))
g2.update_layout(title="Enrollment by beneficiary segment (2013–2025)",
                 yaxis_title="Beneficiaries (millions)", height=420,
                 legend=dict(orientation="h", y=1.12, x=0), **L)

# 3. State churn-risk map
st = fc.groupby("state").apply(
        lambda d: pd.Series({"prob": (d["churn_probability"]*d["TOT_BENES"]).sum()/d["TOT_BENES"].sum()}),
        include_groups=False).reset_index()
g3 = go.Figure(go.Choropleth(
        locations=st["state"], z=st["prob"]*100, locationmode="USA-states",
        colorscale=[[i/(len(SEQ_BLUE)-1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar_title="Risk (%)"))
g3.update_layout(title=f"Predicted churn risk by state (Dec {PREDICT_FROM_YEAR} → Feb {PREDICT_FROM_YEAR+1})",
                 geo=dict(scope="usa", bgcolor="#fcfcfb"), height=420, **{k: v for k, v in L.items() if k != "plot_bgcolor"})

# 4. Top 25 counties
top = fc[fc["TOT_BENES"] >= 1000].head(25).iloc[::-1]
g4 = go.Figure(go.Bar(x=top["churn_probability"]*100, y=top["county"]+", "+top["state"],
                      orientation="h", marker_color=PAL["red"]))
g4.update_layout(title="Top 25 counties by predicted churn risk (≥1,000 benes)",
                 xaxis_title="Churn probability (%)", height=680, **L)

# 5. Feature importance
imp = importance.sort_values("permutation_importance").tail(14)
g5 = go.Figure(go.Bar(x=imp["permutation_importance"], y=imp["feature"], orientation="h",
                      marker_color=PAL["blue"],
                      error_x=dict(type="data", array=imp["permutation_std"], color="#898781")))
g5.update_layout(title="Churn drivers — permutation importance", xaxis_title="Mean decrease in ROC-AUC",
                 height=480, **L)

# 6. Segment churn rates
SEG = {"High dual-eligible share": "dual_ratio", "High disabled share": "disabled_ratio",
       "High aged share": "aged_ratio", "High Medicare Advantage share": "ma_ratio",
       "High ESRD share": "esrd_ratio", "High low-income-subsidy share": "lis_ratio",
       "Large county (benes)": "log_tot_benes"}
srows = [{"segment": "All counties", "rate": labelled["churn"].mean()}]
for nm, r in SEG.items():
    sub = labelled[labelled[r] > labelled[r].median()]
    srows.append({"segment": nm, "rate": sub["churn"].mean()})
segdf = pd.DataFrame(srows).sort_values("rate")
g6 = go.Figure(go.Bar(x=segdf["rate"]*100, y=segdf["segment"], orientation="h", marker_color=PAL["violet"]))
g6.update_layout(title="Historical churn rate by county segment", xaxis_title="Churn rate (%)", height=420, **L)

# Stitch all six into one card grid (plotly.js loaded once, on the first div).
figs = [g1, g2, g3, g4, g5, g6]
parts = [f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False),
                   default_width="100%") for i, f in enumerate(figs)]
cards = "\n".join(f'<section>{p}</section>' for p in parts)
dashboard_html = f"""
<div style="background:#f9f9f7;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;padding:8px;">
  <div style="padding:16px 20px 4px;">
    <h1 style="font-size:22px;margin:0 0 4px;color:#0b0b0b;">Medicare County-Level Churn Prediction</h1>
    <p style="color:#52514e;margin:0 0 8px;">Enrollment trends 2013–2025 and predicted churn for the
       Dec {PREDICT_FROM_YEAR} → Feb {PREDICT_FROM_YEAR+1} window. Source: CMS Medicare Monthly Enrollment.</p>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px;padding:12px 20px 24px;">
    {cards}
  </div>
</div>
<style>section{{background:#fcfcfb;border:1px solid rgba(11,11,11,0.10);border-radius:12px;overflow:hidden;padding:4px;}}</style>
"""
displayHTML(dashboard_html)
