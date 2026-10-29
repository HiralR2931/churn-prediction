# Databricks notebook source
# MAGIC %md
# MAGIC # Medicare County-Level Churn Prediction (Dec 2025 → Feb 2026)
# MAGIC
# MAGIC PySpark implementation of the cleaning + feature-engineering stage, mirroring
# MAGIC the local `src/` pipeline. Run cell-by-cell on a Databricks cluster; the
# MAGIC modelling / dashboard cells hand off to pandas + scikit-learn + Plotly.
# MAGIC
# MAGIC **Data:** CMS Medicare Monthly Enrollment, county level, 2013–2025.

# COMMAND ----------

# MAGIC %md ## 1. Ingest raw CMS extract

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

RAW_PATH = "/Volumes/main/medicare/enrollment/medicare_enrollment_2013_2025.csv"

raw = (
    spark.read.option("header", True).option("inferSchema", False).csv(RAW_PATH)
)
# County-level monthly rows only (drop National/State and annual "Year" roll-ups).
df = raw.filter((F.col("BENE_GEO_LVL") == "County") & (F.col("MONTH") != "Year"))
print("county-monthly rows:", df.count())

# COMMAND ----------

# MAGIC %md ## 2. Resolve suppressed cells & cast numerics
# MAGIC CMS masks counts 1–10 with `*`; replace with the range mid-point (5) and cast.

# COMMAND ----------

ID_COLS = {"BENE_STATE_ABRVTN", "BENE_STATE_DESC", "BENE_COUNTY_DESC",
           "BENE_FIPS_CD", "BENE_GEO_LVL", "MONTH", "YEAR"}
num_cols = [c for c in df.columns if c not in ID_COLS]

for c in num_cols:
    df = df.withColumn(
        c,
        F.when(F.trim(F.col(c)) == "*", F.lit(5.0))
         .otherwise(F.col(c).cast("double")),
    )
df = df.withColumn("YEAR", F.col("YEAR").cast("int"))

# COMMAND ----------

# MAGIC %md ## 3. Validate: drop malformed rows, flag inconsistent splits

# COMMAND ----------

df = df.filter(F.col("TOT_BENES").isNotNull() & (F.col("TOT_BENES") > 0))
df = df.filter(F.col("BENE_FIPS_CD").rlike(r"^\d{5}$"))
# Coverage split should not exceed the total (data-integrity guard, 2% tolerance).
df = df.withColumn(
    "flag_inconsistent_split",
    (F.col("ORGNL_MDCR_BENES") + F.col("MA_AND_OTH_BENES")) > F.col("TOT_BENES") * 1.02,
)
print("inconsistent-split rows:", df.filter("flag_inconsistent_split").count())

# COMMAND ----------

# MAGIC %md ## 4. Normalise time & engineer features

# COMMAND ----------

MONTHS = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
          "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
          "December": 12}
month_map = F.create_map([F.lit(x) for kv in MONTHS.items() for x in kv])
df = (df.withColumn("month_num", month_map[F.col("MONTH")])
        .withColumn("date", F.to_date(F.concat_ws("-", F.col("YEAR"),
                                                   F.col("month_num"), F.lit(1)))))

def ratio(num, den):
    return F.when(F.col(den) > 0, F.col(num) / F.col(den)).otherwise(0.0)

df = (df
    .withColumn("aged_ratio", ratio("AGED_TOT_BENES", "TOT_BENES"))
    .withColumn("disabled_ratio", ratio("DSBLD_TOT_BENES", "TOT_BENES"))
    .withColumn("dual_ratio", ratio("DUAL_TOT_BENES", "TOT_BENES"))
    .withColumn("ma_ratio", ratio("MA_AND_OTH_BENES", "TOT_BENES"))
    .withColumn("rx_ratio", ratio("PRSCRPTN_DRUG_TOT_BENES", "TOT_BENES"))
    .withColumn("log_tot_benes", F.log1p("TOT_BENES")))

# Trend features via per-county time windows.
w = Window.partitionBy("BENE_FIPS_CD").orderBy("date")
df = (df
    .withColumn("tot_lag3", F.lag("TOT_BENES", 3).over(w))
    .withColumn("tot_lag12", F.lag("TOT_BENES", 12).over(w))
    .withColumn("momentum_3m",
                F.when(F.col("tot_lag3") > 0,
                       (F.col("TOT_BENES") - F.col("tot_lag3")) / F.col("tot_lag3"))
                 .otherwise(0.0))
    .withColumn("prior_window_change",
                F.when(F.col("tot_lag12") > 0,
                       (F.col("TOT_BENES") - F.col("tot_lag12")) / F.col("tot_lag12"))
                 .otherwise(0.0)))

# COMMAND ----------

# MAGIC %md ## 5. Build churn label (December Y → February Y+1)

# COMMAND ----------

dec = df.filter(F.col("month_num") == 12).withColumn("base_year", F.col("YEAR"))
feb = (df.filter(F.col("month_num") == 2)
         .select("BENE_FIPS_CD",
                 (F.col("YEAR") - 1).alias("base_year"),
                 F.col("TOT_BENES").alias("target_benes")))
tbl = dec.join(feb, ["BENE_FIPS_CD", "base_year"], "left")
tbl = tbl.withColumn(
    "forward_change",
    (F.col("target_benes") - F.col("TOT_BENES")) / F.col("TOT_BENES"),
).withColumn(
    "churn",
    F.when(F.col("target_benes").isNull(), F.lit(None))
     .otherwise((F.col("forward_change") < -0.005).cast("int")),
)

# Persist the modelling table for the scikit-learn / Plotly stage.
(tbl.write.mode("overwrite").format("delta")
    .saveAsTable("main.medicare.churn_modeling_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model + dashboards
# MAGIC The Random Forest, statistical tests, and Plotly dashboards run on the
# MAGIC collected pandas frame — see `src/model.py`, `src/stats_analysis.py`, and
# MAGIC `src/dashboards.py`. On Databricks:
# MAGIC ```python
# MAGIC pdf = spark.table("main.medicare.churn_modeling_table").toPandas()
# MAGIC from src.model import train_model, evaluate, forecast
# MAGIC # ... same calls as run_pipeline.py steps 4–7
# MAGIC ```
