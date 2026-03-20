# Netflix Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-f59e0b?style=for-the-badge)

An end-to-end machine learning pipeline for predicting customer churn using Netflix subscription behavioral data. The project covers the full data lifecycle — exploratory analysis, feature engineering, and ensemble model benchmarking.

---

## Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses. This project builds a complete ML pipeline that:

- Processes and validates **5,000 Netflix customer records** across 14 features
- Performs comprehensive EDA with multiple visualization types to uncover churn drivers
- Engineers features through label encoding, standard scaling, and outlier treatment
- Trains and benchmarks **6 classification models**, achieving **99.10% accuracy (Gradient Boosting)**
- Extracts feature importance scores to identify the strongest churn predictors

---

## Dataset

| Attribute | Details |
|---|---|
| **Source** | [Netflix Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) |
| **Records** | 5,000 customers |
| **Features** | 14 variables (mixed numeric & categorical) |
| **Target** | `churned` (1 = churned, 0 = retained) |
| **Class Balance** | ~2,515 churned · ~2,485 retained (balanced — no SMOTE required) |

**Feature Categories:**

| Category | Variables |
|---|---|
| **Demographics** | Age, Gender, Region |
| **Subscription** | Subscription Type, Monthly Fee, Number of Profiles |
| **Behavioral** | Watch Hours, Avg Watch Time Per Day, Last Login Days |
| **Preferences** | Favorite Genre, Device, Payment Method |

---

## Data Processing Pipeline

### Data Cleaning & Validation
- Stripped whitespace from column names and categorical labels
- Corrected decimal-point entry errors in `avg_watch_time_per_day` (values > 10 hrs ÷ 10)
- Validated data types; confirmed zero missing values — no imputation required

### Feature Engineering
- **Label Encoding** for all categorical variables (gender, region, device, subscription type, payment method, genre)
- Outlier flagging via box plots across numeric variables
- Stratified 80/20 train-test split (4,000 train / 1,000 test) preserving churn class balance

---

## Key EDA Findings

### Strongest Churn Predictors
| Feature | Correlation with Churn |
|---|---|
| **Watch Hours** | −0.48 |
| **Last Login Days** | +0.47 |
| **Avg Watch Time Per Day** | −0.27 |

### Other Insights
- Balanced classes (50/50 split) — accuracy is a reliable metric without resampling
- **Behavioral engagement variables** far outperform demographics as predictors
- Age and Monthly Fee show near-zero churn correlation
- Low multicollinearity across features — each contributes distinct predictive signal

---

## Model Results

| Model | Test Accuracy |
|---|---|
| **Gradient Boosting** | **99.10%** |
| Random Forest | 97.00% |
| Decision Tree | 96.90% |
| Logistic Regression | 88.40% |
| KNN (K=5) | 87.80% |
| Naive Bayes | 83.10% |

### Baseline — Logistic Regression (Test Set, n=1,000)

| | Predicted Stay | Predicted Churn |
|---|---|---|
| **Actual Stay** | 429 | 68 |
| **Actual Churn** | 48 | 455 |

- **Accuracy:** 88.40%
- **Precision / Recall / F1 (Churn class):** 0.87 / 0.90 / 0.89

### Best Model — Gradient Boosting (Test Set, n=1,000)

| | Predicted Stay | Predicted Churn |
|---|---|---|
| **Actual Stay** | 494 | 3 |
| **Actual Churn** | 6 | 497 |

- **Accuracy:** 99.10%
- **Precision / Recall / F1 (Churn class):** 0.99 / 0.99 / 0.99
- Only **9 total errors** out of 1,000 test customers

### Why Tree-Based Models Dominate
Ensemble models capture nonlinear relationships and complex feature interactions that linear models cannot. Naive Bayes underperformed due to violations of its feature independence assumption among correlated engagement variables.

---

## Project Structure

```
customer-churn/
├── NetflixdataTesting.ipynb              # Full pipeline: EDA → modeling → evaluation
├── DataCollection,DataVisualization,
│   DataExplorationandDataProcessing.pdf  # Milestone report — data collection & EDA
├── ModelExplorationReport.pdf            # Model comparison & evaluation report
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **ML Models** | Scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, Gradient Boosting) |
| **Environment** | Jupyter Notebook · Google Colab |

---

## Setup

```bash
git clone https://github.com/HiralR2931/customer-churn.git
cd customer-churn

pip install pandas numpy matplotlib seaborn scikit-learn

# Open notebook
jupyter notebook NetflixdataTesting.ipynb
```

> **Note:** The notebook uses `google.colab` for file upload. When running locally, replace the `files.upload()` call with `pd.read_csv('path/to/netflix_customer_churn.csv')`.

---

## Next Steps

- [ ] Hyperparameter tuning (GridSearchCV) for Gradient Boosting and Random Forest
- [ ] Churn probability scoring system for risk-level customer prioritization
- [ ] Customer segmentation analysis (identify high-risk behavioral groups)
- [ ] Dynamic pricing optimization model based on churn probability scores
- [ ] Interactive Streamlit dashboard for stakeholder exploration

---

## Author

**Hiral Rana**
MS Data Analytics Engineering · Northeastern University

---

## License

This project is for educational and portfolio purposes.