# Customer Churn Prediction

A machine learning project to predict customer churn and help businesses improve
retention, reduce costs, and optimize marketing strategies.
Built using **Python**, **scikit-learn**, and **Jupyter Notebook**.

## Project Overview

Customer churn directly affects a company's revenue and profitability.
This project builds a **classification model** to predict which customers are at
risk of leaving, using the IBM Telco Customer Churn dataset (7,043 customers).

The workflow includes:
- Data cleaning and preprocessing
- **Feature engineering** — tenure bands, add-on service count, charge-per-month
  ratio, tenure × monthly-charge interaction, one-hot encoding
- Model training and evaluation (logistic regression baseline + random forest)
- Saving the trained model + feature list with joblib
- **Inference demo** — load the saved model and score new customers with
  `predict_proba`

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Environment:** Jupyter Notebook

## How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/yangchenai02/mle-customer-churn-prediction.git
cd mle-customer-churn-prediction

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook churn_prediction.ipynb
```

## Inference service (FastAPI)

The trained model is served over HTTP — see `service/`:

```bash
# 1. Run it
docker compose up --build
# or without Docker:
pip install -r service/requirements.txt
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — liveness + model info (model version, feature count)
- `POST /predict` — score one raw customer row (same columns as the CSV),
  returns `churn_probability`, `prediction`, and `model_version`
- `GET /metrics` — Prometheus metrics (request latency histogram)

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85, "TotalCharges": 29.85
}'
```

**Train/serve consistency:** the service's feature code (`service/features.py`)
is the single shared implementation — the notebook's inference demo imports
it. It is verified in two ways: the service reproduces the notebook's
holdout metrics (accuracy 0.793, ROC-AUC 0.835), and the three demo rows
score identically in the notebook and over HTTP (62.8% / 21.2% / 4.6%).

Two skew traps this avoids, both found while building it:

- `pd.get_dummies(drop_first=True)` is batch-dependent: on a single-row
  request it drops the row's own category. The service builds dummies
  deterministically from the saved feature list instead.
- Per-request pandas overhead: the hot path builds a numpy vector
  directly (`featurize_row`), verified row-by-row identical to the
  DataFrame path on all 7,032 customers.

**Load test** (`python service/loadtest.py --n 500 --workers 20`,
measured on a small 1.5-CPU shared VM, single uvicorn worker):
500 requests, 0 errors, ~222 req/s, p50 49 ms, p95 143 ms.

## What this is — and what it isn't

This is a portfolio project on a public dataset: real feature engineering, a
trained and evaluated model, and a saved model that scores new rows. It is not
a production serving system — no online feature store, no monitoring, no
A/B-tested business impact.
