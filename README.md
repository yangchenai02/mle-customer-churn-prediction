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

## What this is — and what it isn't

This is a portfolio project on a public dataset: real feature engineering, a
trained and evaluated model, and a saved model that scores new rows. It is not
a production serving system — no online feature store, no monitoring, no
A/B-tested business impact.
