"""Churn prediction inference service (FastAPI).

Loads the logistic-regression model trained in churn_prediction.ipynb
(churn_model.joblib) plus its scaler and feature list, and serves
predictions over HTTP.

Endpoints:
  GET  /health   — liveness + model info (model file, feature count)
  POST /predict  — score one raw customer row, returns churn probability
  GET  /metrics  — Prometheus metrics (request count, latency histogram)

Run locally:
  uvicorn service.main:app --host 0.0.0.0 --port 8000
Or with Docker Compose (see docker-compose.yml at the repo root).
"""

import json
import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel, Field

from service.features import NUM_COLS, featurize_row

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(os.path.dirname(__file__)))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "logreg-v1")

REQUEST_LATENCY = Histogram(
    "churn_predict_latency_seconds",
    "Latency of POST /predict (seconds)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["model"] = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
    state["scaler"] = joblib.load(os.path.join(MODEL_DIR, "churn_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "churn_features.json")) as f:
        state["features"] = json.load(f)["features"]
    yield


app = FastAPI(title="Churn Prediction Service", lifespan=lifespan)


class Customer(BaseModel):
    """One raw Telco customer row, same columns as the training CSV."""

    customerID: str | None = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | str = 0.0  # blank string in the raw CSV = new customer


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    prediction: str  # "likely churn" | "likely stays"
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "n_features": len(state["features"]),
        "model": "LogisticRegression (accuracy 0.793, ROC-AUC 0.835 on holdout)",
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    start = time.perf_counter()
    try:
        # Dict -> numpy vector, no per-request DataFrame (see features.py).
        vec = featurize_row(customer.model_dump(), state["features"])
        num_idx = [state["features"].index(c) for c in NUM_COLS]
        vec[num_idx] = state["scaler"].transform(
            vec[num_idx].reshape(1, -1))[0]
        proba = float(state["model"].predict_proba(vec.reshape(1, -1))[0, 1])
        return Prediction(
            customerID=customer.customerID,
            churn_probability=round(proba, 4),
            prediction="likely churn" if proba >= 0.5 else "likely stays",
            model_version=MODEL_VERSION,
        )
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
