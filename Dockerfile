# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Service deps first (cached layer); notebook deps are not needed to serve.
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# Model artifacts produced by churn_prediction.ipynb
COPY churn_model.joblib churn_scaler.joblib churn_features.json ./

COPY service/ service/

EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
