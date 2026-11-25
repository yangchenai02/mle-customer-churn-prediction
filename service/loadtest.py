"""Minimal load test for the churn service (stdlib only).

Usage:
  python service/loadtest.py [--url http://localhost:8000/predict] [--n 500] [--workers 20]

Reads a sample customer row from the Telco CSV, fires N POST /predict
requests across a thread pool, and prints p50/p95/p99 latency, mean
latency, throughput, and error count.
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

PAYLOAD_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
                "TotalCharges"]


def one_request(url: str, payload: dict) -> tuple[float, int]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            return time.perf_counter() - start, resp.status
    except Exception:
        return time.perf_counter() - start, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    row = df.iloc[0][PAYLOAD_COLS].to_dict()
    row["TotalCharges"] = float(row["TotalCharges"])
    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    row["tenure"] = int(row["tenure"])

    latencies: list[float] = []
    errors = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for latency, status in pool.map(
                lambda _: one_request(args.url, row), range(args.n)):
            latencies.append(latency)
            if status != 200:
                errors += 1
    total = time.perf_counter() - start

    latencies.sort()
    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))] * 1000

    print(f"requests: {args.n}  workers: {args.workers}  errors: {errors}")
    print(f"throughput: {args.n / total:.1f} req/s")
    print(f"latency ms — mean {statistics.mean(latencies) * 1000:.1f}  "
          f"p50 {pct(0.50):.1f}  p95 {pct(0.95):.1f}  p99 {pct(0.99):.1f}")


if __name__ == "__main__":
    main()
