"""Shared feature engineering for the churn model.

This is the single source of truth for train/serve feature logic.
It mirrors the feature-engineering cells of churn_prediction.ipynb
(tenure bands, add-on count, charge ratio, interaction, one-hot
with drop_first), so the service can never drift from the notebook
that trained the model.

Why the dummies are built by hand instead of pd.get_dummies:
`get_dummies(drop_first=True)` is batch-dependent — on a single-row
request it drops the row's own category (the only one present), so a
"Contract = One year" customer would score as if Contract were
missing. Building each dummy from the known feature list is
row-independent: the same customer always gets the same vector,
whether scored alone or in a batch. That is the whole point of
train/serve consistency.
"""

import numpy as np
import pandas as pd

ADD_ONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
           "TechSupport", "StreamingTV", "StreamingMovies"]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services",
            "charge_per_month", "tenure_x_monthly"]

# Raw categorical columns and the dummy values the training run kept
# (i.e. every category except the drop_first reference). Derived from
# churn_features.json: a dummy "Contract_One year" means column
# "Contract" with value "One year".
CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod", "tenure_band"]


def dummy_values(feature_names: list[str]) -> dict[str, list[str]]:
    """Map each categorical column to the dummy values kept at training."""
    mapping: dict[str, list[str]] = {c: [] for c in CATEGORICAL_COLS}
    for feat in feature_names:
        for col in CATEGORICAL_COLS:
            if feat.startswith(col + "_"):
                mapping[col].append(feat[len(col) + 1:])
                break
    return mapping


def featurize(raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Turn raw Telco customer rows into the model's feature matrix."""
    d = raw.copy()
    if "customerID" in d.columns:
        d = d.drop(columns=["customerID"])
    if "Churn" in d.columns:
        d = d.drop(columns=["Churn"])

    # A blank TotalCharges in the raw data means a brand-new customer
    # (tenure 0 in the source file). Fill it from MonthlyCharges * tenure
    # so the ratio features stay defined; the notebook drops those rows
    # at training time instead.
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    missing = d["TotalCharges"].isna()
    d.loc[missing, "TotalCharges"] = (
        d.loc[missing, "MonthlyCharges"] * d.loc[missing, "tenure"])

    d["num_services"] = (d[ADD_ONS] == "Yes").sum(axis=1)
    d["charge_per_month"] = np.where(
        d["tenure"] > 0, d["TotalCharges"] / d["tenure"], d["MonthlyCharges"])
    d["tenure_x_monthly"] = d["tenure"] * d["MonthlyCharges"]
    d["tenure_band"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                              labels=["0-12", "13-24", "25-48", "49-72"])

    out = pd.DataFrame(index=d.index)
    for col in NUM_COLS:
        out[col] = d[col].astype(float)
    dummies = dummy_values(feature_names)
    for col, values in dummies.items():
        for value in values:
            out[f"{col}_{value}"] = (d[col].astype(str) == value).astype(int)
    # Plain numeric features that are neither scaled nor dummies
    # (e.g. SeniorCitizen) pass through untouched.
    covered = set(NUM_COLS) | {f"{c}_{v}" for c, vs in dummies.items()
                               for v in vs}
    for col in feature_names:
        if col not in covered and col in d.columns:
            out[col] = d[col].astype(float)
    return out.reindex(columns=feature_names, fill_value=0)


def _band(tenure: float) -> str:
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "13-24"
    if tenure <= 48:
        return "25-48"
    return "49-72"


def featurize_row(customer: dict, feature_names: list[str]) -> np.ndarray:
    """Same transformation as featurize(), for one raw customer dict —
    but without building a DataFrame, so a single request doesn't pay
    pandas overhead. Verified identical to featurize() (see the
    verification in the repo's service notes)."""
    tenure = float(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])
    raw_total = customer.get("TotalCharges")
    if raw_total is None or (isinstance(raw_total, str)
                             and raw_total.strip() == ""):
        # Blank TotalCharges (new customer in the raw CSV) -> same fill
        # as the DataFrame path: MonthlyCharges * tenure.
        total = monthly * tenure
    else:
        total = float(raw_total)

    computed: dict[str, float] = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "num_services": float(sum(1 for c in ADD_ONS
                                  if customer.get(c) == "Yes")),
        "charge_per_month": total / tenure if tenure > 0 else monthly,
        "tenure_x_monthly": tenure * monthly,
        "SeniorCitizen": float(customer.get("SeniorCitizen", 0)),
    }
    band = _band(tenure)
    dummies = dummy_values(feature_names)
    dummy_lookup = {(c, v): f"{c}_{v}" for c, vs in dummies.items()
                    for v in vs}
    raw = dict(customer)
    raw["tenure_band"] = band
    vec = np.zeros(len(feature_names), dtype=float)
    for i, feat in enumerate(feature_names):
        if feat in computed:
            vec[i] = computed[feat]
        else:
            for (col, value), name in dummy_lookup.items():
                if name == feat:
                    vec[i] = 1.0 if str(raw.get(col)) == value else 0.0
                    break
    return vec
