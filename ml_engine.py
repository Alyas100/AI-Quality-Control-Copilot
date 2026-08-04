"""
Module 2: Predictive Engine (Tabular ML)
================================================================================
Trains a fast XGBoost Regressor on the synthetic batch dataset to predict
`ffa_percentage` from environmental + vision-derived features, and exposes a
small inference / risk-classification API for the Streamlit app. The trained
model is cached via `st.cache_resource` so it trains once per server session,
not on every rerun.
"""

import os

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import generate_data

FEATURE_COLUMNS = ["harvest_delay_hours", "storage_temp_c", "humidity_percent", "ripeness_score"]
FEATURE_LABELS = {
    "harvest_delay_hours": "Harvest Delay (hrs)",
    "storage_temp_c": "Storage Temp (°C)",
    "humidity_percent": "Humidity (%)",
    "ripeness_score": "Ripeness Score",
}
TARGET_COLUMN = "ffa_percentage"

SAFE_MAX = 2.5
WARNING_MAX = 3.5


def ensure_dataset(csv_path: str = "synthetic_batch_data.csv") -> str:
    """Create the synthetic dataset if it doesn't already exist on disk."""
    if not os.path.exists(csv_path):
        df = generate_data.generate_dataset()
        df.to_csv(csv_path, index=False)
    return csv_path


@st.cache_resource(show_spinner="Training predictive engine on synthetic batch data...")
def train_model(csv_path: str = "synthetic_batch_data.csv"):
    """Train (once, cached) and return (model, metrics, feature_importances)."""
    csv_path = ensure_dataset(csv_path)
    df = pd.read_csv(csv_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    importances = dict(zip(FEATURE_COLUMNS, (float(v) for v in model.feature_importances_)))
    return model, metrics, importances


def predict_ffa(model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score) -> float:
    """Run real-time inference for a single batch scenario."""
    X = pd.DataFrame([{
        "harvest_delay_hours": harvest_delay_hours,
        "storage_temp_c": storage_temp_c,
        "humidity_percent": humidity_percent,
        "ripeness_score": ripeness_score,
    }])[FEATURE_COLUMNS]
    pred = float(model.predict(X)[0])
    return max(0.0, pred)


def get_risk_level(ffa_percentage: float) -> dict:
    """Classify a predicted FFA% into the Safe / Warning / Critical bands."""
    if ffa_percentage < SAFE_MAX:
        return {
            "level": "Safe", "color": "green", "icon": "✅",
            "message": "Within safe FFA range. Standard processing schedule is fine.",
        }
    elif ffa_percentage <= WARNING_MAX:
        return {
            "level": "Warning", "color": "amber", "icon": "⚠️",
            "message": "Approaching the FFA threshold. Prioritize processing soon.",
        }
    else:
        return {
            "level": "Critical", "color": "red", "icon": "🚨",
            "message": "FFA threshold exceeded. Immediate action required.",
        }
