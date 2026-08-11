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
import xgboost as xgb

FEATURE_COLUMNS = [
    "ripeness_score",
    "harvest_delay_hrs", 
    "storage_temp_c", 
    "humidity_pct"
]

FEATURE_LABELS = {
    "harvest_delay_hrs": "Harvest Delay (hrs)",
    "storage_temp_c": "Storage Temp (°C)",
    "humidity_pct": "Humidity (%)",
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



def predict_ffa(
    model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score
) -> float:
  """Run real-time inference for a single batch scenario."""
  X = pd.DataFrame([{
      "harvest_delay_hrs": harvest_delay_hours,
      "storage_temp_c": storage_temp_c,
      "humidity_pct": humidity_percent,
      "ripeness_score": ripeness_score,
  }])[FEATURE_COLUMNS]

  # Wrap DataFrame into a DMatrix object
  dmatrix_data = xgb.DMatrix(X)
  pred = float(model.predict(dmatrix_data)[0])
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


def predict_moisture(
    model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score
) -> float:
    """Run real-time inference for Moisture Content."""
    X = pd.DataFrame([{
        "ripeness_score": ripeness_score,
        "harvest_delay_hrs": harvest_delay_hours,
        "storage_temp_c": storage_temp_c,
        "humidity_pct": humidity_percent,
    }])[FEATURE_COLUMNS]

    # Wrap DataFrame into a DMatrix object
    dmatrix_data = xgb.DMatrix(X)
    pred = float(model.predict(dmatrix_data)[0])
    return max(0.0, pred)  # Moisture can't be negative


def predict_purity(
    model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score
) -> float:
    """Run real-time inference for Purity (DOBI Index)."""
    X = pd.DataFrame([{
        "ripeness_score": ripeness_score,
        "harvest_delay_hrs": harvest_delay_hours,
        "storage_temp_c": storage_temp_c,
        "humidity_pct": humidity_percent,
    }])[FEATURE_COLUMNS]

    # Wrap DataFrame into a DMatrix object
    dmatrix_data = xgb.DMatrix(X)
    pred = float(model.predict(dmatrix_data)[0])
    return max(0.0, pred)  # Purity score shouldn't be negative