"""
Pearls AQI Predictor — Dashboard
==================================
Shows current AQI for Karachi, a 72h-ahead forecast from the trained model,
and a hazard alert banner. Deploy on Streamlit Community Cloud.

Secrets needed (Streamlit Cloud -> App settings -> Secrets):
    MONGODB_URI = "mongodb+srv://..."
    OPENWEATHER_API_KEY = "..."
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.database import Database
from src.aqi_utils import aqi_category

# ---------------------------------------------------------------------------
# Streamlit secrets -> environment variables (so src/config.py keeps working
# unchanged whether run locally with `set`/`export` or deployed on Cloud)
# ---------------------------------------------------------------------------
if "MONGODB_URI" in os.environ:
    pass  # already set locally via `set`/`export`
else:
    try:
        if "MONGODB_URI" in st.secrets:
            os.environ["MONGODB_URI"] = st.secrets["MONGODB_URI"]
        if "OPENWEATHER_API_KEY" in st.secrets:
            os.environ["OPENWEATHER_API_KEY"] = st.secrets["OPENWEATHER_API_KEY"]
    except Exception:
        pass  # no secrets.toml locally — fine, rely on real env vars instead

st.set_page_config(page_title="Pearls AQI Predictor — Karachi", page_icon="🌫️", layout="centered")

MODEL_PATH = "models/best_model.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"


# ---------------------------------------------------------------------------
# Cached loaders — avoid re-hitting Mongo / re-reading disk on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_cols


@st.cache_data(ttl=600)  # refresh at most every 10 minutes
def load_latest_features():
    db = Database()
    df = db.load_all_features()
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def hazard_alert(aqi_value: float):
    if aqi_value is None or np.isnan(aqi_value):
        return None
    if aqi_value <= 100:
        return None  # no alert needed for Good/Moderate
    elif aqi_value <= 150:
        return ("warning", "Unhealthy for Sensitive Groups — people with respiratory issues should limit outdoor exposure.")
    elif aqi_value <= 200:
        return ("warning", "Unhealthy — consider reducing prolonged outdoor activity.")
    elif aqi_value <= 300:
        return ("error", "Very Unhealthy — avoid outdoor activity where possible.")
    else:
        return ("error", "Hazardous — stay indoors, use air purification if available.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌫️ Pearls AQI Predictor")
st.caption("Karachi, Pakistan — 72-hour air quality forecast")

try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error(
        "No trained model found yet. Run the training pipeline at least once "
        "(`python -m src.training_pipeline`) so `models/best_model.joblib` exists "
        "and is committed to the repo."
    )
    st.stop()

try:
    df = load_latest_features()
except Exception as e:
    st.error(f"Could not load data from MongoDB: {e}")
    st.stop()

if df.empty:
    st.warning("Feature store is empty. Run the feature pipeline / backfill first.")
    st.stop()

latest = df.iloc[-1]
current_aqi = latest["aqi"]
current_category = aqi_category(current_aqi)
latest_time = latest["datetime"]

# --- Current AQI ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Current AQI", f"{current_aqi:.0f}", help=current_category)
    st.caption(f"Category: **{current_category}**")
with col2:
    st.metric("Last updated", latest_time.strftime("%Y-%m-%d %H:%M UTC"))

# --- 72h Forecast ---
st.subheader("72-Hour Forecast")

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Feature mismatch — missing columns for prediction: {missing}")
    st.stop()

latest_row = df[feature_cols].iloc[[-1]]  # keep as DataFrame (1 row)
if latest_row.isnull().any(axis=1).iloc[0]:
    st.warning(
        "The most recent row has missing lag/rolling features (needs 72h of prior "
        "history). Forecast may be unreliable until more hourly data accumulates."
    )

forecast_aqi = float(model.predict(latest_row)[0])
forecast_category = aqi_category(forecast_aqi)

delta = forecast_aqi - current_aqi
st.metric(
    "Predicted AQI (72h from now)",
    f"{forecast_aqi:.0f}",
    delta=f"{delta:+.0f} vs current",
    delta_color="inverse",  # rising AQI (worse air) shown as "bad" in red
)
st.caption(f"Predicted category: **{forecast_category}**")

# --- Hazard alert ---
alert = hazard_alert(forecast_aqi)
if alert:
    level, message = alert
    if level == "warning":
        st.warning(f"⚠️ Forecast alert: {message}")
    else:
        st.error(f"🚨 Forecast alert: {message}")
else:
    st.success("✅ No hazard expected in the 72h forecast.")

# --- Recent trend ---
st.subheader("Recent AQI Trend")
recent = df.tail(72)[["datetime", "aqi"]].set_index("datetime")
st.line_chart(recent)

st.caption(
    "AQI computed on the US EPA 0-500 scale from PM2.5 concentration. "
    "Forecast generated by the best-performing model (selected via time-series "
    "cross-validation) retrained daily via GitHub Actions."
)
