"""
Feature pipeline for Pearls AQI Predictor (Karachi).

- Weather: Open-Meteo (free, no key, with historical archive access)
- Pollutants: OpenWeather Air Pollution API
- Target: US EPA AQI (0-500) computed from PM2.5
- Storage: MongoDB Atlas, with the features collection acting as the store

Usage:
    python -m src.feature_pipeline                 # current hour for the cron job
    python -m src.feature_pipeline --backfill 60    # backfill the last 60 days
"""
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd

from . import config
from .database import Database
from .aqi_utils import pm25_to_aqi


# ---------------------------------------------------------------------------
# Fetching pollutants from OpenWeather.
# ---------------------------------------------------------------------------
def fetch_current_pollution():
    params = {"lat": config.LATITUDE, "lon": config.LONGITUDE, "appid": config.OPENWEATHER_API_KEY}
    r = requests.get(config.OW_AIR_POLLUTION_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["list"]


def fetch_historical_pollution(start_unix, end_unix):
    params = {
        "lat": config.LATITUDE, "lon": config.LONGITUDE,
        "start": start_unix, "end": end_unix, "appid": config.OPENWEATHER_API_KEY,
    }
    r = requests.get(config.OW_AIR_POLLUTION_HISTORY_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["list"]


def pollution_list_to_df(records):
    rows = []
    for rec in records:
        dt = datetime.fromtimestamp(rec["dt"], tz=timezone.utc)
        comp = rec["components"]
        rows.append({
            "datetime": dt,
            "co": comp.get("co"), "no": comp.get("no"), "no2": comp.get("no2"),
            "o3": comp.get("o3"), "so2": comp.get("so2"),
            "pm2_5": comp.get("pm2_5"), "pm10": comp.get("pm10"), "nh3": comp.get("nh3"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fetching weather from Open-Meteo, which does not need a key.
# ---------------------------------------------------------------------------
def fetch_weather_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather for a date range given as 'YYYY-MM-DD'.

    This uses the historical archive endpoint, which also covers today with a
    short delay, so it works for both backfills and hourly runs.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover,surface_pressure",
        "timezone": "UTC",
    }
    r = requests.get(config.OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["time"], utc=True),
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "wind_speed": data["wind_speed_10m"],
        "wind_direction": data["wind_direction_10m"],
        "cloud_cover": data["cloud_cover"],
        "pressure": data["surface_pressure"],
    })
    return df


# ---------------------------------------------------------------------------
# Feature engineering.
# ---------------------------------------------------------------------------
def add_time_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    # Cyclical encodings help the model see that hour 23 is close to hour 0.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_target_and_lags(df):
    """Assume df is sorted by datetime and roughly follows an hourly cadence."""
    df = df.sort_values("datetime").reset_index(drop=True)

    # Build the AQI target from PM2.5.
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)

    # Lag features from 24, 48, and 72 hours ago keep us from leaking the future.
    for lag_h in (24, 48, 72):
        df[f"aqi_lag_{lag_h}h"] = df["aqi"].shift(lag_h)
        df[f"pm2_5_lag_{lag_h}h"] = df["pm2_5"].shift(lag_h)

    # Rolling stats over a 72-hour window.
    df["aqi_rolling_mean_72h"] = df["aqi"].rolling(window=72, min_periods=1).mean()
    df["aqi_rolling_std_72h"] = df["aqi"].rolling(window=72, min_periods=1).std()
    df["pm2_5_rolling_mean_72h"] = df["pm2_5"].rolling(window=72, min_periods=1).mean()

    # Short-term change rate.
    df["aqi_change_rate_1h"] = df["aqi"].diff(1)

    # The training target is AQI 72 hours ahead.
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    return df


def build_feature_frame(pollution_df: pd.DataFrame, weather_df: pd.DataFrame, recompute_derived: bool = True) -> pd.DataFrame:
    # Snap both sources to the hour before merging.
    pollution_df["datetime"] = pollution_df["datetime"].dt.floor("h")
    weather_df["datetime"] = weather_df["datetime"].dt.floor("h")
    merged = pd.merge(pollution_df, weather_df, on="datetime", how="inner")
    merged = add_time_features(merged)
    if recompute_derived:
        merged = add_target_and_lags(merged)
    else:
        # Only the AQI target is needed here; lag and rolling features are
        # rebuilt later against the full history in run_current.
        merged["aqi"] = merged["pm2_5"].apply(pm25_to_aqi)
    return merged


# ---------------------------------------------------------------------------
# Main flows.
# ---------------------------------------------------------------------------
def run_current(db: Database):
    """Fetch the latest pollution and weather, rebuild derived features, and
    store the newest row.

    Lag and rolling features need prior history to be correct. If we computed
    them from only the fresh row, they would all be NaN, so this pulls about
    80 hours of history from Mongo first and then rebuilds the features.
    """
    pollution_records = fetch_current_pollution()
    pollution_df = pollution_list_to_df(pollution_records)

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    weather_df = fetch_weather_range(today, today)

    new_rows = build_feature_frame(pollution_df, weather_df, recompute_derived=False)
    if new_rows.empty:
        print("[current] No matching pollution+weather rows this hour (weather data may lag). Skipping.")
        return

    # Pull a small history buffer so the lag and rolling features can be
    # rebuilt correctly for the new row(s).
    history_df = db.load_recent_features(hours=80)
    combined = pd.concat([history_df, new_rows[["datetime"] + [c for c in new_rows.columns if c not in history_df.columns or c == "datetime"]]], ignore_index=True) if not history_df.empty else new_rows
    combined = combined.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    combined = add_target_and_lags(combined)

    # Only insert the newest rows that match the timestamps we just fetched.
    new_timestamps = set(new_rows["datetime"])
    to_insert = combined[combined["datetime"].isin(new_timestamps)]

    records = to_insert.replace({np.nan: None}).to_dict("records")
    inserted = db.upsert_features(records)
    print(f"[current] Fetched {len(records)} row(s), inserted {inserted} new. Total in store: {db.count_features()}")


def run_backfill(db: Database, days: int):
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    print(f"[backfill] {config.CITY_NAME}: {start.date()} -> {end.date()}")

    # Weather data comes from one call for the whole range.
    weather_df = fetch_weather_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"[backfill] weather rows: {len(weather_df)}")

    # Pollutant data is fetched in 7-day chunks to stay within API limits.
    all_pollution = []
    cursor = start
    chunk = timedelta(days=7)
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        try:
            recs = fetch_historical_pollution(int(cursor.timestamp()), int(chunk_end.timestamp()))
            all_pollution.extend(recs)
            print(f"  pollution {cursor.date()} -> {chunk_end.date()}: {len(recs)} records")
        except requests.HTTPError as e:
            print(f"  [warn] pollution fetch failed {cursor.date()} -> {chunk_end.date()}: {e}")
        cursor = chunk_end
        time.sleep(1)

    if not all_pollution:
        print("[backfill] No pollution data retrieved — check API key / plan.")
        return

    pollution_df = pollution_list_to_df(all_pollution)
    merged = build_feature_frame(pollution_df, weather_df)
    print(f"[backfill] merged feature rows: {len(merged)}")

    records = merged.replace({np.nan: None}).to_dict("records")
    inserted = db.upsert_features(records)
    print(f"[backfill] inserted {inserted} new rows. Total in store: {db.count_features()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=0, help="days of history to backfill")
    args = parser.parse_args()

    config.require_env()
    db = Database()

    if args.backfill > 0:
        run_backfill(db, args.backfill)
    else:
        run_current(db)
