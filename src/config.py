"""Central config for the AQI Predictor project. Reads secrets from env vars."""
import os

# --- Location ---
CITY_NAME = "Karachi"
LATITUDE = 24.8607
LONGITUDE = 67.0011

# --- API keys ---
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# --- MongoDB ---
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "aqi_karachi")
FEATURES_COLLECTION = "features"
MODELS_COLLECTION = "models"
PREDICTIONS_COLLECTION = "predictions"

# --- Open-Meteo (weather, no key needed) ---
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --- OpenWeather (pollutants) ---
OW_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OW_AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"


def require_env():
    missing = []
    if not OPENWEATHER_API_KEY:
        missing.append("OPENWEATHER_API_KEY")
    if not MONGODB_URI:
        missing.append("MONGODB_URI")
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
