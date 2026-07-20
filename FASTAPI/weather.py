"""
weather.py
Pulls REAL pollutant data for Bengaluru from OpenWeatherMap's Air Pollution
API (current + forecast), instead of the random `random.uniform(...)`
placeholders that were in background.py before.

Requires an env var OPENWEATHER_API_KEY (get one free at
https://openweathermap.org/api/air-pollution).

OpenWeatherMap returns raw pollutant concentrations in ug/m3:
  co, no, no2, o3, so2, pm2_5, pm10, nh3
which map 1:1 onto the model's expected feature names (CO, NO, NO2,
O3, SO2, PM2.5, PM10, NH3). It also returns its own 1-5 AQI bucket,
which we ignore — the trained XGBoost/Bi-GRU model predicts the real
0-500 Indian-scale AQI from the raw pollutant features itself.
"""

import os
import requests

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

# Maps OpenWeatherMap's component keys -> the feature names the model expects
COMPONENT_MAP = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "no": "NO",
    "no2": "NO2",
    "nh3": "NH3",
    "co": "CO",
    "so2": "SO2",
    "o3": "O3",
}


def _api_key() -> str:
    key = os.environ.get("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not set. Add it as an environment "
            "variable on the backend host (Render dashboard -> Environment)."
        )
    return key


def fetch_current_pollution() -> dict:
    """
    Returns the current raw pollutant readings for Bengaluru as a dict
    with keys PM2.5, PM10, NO, NO2, NH3, CO, SO2, O3.
    Raises requests.HTTPError / RuntimeError on failure — callers should
    catch this and skip the ingestion cycle rather than insert junk data.
    """
    resp = requests.get(
        AIR_POLLUTION_URL,
        params={"lat": BENGALURU_LAT, "lon": BENGALURU_LON, "appid": _api_key()},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    components = data["list"][0]["components"]
    return {feature: float(components[owm_key]) for owm_key, feature in COMPONENT_MAP.items()}


def fetch_forecast_pollution(hours: int = 8) -> list[dict]:
    """
    Returns up to `hours` real forecasted hourly pollutant readings
    (OpenWeatherMap forecasts up to 96h ahead), each as:
      {"dt": <unix ts>, "PM2.5": ..., "PM10": ..., ...}
    """
    resp = requests.get(
        FORECAST_URL,
        params={"lat": BENGALURU_LAT, "lon": BENGALURU_LON, "appid": _api_key()},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    out = []
    for entry in data["list"][:hours]:
        components = entry["components"]
        row = {feature: float(components[owm_key]) for owm_key, feature in COMPONENT_MAP.items()}
        row["dt"] = entry["dt"]
        out.append(row)
    return out
