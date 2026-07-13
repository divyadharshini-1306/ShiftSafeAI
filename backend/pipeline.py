
import os
import sqlite3

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "aqi_sensor.db")

# Connect once when the module is imported, not per API request.
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

FEATURE_COLS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "is_shift_hour",
    "AQI_lag1",
    "AQI_lag3",
    "PM25_rolling6",
    "AQI_rolling6",
]

INT_COLS = {
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "is_shift_hour",
}


def get_latest_features() -> dict:
    """
    Return the newest database row as exactly 17 raw, unscaled,
    model-ready features with correct Python numeric types.
    """
    row = pd.read_sql_query(
        "SELECT * FROM sensor_data ORDER BY rowid DESC LIMIT 1",
        conn,
    )

    if row.empty:
        raise RuntimeError("sensor_data table is empty.")

    features = {}
    for column in FEATURE_COLS:
        value = row[column].iloc[0]
        features[column] = int(value) if column in INT_COLS else float(value)

    return features


def get_live_data() -> dict:
    """Alias retained for endpoint readability."""
    return get_latest_features()
