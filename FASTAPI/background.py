"""
background.py — Live ingestion loop, backed by real OpenWeatherMap data
=========================================================================
Runs as an asyncio background task started at FastAPI startup. Every 5
minutes it pulls REAL current pollutant readings for Bengaluru from
OpenWeatherMap, computes time + lag/rolling features from what's already
in sensor_data, runs the trained model, and inserts the new row.

Replaces the previous version, which wrote random.uniform() placeholders
into a different database/table (live_iot.db / live_readings) that
get_latest_features() in pipeline.py never read — which is why every
endpoint always returned the exact same stale prediction regardless of
role, duration, or time clicked.
"""

import asyncio
import os
import sqlite3
from datetime import datetime, timezone

import weather
from predict import predict_aqi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "aqi_sensor.db")

FEATURE_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3",
    "hour", "month", "day_of_week", "is_weekend", "is_shift_hour",
    "AQI_lag1", "AQI_lag3", "PM25_rolling6", "AQI_rolling6",
]

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)


def _time_features(dt: datetime) -> dict:
    return {
        "hour": dt.hour,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "is_weekend": int(dt.weekday() >= 5),
        "is_shift_hour": int(6 <= dt.hour <= 18),
    }


def _history_features(cur: sqlite3.Cursor) -> dict:
    cur.execute('SELECT AQI, "PM2.5" FROM sensor_data ORDER BY rowid DESC LIMIT 6')
    rows = cur.fetchall()

    if not rows:
        return {"AQI_lag1": 0.0, "AQI_lag3": 0.0, "PM25_rolling6": 0.0, "AQI_rolling6": 0.0}

    aqi_values = [r[0] for r in rows]
    pm25_values = [r[1] for r in rows]

    return {
        "AQI_lag1": float(aqi_values[0]),
        "AQI_lag3": float(aqi_values[2]) if len(aqi_values) > 2 else float(aqi_values[-1]),
        "PM25_rolling6": round(sum(pm25_values) / len(pm25_values), 4),
        "AQI_rolling6": round(sum(aqi_values) / len(aqi_values), 4),
    }


def _ingest_one_cycle() -> None:
    cur = _conn.cursor()

    pollutants = weather.fetch_current_pollution()
    now = datetime.now(timezone.utc)

    features = {}
    features.update(pollutants)
    features.update(_time_features(now))
    features.update(_history_features(cur))

    predicted_aqi = predict_aqi({col: features[col] for col in FEATURE_COLS})

    cur.execute(
        """
        INSERT INTO sensor_data
            ("City", "Datetime", "PM2.5", "PM10", "NO", "NO2", "NH3", "CO",
             "SO2", "O3", "AQI", "hour", "month", "day_of_week",
             "is_weekend", "is_shift_hour", "AQI_lag1", "AQI_lag3",
             "PM25_rolling6", "AQI_rolling6")
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Bengaluru",
            now.isoformat(timespec="seconds"),
            features["PM2.5"], features["PM10"], features["NO"], features["NO2"],
            features["NH3"], features["CO"], features["SO2"], features["O3"],
            predicted_aqi,
            features["hour"], features["month"], features["day_of_week"],
            features["is_weekend"], features["is_shift_hour"],
            features["AQI_lag1"], features["AQI_lag3"],
            features["PM25_rolling6"], features["AQI_rolling6"],
        ),
    )
    _conn.commit()


async def run_ingestion_loop() -> None:
    print("[ShiftSafe] Live OpenWeatherMap ingestion loop started - polling every 5 minutes")

    while True:
        try:
            _ingest_one_cycle()
        except Exception as exc:
            print(f"[ShiftSafe] Ingestion error (continuing): {exc}")

        await asyncio.sleep(300)
