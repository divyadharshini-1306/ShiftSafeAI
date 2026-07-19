"""
background.py — Live OpenWeather AQI ingestion loop for FastAPI
================================================================
Fetches real air quality data from OpenWeather API every 10 minutes
and inserts a new row into aqi_sensor.db so that get_latest_features()
always returns current Bengaluru conditions for model inference.

Environment variable required (set in Render dashboard):
    OPENWEATHER_API_KEY = your_key_here

The table sensor_data already exists in aqi_sensor.db — no creation needed.
"""

import asyncio
import os
import sqlite3
from datetime import datetime, timezone, timedelta

import requests

# ── Constants ────────────────────────────────────────────────────────────────

# Bengaluru coordinates
LAT = 12.9716
LON = 77.5946

# India Standard Time offset
IST = timezone(timedelta(hours=5, minutes=30))

# OpenWeather API key — set this in Render environment variables
# Never hardcode the key in source code
API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# Database path — same file pipeline.py reads from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "aqi_sensor.db")

# ── Database connection ───────────────────────────────────────────────────────
# check_same_thread=False because asyncio runs on a different thread than
# the connection creator in some FastAPI configurations
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)


# ── Step 1: Fetch from OpenWeather ───────────────────────────────────────────

def _fetch_openweather() -> dict | None:
    """
    Call the OpenWeather Air Pollution API for Bengaluru.
    Returns the raw components dict or None if the call fails.

    The components dict contains:
        co, no, no2, o3, so2, pm2_5, pm10, nh3
    All values are in µg/m³ (micrograms per cubic metre).
    """
    if not API_KEY:
        print("[ShiftSafe] OPENWEATHER_API_KEY not set — skipping fetch")
        return None

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # components is nested under list[0]
        components = data["list"][0]["components"]
        return components

    except requests.exceptions.Timeout:
        print("[ShiftSafe] OpenWeather API timed out — will retry next cycle")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"[ShiftSafe] OpenWeather API error: {exc} — will retry next cycle")
        return None
    except (KeyError, IndexError) as exc:
        print(f"[ShiftSafe] Unexpected API response structure: {exc}")
        return None


# ── Step 2: Map OpenWeather fields to model column names ─────────────────────

def _map_pollutants(components: dict) -> dict:
    """
    Rename OpenWeather component keys to match the exact column names
    the ML model was trained on.

    OpenWeather key  →  Model column
    pm2_5            →  PM2.5      (underscore vs dot — critical mapping)
    pm10             →  PM10
    no               →  NO
    no2              →  NO2
    nh3              →  NH3
    co               →  CO
    so2              →  SO2
    o3               →  O3
    """
    return {
        "PM2.5": float(components["pm2_5"]),
        "PM10":  float(components["pm10"]),
        "NO":    float(components["no"]),
        "NO2":   float(components["no2"]),
        "NH3":   float(components["nh3"]),
        "CO":    float(components["co"]),
        "SO2":   float(components["so2"]),
        "O3":    float(components["o3"]),
    }


# ── Step 3: Compute India CPCB AQI from PM2.5 ────────────────────────────────

def _compute_cpcb_aqi(pm25: float) -> float:
    """
    Compute India's CPCB AQI sub-index from PM2.5 concentration (µg/m³).

    OpenWeather's data['main']['aqi'] uses a 1–5 European scale — NOT
    India's 0–500 CPCB scale. We must compute the Indian AQI ourselves
    so that AQI_lag1, AQI_lag3, and AQI_rolling6 are on the correct scale.

    CPCB PM2.5 breakpoints:
        PM2.5 (µg/m³)    AQI range
        0   – 30         0   – 50
        30  – 60         51  – 100
        60  – 90         101 – 200
        90  – 120        201 – 300
        120 – 250        301 – 400
        250+             401 – 500
    """
    breakpoints = [
        # (pm25_low, pm25_high, aqi_low, aqi_high)
        (0,   30,  0,   50),
        (30,  60,  51,  100),
        (60,  90,  101, 200),
        (90,  120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ]

    pm25 = max(0.0, pm25)  # clamp negative values

    for (c_low, c_high, i_low, i_high) in breakpoints:
        if pm25 <= c_high:
            # Linear interpolation within the breakpoint band
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)

    # Above 500 µg/m³ — cap at 500
    return 500.0


# ── Step 4: Compute time-based features ──────────────────────────────────────

def _compute_time_features() -> dict:
    """
    Derive the 5 time-based model features from current IST datetime.

    All time features use Indian Standard Time (UTC+5:30) because the
    model was trained on Bengaluru data where timestamps are in IST.
    Using UTC here would shift hour by 5.5 hours and corrupt predictions.
    """
    now_ist = datetime.now(IST)

    hour        = now_ist.hour                          # 0–23
    month       = now_ist.month                         # 1–12
    day_of_week = now_ist.weekday()                     # 0=Monday, 6=Sunday
    is_weekend  = 1 if day_of_week >= 5 else 0          # 1 if Sat/Sun
    is_shift_hour = 1 if 6 <= hour <= 18 else 0         # 1 if industrial shift

    return {
        "hour":          hour,
        "month":         month,
        "day_of_week":   day_of_week,
        "is_weekend":    is_weekend,
        "is_shift_hour": is_shift_hour,
    }


# ── Step 5: Compute lag and rolling features from existing DB rows ────────────

def _compute_lag_rolling(current_pm25: float, current_aqi: float) -> dict:
    """
    Compute AQI_lag1, AQI_lag3, PM25_rolling6, AQI_rolling6
    from the most recent rows already in sensor_data.

    Edge case — fewer than N rows available (e.g. first few API calls):
    We use whatever rows exist. If zero rows exist we fall back to the
    current values so the insert can still proceed without NaN.
    """
    # Fetch the 6 most recent AQI and PM2.5 values from sensor_data
    # ORDER BY rowid DESC gives newest first
    cursor = _conn.execute(
        """
        SELECT "AQI", "PM2.5"
        FROM sensor_data
        ORDER BY rowid DESC
        LIMIT 6
        """
    )
    rows = cursor.fetchall()
    # rows[0] = most recent, rows[1] = second most recent, etc.

    # AQI_lag1: AQI of the most recent existing row
    # If no rows exist yet, use current AQI as the initial value
    aqi_lag1 = float(rows[0][0]) if len(rows) >= 1 else current_aqi

    # AQI_lag3: AQI of the 3rd most recent existing row
    aqi_lag3 = float(rows[2][0]) if len(rows) >= 3 else current_aqi

    # PM25_rolling6: average PM2.5 over last 6 rows
    # If fewer than 6 rows available, average whatever we have
    # Include the current reading in the average for better accuracy
    if rows:
        pm25_values = [float(r[1]) for r in rows] + [current_pm25]
        aqi_values  = [float(r[0]) for r in rows] + [current_aqi]
    else:
        # First ever call — no history yet
        pm25_values = [current_pm25]
        aqi_values  = [current_aqi]

    pm25_rolling6 = round(sum(pm25_values[-6:]) / len(pm25_values[-6:]), 3)
    aqi_rolling6  = round(sum(aqi_values[-6:])  / len(aqi_values[-6:]),  3)

    return {
        "AQI_lag1":      round(aqi_lag1, 3),
        "AQI_lag3":      round(aqi_lag3, 3),
        "PM25_rolling6": pm25_rolling6,
        "AQI_rolling6":  aqi_rolling6,
    }


# ── Step 6: Assemble and insert the new row ───────────────────────────────────

def _insert_new_row(
    pollutants: dict,
    time_features: dict,
    lag_rolling: dict,
    aqi: float,
) -> None:
    """
    Insert one complete row into sensor_data.

    Column order matches the table schema in aqi_sensor.db exactly.
    City is always Bengaluru. Datetime is stored as IST ISO string
    to be consistent with the original training data timestamps.
    """
    now_ist = datetime.now(IST)

    _conn.execute(
        """
        INSERT INTO sensor_data (
            City, Datetime,
            "PM2.5", PM10, NO, NO2, NH3, CO, SO2, O3,
            AQI,
            hour, month, day_of_week, is_weekend, is_shift_hour,
            AQI_lag1, AQI_lag3, PM25_rolling6, AQI_rolling6
        ) VALUES (
            ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        (
            # Identity
            "Bengaluru",
            now_ist.isoformat(),

            # Pollutants
            pollutants["PM2.5"],
            pollutants["PM10"],
            pollutants["NO"],
            pollutants["NO2"],
            pollutants["NH3"],
            pollutants["CO"],
            pollutants["SO2"],
            pollutants["O3"],

            # Computed AQI (CPCB scale)
            aqi,

            # Time features
            time_features["hour"],
            time_features["month"],
            time_features["day_of_week"],
            time_features["is_weekend"],
            time_features["is_shift_hour"],

            # Lag and rolling features
            lag_rolling["AQI_lag1"],
            lag_rolling["AQI_lag3"],
            lag_rolling["PM25_rolling6"],
            lag_rolling["AQI_rolling6"],
        ),
    )
    _conn.commit()

    print(
        f"[ShiftSafe] Inserted new row — "
        f"IST: {now_ist.strftime('%Y-%m-%d %H:%M')} | "
        f"PM2.5: {pollutants['PM2.5']} | "
        f"AQI: {aqi} | "
        f"hour: {time_features['hour']} | "
        f"lag1: {lag_rolling['AQI_lag1']}"
    )


# ── Step 7: One complete ingestion cycle ──────────────────────────────────────

def _ingest_one_cycle() -> bool:
    """
    One full cycle: fetch → map → compute AQI → time features →
    lag/rolling → insert.

    Returns True if successful, False if the API call failed.
    """
    # Fetch real data from OpenWeather
    components = _fetch_openweather()
    if components is None:
        # API call failed — skip this cycle, keep old data in DB
        # get_latest_features() will continue using the last good row
        return False

    # Map to model column names
    pollutants = _map_pollutants(components)

    # Compute India CPCB AQI from PM2.5
    aqi = _compute_cpcb_aqi(pollutants["PM2.5"])

    # Compute time features using current IST
    time_features = _compute_time_features()

    # Compute lag and rolling from existing DB rows
    lag_rolling = _compute_lag_rolling(
        current_pm25=pollutants["PM2.5"],
        current_aqi=aqi,
    )

    # Insert the complete row
    _insert_new_row(pollutants, time_features, lag_rolling, aqi)

    return True


# ── Step 8: Asyncio background loop ──────────────────────────────────────────

async def run_ingestion_loop() -> None:
    """
    Asyncio background task — runs indefinitely, fetching every 10 minutes.
    Started via asyncio.create_task() in FastAPI's startup event in main.py.

    On startup: fetches immediately so the very first /predict call
    gets current data rather than the last historical row from 2020.

    Uses await asyncio.sleep() NOT time.sleep() so FastAPI continues
    serving HTTP requests normally during the 10-minute wait.
    """
    print("[ShiftSafe] Live OpenWeather ingestion loop started")
    print(f"[ShiftSafe] Fetching Bengaluru AQI every 10 minutes")
    print(f"[ShiftSafe] API key present: {bool(API_KEY)}")

    while True:
        try:
            success = _ingest_one_cycle()
            if not success:
                print("[ShiftSafe] Cycle skipped — will retry in 10 minutes")
        except Exception as exc:
            # Catch anything unexpected — never let the loop die
            print(f"[ShiftSafe] Unexpected error in ingestion cycle: {exc}")

        # Wait 10 minutes before next fetch
        # await yields control back to FastAPI's event loop during the wait
        await asyncio.sleep(600)
        