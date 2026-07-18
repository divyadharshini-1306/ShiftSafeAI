"""
background.py — Live IoT ingestion loop for FastAPI
====================================================
Runs as an asyncio background task started at FastAPI startup.
Polls simulated IoT sensors every 5 minutes and writes to live_iot.db.

Usage in main.py (already wired):
    from background import run_ingestion_loop

    @app.on_event("startup")
    async def start_background_tasks():
        asyncio.create_task(run_ingestion_loop())
"""

import asyncio
import os
import random
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IOT_DB_PATH = os.path.join(BASE_DIR, "live_iot.db")

# Create connection and table once at module import
_conn = sqlite3.connect(IOT_DB_PATH, check_same_thread=False)
_conn.execute("""
    CREATE TABLE IF NOT EXISTS live_readings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT,
        temperature     REAL,
        humidity        REAL,
        toxic_gas_ppm   REAL,
        ambient_pm25    REAL,
        ambient_aqi     REAL,
        source          TEXT,
        sensor_ok       INTEGER
    )
""")
_conn.commit()


def _read_sensors(failure_rate: float = 0.05):
    """
    Simulate ESP8266 + DHT-11 + MQ-135 sensor readings.
    Returns None values with probability = failure_rate
    to mimic real hardware dropout behaviour.
    """
    if random.random() < failure_rate:
        return None, None, None, 0

    temperature   = round(random.uniform(20.0, 34.0), 1)   # °C Bengaluru range
    humidity      = round(random.uniform(40.0, 85.0), 1)   # %
    toxic_gas_ppm = round(random.uniform(5.0, 60.0), 1)    # MQ-135 ppm range
    return temperature, humidity, toxic_gas_ppm, 1


def _ingest_one_cycle() -> None:
    """
    One complete poll cycle:
    read simulated sensors + ambient data, insert one row into live_iot.db.
    """
    temperature, humidity, toxic_gas_ppm, sensor_ok = _read_sensors()

    # Simulated ambient readings (replace with real OpenWeather call if key available)
    ambient_pm25 = round(random.uniform(20.0, 100.0), 1)
    ambient_aqi  = round(random.uniform(50.0, 180.0), 1)

    _conn.execute(
        """
        INSERT INTO live_readings
            (timestamp, temperature, humidity, toxic_gas_ppm,
             ambient_pm25, ambient_aqi, source, sensor_ok)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            temperature,
            humidity,
            toxic_gas_ppm,
            ambient_pm25,
            ambient_aqi,
            "simulated",
            sensor_ok,
        ),
    )
    _conn.commit()


async def run_ingestion_loop() -> None:
    """
    Asyncio background task — runs indefinitely, polling every 5 minutes.
    Started via asyncio.create_task() in FastAPI's startup event.

    Uses await asyncio.sleep() NOT time.sleep() so FastAPI continues
    serving HTTP requests normally during the 5-minute wait between polls.
    """
    print("[ShiftSafe] IoT ingestion loop started — polling every 5 minutes")

    while True:
        try:
            _ingest_one_cycle()
        except Exception as exc:
            # Log the error but never crash the loop —
            # a single sensor failure must not stop all future ingestion.
            print(f"[ShiftSafe] Ingestion error (continuing): {exc}")

        await asyncio.sleep(300)  # 5 minutes
