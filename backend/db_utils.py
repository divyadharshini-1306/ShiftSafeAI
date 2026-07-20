"""
db_utils.py — Database utility functions for demo endpoints
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "aqi_sensor.db")

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)


def get_recent_rows(n: int = 10) -> list[dict]:
    """
    Fetch the n most recent rows from sensor_data.
    Returns a list of dicts ordered newest first.
    """
    cursor = _conn.execute(
        f"""
        SELECT
            Datetime,
            "PM2.5", PM10, NO, NO2, NH3, CO, SO2, O3,
            AQI,
            hour, month, day_of_week, is_weekend, is_shift_hour,
            AQI_lag1, AQI_lag3, PM25_rolling6, AQI_rolling6
        FROM sensor_data
        ORDER BY rowid DESC
        LIMIT {n}
        """
    )

    columns = [
        "datetime",
        "PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3",
        "AQI",
        "hour", "month", "day_of_week", "is_weekend", "is_shift_hour",
        "AQI_lag1", "AQI_lag3", "PM25_rolling6", "AQI_rolling6",
    ]

    rows = []
    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))

    return rows