"""
pipeline.py
Wraps Atharvi's get_latest_features() logic. The only change from the
README's version is that the SQLite connection is opened through an
explicit init_db() call (made once, from main.py's startup event)
instead of as a bare module-level statement — this is README's own
"Mistake 5", just addressed directly.
"""

import sqlite3
import pandas as pd

FEATURE_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3',
    'hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour',
    'AQI_lag1', 'AQI_lag3', 'PM25_rolling6', 'AQI_rolling6'
]

_conn = None


def init_db(db_path: str) -> None:
    """Open the SQLite connection once. Call this from app startup."""
    global _conn
    # check_same_thread=False because FastAPI's sync route handlers run
    # in a thread pool, so this connection may be touched from more than
    # one thread across requests.
    _conn = sqlite3.connect(db_path, check_same_thread=False)


def get_latest_features() -> dict:
    """Returns the latest sensor row as the exact 17-key dict predict_aqi() expects."""
    if _conn is None:
        raise RuntimeError(
            "Database connection not initialised — call init_db(db_path) "
            "at app startup before hitting any endpoint that needs it."
        )

    row = pd.read_sql_query(
        'SELECT * FROM sensor_data ORDER BY rowid DESC LIMIT 1',
        _conn
    )
    if row.empty:
        raise RuntimeError('No data in sensor_data table yet.')

    features = {}
    for col in FEATURE_COLS:
        val = row[col].iloc[0]
        if col in ('hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour'):
            features[col] = int(val)
        else:
            features[col] = float(val)
    return features
