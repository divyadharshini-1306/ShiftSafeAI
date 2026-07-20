import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import sqlite3

from background import run_ingestion_loop, _history_features, _time_features, _ingest_one_cycle
from exposure import MET_VALUES, calculate_exposure_score, get_risk_tier
from pipeline import get_latest_features
from predict import predict_aqi
import weather

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aqi_sensor.db")


app = FastAPI(
    title="ShiftSafe AI Backend",
    description="AQI prediction and worker exposure-risk API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: begin live IoT ingestion loop ───────────────────────────────────
@app.on_event("startup")
async def start_background_tasks():
    # Populate one real row immediately so the first request after a cold
    # start already has live data instead of waiting up to 5 minutes.
    try:
        await asyncio.get_event_loop().run_in_executor(None, _ingest_one_cycle)
    except Exception as exc:
        print(f"[ShiftSafe] Initial ingestion failed (will retry on schedule): {exc}")
    asyncio.create_task(run_ingestion_loop())


# ── Pydantic request models ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    worker_role: str = Field(
        ...,
        description="construction, traffic_police, factory, or delivery",
        examples=["construction"],
    )


class RiskScoreRequest(BaseModel):
    worker_role: str = Field(
        ...,
        description="construction, traffic_police, factory, or delivery",
        examples=["construction"],
    )
    shift_duration_hours: float = Field(
        ...,
        gt=0,
        le=24,
        description="Shift duration in hours, greater than 0 and at most 24",
        examples=[8],
    )


# ── Shared helpers ───────────────────────────────────────────────────────────
def validate_worker_role(worker_role: str) -> str:
    role = worker_role.strip().lower()
    if role not in MET_VALUES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported worker_role '{worker_role}'. "
                f"Choose one of: {', '.join(MET_VALUES)}"
            ),
        )
    return role


def get_current_prediction() -> tuple[float, dict]:
    """Database → 17 raw features → XGBoost + Bi-GRU ensemble prediction."""
    try:
        features = get_latest_features()
        predicted_aqi = predict_aqi(features)
        return predicted_aqi, features
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )


def get_shift_risk_level(predicted_aqi: float) -> str:
    if predicted_aqi < 100:
        return "Safe"
    if predicted_aqi <= 150:
        return "Moderate"
    return "High"


def recommended_intensity(risk_level: str, hour: int) -> str:
    if risk_level == "Safe":
        return "Heavy"
    if risk_level == "Moderate":
        return "Moderate"
    return "Rest" if hour >= 18 else "Light"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ShiftSafe AI Backend",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/risk-score", "/shift-plan", "/health", "/docs"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost + Bi-GRU ensemble",
        "ensemble_weights": {"xgboost": 0.9, "bigru": 0.1},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    validate_worker_role(request.worker_role)
    predicted_aqi, features = get_current_prediction()
    return {
        "predicted_aqi": predicted_aqi,
        "hour": features["hour"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/risk-score")
def risk_score(request: RiskScoreRequest):
    role = validate_worker_role(request.worker_role)
    predicted_aqi, _ = get_current_prediction()

    exposure_score = calculate_exposure_score(
        predicted_aqi=predicted_aqi,
        shift_duration_hours=request.shift_duration_hours,
        worker_role=role,
    )
    risk = get_risk_tier(exposure_score)

    return {
        "exposure_score": exposure_score,
        "risk_tier": risk["tier"],
        "directive": risk["directive"],
        "predicted_aqi": predicted_aqi,
    }


@app.get("/shift-plan")
def shift_plan(
    worker_role: str,
    shift_start_hour: int = Query(
        ...,
        ge=0,
        le=23,
        description="Local shift starting hour in 24-hour format",
        examples=[6],
    ),
):
    role = validate_worker_role(worker_role)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    try:
        forecast_rows = weather.fetch_forecast_pollution(hours=8)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch OpenWeatherMap forecast: {exc}",
        )

    history = _history_features(cur)
    aqi_history = [history["AQI_lag1"]] * 6
    pm25_history = []

    schedule = []
    for offset, forecast_row in enumerate(forecast_rows):
        hour = (shift_start_hour + offset) % 24
        dt = datetime.fromtimestamp(forecast_row["dt"], tz=timezone.utc)

        pm25_history.append(forecast_row["PM2.5"])
        pm25_window = pm25_history[-6:]

        features = {
            "PM2.5": forecast_row["PM2.5"], "PM10": forecast_row["PM10"],
            "NO": forecast_row["NO"], "NO2": forecast_row["NO2"],
            "NH3": forecast_row["NH3"], "CO": forecast_row["CO"],
            "SO2": forecast_row["SO2"], "O3": forecast_row["O3"],
            **_time_features(dt),
            "AQI_lag1": aqi_history[-1],
            "AQI_lag3": aqi_history[-3] if len(aqi_history) >= 3 else aqi_history[-1],
            "PM25_rolling6": round(sum(pm25_window) / len(pm25_window), 4),
            "AQI_rolling6": round(sum(aqi_history[-6:]) / len(aqi_history[-6:]), 4),
        }

        hour_aqi = predict_aqi(features)
        aqi_history.append(hour_aqi)

        risk_level = get_shift_risk_level(hour_aqi)
        schedule.append({
            "hour": hour,
            "predicted_aqi": hour_aqi,
            "risk_level": risk_level,
            "recommended_intensity": recommended_intensity(risk_level, hour),
        })

    return {
        "worker_role": role,
        "shift_start_hour": shift_start_hour,
        "schedule": schedule,
    }
