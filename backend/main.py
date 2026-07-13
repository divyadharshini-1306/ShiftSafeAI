
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from exposure import MET_VALUES, calculate_exposure, get_risk_tier
from pipeline import get_latest_features
from predict import predict_aqi


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
    """Database -> 17 raw features -> loaded XGBoost + Bi-GRU ensemble."""
    try:
        features = get_latest_features()
        predicted_aqi = predict_aqi(features)
        return predicted_aqi, features

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )


def hourly_aqi_offset(hour: int) -> float:
    """
    Simple schedule adjustment based on observed hourly AQI patterns.
    Hours are in 24-hour local time.
    """
    if 7 <= hour <= 10:
        return -8.0
    if 11 <= hour <= 16:
        return 0.0
    if 17 <= hour <= 22:
        return 12.0
    return -3.0


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


@app.get("/")
def root():
    return {
        "service": "ShiftSafe AI Backend",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost + Bi-GRU ensemble",
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

    exposure_score = calculate_exposure(
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
    base_aqi, _ = get_current_prediction()

    schedule = []

    for offset in range(8):
        hour = (shift_start_hour + offset) % 24
        hour_aqi = round(max(0.0, base_aqi + hourly_aqi_offset(hour)), 1)
        risk_level = get_shift_risk_level(hour_aqi)

        schedule.append(
            {
                "hour": hour,
                "predicted_aqi": hour_aqi,
                "risk_level": risk_level,
                "recommended_intensity": recommended_intensity(risk_level, hour),
            }
        )

    return {
        "worker_role": role,
        "shift_start_hour": shift_start_hour,
        "schedule": schedule,
    }
