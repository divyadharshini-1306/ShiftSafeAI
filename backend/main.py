import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from background import run_ingestion_loop
from exposure import MET_VALUES, calculate_exposure, get_risk_tier
from pipeline import get_latest_features
from predict import predict_aqi
from db_utils import get_recent_rows
from exposure import MET_VALUES, calculate_exposure, calculate_shift_exposure, get_risk_tier

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


def hourly_aqi_offset(hour: int) -> float:
    """
    AQI offset per hour based on observed Bengaluru patterns (EDA finding).
    Evening hours 17-23 carry the highest exposure risk.
    Morning hours 7-10 are the cleanest window.
    """
    if 7 <= hour <= 10:
        return -8.0   # morning clean window — best time for heavy outdoor work
    if 11 <= hour <= 16:
        return 0.0    # midday baseline
    if 17 <= hour <= 23:
        return 12.0   # evening peak — worst for workers (FIX 3: was 17-22)
    return -3.0       # midnight to 6am — low activity, moderate AQI


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

@app.get("/demo/latest-readings")
def latest_readings(n: int = 10):
    """
    Demo endpoint — shows the n most recent rows in aqi_sensor.db.
    Use this to verify the OpenWeather ingestion pipeline is working.
    The newest row should have a Datetime close to the current IST time.
    Default: last 10 rows. Max recommended: 50.
    """
    if n > 50:
        raise HTTPException(
            status_code=400,
            detail="n must be 50 or less to keep the response readable."
        )

    rows = get_recent_rows(n)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No rows found in sensor_data table."
        )

    return {
        "total_rows_returned": len(rows),
        "note": "Ordered newest first. Datetime is IST. AQI is CPCB scale (0-500).",
        "readings": rows,
    }
@app.post("/risk-score")
def risk_score(request: RiskScoreRequest):
    role = validate_worker_role(request.worker_role)
    predicted_aqi, _ = get_current_prediction()
    now_utc = datetime.now(timezone.utc)

    # Use new calculate_exposure — now accounts for fatigue and AQI penalty
    exposure_score = calculate_exposure(
        predicted_aqi=predicted_aqi,
        shift_duration_hours=request.shift_duration_hours,
        worker_role=role,
    )
    risk = get_risk_tier(exposure_score)

    return {
        "exposure_score":  exposure_score,
        "risk_tier":       risk["tier"],
        "directive":       risk["directive"],
        "predicted_aqi":   predicted_aqi,
        "timestamp_utc":   now_utc.isoformat(),
        "timestamp_ist":   now_utc.astimezone(IST).isoformat(),
        "model_note": (
            "Exposure score uses non-linear AQI penalty (power 1.2) "
            "and step-based fatigue progression."
        ),
    }
class ShiftExposureRequest(BaseModel):
    worker_role: str = Field(
        ...,
        description="construction, traffic_police, factory, or delivery",
        examples=["construction"],
    )
    hourly_aqi_list: list[float] = Field(
        ...,
        description="Predicted AQI for each hour of the shift. Length = shift duration.",
        examples=[[87.2, 89.1, 94.5, 98.3, 92.1, 88.7, 105.2, 110.4]],
    )
    breaks: list[dict] | None = Field(
        default=None,
        description=(
            "Optional indoor recovery breaks. Each break: "
            "{'after_hour': int, 'duration_min': float}"
        ),
        examples=[[{"after_hour": 4, "duration_min": 45}]],
    )


@app.post("/shift-exposure")
def shift_exposure(request: ShiftExposureRequest):
    """
    Detailed per-hour cumulative exposure breakdown for a full shift.
    Accounts for fatigue progression, non-linear AQI penalty, and
    recovery breaks. Use this endpoint to power the dashboard's
    Exposure Budget Meter with full hourly detail.
    """
    role = validate_worker_role(request.worker_role)

    if len(request.hourly_aqi_list) > 12:
        raise HTTPException(
            status_code=400,
            detail="hourly_aqi_list cannot exceed 12 hours."
        )

    result = calculate_shift_exposure(
        hourly_aqi_list=request.hourly_aqi_list,
        worker_role=role,
        breaks=request.breaks,
    )

    return result

