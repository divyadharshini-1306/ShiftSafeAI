"""
main.py
FastAPI app exposing the 3 ShiftSafe AI endpoints described in README
Section 5. Wires together predict.py (Divyadarshini's models),
pipeline.py (Atharvi's sensor data) and exposure.py (the risk-scoring
logic, written here since the README only describes its rules, not
its code).
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import predict_aqi
from pipeline import init_db, get_latest_features
import exposure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "aqi_sensor.db")

app = FastAPI(title="ShiftSafe AI API")

# Section 7 — CORS, so Likhita's React frontend isn't blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this to her deployed frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ShiftSafe AI API is running. Visit /docs to explore the endpoints."
    }


@app.on_event("startup")
def startup_event():
    # README Mistake 5: open the DB connection once here, not per-request.
    init_db(DB_PATH)


# ── Request schemas ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    worker_role: str


class RiskScoreRequest(BaseModel):
    worker_role: str
    shift_duration_hours: float = Field(gt=0, le=24)


# ── Endpoint 1 — POST /predict ───────────────────────────────────────

@app.post("/predict")
def predict(req: PredictRequest):
    features = get_latest_features()
    predicted_aqi = predict_aqi(features)
    return {
        "predicted_aqi": predicted_aqi,
        "hour": features["hour"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ── Endpoint 2 — POST /risk-score ────────────────────────────────────

@app.post("/risk-score")
def risk_score(req: RiskScoreRequest):
    if req.worker_role not in exposure.MET_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown worker_role '{req.worker_role}'. "
                   f"Must be one of: {list(exposure.MET_VALUES.keys())}",
        )

    features = get_latest_features()
    predicted_aqi = predict_aqi(features)

    exposure_score = exposure.calculate_exposure_score(
        predicted_aqi, req.shift_duration_hours, req.worker_role
    )
    risk_tier, directive = exposure.get_risk_tier(exposure_score)

    return {
        "exposure_score": exposure_score,
        "risk_tier": risk_tier,
        "directive": directive,
        "predicted_aqi": predicted_aqi,
    }


# ── Endpoint 3 — GET /shift-plan ─────────────────────────────────────

@app.get("/shift-plan")
def shift_plan(
    worker_role: str,
    shift_start_hour: int = Query(ge=0, le=23),
):
    if worker_role not in exposure.MET_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown worker_role '{worker_role}'. "
                   f"Must be one of: {list(exposure.MET_VALUES.keys())}",
        )

    features = get_latest_features()
    baseline_aqi = predict_aqi(features)

    schedule = []
    for i in range(8):
        hour = (shift_start_hour + i) % 24
        projected_aqi = exposure.project_aqi_for_hour(baseline_aqi, hour)
        risk_level, intensity = exposure.get_shift_risk_level(projected_aqi)
        schedule.append({
            "hour": hour,
            "predicted_aqi": projected_aqi,
            "risk_level": risk_level,
            "recommended_intensity": intensity,
        })

    return {
        "worker_role": worker_role,
        "shift_start_hour": shift_start_hour,
        "schedule": schedule,
    }
