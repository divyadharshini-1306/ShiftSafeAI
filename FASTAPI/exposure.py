"""
exposure.py
MET values, the exposure-score formula, and risk-tier / intensity
mappings used by /risk-score and /shift-plan.

A note on the thresholds below: they follow the *stated rules* in
README Section 5, not the worked examples shown in the same section.
The worked examples don't actually satisfy the rules as written:
  - Endpoint 2 example: 94.5 (AQI) x 8 (hours) x 5.0 (construction MET)
    = 3780, not the 423.6 shown in the sample response.
  - Endpoint 3 example: hours with AQI 94.5/98.3 (under 100) are
    labelled "Moderate" instead of "Safe", and hours with AQI
    105.2/110.4 (under 150) are labelled "High" instead of "Moderate".
These look like placeholder numbers that were never recalculated
against the formulas, rather than an intentional alternate rule. This
file implements the explicit formulas/thresholds from Section 5 since
those are the only unambiguous spec. Flag this to Divyadarshini /
Atharvi if the example numbers were meant to be authoritative instead.
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Endpoint 2 — POST /risk-score
# ---------------------------------------------------------------------------

MET_VALUES = {
    "construction": 5.0,     # heavy outdoor physical labour
    "traffic_police": 3.0,   # standing and walking outdoors
    "factory": 4.0,           # moderate indoor/outdoor assembly
    "delivery": 6.0,           # cycling with load
}


def calculate_exposure_score(
    predicted_aqi: float, shift_duration_hours: float, worker_role: str
) -> float:
    """exposure_score = predicted_aqi x shift_duration_hours x MET_value"""
    if worker_role not in MET_VALUES:
        raise ValueError(
            f"Unknown worker_role '{worker_role}'. "
            f"Must be one of: {list(MET_VALUES.keys())}"
        )
    met = MET_VALUES[worker_role]
    return round(predicted_aqi * shift_duration_hours * met, 1)


def get_risk_tier(exposure_score: float) -> Tuple[str, str]:
    """
    Maps a cumulative exposure score to (risk_tier, directive).
    Boundaries per README Section 5, Endpoint 2:
      Safe:     < 200
      Moderate: 200 - 400
      High:     400 - 600
      Critical: > 600
    """
    if exposure_score < 200:
        return "Safe", "Normal operations."
    elif exposure_score < 400:
        return "Moderate", "Wear dust mask. Hydrate regularly."
    elif exposure_score < 600:
        return "High", "Wear N95 mask. Limit outdoor tasks to 30-minute intervals."
    else:
        return "Critical", "Halt all outdoor operations immediately."


# ---------------------------------------------------------------------------
# Endpoint 3 — GET /shift-plan
# ---------------------------------------------------------------------------

# Placeholder diurnal pattern: a generic two-peak urban traffic curve
# (morning rush ~8am, evening rush ~6-7pm). predict_aqi() only forecasts
# ONE hour ahead from current sensor readings, so there's no real
# multi-hour-ahead model yet — this is a stand-in multiplier applied to
# that single baseline prediction, exactly as README Section 5 Endpoint 3
# describes ("apply a simple offset to simulate how AQI changes").
#
# Replace this with Atharvi's actual finding once she has it — a one-line
# `df.groupby('hour')['AQI'].mean()` on the 48,189-row dataset will give
# you the real per-hour pattern instead of this hand-picked curve.
HOURLY_OFFSET_PCT = {
    0: -15, 1: -18, 2: -20, 3: -20, 4: -15, 5: -10,
    6: -5,  7: 5,   8: 12,  9: 8,   10: 0,  11: -3,
    12: -5, 13: -2, 14: 2,  15: 5,  16: 10, 17: 15,
    18: 18, 19: 15, 20: 10, 21: 5,  22: -5, 23: -10,
}


def project_aqi_for_hour(baseline_aqi: float, hour: int) -> float:
    offset_pct = HOURLY_OFFSET_PCT[hour % 24]
    return round(baseline_aqi * (1 + offset_pct / 100), 1)


def get_shift_risk_level(predicted_aqi: float) -> Tuple[str, str]:
    """
    (risk_level, recommended_intensity) for one hour of the schedule.
    Thresholds per README Section 5, Endpoint 3's stated rule:
      Safe:     AQI < 100   -> Heavy
      Moderate: 100 <= AQI <= 150 -> Moderate
      High:     AQI > 150   -> Light
    """
    if predicted_aqi < 100:
        return "Safe", "Heavy"
    elif predicted_aqi <= 150:
        return "Moderate", "Moderate"
    else:
        return "High", "Light"
