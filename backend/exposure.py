
MET_VALUES = {
    "construction": 5.0,
    "traffic_police": 3.0,
    "factory": 4.0,
    "delivery": 6.0,
}


def calculate_exposure(
    predicted_aqi: float,
    shift_duration_hours: float,
    worker_role: str,
) -> float:
    """
    exposure_score = AQI x shift duration x activity MET value
    """
    role = worker_role.strip().lower()

    if role not in MET_VALUES:
        raise ValueError(
            f"Unsupported worker role: {worker_role}. "
            f"Choose one of: {', '.join(MET_VALUES)}"
        )

    if not 0 < shift_duration_hours <= 24:
        raise ValueError("shift_duration_hours must be greater than 0 and at most 24.")

    if predicted_aqi < 0:
        raise ValueError("predicted_aqi cannot be negative.")

    score = predicted_aqi * shift_duration_hours * MET_VALUES[role]
    return round(score, 1)


def get_risk_tier(exposure_score: float) -> dict:
    """
    Return the ShiftSafe risk tier and worker safety directive.
    """
    if exposure_score < 200:
        return {
            "tier": "Safe",
            "directive": "Normal operations. Standard PPE applies.",
        }

    if exposure_score <= 400:
        return {
            "tier": "Moderate",
            "directive": "Wear dust mask. Hydrate every 30 minutes.",
        }

    if exposure_score <= 600:
        return {
            "tier": "High",
            "directive": "Wear N95 mask. Limit outdoor tasks to 30-minute intervals.",
        }

    return {
        "tier": "Critical",
        "directive": "Halt all outdoor operations immediately.",
    }
