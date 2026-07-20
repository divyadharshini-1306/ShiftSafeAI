"""
exposure.py — ShiftSafe AI Health Risk Engine
=============================================
Implements physiologically-grounded cumulative exposure scoring with:
  1. Role-specific MET values
  2. Non-linear AQI penalty (power function, exponent 1.2)
  3. Step-based fatigue progression across the shift
  4. Time-weighted recovery credit for indoor breaks

Formula per hour:
    hourly_exposure = aqi_penalty(aqi) × met(role) × fatigue(hour_in_shift)

Total shift exposure:
    total = sum(hourly_exposure for each hour)
    total = total × (1 - recovery_reduction)
"""

# ── MET values (Metabolic Equivalent of Task) ─────────────────────────────────
# Source: Compendium of Physical Activities (Ainsworth et al.)
# Higher MET = faster breathing = more pollutant absorbed per unit time

MET_VALUES = {
    "construction":   5.0,   # heavy outdoor physical labour
    "traffic_police": 3.0,   # standing and walking outdoors
    "factory":        4.0,   # moderate indoor/outdoor assembly
    "delivery":       6.0,   # cycling with load — highest exertion
}

# ── Fatigue multiplier configuration ─────────────────────────────────────────
# Step-based: reflects rising ventilation rate as the body fatigues.
# Research basis: sustained physical work increases breathing rate
# by 10-20% after 4 hours and up to 30% after 8 hours (NIOSH, 2016).
#
# Role-specific ceiling: high-exertion roles (construction, delivery)
# fatigue faster and reach a higher ceiling than moderate roles.

_FATIGUE_STEPS = {
    # hour_in_shift (1-indexed) → base fatigue multiplier
    # Applied before role-specific ceiling adjustment
    "low":    {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00,   # hours 1-4: normal
               5: 1.10, 6: 1.10, 7: 1.10,              # hours 5-7: moderate
               8: 1.20, 9: 1.20, 10: 1.20},            # hours 8+:  high

    "high":   {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00,   # hours 1-4: normal
               5: 1.15, 6: 1.15, 7: 1.15,              # hours 5-7: moderate
               8: 1.30, 9: 1.30, 10: 1.30},            # hours 8+:  high
}

# Which fatigue profile each role uses
_FATIGUE_PROFILE = {
    "construction":   "high",   # physically demanding outdoor role
    "traffic_police": "low",    # standing/walking — less intense
    "factory":        "low",    # moderate — controlled environment
    "delivery":       "high",   # cycling with load — high exertion
}

# AQI power function exponent
# 1.2 reflects WHO concentration-response curve for PM2.5:
# health damage is super-linear at higher concentrations because
# the respiratory system's filtration is overwhelmed above AQI 100
_AQI_EXPONENT = 1.2

# Recovery parameters
_RECOVERY_PER_30_MIN = 0.15   # 15% reduction per 30 minutes of indoor break
_RECOVERY_MAX        = 0.40   # never reduce more than 40% regardless of break length
_RECOVERY_AQI_MIN    = 100.0  # breaks only count during Moderate/High/Critical hours


# ── Core helper functions ─────────────────────────────────────────────────────

def _aqi_penalty(aqi: float) -> float:
    """
    Non-linear AQI penalty using a power function.

    Formula: (aqi / 100) ^ 1.2

    At AQI 50  → penalty = 0.44  (lower than linear — relatively safe)
    At AQI 100 → penalty = 1.00  (baseline — exactly linear)
    At AQI 200 → penalty = 2.30  (worse than linear — disproportionate harm)
    At AQI 300 → penalty = 3.74  (severe — respiratory system overwhelmed)

    This captures the scientific consensus that PM2.5 health effects
    are not proportional at high concentrations.
    """
    aqi = max(0.0, aqi)   # clamp negative values
    return (aqi / 100.0) ** _AQI_EXPONENT


def _fatigue_multiplier(role: str, hour_in_shift: int) -> float:
    """
    Return the fatigue multiplier for a given role and hour number in the shift.

    hour_in_shift is 1-indexed:
        hour_in_shift=1 → first hour of the shift
        hour_in_shift=8 → eighth hour of the shift

    Hours beyond 10 use the hour-10 value (maximum fatigue).
    """
    profile  = _FATIGUE_PROFILE[role]
    steps    = _FATIGUE_STEPS[profile]
    # Cap at hour 10 — beyond that fatigue does not increase further
    capped   = min(hour_in_shift, 10)
    return steps.get(capped, steps[10])


def _recovery_reduction(
    break_minutes: float,
    ambient_aqi_during_break: float,
) -> float:
    """
    Compute the fractional reduction in cumulative exposure due to a
    single indoor recovery break.

    Formula: min(MAX, (break_minutes / 30) × RATE_PER_30_MIN)

    A break only counts if it occurs during a Moderate/High/Critical
    AQI period (ambient_aqi_during_break >= 100). Breaks during clean
    air give zero credit — there is nothing to recover from.

    Returns a fraction between 0.0 and _RECOVERY_MAX.
    """
    if ambient_aqi_during_break < _RECOVERY_AQI_MIN:
        return 0.0

    if break_minutes <= 0:
        return 0.0

    reduction = (break_minutes / 30.0) * _RECOVERY_PER_30_MIN
    return min(reduction, _RECOVERY_MAX)


# ── Public API ────────────────────────────────────────────────────────────────

def calculate_exposure(
    predicted_aqi: float,
    shift_duration_hours: float,
    worker_role: str,
) -> float:
    """
    Calculate cumulative exposure score for a worker doing a continuous
    shift at a single predicted AQI level.

    This is the simple single-AQI version used by the /risk-score endpoint
    when only one prediction is available.

    For a more accurate multi-hour calculation use calculate_shift_exposure().

    Formula per hour:
        hourly = aqi_penalty(aqi) × MET(role) × fatigue_multiplier(role, hour)

    Total = sum of all hourly contributions.
    """
    role = worker_role.strip().lower()

    if role not in MET_VALUES:
        raise ValueError(
            f"Unsupported worker role: '{worker_role}'. "
            f"Choose one of: {', '.join(MET_VALUES)}"
        )

    if not 0 < shift_duration_hours <= 24:
        raise ValueError(
            "shift_duration_hours must be greater than 0 and at most 24."
        )

    if predicted_aqi < 0:
        raise ValueError("predicted_aqi cannot be negative.")

    met     = MET_VALUES[role]
    penalty = _aqi_penalty(predicted_aqi)
    total   = 0.0

    # Accumulate hour by hour so fatigue multiplier changes correctly
    # We treat each fractional hour at the end the same as a full hour
    full_hours     = int(shift_duration_hours)
    remaining_frac = shift_duration_hours - full_hours

    for h in range(1, full_hours + 1):
        fatigue = _fatigue_multiplier(role, h)
        total  += penalty * met * fatigue * 1.0   # 1.0 = one full hour

    # Partial final hour
    if remaining_frac > 0:
        fatigue = _fatigue_multiplier(role, full_hours + 1)
        total  += penalty * met * fatigue * remaining_frac

    return round(total, 1)


def calculate_shift_exposure(
    hourly_aqi_list: list[float],
    worker_role: str,
    breaks: list[dict] | None = None,
) -> dict:
    """
    Calculate cumulative exposure for a full shift with per-hour AQI values,
    fatigue progression, and optional recovery breaks.

    Parameters
    ----------
    hourly_aqi_list : list of floats
        Predicted AQI for each hour of the shift.
        Length = number of shift hours (e.g. 8 values for an 8-hour shift).

    worker_role : str
        One of: construction, traffic_police, factory, delivery

    breaks : list of dicts, optional
        Each dict describes one indoor recovery break:
        {
            "after_hour":   int,    # which shift hour the break follows (1-indexed)
            "duration_min": float,  # break length in minutes
        }
        The ambient AQI during the break is taken from the AQI of that hour.

    Returns
    -------
    dict with keys:
        total_exposure      float   final cumulative score after recovery
        pre_recovery        float   score before any break credits
        recovery_credit     float   total score reduction from breaks
        hourly_breakdown    list    per-hour contribution details
        risk_tier           str     Safe / Moderate / High / Critical
        directive           str     safety instruction
    """
    role = worker_role.strip().lower()

    if role not in MET_VALUES:
        raise ValueError(
            f"Unsupported worker role: '{worker_role}'. "
            f"Choose one of: {', '.join(MET_VALUES)}"
        )

    if not hourly_aqi_list:
        raise ValueError("hourly_aqi_list cannot be empty.")

    met       = MET_VALUES[role]
    breaks    = breaks or []
    breakdown = []
    pre_recovery_total = 0.0

    for i, aqi in enumerate(hourly_aqi_list):
        hour_in_shift = i + 1   # 1-indexed
        aqi           = max(0.0, aqi)
        penalty       = _aqi_penalty(aqi)
        fatigue       = _fatigue_multiplier(role, hour_in_shift)
        contribution  = penalty * met * fatigue * 1.0

        pre_recovery_total += contribution

        breakdown.append({
            "hour_in_shift":     hour_in_shift,
            "predicted_aqi":     round(aqi, 1),
            "aqi_penalty":       round(penalty, 3),
            "fatigue_multiplier":round(fatigue, 2),
            "met_value":         met,
            "hourly_contribution": round(contribution, 2),
            "running_total":     round(pre_recovery_total, 2),
        })

    # Apply recovery breaks
    total_recovery_fraction = 0.0

    for brk in breaks:
        after_hour   = brk.get("after_hour", 0)
        duration_min = brk.get("duration_min", 0)

        # Get the AQI of the hour this break follows
        # If after_hour is out of range, use last hour's AQI
        idx = min(after_hour - 1, len(hourly_aqi_list) - 1)
        idx = max(idx, 0)
        break_aqi = hourly_aqi_list[idx]

        fraction = _recovery_reduction(duration_min, break_aqi)
        total_recovery_fraction += fraction

    # Cap total recovery at maximum allowed
    total_recovery_fraction = min(total_recovery_fraction, _RECOVERY_MAX)
    recovery_credit         = pre_recovery_total * total_recovery_fraction
    final_total             = pre_recovery_total - recovery_credit

    risk = get_risk_tier(final_total)

    return {
        "total_exposure":   round(final_total, 1),
        "pre_recovery":     round(pre_recovery_total, 1),
        "recovery_credit":  round(recovery_credit, 1),
        "hourly_breakdown": breakdown,
        "risk_tier":        risk["tier"],
        "directive":        risk["directive"],
    }


def get_risk_tier(exposure_score: float) -> dict:
    """
    Map a cumulative exposure score to a risk tier and safety directive.

    Thresholds are calibrated to the new non-linear formula.
    Scores are higher than the old linear formula at the same AQI
    because the power function and fatigue multiplier both increase totals.

    Tier      Score range     Meaning
    Safe      below 300       Within acceptable occupational limits
    Moderate  300 – 600       Elevated — precautionary action needed
    High      600 – 1000      Significant risk — protective equipment required
    Critical  above 1000      Dangerous — immediate action required
    """
    if exposure_score < 300:
        return {
            "tier":      "Safe",
            "directive": "Normal operations. Standard PPE applies.",
        }

    if exposure_score <= 600:
        return {
            "tier":      "Moderate",
            "directive": (
                "Elevated exposure detected. Wear dust mask. "
                "Hydrate every 30 minutes. Consider task rotation."
            ),
        }

    if exposure_score <= 1000:
        return {
            "tier":      "High",
            "directive": (
                "Significant exposure risk. Wear N95 mask. "
                "Limit continuous outdoor tasks to 30-minute intervals. "
                "Mandatory 15-minute indoor break after every outdoor hour."
            ),
        }

    return {
        "tier":      "Critical",
        "directive": (
            "Dangerous exposure level. Halt all outdoor operations immediately. "
            "Move all workers indoors. Do not resume outdoor work until "
            "AQI drops below 100."
        ),
    }