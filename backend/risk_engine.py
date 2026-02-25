from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskAssessment:
    risk_level: str
    recommended_action: str


def assess_risk(type_ii_probability: float, storm_probability: float) -> RiskAssessment:
    """
    Map storm probability to a discrete risk level and recommended action.
    """
    p = float(max(0.0, min(1.0, storm_probability)))

    if p < 0.3:
        level = "LOW"
        action = (
            "Nominal operations. Continue monitoring solar radio data; "
            "no immediate mitigation required."
        )
    elif p < 0.7:
        level = "MEDIUM"
        action = (
            "Elevated space weather risk. Review mission timelines and "
            "prepare contingency procedures for critical assets."
        )
    else:
        level = "HIGH"
        action = (
            "High probability of geomagnetic disturbance. Consider "
            "safing sensitive spacecraft subsystems and coordinating with "
            "ground infrastructure operators."
        )

    return RiskAssessment(risk_level=level, recommended_action=action)


def assess_surya_risk(
    flare_probability: float,
    solar_wind_speed: float,
    entropy: float,
    drift_rate: float,
    max_intensity: float,
) -> RiskAssessment:
    """
    Dynamic risk assessment combining Surya-like outputs and auxiliary
    spectrogram metrics into LOW / MEDIUM / HIGH / EXTREME levels.
    """
    fp = float(max(0.0, min(1.0, flare_probability)))
    wind = float(solar_wind_speed)
    drift_mag = abs(float(drift_rate))

    score = (
        0.5 * fp
        + 0.25 * np.tanh((wind - 400.0) / 200.0)
        + 0.15 * np.tanh(drift_mag / 50.0)
        + 0.10 * np.tanh(max_intensity / 5.0)
    )

    if score > 0.85 or (fp > 0.9 and wind > 800):
        level = "EXTREME"
        action = (
            "EXTREME solar activity expected. Immediately activate satellite "
            "protection protocols, defer non-critical operations, and "
            "coordinate with ground infrastructure operators."
        )
    elif score > 0.7:
        level = "HIGH"
        action = (
            "High probability of disruptive space weather. Safeguard sensitive "
            "payloads, switch to hardened modes where available, and increase "
            "telemetry cadence."
        )
    elif score > 0.4:
        level = "MEDIUM"
        action = (
            "Elevated risk. Review mission timelines, avoid scheduling "
            "high-risk maneuvers, and prepare contingency plans."
        )
    else:
        level = "LOW"
        action = (
            "Nominal space weather conditions. Continue routine monitoring."
        )

    return RiskAssessment(risk_level=level, recommended_action=action)


