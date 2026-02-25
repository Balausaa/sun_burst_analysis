from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from .processing.signal_processor import SignalAnalysisResult, analyze_signal_matrix
from .processing.surya_processor import SuryaAnalysisResult, analyze_with_surya
from .risk_engine import assess_surya_risk


@dataclass
class CombinedAnalysisResult:
    timestamp: datetime
    satellite_id: str | None
    entropy: float
    drift_rate_mhz_s: float
    max_intensity: float
    mean_intensity: float
    type_ii_probability: float
    solar_wind_speed_kms: float
    geomagnetic_risk: str
    final_risk_level: str
    recommendation: str


def _safe_float(x: float, default: float = 0.0) -> float:
    return float(np.nan_to_num(x, nan=default, posinf=default, neginf=default))


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (value - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))


def analyze_csv_combined(file_bytes: bytes, satellite_id: str | None = None) -> CombinedAnalysisResult:
    """
    Run both the Type II (CNN+LSTM) pipeline and the Surya-based pipeline,
    then compute a hybrid risk score.
    """
    surya_res: SuryaAnalysisResult = analyze_with_surya(file_bytes)

    # Reuse the spectrogram matrix from Surya for the Type II module.
    t2_matrix = surya_res.spectrogram
    t2_res: SignalAnalysisResult = analyze_signal_matrix(t2_matrix)

    ts = datetime.now(timezone.utc)

    type_ii_prob = _safe_float(t2_res.type_ii_probability, default=0.0)
    drift = _safe_float(t2_res.drift_rate, default=0.0)
    max_int = _safe_float(t2_res.max_intensity, default=0.0)
    mean_int = _safe_float(t2_res.mean_intensity, default=0.0)

    solar_wind = _safe_float(surya_res.solar_wind_speed, default=0.0)
    flare_prob = _safe_float(surya_res.flare_probability, default=0.0)

    surya_risk = assess_surya_risk(
        flare_probability=surya_res.flare_probability,
        solar_wind_speed=solar_wind,
        entropy=surya_res.entropy,
        drift_rate=surya_res.drift_rate,
        max_intensity=surya_res.max_intensity,
    )

    # Normalize supporting metrics. Wider ranges reduce "always medium" behavior.
    drift_norm = _normalize(abs(drift), lo=0.0, hi=30.0)
    wind_norm = _normalize(solar_wind, lo=300.0, hi=900.0)

    # Non‑linear weighting to de‑emphasize mid‑range values and
    # make clearly low/high situations stand out more.
    type_ii_term = type_ii_prob ** 1.3
    flare_term = flare_prob ** 1.3
    drift_term = drift_norm ** 1.1
    wind_term = wind_norm ** 1.1

    final_risk_score = float(
        0.4 * type_ii_term
        + 0.3 * flare_term
        + 0.2 * drift_term
        + 0.1 * wind_term
    )
    final_risk_score = float(np.clip(final_risk_score, 0.0, 1.0))

    # Tighter thresholds to create more LOW and HIGH/EXTREME outcomes,
    # with MEDIUM reserved for genuinely borderline conditions.
    if final_risk_score < 0.20:
        final_risk_level = "LOW"
    elif final_risk_score < 0.45:
        final_risk_level = "MEDIUM"
    elif final_risk_score < 0.70:
        final_risk_level = "HIGH"
    else:
        final_risk_level = "EXTREME"

    if final_risk_level == "LOW":
        recommendation = (
            "Nominal space weather conditions. Continue routine monitoring."
        )
    elif final_risk_level == "MEDIUM":
        recommendation = (
            "Elevated risk. Review mission timelines, avoid high-risk maneuvers, "
            "and prepare contingency plans."
        )
    elif final_risk_level == "HIGH":
        recommendation = (
            "High probability of impactful space weather. Safeguard sensitive "
            "payloads and consider deferring non-critical operations."
        )
    else:
        recommendation = (
            "EXTREME risk. Immediately activate satellite protection protocols, "
            "defer high-risk maneuvers, and coordinate with ground infrastructure."
        )

    return CombinedAnalysisResult(
        timestamp=ts,
        satellite_id=satellite_id,
        entropy=_safe_float(surya_res.entropy, default=0.0),
        drift_rate_mhz_s=drift,
        max_intensity=max_int,
        mean_intensity=mean_int,
        type_ii_probability=type_ii_prob,
        solar_wind_speed_kms=solar_wind,
        geomagnetic_risk=surya_risk.risk_level,
        final_risk_level=final_risk_level,
        recommendation=recommendation,
    )

