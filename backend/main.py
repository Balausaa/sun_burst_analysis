from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .processing.signal_processor import (
    SignalAnalysisResult,
    analyze_signal_matrix,
)
from .processing.surya_processor import SuryaAnalysisResult, analyze_with_surya
from .hybrid_analyzer import CombinedAnalysisResult, analyze_csv_combined
from .risk_engine import RiskAssessment, assess_risk, assess_surya_risk
from .synthetic_generator import SyntheticConfig, generate_synthetic_signal


app = FastAPI(
    title="HelioGuard AI - Space Weather Early Warning API",
    version="0.1.0",
    description=(
        "MVP API for detecting Type II solar radio bursts and estimating "
        "geomagnetic storm risk using a hybrid CNN + LSTM architecture."
    ),
)


class AnalyzeRequest(BaseModel):
    satellite_id: str = Field(..., description="ID of the observing spacecraft or instrument.")
    signal_data: List[List[float]] = Field(
        ..., description="2D matrix [time, frequency] of intensity values."
    )
    timestamp: datetime = Field(..., description="Acquisition time in ISO 8601 format.")


class AnalyzeResponse(BaseModel):
    type_ii_detected: bool
    detection_probability: float
    storm_probability: float
    risk_level: str
    recommended_action: str


class SyntheticRequest(BaseModel):
    config: SyntheticConfig = Field(
        default_factory=SyntheticConfig,
        description="Configuration for the synthetic burst generator.",
    )


class SyntheticResponse(BaseModel):
    time: List[float]
    frequency: List[float]
    intensity: List[List[float]]
    has_type_ii_pattern: bool


class RiskHistoryItem(BaseModel):
    satellite_id: str
    timestamp: datetime
    detection_probability: float
    storm_probability: float
    risk_level: str


class PredictResponse(BaseModel):
    timestamp: datetime
    satellite_id: Optional[str]
    signal_metrics: Dict[str, float]
    type_ii_probability: float
    solar_wind_speed_kms: float
    geomagnetic_risk: str
    final_risk_level: str
    recommendation: str
    event_history: List[Dict[str, object]]


risk_history: List[RiskHistoryItem] = []
combined_history: List[Dict[str, object]] = []


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "HelioGuard AI"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    matrix = np.array(request.signal_data, dtype=np.float32)

    analysis: SignalAnalysisResult = analyze_signal_matrix(matrix)
    risk: RiskAssessment = assess_risk(
        type_ii_probability=analysis.type_ii_probability,
        storm_probability=analysis.storm_probability,
    )

    item = RiskHistoryItem(
        satellite_id=request.satellite_id,
        timestamp=request.timestamp,
        detection_probability=analysis.type_ii_probability,
        storm_probability=analysis.storm_probability,
        risk_level=risk.risk_level,
    )
    risk_history.append(item)

    return AnalyzeResponse(
        type_ii_detected=analysis.type_ii_probability >= 0.5,
        detection_probability=analysis.type_ii_probability,
        storm_probability=analysis.storm_probability,
        risk_level=risk.risk_level,
        recommended_action=risk.recommended_action,
    )


@app.post("/generate-synthetic", response_model=SyntheticResponse)
def generate_synthetic(req: SyntheticRequest) -> SyntheticResponse:
    data = generate_synthetic_signal(req.config)

    return SyntheticResponse(
        time=data["time"].tolist(),
        frequency=data["frequency"].tolist(),
        intensity=data["intensity"].tolist(),
        has_type_ii_pattern=bool(data["has_type_ii_pattern"]),
    )


@app.get("/risk-history", response_model=List[RiskHistoryItem])
def get_risk_history(limit: int = 50) -> List[RiskHistoryItem]:
    return risk_history[-limit:]


@app.post("/predict", response_model=PredictResponse)
async def predict(
    csv_file: UploadFile = File(...),
    satellite_id: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
) -> PredictResponse:
    """
    Combined Type II + Surya prediction endpoint.
    Accepts a CSV file with columns: time, frequency, intensity.
    """
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await csv_file.read()
    try:
        combined: CombinedAnalysisResult = analyze_csv_combined(
            content, satellite_id=satellite_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event = {
        "timestamp": combined.timestamp.isoformat(),
        "satellite_id": combined.satellite_id,
        "session_id": session_id,
        "type_ii_probability": combined.type_ii_probability,
        "final_risk_level": combined.final_risk_level,
    }
    combined_history.append(event)

    metrics = {
        "entropy": float(combined.entropy),
        "drift_rate_mhz_s": float(combined.drift_rate_mhz_s),
        "max_intensity": float(combined.max_intensity),
        "mean_intensity": float(combined.mean_intensity),
    }

    history_view = combined_history[-100:]

    return PredictResponse(
        timestamp=combined.timestamp,
        satellite_id=combined.satellite_id,
        signal_metrics=metrics,
        type_ii_probability=float(combined.type_ii_probability),
        solar_wind_speed_kms=float(combined.solar_wind_speed_kms),
        geomagnetic_risk=combined.geomagnetic_risk,
        final_risk_level=combined.final_risk_level,
        recommendation=combined.recommendation,
        event_history=history_view,
    )


def get_app() -> FastAPI:
    """Convenience accessor for ASGI servers."""
    return app

