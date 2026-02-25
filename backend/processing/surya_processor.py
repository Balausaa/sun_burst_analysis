from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .entropy import spectral_entropy
from ..models.surya_model import SuryaOutputs, SuryaWrapper


_SURYA_WRAPPER: SuryaWrapper | None = None


def _get_surya() -> SuryaWrapper:
    global _SURYA_WRAPPER
    if _SURYA_WRAPPER is None:
        _SURYA_WRAPPER = SuryaWrapper()
    return _SURYA_WRAPPER


@dataclass
class SuryaAnalysisResult:
    spectrogram: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    entropy: float
    drift_rate: float
    max_intensity: float
    mean_intensity: float
    flare_probability: float
    solar_wind_speed: float


def csv_to_matrix(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a CSV (time, frequency, intensity) into a 2D matrix.
    """
    df = pd.read_csv(BytesIO(file_bytes))
    required_cols = {"time", "frequency", "intensity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    pivot = df.pivot(index="time", columns="frequency", values="intensity").sort_index()
    matrix = pivot.to_numpy(dtype=np.float32)
    times = pivot.index.to_numpy(dtype=np.float32)
    freqs = pivot.columns.to_numpy(dtype=np.float32)
    return matrix, times, freqs


def preprocess_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    x = x - np.mean(x)
    std = float(np.std(x) + 1e-6)
    x = x / std
    x = signal.medfilt(x, kernel_size=3)
    return x


def estimate_drift_rate(spec: np.ndarray, times: np.ndarray, freqs: np.ndarray) -> float:
    if spec.size == 0 or times.size < 2:
        return 0.0
    max_idx = np.argmax(spec, axis=1)
    peak_freqs = freqs[max_idx]
    t = times - times.mean()
    denom = float(np.sum(t * t) + 1e-8)
    slope = float(np.sum(t * (peak_freqs - peak_freqs.mean())) / denom)
    return slope


def analyze_with_surya(file_bytes: bytes) -> SuryaAnalysisResult:
    """
    Full preprocessing + Surya wrapper pipeline used by the /predict API
    and the Streamlit UI.
    """
    matrix, times, freqs = csv_to_matrix(file_bytes)
    spec = preprocess_matrix(matrix)

    ent = float(spectral_entropy(spec))
    drift = float(estimate_drift_rate(spec, times, freqs))
    max_int = float(np.max(spec)) if spec.size else 0.0
    mean_int = float(np.mean(spec)) if spec.size else 0.0

    surya = _get_surya()
    surya_out: SuryaOutputs = surya.predict_from_spectrogram(spec)

    flare_prob = float(np.nan_to_num(surya_out.flare_probability, nan=0.0, posinf=1.0, neginf=0.0))
    solar_wind = float(np.nan_to_num(surya_out.solar_wind_speed, nan=0.0))

    ent = float(np.nan_to_num(ent, nan=0.0))
    drift = float(np.nan_to_num(drift, nan=0.0))
    max_int = float(np.nan_to_num(max_int, nan=0.0))
    mean_int = float(np.nan_to_num(mean_int, nan=0.0))

    return SuryaAnalysisResult(
        spectrogram=spec,
        times=times,
        freqs=freqs,
        entropy=ent,
        drift_rate=drift,
        max_intensity=max_int,
        mean_intensity=mean_int,
        flare_probability=flare_prob,
        solar_wind_speed=solar_wind,
    )

