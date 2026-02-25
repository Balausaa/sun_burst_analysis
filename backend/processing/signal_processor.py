from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy import signal

from ..models.cnn_model import ResNetFeatureExtractor, TypeIIDetectorHead, build_cnn_detector
from ..models.lstm_model import StormLSTM, build_storm_lstm
from .entropy import spectral_entropy


_CNN_EXTRACTOR: ResNetFeatureExtractor | None = None
_CNN_HEAD: TypeIIDetectorHead | None = None
_LSTM: StormLSTM | None = None


def _get_models(feature_dim: int) -> Tuple[ResNetFeatureExtractor, TypeIIDetectorHead, StormLSTM]:
    global _CNN_EXTRACTOR, _CNN_HEAD, _LSTM
    if _CNN_EXTRACTOR is None or _CNN_HEAD is None:
        _CNN_EXTRACTOR, _CNN_HEAD = build_cnn_detector()
    if _LSTM is None:
        # CNN feature vector + [entropy, drift rate, duration]
        _LSTM = build_storm_lstm(input_dim=feature_dim + 3)
    return _CNN_EXTRACTOR, _CNN_HEAD, _LSTM


@dataclass
class SignalAnalysisResult:
    spectrogram: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    entropy: float
    drift_rate: float
    duration: float
    type_ii_probability: float
    storm_probability: float
    max_intensity: float
    mean_intensity: float


def preprocess_signal(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize intensity and apply a simple noise filter.
    matrix: (time, frequency)
    """
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Signal matrix must be 2D [time, frequency].")

    x = x - np.mean(x)
    std = float(np.std(x) + 1e-6)
    x = x / std

    x = signal.medfilt(x, kernel_size=3)
    return x


def compute_spectrogram(matrix: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a time-frequency spectrogram from the intensity matrix by
    treating each frequency channel as a separate sensor and averaging.
    """
    time_len, freq_len = matrix.shape
    freqs = np.linspace(245.0, 500.0, freq_len, dtype=np.float32)
    times = np.arange(time_len, dtype=np.float32) / fs

    spec = matrix.astype(np.float32)
    return spec, times, freqs


def estimate_drift_rate(spec: np.ndarray, times: np.ndarray, freqs: np.ndarray) -> float:
    """
    Estimate spectral drift rate by tracking the frequency of maximum
    intensity over time and fitting a straight line.
    """
    if spec.size == 0 or times.size < 2:
        return 0.0

    max_indices = np.argmax(spec, axis=1)
    peak_freqs = freqs[max_indices]

    t = times
    f = peak_freqs
    t = t - t.mean()
    denom = float(np.sum(t * t) + 1e-8)
    slope = float(np.sum(t * (f - f.mean())) / denom)
    return slope


def _heuristic_type_ii_score(ent: float, drift_rate: float) -> float:
    ent_norm = 1.0 - np.clip(ent / 6.0, 0.0, 1.0)
    drift_norm = np.clip(abs(drift_rate) / 50.0, 0.0, 1.0)
    score = 0.6 * ent_norm + 0.4 * drift_norm
    return float(score)


def _normalize_feature(value: float, scale: float) -> float:
    return float(np.clip(value / scale, 0.0, 1.0))


def analyze_signal_matrix(matrix: np.ndarray) -> SignalAnalysisResult:
    """
    End-to-end analysis pipeline used by both the API and dashboard.
    """
    processed = preprocess_signal(matrix)

    spec, times, freqs = compute_spectrogram(processed)
    ent = float(spectral_entropy(spec))
    drift = float(estimate_drift_rate(spec, times, freqs))
    duration = float(times[-1] - times[0]) if times.size > 1 else 0.0
    max_int = float(np.max(spec)) if spec.size else 0.0
    mean_int = float(np.mean(spec)) if spec.size else 0.0

    extractor, head, lstm = _get_models(feature_dim=512)

    features = extractor(spec)
    with torch.no_grad():
        type_ii_prob_cnn = float(head(features).cpu().item())

    heuristic_score = _heuristic_type_ii_score(ent, drift)
    type_ii_probability = float(0.3 * type_ii_prob_cnn + 0.7 * heuristic_score)

    ent_norm = _normalize_feature(ent, scale=6.0)
    drift_norm = _normalize_feature(abs(drift), scale=50.0)
    dur_norm = _normalize_feature(duration, scale=600.0)

    fused_vector = torch.cat(
        [
            features,
            torch.tensor([ent_norm, drift_norm, dur_norm], dtype=torch.float32),
        ],
        dim=0,
    )

    sequence = fused_vector.unsqueeze(0).unsqueeze(0).repeat(1, 5, 1)

    with torch.no_grad():
        storm_prob_lstm = float(lstm(sequence).cpu().item())

    storm_probability = float(
        0.5 * storm_prob_lstm + 0.25 * ent_norm + 0.25 * drift_norm
    )

    # Guard against NaNs / infs by replacing with safe defaults.
    type_ii_probability = float(np.nan_to_num(type_ii_probability, nan=0.0, posinf=1.0, neginf=0.0))
    storm_probability = float(np.nan_to_num(storm_probability, nan=0.0, posinf=1.0, neginf=0.0))
    ent = float(np.nan_to_num(ent, nan=0.0))
    drift = float(np.nan_to_num(drift, nan=0.0))
    duration = float(np.nan_to_num(duration, nan=0.0))
    max_int = float(np.nan_to_num(max_int, nan=0.0))
    mean_int = float(np.nan_to_num(mean_int, nan=0.0))

    return SignalAnalysisResult(
        spectrogram=spec,
        times=times,
        frequencies=freqs,
        entropy=ent,
        drift_rate=drift,
        duration=duration,
        type_ii_probability=type_ii_probability,
        storm_probability=storm_probability,
        max_intensity=max_int,
        mean_intensity=mean_int,
    )

