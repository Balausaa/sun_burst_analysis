from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


@dataclass
class SyntheticConfig:
    """
    Configuration for generating synthetic radio dynamic spectra.
    """

    n_time: int = 256
    n_freq: int = 128
    base_noise_level: float = 1.0
    include_type_ii: bool = True
    type_ii_start_freq_mhz: float = 450.0
    type_ii_end_freq_mhz: float = 250.0
    type_ii_duration_fraction: float = 0.6
    random_seed: int = 42


def generate_synthetic_signal(config: SyntheticConfig) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic time-frequency intensity matrix with optional
    Type II-like drifting feature plus background noise.
    """
    cfg = config
    rng = np.random.default_rng(cfg.random_seed)

    time = np.linspace(0.0, 600.0, cfg.n_time, dtype=np.float32)
    freq = np.linspace(245.0, 500.0, cfg.n_freq, dtype=np.float32)

    intensity = rng.normal(loc=0.0, scale=cfg.base_noise_level, size=(cfg.n_time, cfg.n_freq)).astype(
        np.float32
    )

    has_type_ii = False
    if cfg.include_type_ii:
        start_idx = int(0.1 * cfg.n_time)
        end_idx = int(start_idx + cfg.type_ii_duration_fraction * cfg.n_time)
        end_idx = min(end_idx, cfg.n_time)
        t_segment = np.linspace(0.0, 1.0, end_idx - start_idx, dtype=np.float32)
        drift_freqs = cfg.type_ii_start_freq_mhz + t_segment * (
            cfg.type_ii_end_freq_mhz - cfg.type_ii_start_freq_mhz
        )

        for i, f0 in enumerate(drift_freqs):
            t_idx = start_idx + i
            band_width = int(0.05 * cfg.n_freq)
            center_idx = int(np.argmin(np.abs(freq - f0)))
            low = max(0, center_idx - band_width // 2)
            high = min(cfg.n_freq, center_idx + band_width // 2)
            burst_power = rng.normal(loc=6.0, scale=1.0)
            intensity[t_idx, low:high] += burst_power

        has_type_ii = True

    return {
        "time": time,
        "frequency": freq,
        "intensity": intensity,
        "has_type_ii_pattern": np.array(has_type_ii, dtype=bool),
    }

