from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


def spectral_entropy(intensity: np.ndarray, num_bins: int = 64) -> float:
    """
    Compute a simple Shannon entropy over the intensity histogram.
    Lower entropy corresponds to more structured signals.
    """
    flat = np.asarray(intensity, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0

    flat = flat - np.min(flat)
    if np.max(flat) > 0:
        flat = flat / (np.max(flat) + 1e-8)

    hist, _ = np.histogram(flat, bins=num_bins, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    ent = float(scipy_entropy(hist, base=2))
    return ent

