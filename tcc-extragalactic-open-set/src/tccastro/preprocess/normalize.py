from __future__ import annotations

import numpy as np


def robust_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize per-channel with robust stats.

    Formula: (x - median) / (p95 - p5 + eps)
    """
    if x.ndim != 3:
        raise ValueError("Expected shape [C,H,W]")
    x = x.astype(np.float32)
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        channel = x[c]
        median = np.median(channel)
        p5 = np.percentile(channel, 5)
        p95 = np.percentile(channel, 95)
        out[c] = (channel - median) / (p95 - p5 + eps)
    return out
