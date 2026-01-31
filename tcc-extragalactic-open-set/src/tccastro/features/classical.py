from __future__ import annotations

import numpy as np


def _sobel_gradients(channel: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(channel)
    gy = np.zeros_like(channel)
    gx[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    gy[1:-1, :] = channel[2:, :] - channel[:-2, :]
    return np.sqrt(gx**2 + gy**2)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """Extract simple interpretable features from a patch [C,H,W]."""
    features: list[float] = []
    for c in range(patch.shape[0]):
        channel = patch[c]
        features.extend(
            [
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.median(channel)),
                float(np.percentile(channel, 5)),
                float(np.percentile(channel, 95)),
            ]
        )
        features.append(float(np.sum(np.abs(channel))))
        grad = _sobel_gradients(channel)
        features.append(float(np.mean(grad)))
        h, w = channel.shape
        h0, h1 = h // 4, 3 * h // 4
        w0, w1 = w // 4, 3 * w // 4
        center = np.sum(channel[h0:h1, w0:w1])
        total = np.sum(channel) + 1e-6
        features.append(float(center / total))
    return np.array(features, dtype=np.float32)


def batch_extract_features(patches: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(p) for p in patches], axis=0)
