from __future__ import annotations

import numpy as np


def entropy_score(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy; larger means more unknown."""
    clipped = np.clip(probs, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy score from logits; larger means more unknown."""
    scaled = logits / temperature
    max_logits = np.max(scaled, axis=1, keepdims=True)
    lse = max_logits + np.log(np.sum(np.exp(scaled - max_logits), axis=1, keepdims=True))
    return lse.squeeze(1)


def prototype_distance(
    embeddings: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    """Compute distance to nearest prototype (optional)."""
    diffs = embeddings[:, None, :] - prototypes[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.min(dists, axis=1)
