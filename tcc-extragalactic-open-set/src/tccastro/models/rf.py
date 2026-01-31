from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from tccastro.features.classical import batch_extract_features


@dataclass
class RFTrainingResult:
    model: RandomForestClassifier
    metrics: dict[str, float]


def train_rf(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> RFTrainingResult:
    feats = batch_extract_features(X)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(feats, y)
    preds = model.predict(feats)
    metrics = {
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
    }
    return RFTrainingResult(model=model, metrics=metrics)


def save_model(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)
