from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class EvalResult:
    report: dict
    confusion: np.ndarray
    macro_f1: float
    balanced_accuracy: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    report = classification_report(y_true, y_pred, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    return EvalResult(
        report=report,
        confusion=confusion,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
    )
