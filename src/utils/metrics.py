from __future__ import annotations
from collections import defaultdict

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

import numpy as np


def compute_metrics(y_true, y_pred, task: str = "classification") -> dict[str, float]:
    """Compute standard metrics for classification or regression.

    Args:
        y_true: Ground-truth labels or values.
        y_pred: Model predictions.
        task:   ``"classification"`` or ``"regression"``.

    Returns:
        Dict of metric names → float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "classification":
        return {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    elif task == "regression":
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2":  float(r2_score(y_true, y_pred)),
        }
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'classification' or 'regression'.")


class MetricTracker:
    """Accumulate batch-level metrics across an epoch.

    Usage::

        tracker = MetricTracker()
        for batch in loader:
            tracker.update("loss", loss.item(), n=len(batch))
        epoch_metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self._sums:   dict[str, float] = defaultdict(float)
        self._counts: dict[str, int]   = defaultdict(int)

    def update(self, metric_name: str, value: float, n: int = 1):
        """Weighted update.

        Args:
            metric_name: Metric key, e.g. ``"loss"``.
            value:       Scalar value for this batch.
            n:           Number of samples this batch represents.
        """
        self._sums[metric_name]   += value * n
        self._counts[metric_name] += n

    def compute(self) -> dict[str, float]:
        """Return weighted average of all accumulated metrics."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self):
        """Clear all state — call at the start of each epoch."""
        self._sums.clear()
        self._counts.clear()
