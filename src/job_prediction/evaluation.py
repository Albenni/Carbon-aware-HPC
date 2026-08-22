"""Small, unit-aware regression metrics for job predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    count: int
    mae: float
    rmse: float
    weighted_absolute_relative_error: float
    mean_absolute_relative_error: float
    median_absolute_relative_error: float

    def as_row(self) -> dict[str, int | float]:
        return asdict(self)


def regression_metrics(actual: object, predicted: object) -> RegressionMetrics:
    """Return raw-unit and relative errors for positive targets."""

    observed = np.asarray(actual, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or estimates.ndim != 1 or observed.shape != estimates.shape:
        raise ValueError("actual and predicted must be one-dimensional and equally sized")
    if observed.size == 0:
        raise ValueError("at least one prediction is required")
    if not np.all(np.isfinite(observed)) or not np.all(observed > 0.0):
        raise ValueError("actual values must be finite and greater than zero")
    if not np.all(np.isfinite(estimates)) or not np.all(estimates > 0.0):
        raise ValueError("predictions must be finite and greater than zero")

    absolute = np.abs(estimates - observed)
    relative = absolute / observed
    return RegressionMetrics(
        count=int(observed.size),
        mae=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(estimates - observed)))),
        weighted_absolute_relative_error=float(np.sum(absolute) / np.sum(observed)),
        mean_absolute_relative_error=float(np.mean(relative)),
        median_absolute_relative_error=float(np.median(relative)),
    )
