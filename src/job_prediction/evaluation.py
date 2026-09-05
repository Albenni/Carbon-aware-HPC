"""Small, unit-aware regression metrics for job predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


# Reporting bands for actual duration. The last two split the ">=1 h" cohort
# that carries almost all of the workload's runtime seconds, and therefore
# almost all of WAPE and of the carbon-aware scheduling decision.
DURATION_BANDS = (
    ("<10 s", 0.0, 10.0),
    ("10-60 s", 10.0, 60.0),
    ("1-10 min", 60.0, 600.0),
    ("10-60 min", 600.0, 3_600.0),
    ("1-3 h", 3_600.0, 10_800.0),
    (">=3 h", 10_800.0, float("inf")),
)
LONG_JOB_SECONDS = 3_600.0


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    count: int
    mae: float
    rmse: float
    weighted_absolute_relative_error: float
    mean_absolute_relative_error: float
    median_absolute_relative_error: float
    bias: float
    relative_bias: float
    p95_absolute_error: float

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

    residual = estimates - observed
    absolute = np.abs(residual)
    relative = absolute / observed
    total = float(np.sum(observed))
    return RegressionMetrics(
        count=int(observed.size),
        mae=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        weighted_absolute_relative_error=float(np.sum(absolute) / total),
        mean_absolute_relative_error=float(np.mean(relative)),
        median_absolute_relative_error=float(np.median(relative)),
        bias=float(np.mean(residual)),
        relative_bias=float(np.sum(residual) / total),
        p95_absolute_error=float(np.percentile(absolute, 95.0)),
    )


def banded_metrics(
    actual: object,
    predicted: object,
    *,
    by: object | None = None,
) -> pd.DataFrame:
    """Report the metrics separately inside each duration band.

    ``by`` selects the banding quantity; it defaults to the actual target, which
    is what the duration report needs. Power and energy are banded by the actual
    duration instead, so pass it explicitly there.
    """

    observed = np.asarray(actual, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    grouping = observed if by is None else np.asarray(by, dtype=float)
    if grouping.shape != observed.shape:
        raise ValueError("banding values must match the target shape")

    rows: list[dict[str, object]] = []
    for label, lower, upper in DURATION_BANDS:
        selected = (grouping >= lower) & (grouping < upper)
        if not selected.any():
            continue
        rows.append(
            {"band": label, **regression_metrics(
                observed[selected], estimates[selected]
            ).as_row()}
        )
    return pd.DataFrame(rows)


def long_job_metrics(
    actual: object,
    predicted: object,
    *,
    durations: object | None = None,
) -> RegressionMetrics:
    """Metrics restricted to jobs lasting at least one hour."""

    observed = np.asarray(actual, dtype=float)
    grouping = observed if durations is None else np.asarray(durations, dtype=float)
    selected = grouping >= LONG_JOB_SECONDS
    if not selected.any():
        raise ValueError("no jobs of at least one hour are present")
    return regression_metrics(
        observed[selected],
        np.asarray(predicted, dtype=float)[selected],
    )
