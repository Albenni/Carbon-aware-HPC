"""Small, unit-aware regression metrics for job predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import (
    AVERAGE_POWER_WATTS,
    DURATION_SECONDS,
    TemporalSplit,
    completed_by,
    require_columns,
)


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


def job_metric_row(frame, estimate, target, unit, predicted) -> dict[str, object]:
    """The same overall and long-job metrics for learned and simple estimates."""

    actual = frame[target].to_numpy(dtype=float)
    durations = frame[DURATION_SECONDS].to_numpy(dtype=float)
    long_count = int((durations >= LONG_JOB_SECONDS).sum())
    long_jobs = (
        long_job_metrics(actual, predicted, durations=durations) if long_count else None
    )
    return {
        "estimate": estimate,
        "target": target,
        "unit": unit,
        **regression_metrics(actual, predicted).as_row(),
        "long_job_count": long_count,
        "long_job_mae": long_jobs.mae if long_jobs else float("nan"),
        "long_job_wape": (
            long_jobs.weighted_absolute_relative_error if long_jobs else float("nan")
        ),
        "long_job_bias": long_jobs.bias if long_jobs else float("nan"),
    }


def runtime_band_metrics(actual: object, predicted: object) -> pd.DataFrame:
    """Group only by actual seconds: [0,5m), [5m,30m), [30m,1h), [1h,4h], >4h."""

    observed = np.asarray(actual, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    # Validate even if a malformed value would fall outside every band.
    empty = {key: float("nan") for key in regression_metrics(observed, estimates).as_row()}
    rows = []
    for label, selected in (
        ("< 5 min", observed < 300),
        ("5–30 min", (observed >= 300) & (observed < 1_800)),
        ("30–60 min", (observed >= 1_800) & (observed < 3_600)),
        ("1–4 h", (observed >= 3_600) & (observed <= 14_400)),
        ("> 4 h", observed > 14_400),
    ):
        values = observed[selected]
        metrics = regression_metrics(values, estimates[selected]).as_row() if values.size else empty
        rows.append({
            "band": label, **metrics, "count": int(values.size),
            "actual_runtime_mean_s": float(np.mean(values)) if values.size else float("nan"),
            "actual_runtime_median_s": float(np.median(values)) if values.size else float("nan"),
        })
    return pd.DataFrame(rows)


def baseline_estimates(
    split: TemporalSplit, *, runtime_by_nodes: bool = False, power_by_nodes: bool = False,
) -> list[tuple]:
    """Predict the unchanged test rows; fit medians only on observable train labels.

    Allocated nodes are used only for the optional offline comparison, never
    added to the submission-time models or scheduler inputs.
    """

    train = completed_by(split.train, split.train_until, period_name="training")
    require_columns(train, (DURATION_SECONDS, AVERAGE_POWER_WATTS))
    require_columns(split.test, ("time_limit",))
    runtime = pd.to_numeric(split.test.time_limit, errors="raise").to_numpy(dtype=float) * 60.0
    if not np.all(np.isfinite(runtime)) or not np.all(runtime > 0.0):
        raise ValueError("time_limit * 60 must be finite and greater than zero")
    estimates = [("time_limit * 60", DURATION_SECONDS, "s", runtime)]
    for target, label, unit, grouped in (
        (DURATION_SECONDS, "runtime", "s", runtime_by_nodes),
        (AVERAGE_POWER_WATTS, "power", "W", power_by_nodes),
    ):
        values = pd.to_numeric(train[target], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
            raise ValueError(f"{target} must be finite and greater than zero")
        median = float(np.median(values))
        estimates.append((f"training median {label}", target, unit, np.full(len(split.test), median)))
        if grouped:
            require_columns(train, ("num_nodes_alloc",))
            require_columns(split.test, ("num_nodes_alloc",))
            medians = pd.Series(values).groupby(train.num_nodes_alloc, observed=True).median()
            predicted = split.test.num_nodes_alloc.map(medians).fillna(median).to_numpy(dtype=float)
            estimates.append((f"training median {label} by nodes", target, unit, predicted))
    return estimates
