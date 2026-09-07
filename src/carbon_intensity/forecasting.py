"""Direct ridge regression for a complete day of carbon-intensity forecasts."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .features import (
    FEATURE_NAMES,
    LOOKBACK,
    TARGET_FEATURE_NAMES,
    forecast_features,
    target_features,
)
from .series import CarbonIntensityForecast, CarbonIntensitySample
from .protocol import ForecastExample, TemporalProtocol
from .series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider, aware_utc, bucket_start


HORIZON = timedelta(hours=24)
CADENCE = timedelta(hours=1)
BUCKETS = HORIZON // STEP
MODEL_VERSION = 2
DEFAULT_ALPHA = 0.01
# Shared columns first, then the two columns that vary with the target bucket.
FEATURE_LAYOUT = (*FEATURE_NAMES, *TARGET_FEATURE_NAMES)


def origins(
    protocol: TemporalProtocol, partition: str, *, since: datetime | None = None,
    horizon: timedelta = HORIZON, cadence: timedelta = CADENCE,
) -> Iterator[datetime]:
    """Issue times whose week of history and whole target window stay usable.

    The first origin waits for a complete week after ``train_start``; the last
    one keeps its labels inside the partition, becoming observable before the
    partition ends except on test, matching the evaluator. ``since`` shortens
    the training history without touching the split boundaries.
    """
    lower, upper = protocol.intervals[partition]
    label_delay = timedelta(0) if partition == "test" else protocol.observation_delay
    earliest = max(lower, protocol.train_start + LOOKBACK + protocol.observation_delay)
    if since is not None:
        earliest = max(earliest, aware_utc(since, "since"))
    issue = bucket_start(earliest, cadence)
    if issue < earliest:
        issue += cadence
    while issue + horizon + label_delay <= upper:
        yield issue
        issue += cadence


def design_row(example: ForecastExample, protocol: TemporalProtocol) -> tuple:
    """Shared features, per-target seasonal columns and labels for one origin."""
    return (
        forecast_features(example.features, example.issue_time, protocol),
        target_features(example.features, example.issue_time, protocol, BUCKETS),
        [sample.intensity_gco2e_per_kwh for sample in example.targets],
    )


def fit_direct(
    shared: np.ndarray, seasonal: np.ndarray, labels: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One ridge per target bucket, returned in original feature units.

    Standardisation uses the supplied rows only; converting the weights back
    keeps a single equation readable without carrying a scaler alongside it.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if isinstance(alpha, bool) or not isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    buckets = labels.shape[1]
    coefficients = np.empty((buckets, len(FEATURE_LAYOUT)))
    intercept = np.empty(buckets)
    for bucket in range(buckets):
        design = np.hstack([shared, seasonal[:, bucket, :]])
        scaler = StandardScaler().fit(design)
        estimator = Ridge(alpha=alpha, solver="svd").fit(
            scaler.transform(design), labels[:, bucket],
        )
        coefficients[bucket] = estimator.coef_ / scaler.scale_
        intercept[bucket] = estimator.intercept_ - coefficients[bucket] @ scaler.mean_
    return coefficients, intercept


def design(
    actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol, partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the model matrices once for a partition, on the hourly origin grid.

    Every row is assembled by ``TemporalProtocol.example``, so features stay
    complete at their issue time and labels stay inside the partition. A row
    whose label window would straddle a partition boundary is therefore dropped,
    which costs one day of origins per boundary. Building the matrices once is
    what makes a candidate sweep or a periodic refit affordable: both differ
    only in which rows they may read.
    """
    shared, seasonal, labels, issued = [], [], [], []
    for issue in origins(protocol, partition):
        example = protocol.example(actual, issue, LOOKBACK, HORIZON, partition=partition)
        for collected, value in zip(
            (shared, seasonal, labels), design_row(example, protocol), strict=True,
        ):
            collected.append(value)
        issued.append(issue)
    if not issued:
        raise ValueError(f"{partition} contains no complete forecast windows")
    return np.array(shared), np.array(seasonal), np.array(labels), np.array(issued, dtype=object)


def observable_rows(
    issued: np.ndarray, cutoff: datetime, protocol: TemporalProtocol,
    history_span: timedelta | None = None,
) -> np.ndarray:
    """Rows a model refitted at ``cutoff`` was allowed to see.

    A row enters only once its whole 24-hour label window has been published,
    which is what keeps any walk-forward leak-free by construction rather than
    by convention. ``history_span`` additionally drops the oldest rows.
    """
    rows = issued + HORIZON + protocol.observation_delay <= cutoff
    if history_span is not None:
        rows &= issued >= cutoff - history_span
    if not rows.any():
        raise ValueError(f"no training rows are observable at {cutoff.isoformat()}")
    return rows


def predict(
    coefficients: np.ndarray, intercept: np.ndarray, history: tuple, issue_time: datetime,
    protocol: TemporalProtocol, count: int,
) -> np.ndarray:
    """Evaluate the direct equations on one origin; predictions are floored at zero."""
    shared = forecast_features(history, issue_time, protocol)
    seasonal = target_features(history, issue_time, protocol, count)
    columns = np.hstack([np.broadcast_to(shared, (count, len(FEATURE_NAMES))), seasonal])
    return np.maximum(intercept[:count] + (coefficients[:count] * columns).sum(axis=1), 0)


def trajectory(issue_time: datetime, values: np.ndarray) -> CarbonIntensityForecast:
    return CarbonIntensityForecast(issue_time, tuple(
        CarbonIntensitySample(issue_time + index * STEP, value)
        for index, value in enumerate(values)
    ))


@dataclass
class RidgeCarbonIntensityForecaster:
    """One interpretable linear equation per target bucket; weights stay frozen."""

    actual: TimeSeriesCarbonIntensityProvider
    protocol: TemporalProtocol
    coefficients: np.ndarray
    intercept: np.ndarray
    metadata: dict

    @classmethod
    def fit(
        cls, actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
        metadata_path: str | Path, *, alpha: float = DEFAULT_ALPHA,
        history_span: timedelta | None = None,
    ) -> "RidgeCarbonIntensityForecaster":
        """Train on the origins ending before validation; never on later data.

        ``history_span`` keeps only the most recent training history, measured
        back from the start of validation. ``None`` uses everything available.
        """
        import sklearn

        if actual.granularity != STEP:
            raise ValueError("ML forecasts require 15-minute actuals")
        if history_span is not None and (
            not isinstance(history_span, timedelta) or history_span <= LOOKBACK
        ):
            raise ValueError("history_span must exceed the one-week feature lookback")
        since = None if history_span is None else protocol.validation_start - history_span
        shared, seasonal, labels = [], [], []

        def examples():
            for issue in origins(protocol, "train", since=since):
                example = protocol.example(actual, issue, LOOKBACK, HORIZON, partition="train")
                row = design_row(example, protocol)
                for collected, value in zip((shared, seasonal, labels), row, strict=True):
                    collected.append(value)
                yield example

        metadata = protocol.save_training_metadata(metadata_path, "ridge_direct", examples())
        coefficients, intercept = fit_direct(
            np.array(shared), np.array(seasonal), np.array(labels), alpha,
        )
        metadata.update({
            "model_version": MODEL_VERSION, "alpha": alpha, "solver": "svd",
            "feature_names": list(FEATURE_LAYOUT), "lookback_hours": 168,
            "horizon_hours": 24, "cadence_minutes": 60,
            "history_span_days": None if history_span is None else history_span.days,
            "training_window_start": (since or protocol.train_start).isoformat(),
            "strategy": "direct: one linear equation per 15-minute target bucket",
            "lag_reference": "floor15(issue_time - observation_delay)",
            "seasonal_reference": "last observable day and week at each target bucket",
            "calendar_timezone": "UTC", "prediction_floor": 0,
            "standardization": "mean and standard deviation fitted on train only",
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        })
        Path(metadata_path).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        return cls(actual, protocol, coefficients, intercept, metadata)

    def get_forecast(self, issue_time: datetime, horizon: timedelta) -> CarbonIntensityForecast:
        issue_time = aware_utc(issue_time, "issue_time")
        if not isinstance(horizon, timedelta) or not timedelta(0) < horizon <= HORIZON:
            raise ValueError("horizon must be positive and at most 24 hours")
        if issue_time < datetime.fromisoformat(self.metadata["training_available_at"]):
            raise ValueError("fitted model was not available at issue_time")
        history = self.protocol.history(self.actual, issue_time, LOOKBACK)
        count = (horizon + STEP - timedelta(microseconds=1)) // STEP
        return trajectory(issue_time, predict(
            self.coefficients, self.intercept, history, issue_time, self.protocol, count,
        ))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "metadata": self.metadata, "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path, actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
    ) -> "RidgeCarbonIntensityForecaster":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        if metadata["model_version"] != MODEL_VERSION or metadata["feature_names"] != list(FEATURE_LAYOUT):
            raise ValueError("unsupported forecast model or feature layout")
        if any(metadata[key] != value for key, value in protocol.metadata().items()):
            raise ValueError("model and temporal protocol do not match")
        cutoff = aware_utc(datetime.fromisoformat(metadata["training_cutoff"]))
        available = aware_utc(datetime.fromisoformat(metadata["training_available_at"]))
        if not protocol.train_start <= cutoff < protocol.validation_start or (
            available != cutoff + STEP + protocol.observation_delay or available > protocol.validation_start
        ):
            raise ValueError("model training timestamps violate the temporal protocol")
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        intercept = np.asarray(payload["intercept"], dtype=float)
        if actual.granularity != STEP or coefficients.shape != (BUCKETS, len(FEATURE_LAYOUT)) or (
            intercept.shape != (BUCKETS,)
        ):
            raise ValueError("model requires 15-minute actuals and 96 complete linear equations")
        if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
            raise ValueError("model coefficients must be finite")
        return cls(actual, protocol, coefficients, intercept, metadata)
