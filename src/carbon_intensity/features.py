"""Deterministic forecasting features from completed observations and UTC time."""

from datetime import datetime, timedelta
from math import cos, pi, sin

import numpy as np

from .series import CarbonIntensitySample
from .protocol import TemporalProtocol
from .series import FIFTEEN_MINUTES, aware_utc, bucket_start


LAG_MINUTES = (15, 30, 60, 180, 360, 720, 1440, 2880, 10080)
ROLLING_HOURS = (1, 6, 24)
LOOKBACK = timedelta(days=7)
SEASONAL_PERIODS = {"target_lag_1d": timedelta(days=1), "target_lag_7d": LOOKBACK}
FEATURE_NAMES = (
    *(f"lag_{minutes}min" for minutes in LAG_MINUTES),
    *(f"rolling_{hours}h_{stat}" for hours in ROLLING_HOURS for stat in ("mean", "min", "max")),
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "weekend",
)
TARGET_FEATURE_NAMES = tuple(SEASONAL_PERIODS)


def _observable_week(
    history: tuple[CarbonIntensitySample, ...], issue_time: datetime,
    protocol: TemporalProtocol,
) -> tuple[np.ndarray, datetime, datetime]:
    """Validate the history against the protocol and return its values.

    Causality is not assumed: the protocol rejects any observation that is not
    complete at ``issue_time``, and the week must end exactly on the last
    observable boundary, so a history containing the bucket in progress is
    refused instead of being used silently.
    """
    issue_time = aware_utc(issue_time, "issue_time")
    if issue_time != bucket_start(issue_time):
        raise ValueError("ML forecast issue_time must align with the 15-minute UTC grid")
    protocol.validate_features(history, issue_time)
    end = bucket_start(issue_time - protocol.observation_delay)
    if len(history) != LOOKBACK // FIFTEEN_MINUTES or any(
        sample.timestamp != end - LOOKBACK + i * FIFTEEN_MINUTES
        for i, sample in enumerate(history)
    ):
        raise ValueError("features require a complete week ending at the observable boundary")
    return np.array([sample.intensity_gco2e_per_kwh for sample in history]), issue_time, end


def forecast_features(
    history: tuple[CarbonIntensitySample, ...], issue_time: datetime,
    protocol: TemporalProtocol,
) -> np.ndarray:
    """Lag offsets end at the last observable boundary, including publication delay.

    Calendar features describe the issue time in UTC. Each direct output learns
    its own relationship with that calendar; no future intensity is a feature.
    """
    values, issue_time, _ = _observable_week(history, issue_time, protocol)
    features = [values[-minutes // 15] for minutes in LAG_MINUTES]
    for hours in ROLLING_HOURS:
        window = values[-hours * 4:]
        features.extend((window.mean(), window.min(), window.max()))
    for value, period in (
        (issue_time.hour + issue_time.minute / 60, 24),
        (issue_time.weekday(), 7), (issue_time.month - 1, 12),
    ):
        angle = 2 * pi * value / period
        features.extend((sin(angle), cos(angle)))
    features.append(float(issue_time.weekday() >= 5))
    return np.asarray(features, dtype=float)


def target_features(
    history: tuple[CarbonIntensitySample, ...], issue_time: datetime,
    protocol: TemporalProtocol, count: int,
) -> np.ndarray:
    """Per-target seasonal columns: yesterday and last week at each target bucket.

    The shared vector describes the issue time alone, so a direct equation
    cannot otherwise see the seasonal profile at *its own* target hour: it only
    knows yesterday at the issue hour. Offsets are read from the last observable
    day and week, wrapping the way the seasonal baselines do, so every column
    comes from the same validated history and stays available at issue_time.
    """
    values, issue_time, end = _observable_week(history, issue_time, protocol)
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(values):
        raise ValueError("count must be a positive number of target buckets within the history")
    index = (issue_time - end) // FIFTEEN_MINUTES + np.arange(count)
    return np.column_stack([
        values[len(values) - period + index % period]
        for period in (span // FIFTEEN_MINUTES for span in SEASONAL_PERIODS.values())
    ])
