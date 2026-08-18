from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite


def _aware_utc(timestamp: datetime, field_name: str) -> datetime:
    if not isinstance(timestamp, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp.astimezone(timezone.utc)


def _intensity(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("intensity_gco2e_per_kwh must be a number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("intensity_gco2e_per_kwh must be a real number") from error
    if not isfinite(converted):
        raise ValueError("intensity_gco2e_per_kwh must be finite")
    if converted < 0.0:
        raise ValueError("intensity_gco2e_per_kwh cannot be negative")
    return converted


@dataclass(frozen=True, slots=True)
class CarbonIntensitySample:
    """One actual carbon-intensity bucket, valid from ``timestamp`` onward."""

    timestamp: datetime
    intensity_gco2e_per_kwh: float
    is_estimated: bool | None = None
    estimation_method: str | None = None

    def __post_init__(self) -> None:
        timestamp = _aware_utc(self.timestamp, "timestamp")
        intensity = _intensity(self.intensity_gco2e_per_kwh)

        if self.is_estimated is not None and not isinstance(self.is_estimated, bool):
            raise TypeError("is_estimated must be a boolean or None")
        if self.estimation_method is not None and not isinstance(
            self.estimation_method,
            str,
        ):
            raise TypeError("estimation_method must be text or None")

        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "intensity_gco2e_per_kwh", intensity)


@dataclass(frozen=True, slots=True)
class CarbonIntensityForecast:
    """Forecast values together with the time at which they became available."""

    issue_time: datetime
    samples: tuple[CarbonIntensitySample, ...]

    def __post_init__(self) -> None:
        issue_time = _aware_utc(self.issue_time, "issue_time")
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("a forecast must contain at least one sample")
        if not all(isinstance(sample, CarbonIntensitySample) for sample in samples):
            raise TypeError("forecast samples must be CarbonIntensitySample values")
        samples = tuple(sorted(samples, key=lambda sample: sample.timestamp))
        timestamps = [sample.timestamp for sample in samples]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("forecast samples cannot contain duplicate timestamps")
        object.__setattr__(self, "issue_time", issue_time)
        object.__setattr__(self, "samples", samples)
