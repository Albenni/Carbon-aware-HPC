"""What a carbon-intensity series is, and how a policy reads one.

One module because they are one concept: a :class:`CarbonIntensitySample` is a
bucket of the series, a :class:`CarbonIntensityForecast` is a trajectory of
them, and :class:`TimeSeriesCarbonIntensityProvider` is the series itself with
the lookups accounting and simulation need. Everything works on the same
15-minute UTC grid, which is the resolution Electricity Maps publishes and the
one the whole experiment is defined on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any


FIFTEEN_MINUTES = timedelta(minutes=15)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CACHE_SCHEMA_VERSION = 1


class CarbonIntensityError(ValueError):
    """Base error for invalid or unavailable carbon-intensity data."""


class MissingCarbonIntensityError(CarbonIntensityError):
    """Raised when an actual bucket is absent from the time series."""


class ForecastUnavailableError(CarbonIntensityError):
    """Raised when a provider has no forecast issued at the requested time."""


def aware_utc(timestamp: datetime, field_name: str = "timestamp") -> datetime:
    """Validate a timestamp and represent the same instant in UTC."""

    if not isinstance(timestamp, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp.astimezone(timezone.utc)


def bucket_start(
    timestamp: datetime,
    granularity: timedelta = FIFTEEN_MINUTES,
) -> datetime:
    """Return the UTC start of the bucket containing ``timestamp``.

    The granularity is a parameter because a policy may decide on a coarser
    grid than the series it reads; the series itself is always 15-minute.
    """

    if granularity <= timedelta(0):
        raise ValueError("granularity must be greater than zero")
    return _EPOCH + (aware_utc(timestamp) - _EPOCH) // granularity * granularity


@dataclass(frozen=True, slots=True)
class CarbonIntensitySample:
    """One actual carbon-intensity bucket, valid from ``timestamp`` onward."""

    timestamp: datetime
    intensity_gco2e_per_kwh: float
    is_estimated: bool | None = None
    estimation_method: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.intensity_gco2e_per_kwh, bool):
            raise TypeError("intensity_gco2e_per_kwh must be a number")
        intensity = float(self.intensity_gco2e_per_kwh)
        if not isfinite(intensity):
            raise ValueError("intensity_gco2e_per_kwh must be finite")
        if intensity < 0.0:
            raise ValueError("intensity_gco2e_per_kwh cannot be negative")
        object.__setattr__(self, "timestamp", aware_utc(self.timestamp))
        object.__setattr__(self, "intensity_gco2e_per_kwh", intensity)


@dataclass(frozen=True, slots=True)
class CarbonIntensityForecast:
    """Forecast values together with the time at which they became available."""

    issue_time: datetime
    samples: tuple[CarbonIntensitySample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("a forecast must contain at least one sample")
        if not all(isinstance(sample, CarbonIntensitySample) for sample in samples):
            raise TypeError("forecast samples must be CarbonIntensitySample values")
        samples = tuple(sorted(samples, key=lambda sample: sample.timestamp))
        if len({sample.timestamp for sample in samples}) != len(samples):
            raise ValueError("forecast samples cannot contain duplicate timestamps")
        object.__setattr__(self, "issue_time", aware_utc(self.issue_time, "issue_time"))
        object.__setattr__(self, "samples", samples)


class CarbonIntensityProvider(ABC):
    """Interface used by accounting and simulation code."""

    @property
    @abstractmethod
    def granularity(self) -> timedelta:
        """Duration for which each returned actual sample is valid."""

    @abstractmethod
    def get_actual(self, timestamp: datetime) -> float:
        """Return actual gCO2e/kWh for the bucket containing ``timestamp``."""

    @abstractmethod
    def get_actual_range(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[CarbonIntensitySample, ...]:
        """Return actual buckets overlapping the half-open range [start, end)."""

    def get_forecast(
        self,
        issue_time: datetime,
        horizon: timedelta,
    ) -> CarbonIntensityForecast:
        """Return an issued forecast without substituting future actual data."""

        del issue_time, horizon
        raise ForecastUnavailableError("this provider contains actual data only")


class TimeSeriesCarbonIntensityProvider(CarbonIntensityProvider):
    """In-memory, piecewise-constant actual carbon-intensity series.

    Input samples may be unordered, but every timestamp must lie exactly on the
    15-minute UTC grid. Gaps remain gaps: lookups never interpolate, select a
    nearest value, or extrapolate beyond the available buckets.
    """

    granularity = FIFTEEN_MINUTES

    def __init__(
        self,
        samples: Iterable[CarbonIntensitySample],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        supplied = tuple(samples)
        if not supplied:
            raise ValueError("at least one carbon-intensity sample is required")
        if not all(isinstance(sample, CarbonIntensitySample) for sample in supplied):
            raise TypeError("samples must contain CarbonIntensitySample values")
        ordered = tuple(sorted(supplied, key=lambda sample: sample.timestamp))

        by_timestamp: dict[datetime, CarbonIntensitySample] = {}
        for sample in ordered:
            if bucket_start(sample.timestamp) != sample.timestamp:
                raise ValueError(
                    "sample timestamps must align with the 15-minute UTC grid: "
                    f"{sample.timestamp.isoformat()}"
                )
            if sample.timestamp in by_timestamp:
                raise ValueError(
                    f"duplicate carbon-intensity timestamp: {sample.timestamp.isoformat()}"
                )
            by_timestamp[sample.timestamp] = sample

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        supplied_metadata = dict(metadata or {})
        try:
            # The round trip is the validation. Provenance has to survive the
            # cache unchanged, so anything strict JSON refuses outright, and
            # anything it silently rewrites - an integer key, a tuple - is
            # rejected here rather than discovered as a difference on reload.
            metadata_json = json.dumps(
                supplied_metadata, allow_nan=False, sort_keys=True
            )
            if json.loads(metadata_json) != supplied_metadata:
                raise TypeError("metadata must round-trip through JSON unchanged")
        except ValueError as error:
            raise TypeError("metadata must contain JSON values") from error

        self._samples = ordered
        self._by_timestamp = by_timestamp
        self._metadata_json = metadata_json

    @property
    def samples(self) -> tuple[CarbonIntensitySample, ...]:
        return self._samples

    @property
    def metadata(self) -> Mapping[str, Any]:
        # A fresh decoded copy keeps nested provenance immutable to callers.
        return MappingProxyType(json.loads(self._metadata_json))

    @property
    def coverage_start(self) -> datetime:
        return self._samples[0].timestamp

    @property
    def coverage_end(self) -> datetime:
        return self._samples[-1].timestamp + FIFTEEN_MINUTES

    def _sample_at(self, timestamp: datetime) -> CarbonIntensitySample:
        bucket = bucket_start(timestamp)
        try:
            return self._by_timestamp[bucket]
        except KeyError as error:
            raise MissingCarbonIntensityError(
                f"no actual carbon intensity for UTC bucket {bucket.isoformat()}"
            ) from error

    def get_actual(self, timestamp: datetime) -> float:
        return self._sample_at(timestamp).intensity_gco2e_per_kwh

    def get_actual_range(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[CarbonIntensitySample, ...]:
        start_utc = aware_utc(start, "start")
        end_utc = aware_utc(end, "end")
        if end_utc <= start_utc:
            raise ValueError("end must be later than start")

        result: list[CarbonIntensitySample] = []
        current = bucket_start(start_utc)
        while current < end_utc:
            result.append(self._sample_at(current))
            current += FIFTEEN_MINUTES
        return tuple(result)

    def save(self, path: str | Path) -> Path:
        """Persist samples and source metadata in a normalized JSON cache.

        Written through a temporary file in the same directory and renamed, so
        an interrupted run leaves the previous cache intact rather than a
        half-written one that would fail validation on the next load.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "granularity_seconds": FIFTEEN_MINUTES.total_seconds(),
            "metadata": json.loads(self._metadata_json),
            "samples": [
                {
                    "timestamp": sample.timestamp.isoformat().replace("+00:00", "Z"),
                    "intensity_gco2e_per_kwh": sample.intensity_gco2e_per_kwh,
                    "is_estimated": sample.is_estimated,
                    "estimation_method": sample.estimation_method,
                }
                for sample in self._samples
            ],
        }

        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
                stream.write("\n")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TimeSeriesCarbonIntensityProvider:
        """Load a cache produced by :meth:`save` and validate it again.

        The cache comes from disk rather than from this codebase, so its shape
        is checked before anything is built from it.
        """

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid carbon-intensity cache: {source}") from error
        if not isinstance(payload, dict):
            raise ValueError("carbon-intensity cache root must be an object")
        if payload.get("schema_version") is not _CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported carbon-intensity cache schema")
        try:
            if payload["granularity_seconds"] != FIFTEEN_MINUTES.total_seconds():
                raise ValueError("carbon-intensity caches must use 15-minute buckets")
            raw_samples = payload["samples"]
            metadata = payload.get("metadata", {})
        except KeyError as error:
            raise ValueError("carbon-intensity cache metadata is invalid") from error
        if not isinstance(raw_samples, list) or not isinstance(metadata, dict):
            raise ValueError("cached samples must be a list and metadata an object")

        samples: list[CarbonIntensitySample] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                raise ValueError("each cached sample must be an object")
            try:
                samples.append(
                    CarbonIntensitySample(
                        timestamp=datetime.fromisoformat(raw_sample["timestamp"]),
                        intensity_gco2e_per_kwh=raw_sample["intensity_gco2e_per_kwh"],
                        is_estimated=raw_sample.get("is_estimated"),
                        estimation_method=raw_sample.get("estimation_method"),
                    )
                )
            except KeyError as error:
                raise ValueError("cached sample is missing a required field") from error
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid cached sample: {raw_sample!r}") from error

        return cls(samples, metadata=metadata)
