from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any

from .models import CarbonIntensityForecast, CarbonIntensitySample


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
    """Return the UTC start of the bucket containing ``timestamp``."""

    timestamp_utc = aware_utc(timestamp)
    _validate_granularity(granularity)
    bucket_index = (timestamp_utc - _EPOCH) // granularity
    return _EPOCH + bucket_index * granularity


def _validate_granularity(granularity: timedelta) -> None:
    if not isinstance(granularity, timedelta):
        raise TypeError("granularity must be a timedelta")
    if granularity <= timedelta(0):
        raise ValueError("granularity must be greater than zero")


def _iso_utc(timestamp: datetime) -> str:
    return aware_utc(timestamp).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("cached sample timestamp must be text")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid cached timestamp: {value!r}") from error


def _validate_metadata_value(value: object, path: str = "metadata") -> None:
    """Require metadata that round-trips through strict JSON unchanged."""

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError(f"{path} cannot contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_metadata_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} object keys must be text")
        for key, item in value.items():
            _validate_metadata_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} must contain only JSON values")


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
    configured UTC grid. Gaps remain gaps: lookups never interpolate, select a
    nearest value, or extrapolate beyond the available buckets.
    """

    def __init__(
        self,
        samples: Iterable[CarbonIntensitySample],
        *,
        granularity: timedelta = FIFTEEN_MINUTES,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        _validate_granularity(granularity)
        supplied_samples = tuple(samples)
        if not supplied_samples:
            raise ValueError("at least one carbon-intensity sample is required")
        if not all(
            isinstance(sample, CarbonIntensitySample) for sample in supplied_samples
        ):
            raise TypeError("samples must contain CarbonIntensitySample values")
        ordered_samples = tuple(
            sorted(supplied_samples, key=lambda sample: sample.timestamp)
        )

        by_timestamp: dict[datetime, CarbonIntensitySample] = {}
        for sample in ordered_samples:
            if bucket_start(sample.timestamp, granularity) != sample.timestamp:
                raise ValueError(
                    "sample timestamps must align with the UTC granularity grid: "
                    f"{sample.timestamp.isoformat()}"
                )
            if sample.timestamp in by_timestamp:
                raise ValueError(
                    f"duplicate carbon-intensity timestamp: {sample.timestamp.isoformat()}"
                )
            by_timestamp[sample.timestamp] = sample

        if metadata is None:
            metadata_copy: dict[str, Any] = {}
        elif isinstance(metadata, Mapping):
            metadata_copy = dict(metadata)
        else:
            raise TypeError("metadata must be a mapping")
        _validate_metadata_value(metadata_copy)
        try:
            metadata_json = json.dumps(
                metadata_copy,
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("metadata must contain JSON-serializable values") from error

        self._granularity = granularity
        self._samples = ordered_samples
        self._by_timestamp = by_timestamp
        self._metadata_json = metadata_json

    @property
    def granularity(self) -> timedelta:
        return self._granularity

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
        return self._samples[-1].timestamp + self._granularity

    def _sample_at(self, timestamp: datetime) -> CarbonIntensitySample:
        bucket = bucket_start(timestamp, self._granularity)
        try:
            return self._by_timestamp[bucket]
        except KeyError as error:
            raise MissingCarbonIntensityError(
                "no actual carbon intensity for UTC bucket "
                f"{bucket.isoformat()}"
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
        current = bucket_start(start_utc, self._granularity)
        while current < end_utc:
            result.append(self._sample_at(current))
            current += self._granularity
        return tuple(result)

    def save(self, path: str | Path) -> Path:
        """Persist samples and source metadata in a normalized JSON cache."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "granularity_seconds": self._granularity.total_seconds(),
            "metadata": json.loads(self._metadata_json),
            "samples": [
                {
                    "timestamp": _iso_utc(sample.timestamp),
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
            ) as temporary_file:
                temporary = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                temporary_file.write("\n")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TimeSeriesCarbonIntensityProvider:
        """Load a cache produced by :meth:`save` and validate it again."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid carbon-intensity cache: {source}") from error
        if not isinstance(payload, dict):
            raise ValueError("carbon-intensity cache root must be an object")
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or (  # bool is an int subclass
            schema_version != _CACHE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported carbon-intensity cache schema")

        try:
            raw_granularity = payload["granularity_seconds"]
            raw_samples = payload["samples"]
            metadata = payload.get("metadata", {})
        except KeyError as error:
            raise ValueError("carbon-intensity cache metadata is invalid") from error
        if isinstance(raw_granularity, bool) or not isinstance(
            raw_granularity,
            (int, float),
        ):
            raise ValueError("cached granularity must be a number")
        if not isfinite(raw_granularity) or raw_granularity <= 0:
            raise ValueError("cached granularity must be finite and positive")
        try:
            granularity = timedelta(seconds=raw_granularity)
        except OverflowError as error:
            raise ValueError("cached granularity is outside the supported range") from error
        if not isinstance(raw_samples, list):
            raise ValueError("cached samples must be a list")
        if not isinstance(metadata, dict):
            raise ValueError("cached metadata must be an object")

        samples: list[CarbonIntensitySample] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                raise ValueError("each cached sample must be an object")
            try:
                samples.append(
                    CarbonIntensitySample(
                        timestamp=_parse_timestamp(raw_sample["timestamp"]),
                        intensity_gco2e_per_kwh=raw_sample[
                            "intensity_gco2e_per_kwh"
                        ],
                        is_estimated=raw_sample.get("is_estimated"),
                        estimation_method=raw_sample.get("estimation_method"),
                    )
                )
            except KeyError as error:
                raise ValueError("cached sample is missing a required field") from error

        return cls(samples, granularity=granularity, metadata=metadata)
