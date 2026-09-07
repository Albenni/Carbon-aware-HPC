from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .series import (
    FIFTEEN_MINUTES,
    CarbonIntensitySample,
    TimeSeriesCarbonIntensityProvider,
    aware_utc,
    bucket_start,
)


API_BASE_URL = "https://api.electricitymaps.com/v4"
DEFAULT_ZONE = "IT-NO"
TEMPORAL_GRANULARITY = "15_minutes"

# Longer caller ranges are split into adjacent, end exclusive chunks below.
MAX_REQUEST_SPAN = timedelta(days=2)

JsonObject: TypeAlias = Mapping[str, Any]
Transport: TypeAlias = Callable[[str, Mapping[str, str]], JsonObject]


def read_api_key(env_file: Path, variable_name: str) -> str:
    """The API token, from the environment or from a ``KEY=value`` env file.

    Lives here rather than in a module of its own because it is the client's
    credential and has no other caller. The token is read on each run and never
    written into a cache, a manifest, or a saved provider.
    """

    value = os.environ.get(variable_name)
    if not value and env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip().removeprefix("export ").lstrip()
            if line.startswith("#"):
                continue
            key, separator, candidate = line.partition("=")
            if separator and key.strip() == variable_name:
                value = candidate.strip().strip("'\"")
                break
    if not value:
        raise RuntimeError(
            f"set {variable_name} or add it to {env_file}; the token is never cached"
        )
    return value.strip()


class ElectricityMapsError(RuntimeError):
    """Raised when Electricity Maps cannot return a valid historical series."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep the API token on the configured Electricity Maps origin."""

    def redirect_request(self, request, file_pointer, code, message, headers, url):
        del request, file_pointer, code, message, headers, url
        return None

def _request_json(
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> JsonObject:
    request_headers = {
        **dict(headers),
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    request = Request(
        url,
        headers=request_headers,
        method="GET",
    )

    try:
        opener = build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read()
    except HTTPError as error:
        try:
            error_body = error.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""

        raise ElectricityMapsError(
            f"Electricity Maps returned HTTP {error.code}: {error_body}"
        ) from error
    except (URLError, TimeoutError) as error:
        raise ElectricityMapsError(
            "could not reach Electricity Maps"
        ) from error

    try:
        payload = json.loads(body)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ElectricityMapsError(
            "Electricity Maps returned invalid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise ElectricityMapsError(
            "Electricity Maps response must be an object"
        )

    return payload

def _iso_utc(timestamp: datetime) -> str:
    return aware_utc(timestamp).isoformat().replace("+00:00", "Z")


def _parse_api_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ElectricityMapsError("sample datetime must be text")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ElectricityMapsError(f"invalid sample datetime: {value!r}") from error
    return aware_utc(timestamp, "sample datetime")


def _ceil_bucket(timestamp: datetime) -> datetime:
    timestamp_utc = aware_utc(timestamp)
    floor = bucket_start(timestamp_utc, FIFTEEN_MINUTES)
    return floor if floor == timestamp_utc else floor + FIFTEEN_MINUTES


def _parse_sample(
    raw_sample: object,
    expected_zone: str,
    expected_emission_factor_type: str,
    expected_flow_traced: bool,
) -> CarbonIntensitySample:
    if not isinstance(raw_sample, dict):
        raise ElectricityMapsError("each API sample must be an object")
    sample_zone = raw_sample.get("zone", expected_zone)
    if sample_zone != expected_zone:
        raise ElectricityMapsError(
            f"expected zone {expected_zone}, received {sample_zone}"
        )
    emission_factor_type = raw_sample.get("emissionFactorType")
    if emission_factor_type is not None and (
        emission_factor_type != expected_emission_factor_type
    ):
        raise ElectricityMapsError(
            "Electricity Maps returned an unexpected emission-factor type"
        )
    if raw_sample.get("temporalGranularity", TEMPORAL_GRANULARITY) != TEMPORAL_GRANULARITY:
        raise ElectricityMapsError("API sample has an unexpected temporal granularity")
    if raw_sample.get("flowTraced", expected_flow_traced) is not expected_flow_traced:
        raise ElectricityMapsError("API sample has an unexpected flow-tracing method")

    if "carbonIntensity" in raw_sample:
        intensity = raw_sample["carbonIntensity"]
    elif "value" in raw_sample:
        intensity = raw_sample["value"]
    else:
        raise ElectricityMapsError("API sample has no carbon-intensity value")

    try:
        return CarbonIntensitySample(
            timestamp=_parse_api_timestamp(raw_sample.get("datetime")),
            intensity_gco2e_per_kwh=intensity,
            is_estimated=raw_sample.get("isEstimated"),
            estimation_method=raw_sample.get("estimationMethod"),
        )
    except (TypeError, ValueError) as error:
        raise ElectricityMapsError("API sample is invalid") from error


class ElectricityMapsClient:
    """Download actual carbon intensity and return an offline provider."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty text")
        if isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a real number")
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as error:
            raise TypeError("timeout_seconds must be a real number") from error
        if not isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")

        self._api_key = api_key.strip()
        self._transport = transport or (
            lambda url, headers: _request_json(url, headers, timeout)
        )

    def fetch_actual_range(
        self,
        start: datetime,
        end: datetime,
        *,
        zone: str = DEFAULT_ZONE,
        emission_factor_type: str = "lifecycle",
        flow_traced: bool = True,
        include_estimated: bool = True,
        chunk_size: timedelta = MAX_REQUEST_SPAN,
    ) -> TimeSeriesCarbonIntensityProvider:
        """Fetch and merge an end-exclusive range using two-day API requests."""

        requested_start = aware_utc(start, "start")
        requested_end = aware_utc(end, "end")
        if requested_end <= requested_start:
            raise ValueError("end must be later than start")
        if not isinstance(zone, str) or not zone.strip():
            raise ValueError("zone must be non-empty text")
        if emission_factor_type not in {"lifecycle", "direct"}:
            raise ValueError("emission_factor_type must be 'lifecycle' or 'direct'")
        if not isinstance(flow_traced, bool) or not isinstance(include_estimated, bool):
            raise TypeError("flow_traced and include_estimated must be booleans")
        if not isinstance(chunk_size, timedelta) or not (
            FIFTEEN_MINUTES <= chunk_size <= MAX_REQUEST_SPAN
        ):
            raise ValueError("chunk_size must be between 15 minutes and 2 days")
        if chunk_size % FIFTEEN_MINUTES != timedelta(0):
            raise ValueError("chunk_size must be a multiple of 15 minutes")

        zone = zone.strip()
        fetch_start = bucket_start(requested_start, FIFTEEN_MINUTES)
        fetch_end = _ceil_bucket(requested_end)
        samples_by_time: dict[datetime, CarbonIntensitySample] = {}
        request_count = 0
        duplicate_count = 0

        chunk_start = fetch_start
        while chunk_start < fetch_end:
            chunk_end = min(chunk_start + chunk_size, fetch_end)
            payload = self._fetch_chunk(
                chunk_start,
                chunk_end,
                zone=zone,
                emission_factor_type=emission_factor_type,
                flow_traced=flow_traced,
                include_estimated=include_estimated,
            )
            request_count += 1
            for sample in self._samples_from_payload(
                payload,
                zone,
                emission_factor_type,
                flow_traced,
            ):
                if not chunk_start <= sample.timestamp < chunk_end:
                    raise ElectricityMapsError(
                        "API returned a sample outside the requested chunk"
                    )
                if not include_estimated and sample.is_estimated is not False:
                    raise ElectricityMapsError(
                        "API returned an estimated or unlabelled sample while "
                        "estimations were disabled"
                    )
                previous = samples_by_time.get(sample.timestamp)
                if previous is not None and previous != sample:
                    raise ElectricityMapsError(
                        "API returned conflicting values for "
                        f"{sample.timestamp.isoformat()}"
                    )
                duplicate_count += previous is not None
                samples_by_time[sample.timestamp] = sample
            chunk_start = chunk_end

        if not samples_by_time:
            raise ElectricityMapsError("Electricity Maps returned no samples")

        fetched_at = datetime.now(timezone.utc)
        try:
            provider = TimeSeriesCarbonIntensityProvider(
                samples_by_time.values(),
                metadata={
                    "source": "Electricity Maps API",
                    "api_version": "v4",
                    "endpoint": "/carbon-intensity/past-range",
                    "signal": "actual carbon intensity",
                    "unit": "gCO2eq/kWh",
                    "zone": zone,
                    "temporal_granularity": TEMPORAL_GRANULARITY,
                    "emission_factor_type": emission_factor_type,
                    "flow_traced": flow_traced,
                    "flow_traced_provenance": "requested; checked when returned",
                    "estimated_values_included": include_estimated,
                    "requested_start": _iso_utc(requested_start),
                    "requested_end": _iso_utc(requested_end),
                    "cache_start": _iso_utc(fetch_start),
                    "cache_end": _iso_utc(fetch_end),
                    "request_chunk_seconds": int(chunk_size.total_seconds()),
                    "request_count": request_count,
                    "duplicate_sample_count": duplicate_count,
                    "fetched_at": _iso_utc(fetched_at),
                    "gap_policy": "error",
                },
            )
            # Fail the download if any expected bucket is absent. A partial
            # cache must never look complete to the simulator.
            provider.get_actual_range(fetch_start, fetch_end)
        except (TypeError, ValueError) as error:
            raise ElectricityMapsError(
                "Electricity Maps returned an invalid or incomplete series"
            ) from error
        return provider

    def _fetch_chunk(
        self,
        start: datetime,
        end: datetime,
        *,
        zone: str,
        emission_factor_type: str,
        flow_traced: bool,
        include_estimated: bool,
    ) -> JsonObject:
        query = urlencode(
            {
                "zone": zone,
                "start": _iso_utc(start),
                "end": _iso_utc(end),
                "temporalGranularity": TEMPORAL_GRANULARITY,
                "emissionFactorType": emission_factor_type,
                "flowTraced": str(flow_traced).lower(),
                "disableEstimations": str(not include_estimated).lower(),
            }
        )
        url = f"{API_BASE_URL}/carbon-intensity/past-range?{query}"
        return self._transport(url, {"auth-token": self._api_key})

    @staticmethod
    def _samples_from_payload(
        payload: JsonObject,
        expected_zone: str,
        expected_emission_factor_type: str,
        expected_flow_traced: bool,
    ) -> tuple[CarbonIntensitySample, ...]:
        response_zone = payload.get("zone")
        if response_zone is not None and response_zone != expected_zone:
            raise ElectricityMapsError(
                f"expected zone {expected_zone}, received {response_zone}"
            )
        if payload.get("flowTraced", expected_flow_traced) is not expected_flow_traced:
            raise ElectricityMapsError("API returned an unexpected flow-tracing method")
        response_granularity = payload.get("temporalGranularity")
        if response_granularity is not None and (
            response_granularity != TEMPORAL_GRANULARITY
        ):
            raise ElectricityMapsError(
                "Electricity Maps returned an unexpected temporal granularity"
            )
        response_emission_factor_type = payload.get("emissionFactorType")
        if response_emission_factor_type is not None and (
            response_emission_factor_type != expected_emission_factor_type
        ):
            raise ElectricityMapsError(
                "Electricity Maps returned an unexpected emission-factor type"
            )
        unit = payload.get("unit")
        if unit is not None and str(unit).lower() != "gco2eq/kwh":
            raise ElectricityMapsError(f"unexpected carbon-intensity unit: {unit}")

        raw_samples = payload.get("data")
        if not isinstance(raw_samples, list):
            raise ElectricityMapsError("Electricity Maps response has no data list")
        return tuple(
            _parse_sample(
                raw_sample,
                expected_zone,
                expected_emission_factor_type,
                expected_flow_traced,
            )
            for raw_sample in raw_samples
        )
