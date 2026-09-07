"""Reproducible Electricity Maps actual history for the PM100 experiment."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .electricity_maps import ElectricityMapsClient, MAX_REQUEST_SPAN
from .series import FIFTEEN_MINUTES, TimeSeriesCarbonIntensityProvider, aware_utc, bucket_start


ACTUAL_METADATA = {
    "source": "Electricity Maps API",
    "api_version": "v4",
    "endpoint": "/carbon-intensity/past-range",
    "signal": "actual carbon intensity",
    "unit": "gCO2eq/kWh",
    "zone": "IT-NO",
    "temporal_granularity": "15_minutes",
    "emission_factor_type": "lifecycle",
    "flow_traced": True,
}
HISTORY_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
VALIDATION_START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _validate_actual(provider: TimeSeriesCarbonIntensityProvider) -> None:
    if provider.granularity != FIFTEEN_MINUTES:
        raise ValueError("historical actuals must use 15-minute buckets")
    metadata = provider.metadata
    for key, expected in ACTUAL_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(f"incompatible actual metadata: {key} must be {expected!r}")
    provider.get_actual_range(provider.coverage_start, provider.coverage_end)


def merge_actual_caches(
    paths: list[Path], start: datetime, end: datetime,
) -> TimeSeriesCarbonIntensityProvider:
    """Reject gaps/conflicts and record identical overlaps and source hashes."""
    start, end = aware_utc(start), aware_utc(end)
    if start >= end or any(t != bucket_start(t) for t in (start, end)):
        raise ValueError("historical bounds must be ordered on the 15-minute grid")
    samples = {}
    sources = []
    duplicates = 0
    api_duplicates = 0
    untracked_sources = 0
    for path in paths:
        provider = TimeSeriesCarbonIntensityProvider.load(path)
        _validate_actual(provider)
        sources.append({
            "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
            "metadata": dict(provider.metadata), "buckets": len(provider.samples),
        })
        api_duplicates += provider.metadata.get("duplicate_sample_count", 0)
        untracked_sources += "duplicate_sample_count" not in provider.metadata
        for sample in provider.samples:
            if not start <= sample.timestamp < end:
                raise ValueError(f"source cache extends outside the historical range: {path}")
            previous = samples.get(sample.timestamp)
            if previous is not None and previous != sample:
                raise ValueError(f"conflicting actuals at {sample.timestamp.isoformat()}")
            duplicates += previous is not None
            samples[sample.timestamp] = sample

    yearly = {}
    for year in sorted({timestamp.year for timestamp in samples}):
        points = [sample for sample in samples.values() if sample.timestamp.year == year]
        yearly[str(year)] = {
            "buckets": len(points),
            "estimated": sum(sample.is_estimated is True for sample in points),
            "unlabelled": sum(sample.is_estimated is None for sample in points),
            "estimation_methods": dict(sorted(Counter(
                sample.estimation_method or "unspecified" for sample in points
            ).items())),
        }
    merged = TimeSeriesCarbonIntensityProvider(samples.values(), metadata={
        **ACTUAL_METADATA,
        "cache_start": start.isoformat(), "cache_end": end.isoformat(),
        "gap_policy": "error", "source_caches": sources,
        "methodology_note": (
            "Matching API parameters; no historical methodology version or "
            "as-published vintage is supplied by these caches."
        ),
        "quality": {
            "missing_buckets": 0, "identical_overlaps": duplicates,
            "api_duplicate_samples": api_duplicates, "by_year": yearly,
            "sources_without_api_duplicate_tracking": untracked_sources,
        },
    })
    merged.get_actual_range(start, end)
    return merged


def build_history(
    existing_2020: Path,
    output_dir: Path,
    client: ElectricityMapsClient | None = None,
) -> TimeSeriesCarbonIntensityProvider:
    """Download missing history with one durable cache per two-day request.

    Passing no client makes this an offline rebuild. Existing chunks are always
    revalidated and reused, so an interrupted download resumes on the next run.
    """
    existing = TimeSeriesCarbonIntensityProvider.load(existing_2020)
    _validate_actual(existing)
    if not VALIDATION_START <= existing.coverage_start < existing.coverage_end <= datetime(
        2021, 1, 1, tzinfo=timezone.utc,
    ):
        raise ValueError("the existing actual cache must lie within 2020")

    paths = []
    start = HISTORY_START
    while start < existing.coverage_start:
        next_year = datetime(start.year + 1, 1, 1, tzinfo=timezone.utc)
        end = min(start + MAX_REQUEST_SPAN, next_year, existing.coverage_start)
        path = output_dir / "chunks" / f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
        if path.exists():
            chunk = TimeSeriesCarbonIntensityProvider.load(path)
        else:
            if client is None:
                raise ValueError(f"missing historical cache: {path}; run without --offline")
            chunk = client.fetch_actual_range(start, end)
            chunk.save(path)
        _validate_actual(chunk)
        if (chunk.coverage_start, chunk.coverage_end) != (start, end):
            raise ValueError(f"unexpected historical chunk coverage: {path}")
        paths.append(path)
        if len(paths) == 1 or len(paths) % 20 == 0 or end == existing.coverage_start:
            print(f"Historical cache: {len(paths)} chunks, through {end.isoformat()}", flush=True)
        start = end
    paths.append(existing_2020)
    history = merge_actual_caches(paths, HISTORY_START, existing.coverage_end)
    history.save(output_dir / "actual.json")
    return history

