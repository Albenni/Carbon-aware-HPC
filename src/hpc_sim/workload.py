"""Build simulator jobs from the cleaned PM100 tables.

This is the only module in the package that needs pyarrow; the engine, cluster,
schedulers, and models stay on the standard library.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from carbon_accounting import JobPowerProfile, measured_average_power

from .models import Job


PM100_SAMPLE_INTERVAL_SECONDS = 20.0
SECONDS_PER_MINUTE = 60.0

AveragePowerSource = Literal["weighted", "stored"]

_COLUMNS = (
    "job_id",
    "submit_time",
    "release_time",
    "start_time",
    "run_time",
    "time_limit",
    "num_nodes_alloc",
    "node_power_consumption",
    "node_power_mean_W",
)


def _job_from_row(
    row: dict[str, object],
    average_power_source: AveragePowerSource,
) -> Job:
    duration_seconds = float(row["run_time"])  # type: ignore[arg-type]
    profile = tuple(row["node_power_consumption"])  # type: ignore[arg-type]

    # The stored mean is needed to construct the profile, but it is replaced
    # below when the weighted mean is requested.
    power = JobPowerProfile(
        job_id=row["job_id"],  # type: ignore[arg-type]
        duration_seconds=duration_seconds,
        average_power_watts=float(row["node_power_mean_W"]),  # type: ignore[arg-type]
        power_profile_watts=profile,
        sample_interval_seconds=PM100_SAMPLE_INTERVAL_SECONDS,
    )
    if average_power_source == "weighted":
        # node_power_mean_W is an arithmetic mean of the samples, so it drifts
        # from the profile whenever the final segment is partial. The model
        # formalization requires the duration-weighted mean, which is the only
        # average that makes the average and measured models consume identical
        # energy and therefore isolates the timing effect.
        power = JobPowerProfile(
            job_id=power.job_id,
            duration_seconds=duration_seconds,
            average_power_watts=measured_average_power(power),
            power_profile_watts=profile,
            sample_interval_seconds=PM100_SAMPLE_INTERVAL_SECONDS,
        )

    time_limit = row.get("time_limit")
    time_limit_seconds = (
        float(time_limit) * SECONDS_PER_MINUTE  # type: ignore[arg-type]
        if time_limit is not None and float(time_limit) > 0  # type: ignore[arg-type]
        else None
    )

    return Job(
        job_id=row["job_id"],  # type: ignore[arg-type]
        submit_time=row["submit_time"],  # type: ignore[arg-type]
        release_time=row["release_time"],  # type: ignore[arg-type]
        nodes_required=int(row["num_nodes_alloc"]),  # type: ignore[arg-type]
        actual_duration_seconds=duration_seconds,
        power=power,
        time_limit_seconds=time_limit_seconds,
        trace_start_time=row["start_time"],  # type: ignore[arg-type]
    )


def load_jobs(
    path: str | Path,
    *,
    limit: int | None = None,
    released_from: datetime | None = None,
    released_before: datetime | None = None,
    average_power_source: AveragePowerSource = "weighted",
    batch_size: int = 4096,
) -> tuple[Job, ...]:
    """Read a cleaned PM100 parquet table into simulator jobs.

    Rows are read in file order, which the dataset preparation export already made
    chronological by submission. ``limit`` therefore takes a contiguous prefix
    rather than an arbitrary sample, keeping the arrival process intact.

    ``released_from`` / ``released_before`` select a half-open window on
    ``release_time``, so a run can target one month without materialising the
    whole trace.
    """

    if average_power_source not in ("weighted", "stored"):
        raise ValueError("average_power_source must be 'weighted' or 'stored'")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(Path(path))
    jobs: list[Job] = []

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(_COLUMNS)):
        columns = {name: batch[name].to_pylist() for name in _COLUMNS}
        for position in range(batch.num_rows):
            row = {name: values[position] for name, values in columns.items()}
            release_time = row["release_time"]
            if released_from is not None and release_time < released_from:
                continue
            if released_before is not None and release_time >= released_before:
                continue
            jobs.append(_job_from_row(row, average_power_source))
            if limit is not None and len(jobs) >= limit:
                return tuple(jobs)

    if not jobs:
        raise ValueError(f"no jobs matched the requested window in {path}")
    return tuple(jobs)
