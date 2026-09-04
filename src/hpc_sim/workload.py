"""Build simulator jobs from the cleaned PM100 tables.

This is the only module in the package that needs pyarrow; the engine, cluster,
schedulers, and models stay on the standard library.
"""

from __future__ import annotations

from collections.abc import Set
from datetime import datetime
from pathlib import Path
from typing import Literal

from carbon_accounting import JobIdentifier, JobPowerProfile, measured_average_power

from .models import Job


PM100_SAMPLE_INTERVAL_SECONDS = 20.0
SECONDS_PER_MINUTE = 60.0

AveragePowerSource = Literal["weighted", "stored"]

TERMINAL_CONTENTION_STATES = frozenset(
    {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
)

_COLUMNS = (
    "job_id",
    "submit_time",
    "release_time",
    "start_time",
    "run_time",
    "time_limit",
    "num_nodes_req",
    "node_power_consumption",
    "node_power_mean_W",
)

_CONTENTION_COLUMNS = (
    "job_id",
    "job_state",
    "partition",
    "submit_time",
    "eligible_time",
    "start_time",
    "end_time",
    "time_limit",
    "num_nodes_alloc",
    "nodes",
)


def _time_limit_seconds(value: object) -> float | None:
    return (
        float(value) * SECONDS_PER_MINUTE
        if value is not None and float(value) > 0  # type: ignore[arg-type]
        else None
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

    return Job(
        job_id=row["job_id"],  # type: ignore[arg-type]
        submit_time=row["submit_time"],  # type: ignore[arg-type]
        release_time=row["release_time"],  # type: ignore[arg-type]
        # Requested nodes are known at submission. They equal allocated nodes
        # for every retained PM100 job, but only this field respects the online
        # information boundary used by the prediction models.
        nodes_required=int(row["num_nodes_req"]),  # type: ignore[arg-type]
        actual_duration_seconds=duration_seconds,
        power=power,
        time_limit_seconds=_time_limit_seconds(row.get("time_limit")),
        trace_start_time=row["start_time"],  # type: ignore[arg-type]
    )


def _contention_job_from_row(row: dict[str, object], partition: str) -> Job | None:
    """Build a resource-only job when the raw execution evidence is sound."""

    if (
        row["job_state"] not in TERMINAL_CONTENTION_STATES
        or row["partition"] != partition
    ):
        return None

    submit = row["submit_time"]
    release = row["eligible_time"]
    start = row["start_time"]
    end = row["end_time"]
    if not all(isinstance(value, datetime) for value in (submit, release, start, end)):
        return None
    try:
        if not submit <= release <= start < end:  # type: ignore[operator]
            return None
    except TypeError:
        return None

    allocated = row["num_nodes_alloc"]
    nodes_value = row["nodes"]
    if isinstance(allocated, bool) or not isinstance(allocated, int) or allocated <= 0:
        return None
    if nodes_value is None or isinstance(nodes_value, (str, bytes, bytearray)):
        return None
    try:
        nodes = tuple(nodes_value)  # type: ignore[arg-type]
        if len(nodes) != allocated or len(nodes) != len(set(nodes)):
            return None
    except (TypeError, ValueError):
        return None

    duration_seconds = (end - start).total_seconds()  # type: ignore[operator]
    return Job(
        job_id=row["job_id"],  # type: ignore[arg-type]
        submit_time=submit,  # type: ignore[arg-type]
        release_time=release,  # type: ignore[arg-type]
        nodes_required=allocated,
        actual_duration_seconds=duration_seconds,
        power=None,
        time_limit_seconds=_time_limit_seconds(row.get("time_limit")),
        trace_start_time=start,  # type: ignore[arg-type]
    )


def load_jobs(
    path: str | Path,
    *,
    limit: int | None = None,
    released_from: datetime | None = None,
    released_before: datetime | None = None,
    average_power_source: AveragePowerSource = "weighted",
    batch_size: int = 4096,
    job_ids: Set[JobIdentifier] | None = None,
) -> tuple[Job, ...]:
    """Read a cleaned PM100 parquet table into simulator jobs.

    Rows are read in file order. The committed debug table is chronological, so
    ``limit`` takes its contiguous prefix; the full cleaned table is not sorted,
    and experiments that need a temporal cohort should use ``job_ids`` from a
    chronological split instead of combining that table with ``limit``.

    ``released_from`` / ``released_before`` select a half-open window on
    ``release_time``, so a run can target one month without materialising the
    whole trace. ``job_ids`` restricts loading to a prediction artifact or any
    other explicit cohort.
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
            if job_ids is not None and row["job_id"] not in job_ids:
                continue
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


def load_contention_jobs(
    path: str | Path,
    *,
    partition: str = "1",
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    batch_size: int = 4096,
) -> tuple[Job, ...]:
    """Load terminal PM100 executions as scheduler-only resource demand.

    Only rows with ordered submit, eligible, start and end timestamps and a
    positive allocation matching unique node ids are retained. Their observed
    duration and allocated width govern node release; no timestamp or power
    profile is inferred. When supplied, the half-open window keeps jobs released
    before its end whose observed completion falls after its start.
    """

    if window_start is not None and window_end is not None and window_start >= window_end:
        raise ValueError("contention window must have positive duration")

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(Path(path))
    jobs: list[Job] = []
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(_CONTENTION_COLUMNS),
    ):
        columns = {name: batch[name].to_pylist() for name in _CONTENTION_COLUMNS}
        for position in range(batch.num_rows):
            row = {name: values[position] for name, values in columns.items()}
            job = _contention_job_from_row(row, partition)
            if job is None:
                continue
            observed_end = row["end_time"]
            if window_start is not None and observed_end <= window_start:  # type: ignore[operator]
                continue
            if window_end is not None and job.release_time >= window_end:
                continue
            jobs.append(job)

    if not jobs:
        raise ValueError(f"no terminal contention jobs matched in {path}")
    return tuple(jobs)
