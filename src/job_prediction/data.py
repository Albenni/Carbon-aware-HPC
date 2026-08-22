"""Leak-free PM100 inputs and chronological dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from carbon_accounting import (
    JobPowerProfile,
    energy_from_measured_profile,
)


JOB_ID = "job_id"
SUBMIT_TIME = "submit_time"
COMPLETION_TIME = "end_time"
DURATION_SECONDS = "duration_seconds"
AVERAGE_POWER_WATTS = "average_power_watts"
ENERGY_KWH = "energy_kwh"

# Every entry is present when a job is submitted. Identifiers, allocation
# outcomes, eligibility/start/end timestamps, and measured power are excluded.
NUMERIC_FEATURES = (
    "time_limit",
    "num_nodes_req",
    "num_cores_req",
    "num_tasks",
    "cores_per_task",
    "num_gpus_req",
    "mem_req",
    "priority",
)
CATEGORICAL_FEATURES = ("qos",)
SUBMISSION_FEATURES = (SUBMIT_TIME, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
TARGETS = (DURATION_SECONDS, AVERAGE_POWER_WATTS, ENERGY_KWH)

_SOURCE_COLUMNS = (
    JOB_ID,
    *SUBMISSION_FEATURES,
    COMPLETION_TIME,
    "run_time",
    "node_power_consumption",
)
_POWER_SAMPLE_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Strictly ordered train, validation, and test frames.

    ``train_until`` and ``validation_until`` are half-open boundaries. All jobs
    with the same submission timestamp consequently stay in the same split.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_until: datetime
    validation_until: datetime

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.train), len(self.validation), len(self.test)


def _targets(
    job_ids: list[object],
    durations: list[int],
    profiles: list[list[int]],
) -> tuple[list[float], list[float]]:
    powers: list[float] = []
    energies: list[float] = []
    for job_id, duration, samples in zip(job_ids, durations, profiles, strict=True):
        profile = JobPowerProfile(
            job_id=job_id,  # type: ignore[arg-type]
            duration_seconds=duration,
            average_power_watts=0.0,
            power_profile_watts=tuple(samples),
            sample_interval_seconds=_POWER_SAMPLE_SECONDS,
        )
        energy = energy_from_measured_profile(profile)
        energies.append(energy)
        powers.append(energy * 3_600_000.0 / duration)
    return powers, energies


def load_job_data(
    path: str | Path,
    *,
    batch_size: int = 4_096,
) -> pd.DataFrame:
    """Load submission features and post-run targets from a cleaned trace.

    The measured profile is used only to construct training targets. It is
    discarded before the frame is returned and can never reach the encoder.
    Batches keep the variable-length profiles from being expanded into one
    large Python object graph.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    import pyarrow
    import pyarrow.parquet as parquet

    source = Path(path)
    parquet_file = parquet.ParquetFile(source)
    missing = set(_SOURCE_COLUMNS).difference(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")

    frames: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(_SOURCE_COLUMNS),
    ):
        job_ids = batch[JOB_ID].to_pylist()
        durations = batch["run_time"].to_pylist()
        powers, energies = _targets(
            job_ids,
            durations,
            batch["node_power_consumption"].to_pylist(),
        )
        features = pyarrow.Table.from_batches([batch]).select(
            [JOB_ID, *SUBMISSION_FEATURES, COMPLETION_TIME]
        ).to_pandas()
        features[DURATION_SECONDS] = durations
        features[AVERAGE_POWER_WATTS] = powers
        features[ENERGY_KWH] = energies
        frames.append(features)

    if not frames:
        raise ValueError(f"no jobs found in {source}")

    data = pd.concat(frames, ignore_index=True)
    if data[JOB_ID].duplicated().any():
        raise ValueError("job ids must be unique")

    # The full cleaned parquet is not stored chronologically. Sorting here is
    # therefore part of the anti-leakage contract, not presentation polish.
    return data.sort_values(
        [SUBMIT_TIME, JOB_ID],
        kind="stable",
        ignore_index=True,
    )


def temporal_split(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Make a chronological split without bisecting timestamp ties."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")
    if SUBMIT_TIME not in data or JOB_ID not in data:
        raise ValueError(f"data must contain {SUBMIT_TIME!r} and {JOB_ID!r}")
    if len(data) < 3:
        raise ValueError("at least three jobs are required")

    ordered = data.sort_values(
        [SUBMIT_TIME, JOB_ID],
        kind="stable",
        ignore_index=True,
    )
    timestamps = pd.to_datetime(ordered[SUBMIT_TIME], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError("submit_time cannot be missing")
    train_position = max(1, int(len(ordered) * train_fraction))
    validation_position = max(
        train_position + 1,
        int(len(ordered) * (train_fraction + validation_fraction)),
    )
    if validation_position >= len(ordered):
        raise ValueError("split fractions leave no test jobs")

    train_until = timestamps.iloc[train_position]
    validation_until = timestamps.iloc[validation_position]
    train_mask = timestamps < train_until
    validation_mask = (timestamps >= train_until) & (timestamps < validation_until)
    test_mask = timestamps >= validation_until

    train = ordered.loc[train_mask].reset_index(drop=True)
    validation = ordered.loc[validation_mask].reset_index(drop=True)
    test = ordered.loc[test_mask].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "timestamp ties make one temporal split empty; choose different fractions"
        )

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_until=train_until.to_pydatetime(),
        validation_until=validation_until.to_pydatetime(),
    )
