"""Attach persisted scheduling predictions to simulator jobs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import isclose, isfinite
from pathlib import Path

from hpc_sim.models import Job

from .data import JOB_ID
from .model import (
    PREDICTED_AVERAGE_POWER_WATTS,
    PREDICTED_DURATION_SECONDS,
    PREDICTED_ENERGY_KWH,
    WATT_SECONDS_PER_KILOWATT_HOUR,
)


@dataclass(frozen=True, slots=True)
class SchedulingPrediction:
    duration_seconds: float
    average_power_watts: float
    energy_kwh: float

    def __post_init__(self) -> None:
        values = (
            self.duration_seconds,
            self.average_power_watts,
            self.energy_kwh,
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("prediction values must be finite and greater than zero")
        implied_energy = (
            self.average_power_watts
            * self.duration_seconds
            / WATT_SECONDS_PER_KILOWATT_HOUR
        )
        if not isclose(self.energy_kwh, implied_energy, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "prediction is inconsistent: energy must equal power * duration"
            )


def load_prediction_file(
    path: str | Path,
) -> dict[object, SchedulingPrediction]:
    """Read the prediction-only parquet artifact written by the trainer."""

    import pyarrow.parquet as parquet

    columns = (
        JOB_ID,
        PREDICTED_DURATION_SECONDS,
        PREDICTED_AVERAGE_POWER_WATTS,
        PREDICTED_ENERGY_KWH,
    )
    table = parquet.read_table(Path(path), columns=list(columns))
    values = {name: table[name].to_pylist() for name in columns}
    predictions: dict[object, SchedulingPrediction] = {}
    for job_id, duration, power, energy in zip(
        *(values[name] for name in columns),
        strict=True,
    ):
        if job_id in predictions:
            raise ValueError(f"prediction file contains duplicate job id {job_id}")
        predictions[job_id] = SchedulingPrediction(
            duration_seconds=float(duration),
            average_power_watts=float(power),
            energy_kwh=float(energy),
        )
    if not predictions:
        raise ValueError(f"prediction file {path} is empty")
    return predictions


def attach_predictions(
    jobs: Iterable[Job],
    predictions: Mapping[object, SchedulingPrediction],
) -> tuple[Job, ...]:
    """Return immutable jobs carrying estimates, with no ground-truth changes."""

    attached: list[Job] = []
    missing: list[object] = []
    for job in jobs:
        prediction = predictions.get(job.job_id)
        if prediction is None:
            missing.append(job.job_id)
            continue
        attached.append(
            replace(
                job,
                predicted_duration_seconds=prediction.duration_seconds,
                predicted_average_power_watts=prediction.average_power_watts,
            )
        )
    if missing:
        preview = ", ".join(str(job_id) for job_id in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"missing predictions for job ids: {preview}{suffix}")
    if not attached:
        raise ValueError("at least one job is required")
    return tuple(attached)
