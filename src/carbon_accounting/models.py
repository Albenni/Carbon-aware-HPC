from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import TypeAlias


JobIdentifier: TypeAlias = int | str


class PowerModel(str, Enum):
    """Power representation used to account for a job."""

    AVERAGE = "average"
    MEASURED = "measured"


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number, not a boolean")

    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a real number") from error

    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class JobPowerProfile:
    """The power and duration data needed to account for one job.

    This is intentionally not the simulator's future ``Job`` model. It is a
    small accounting input that can later be embedded in, or constructed from,
    a scheduler job.

    ``average_power_watts`` and every measured sample represent whole-job
    power. For PM100, the node trace is already aggregated over all allocated
    nodes and both sockets.
    """

    duration_seconds: float
    average_power_watts: float
    power_profile_watts: tuple[float, ...] | None = None
    sample_interval_seconds: float = 20.0
    job_id: JobIdentifier | None = None

    def __post_init__(self) -> None:
        duration = _finite_float(self.duration_seconds, "duration_seconds")
        if duration <= 0.0:
            raise ValueError("duration_seconds must be greater than zero")

        average_power = _finite_float(
            self.average_power_watts,
            "average_power_watts",
        )
        if average_power < 0.0:
            raise ValueError("average_power_watts cannot be negative")

        sample_interval = _finite_float(
            self.sample_interval_seconds,
            "sample_interval_seconds",
        )
        if sample_interval <= 0.0:
            raise ValueError("sample_interval_seconds must be greater than zero")

        measured_profile: tuple[float, ...] | None = None
        if self.power_profile_watts is not None:
            if isinstance(self.power_profile_watts, (str, bytes, bytearray)):
                raise TypeError(
                    "power_profile_watts must be an iterable of real numbers, "
                    "not text or bytes"
                )
            try:
                measured_profile = tuple(
                    _finite_float(sample, f"power_profile_watts[{index}]")
                    for index, sample in enumerate(self.power_profile_watts)
                )
            except TypeError as error:
                raise TypeError(
                    "power_profile_watts must be an iterable of real numbers"
                ) from error

            if not measured_profile:
                raise ValueError("power_profile_watts cannot be empty")
            if any(sample < 0.0 for sample in measured_profile):
                raise ValueError("power_profile_watts cannot contain negative values")

        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "average_power_watts", average_power)
        object.__setattr__(self, "sample_interval_seconds", sample_interval)
        object.__setattr__(self, "power_profile_watts", measured_profile)


@dataclass(frozen=True, slots=True)
class AccountingResult:
    """Energy and operational emissions calculated for one job execution."""

    job_id: JobIdentifier | None
    power_model: PowerModel
    start_time: datetime
    end_time: datetime
    energy_kwh: float
    emissions_gco2: float
