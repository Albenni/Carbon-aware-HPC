from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite

from carbon_accounting import JobIdentifier, JobPowerProfile


def _aware_utc(timestamp: object, field_name: str) -> datetime:
    if not isinstance(timestamp, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp.astimezone(timezone.utc)


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not a boolean")
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _positive_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number, not a boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a real number") from error
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    if converted <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return converted


def _optional_positive_finite(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, field_name)


@dataclass(frozen=True, slots=True)
class Job:
    """One schedulable unit of work, with its ground truth and its predictions.

    The simulator decides *when* the job runs. ``power`` carries the accounting
    input that :mod:`carbon_accounting` consumes; that package deliberately keeps
    ``JobPowerProfile`` free of scheduling concerns and expects it to be embedded
    here.

    ``actual_duration_seconds`` always governs when nodes are released. The
    optional ``predicted_*`` fields exist so a later phase can let a scheduler
    decide on estimates while the simulated execution stays truthful; a policy
    must read :attr:`scheduling_duration_seconds` rather than the actual value.
    """

    job_id: JobIdentifier
    submit_time: datetime
    release_time: datetime
    nodes_required: int
    actual_duration_seconds: float
    power: JobPowerProfile
    time_limit_seconds: float | None = None
    trace_start_time: datetime | None = None
    predicted_duration_seconds: float | None = None
    predicted_average_power_watts: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.power, JobPowerProfile):
            raise TypeError("power must be a JobPowerProfile")

        submit_time = _aware_utc(self.submit_time, "submit_time")
        release_time = _aware_utc(self.release_time, "release_time")
        if release_time < submit_time:
            raise ValueError(
                f"release_time {release_time.isoformat()} precedes submit_time "
                f"{submit_time.isoformat()}"
            )

        duration = _positive_finite(
            self.actual_duration_seconds,
            "actual_duration_seconds",
        )
        # The accounting profile and the schedule must describe one execution.
        if duration != self.power.duration_seconds:
            raise ValueError(
                "actual_duration_seconds must equal power.duration_seconds: "
                f"{duration:g}s vs {self.power.duration_seconds:g}s"
            )

        trace_start_time: datetime | None = None
        if self.trace_start_time is not None:
            trace_start_time = _aware_utc(self.trace_start_time, "trace_start_time")
            if trace_start_time < release_time:
                raise ValueError(
                    f"trace_start_time {trace_start_time.isoformat()} precedes "
                    f"release_time {release_time.isoformat()}"
                )

        object.__setattr__(self, "submit_time", submit_time)
        object.__setattr__(self, "release_time", release_time)
        object.__setattr__(
            self,
            "nodes_required",
            _positive_int(self.nodes_required, "nodes_required"),
        )
        object.__setattr__(self, "actual_duration_seconds", duration)
        object.__setattr__(
            self,
            "time_limit_seconds",
            _optional_positive_finite(self.time_limit_seconds, "time_limit_seconds"),
        )
        object.__setattr__(self, "trace_start_time", trace_start_time)
        object.__setattr__(
            self,
            "predicted_duration_seconds",
            _optional_positive_finite(
                self.predicted_duration_seconds,
                "predicted_duration_seconds",
            ),
        )
        object.__setattr__(
            self,
            "predicted_average_power_watts",
            _optional_positive_finite(
                self.predicted_average_power_watts,
                "predicted_average_power_watts",
            ),
        )

    @property
    def scheduling_duration_seconds(self) -> float:
        """Duration a scheduler is allowed to use when deciding.

        Falls back to the actual duration only while no prediction exists, which
        is the perfect-information setting of the early phases.
        """

        if self.predicted_duration_seconds is not None:
            return self.predicted_duration_seconds
        return self.actual_duration_seconds

    @property
    def scheduling_average_power_watts(self) -> float:
        """Average power a scheduler is allowed to use when deciding.

        The measured-profile mean is ground truth that only exists after the
        run; it stands in here until a later addition supplies a prediction, exactly as
        :attr:`scheduling_duration_seconds` does for duration.
        """

        if self.predicted_average_power_watts is not None:
            return self.predicted_average_power_watts
        return self.power.average_power_watts

    @property
    def node_seconds(self) -> float:
        return self.nodes_required * self.actual_duration_seconds


@dataclass(frozen=True, slots=True)
class JobRecord:
    """What the simulator observed for one job execution.

    Waiting and turnaround are reported from both reference points because the
    project has not yet fixed the convention: eligibility is the natural release
    instant for a delay budget, while submission is what a user perceives.

    The energy and emission fields stay ``None`` until
    :func:`hpc_sim.emissions.account_schedule` fills them in from a
    carbon-intensity provider; the event loop itself is carbon-agnostic.
    """

    job_id: JobIdentifier
    nodes_required: int
    submit_time: datetime
    release_time: datetime
    start_time: datetime
    end_time: datetime
    trace_start_time: datetime | None = None
    energy_kwh: float | None = None
    emissions_gco2e: float | None = None
    energy_kwh_average_model: float | None = None
    emissions_gco2e_average_model: float | None = None

    @property
    def runtime_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def waiting_seconds(self) -> float:
        """Delay between becoming eligible and starting."""

        return (self.start_time - self.release_time).total_seconds()

    @property
    def waiting_seconds_from_submit(self) -> float:
        return (self.start_time - self.submit_time).total_seconds()

    @property
    def turnaround_seconds(self) -> float:
        return (self.end_time - self.release_time).total_seconds()

    @property
    def turnaround_seconds_from_submit(self) -> float:
        return (self.end_time - self.submit_time).total_seconds()

    @property
    def delay_vs_trace_seconds(self) -> float | None:
        """How much later than the historical schedule this job started."""

        if self.trace_start_time is None:
            return None
        return (self.start_time - self.trace_start_time).total_seconds()

    def with_accounting(
        self,
        *,
        energy_kwh: float,
        emissions_gco2e: float,
        energy_kwh_average_model: float,
        emissions_gco2e_average_model: float,
    ) -> JobRecord:
        return JobRecord(
            job_id=self.job_id,
            nodes_required=self.nodes_required,
            submit_time=self.submit_time,
            release_time=self.release_time,
            start_time=self.start_time,
            end_time=self.end_time,
            trace_start_time=self.trace_start_time,
            energy_kwh=energy_kwh,
            emissions_gco2e=emissions_gco2e,
            energy_kwh_average_model=energy_kwh_average_model,
            emissions_gco2e_average_model=emissions_gco2e_average_model,
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Outcome of one simulated run."""

    scheduler_name: str
    total_nodes: int
    records: tuple[JobRecord, ...]
    first_event_time: datetime
    last_event_time: datetime
    busy_node_seconds: float
    peak_busy_nodes: int

    @property
    def makespan_seconds(self) -> float:
        return (self.last_event_time - self.first_event_time).total_seconds()

    @property
    def schedule_start(self) -> datetime:
        return min(record.start_time for record in self.records)

    @property
    def schedule_end(self) -> datetime:
        return max(record.end_time for record in self.records)

    @property
    def utilisation(self) -> float:
        """Mean fraction of the cluster held by jobs over the makespan."""

        available = self.total_nodes * self.makespan_seconds
        if available <= 0.0:
            return 0.0
        return self.busy_node_seconds / available

    def replace_records(self, records: tuple[JobRecord, ...]) -> SimulationResult:
        return SimulationResult(
            scheduler_name=self.scheduler_name,
            total_nodes=self.total_nodes,
            records=records,
            first_event_time=self.first_event_time,
            last_event_time=self.last_event_time,
            busy_node_seconds=self.busy_node_seconds,
            peak_busy_nodes=self.peak_busy_nodes,
        )


def seconds(value: float) -> timedelta:
    """Build a timedelta without repeating the keyword at every call site."""

    return timedelta(seconds=value)
