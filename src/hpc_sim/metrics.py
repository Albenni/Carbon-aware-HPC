"""Evaluation metrics for a finished simulation.

Everything here is derived from :class:`~hpc_sim.models.SimulationResult`, so a
metric never depends on which policy produced the schedule. That is the point:
the baselines and the carbon-aware policies of later phases are scored by the
same function on the same workload.

Emission and energy fields require the run to have gone through
:func:`hpc_sim.emissions.account_schedule` first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from statistics import fmean, median
from typing import Literal

from .models import JobRecord, SimulationResult


#: Runtimes shorter than this are treated as this long when computing bounded
#: slowdown, so a job lasting a few seconds cannot report a slowdown of
#: thousands and dominate the mean. Ten seconds is the value used throughout
#: the parallel-workload literature.
BOUNDED_SLOWDOWN_THRESHOLD_SECONDS = 10.0

SECONDS_PER_HOUR = 3_600.0
WATTS_PER_KILOWATT = 1_000.0
GRAMS_PER_TONNE = 1e6

#: Waiting and turnaround can be measured from either reference point, and the
#: project has not fixed the convention: eligibility is the natural origin for
#: a delay budget, submission is what a user perceives.
TimeReference = Literal["release", "submit"]


@dataclass(frozen=True, slots=True)
class Distribution:
    """Summary of one per-job quantity.

    The tail matters as much as the mean here: a policy that delays jobs to
    chase clean electricity can improve an average while punishing a minority
    of jobs badly, so p95, p99, and the maximum are reported alongside it.
    """

    mean: float
    median: float
    p95: float
    p99: float
    maximum: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "Distribution":
        if not values:
            raise ValueError("a distribution needs at least one value")
        ordered = sorted(values)
        return cls(
            mean=fmean(ordered),
            median=median(ordered),
            p95=_percentile(ordered, 0.95),
            p99=_percentile(ordered, 0.99),
            maximum=ordered[-1],
        )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Nearest rank percentile over an already sorted sequence."""

    if not ordered:
        raise ValueError("percentile of an empty sequence")
    rank = max(1, min(len(ordered), ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def waiting_seconds(record: JobRecord, reference: TimeReference) -> float:
    if reference == "release":
        return record.waiting_seconds
    if reference == "submit":
        return record.waiting_seconds_from_submit
    raise ValueError("reference must be 'release' or 'submit'")


def turnaround_seconds(record: JobRecord, reference: TimeReference) -> float:
    if reference == "release":
        return record.turnaround_seconds
    if reference == "submit":
        return record.turnaround_seconds_from_submit
    raise ValueError("reference must be 'release' or 'submit'")


def bounded_slowdown(
    record: JobRecord,
    reference: TimeReference = "release",
    threshold_seconds: float = BOUNDED_SLOWDOWN_THRESHOLD_SECONDS,
) -> float:
    """Turnaround relative to runtime, floored so short jobs cannot explode.

    ``max(1, (wait + runtime) / max(runtime, threshold))``. A job that starts
    immediately scores 1; a job that waits as long as it runs scores 2.
    """

    runtime = record.runtime_seconds
    inflated = waiting_seconds(record, reference) + runtime
    return max(1.0, inflated / max(runtime, threshold_seconds))


def peak_power_watts(records: Sequence[JobRecord]) -> float:
    """Highest aggregate power the schedule ever draws, at job-average power.

    Each job is represented by its duration weighted mean power, recovered from
    its accounted energy, so this is a floor on the true peak: the 20 second
    PM100 profile fluctuates inside every job. It is the quantity a power capped
    policy actually controls, which is what makes it worth reporting.
    """

    deltas: list[tuple[datetime, int, float]] = []
    for record in records:
        if record.energy_kwh is None or record.runtime_seconds <= 0.0:
            continue
        watts = record.energy_kwh * SECONDS_PER_HOUR * WATTS_PER_KILOWATT / record.runtime_seconds
        # A release at an instant is applied before an allocation at the same instant.
        deltas.append((record.end_time, 0, -watts))
        deltas.append((record.start_time, 1, watts))
    if not deltas:
        return 0.0

    deltas.sort(key=lambda item: (item[0], item[1]))
    drawn = 0.0
    peak = 0.0
    for _, _, delta in deltas:
        drawn += delta
        peak = max(peak, drawn)
    return peak


@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    """Every metric for one run, in one comparable record."""

    scheduler_name: str
    job_count: int
    reference: TimeReference

    # Carbon and energy
    total_emissions_gco2e: float
    mean_emissions_gco2e: float
    total_emissions_gco2e_average_model: float
    total_energy_kwh: float
    peak_power_watts: float

    # Quality of service
    waiting: Distribution
    turnaround: Distribution
    bounded_slowdown: Distribution
    delay_vs_trace: Distribution | None

    # System
    total_nodes: int
    peak_busy_nodes: int
    utilisation: float
    makespan_seconds: float
    throughput_jobs_per_hour: float

    @property
    def total_emissions_tco2e(self) -> float:
        return self.total_emissions_gco2e / GRAMS_PER_TONNE

    @property
    def total_energy_mwh(self) -> float:
        return self.total_energy_kwh / WATTS_PER_KILOWATT

    @property
    def peak_power_mw(self) -> float:
        return self.peak_power_watts / WATTS_PER_KILOWATT / WATTS_PER_KILOWATT

    @property
    def average_model_gap(self) -> float:
        """Relative error of the constant-average-power representation.

        Positive means the simple model a scheduler plans with overstates the
        emissions the measured profile actually produces.
        """

        if self.total_emissions_gco2e == 0.0:
            return 0.0
        return (
            self.total_emissions_gco2e_average_model - self.total_emissions_gco2e
        ) / self.total_emissions_gco2e

    def as_row(self) -> dict[str, float | str | int]:
        """Flat mapping for a comparison table or a CSV export."""

        row: dict[str, float | str | int] = {
            "scheduler": self.scheduler_name,
            "jobs": self.job_count,
            "total_emissions_tco2e": self.total_emissions_tco2e,
            "mean_emissions_gco2e": self.mean_emissions_gco2e,
            "total_energy_mwh": self.total_energy_mwh,
            "peak_power_mw": self.peak_power_mw,
            "average_model_gap": self.average_model_gap,
            "waiting_mean_s": self.waiting.mean,
            "waiting_median_s": self.waiting.median,
            "waiting_p95_s": self.waiting.p95,
            "waiting_p99_s": self.waiting.p99,
            "waiting_max_s": self.waiting.maximum,
            "turnaround_mean_s": self.turnaround.mean,
            "turnaround_median_s": self.turnaround.median,
            "bounded_slowdown_mean": self.bounded_slowdown.mean,
            "bounded_slowdown_median": self.bounded_slowdown.median,
            "bounded_slowdown_p95": self.bounded_slowdown.p95,
            "bounded_slowdown_max": self.bounded_slowdown.maximum,
            "total_nodes": self.total_nodes,
            "peak_busy_nodes": self.peak_busy_nodes,
            "utilisation": self.utilisation,
            "makespan_days": self.makespan_seconds / 86_400.0,
            "throughput_jobs_per_hour": self.throughput_jobs_per_hour,
        }
        if self.delay_vs_trace is not None:
            row["delay_vs_trace_mean_s"] = self.delay_vs_trace.mean
            row["delay_vs_trace_max_s"] = self.delay_vs_trace.maximum
        return row


def schedule_metrics(
    result: SimulationResult,
    *,
    reference: TimeReference = "release",
    bounded_slowdown_threshold_seconds: float = BOUNDED_SLOWDOWN_THRESHOLD_SECONDS,
) -> ScheduleMetrics:
    """Score one finished, accounted run.

    ``reference`` chooses the origin for waiting, turnaround, and bounded
    slowdown. It defaults to eligibility, the origin the model formalization
    proposes for a delay budget; ``"submit"`` gives the user-perceived figures.
    """

    records = result.records
    if not records:
        raise ValueError("cannot score a run with no records")

    emissions = [record.emissions_gco2e or 0.0 for record in records]
    delays = [
        record.delay_vs_trace_seconds
        for record in records
        if record.delay_vs_trace_seconds is not None
    ]
    makespan = result.makespan_seconds

    return ScheduleMetrics(
        scheduler_name=result.scheduler_name,
        job_count=len(records),
        reference=reference,
        total_emissions_gco2e=sum(emissions),
        mean_emissions_gco2e=fmean(emissions),
        total_emissions_gco2e_average_model=sum(
            record.emissions_gco2e_average_model or 0.0 for record in records
        ),
        total_energy_kwh=sum(record.energy_kwh or 0.0 for record in records),
        peak_power_watts=peak_power_watts(records),
        waiting=Distribution.of(
            [waiting_seconds(record, reference) for record in records]
        ),
        turnaround=Distribution.of(
            [turnaround_seconds(record, reference) for record in records]
        ),
        bounded_slowdown=Distribution.of(
            [
                bounded_slowdown(record, reference, bounded_slowdown_threshold_seconds)
                for record in records
            ]
        ),
        delay_vs_trace=Distribution.of(delays) if delays else None,
        total_nodes=result.total_nodes,
        peak_busy_nodes=result.peak_busy_nodes,
        utilisation=result.utilisation,
        makespan_seconds=makespan,
        throughput_jobs_per_hour=(
            len(records) / (makespan / SECONDS_PER_HOUR) if makespan > 0.0 else 0.0
        ),
    )


def format_metrics(metrics: ScheduleMetrics) -> str:
    """Human-readable block, matching the layout used by the run scripts."""

    lines = [
        f"scheduler                {metrics.scheduler_name}",
        f"jobs                     {metrics.job_count:,}",
        f"cluster capacity         {metrics.total_nodes:,} nodes",
        f"peak nodes in use        {metrics.peak_busy_nodes:,}",
        f"node utilisation         {metrics.utilisation:.1%}",
        f"makespan                 {metrics.makespan_seconds / 86_400:.2f} days",
        f"throughput               {metrics.throughput_jobs_per_hour:,.1f} jobs/h",
        "",
        f"waiting mean ({metrics.reference})    {metrics.waiting.mean:,.1f} s",
        f"waiting median           {metrics.waiting.median:,.1f} s",
        f"waiting p95 / p99        {metrics.waiting.p95:,.1f} / {metrics.waiting.p99:,.1f} s",
        f"waiting max              {metrics.waiting.maximum:,.1f} s",
        f"turnaround mean          {metrics.turnaround.mean:,.1f} s",
        f"bounded slowdown mean    {metrics.bounded_slowdown.mean:,.2f}",
        f"bounded slowdown p95     {metrics.bounded_slowdown.p95:,.2f}",
        f"bounded slowdown max     {metrics.bounded_slowdown.maximum:,.2f}",
    ]
    if metrics.delay_vs_trace is not None:
        lines.append(
            f"delay vs trace mean/max  {metrics.delay_vs_trace.mean:,.1f} / "
            f"{metrics.delay_vs_trace.maximum:,.1f} s"
        )
    lines.extend(
        [
            "",
            f"total energy             {metrics.total_energy_mwh:,.2f} MWh",
            f"peak power               {metrics.peak_power_mw:,.3f} MW",
            f"total emissions          {metrics.total_emissions_tco2e:,.3f} tCO2e",
            f"mean emissions per job   {metrics.mean_emissions_gco2e:,.1f} gCO2e",
            f"average-power model      "
            f"{metrics.total_emissions_gco2e_average_model / GRAMS_PER_TONNE:,.3f} "
            f"tCO2e ({metrics.average_model_gap:+.2%})",
        ]
    )
    return "\n".join(lines)
