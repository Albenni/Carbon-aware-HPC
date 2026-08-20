"""Carbon-aware scheduling with perfect knowledge of the grid signal.

The policy here answers the question the baselines cannot: if a job may be held
back for a bounded amount of time, when should it run? Energy is invariant to
the schedule (§2.3 of the model formalization), so the only lever is *when* the
work meets the grid, and the only cost of pulling it is delay.

Everything in this module assumes the actual future carbon intensity is
readable at decision time. That makes it the benchmark of
§2.7: it measures how much carbon is available
to save before any forecast error is introduced. Swapping
:class:`CarbonSignal` for one built on issued forecasts is the only change
needed to move to the realistic setting.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta
from types import MappingProxyType

from carbon_accounting import energy_from_constant_power
from carbon_intensity import (
    CarbonIntensityProvider,
    MissingCarbonIntensityError,
    aware_utc,
    bucket_start,
)

from .engine import PendingQueue, Simulator
from .models import Job, seconds
from .schedulers import EASYBackfillScheduler, RuntimeEstimateSource


class CarbonSignal:
    """Random-access integral of an actual carbon-intensity series.

    A greedy policy scores many candidate start times per job, and the windows
    overlap heavily, so integrating bucket by bucket every time is wasteful.
    The series is materialised once and kept as a cumulative integral, which
    turns any window into two lookups regardless of how long the job runs.

    Buckets are piecewise constant, exactly as the provider defines them, so
    this is the same signal the accounting sees and not a smoothed version of
    it. A window reaching outside the available series raises rather than being
    clamped: a policy that plans on invented data would report a saving that
    the evaluation could never confirm.
    """

    __slots__ = ("_start", "_end", "_step_seconds", "_intensity", "_cumulative")

    def __init__(self, provider: CarbonIntensityProvider) -> None:
        coverage_start = getattr(provider, "coverage_start", None)
        coverage_end = getattr(provider, "coverage_end", None)
        if coverage_start is None or coverage_end is None:
            raise TypeError(
                "a carbon-aware policy needs a provider that declares its "
                "coverage_start and coverage_end"
            )

        samples = provider.get_actual_range(coverage_start, coverage_end)
        step_seconds = provider.granularity.total_seconds()
        intensity = tuple(sample.intensity_gco2e_per_kwh for sample in samples)

        cumulative = [0.0]
        for value in intensity:
            cumulative.append(cumulative[-1] + value * step_seconds)

        self._start = aware_utc(coverage_start, "coverage_start")
        self._end = self._start + timedelta(seconds=step_seconds * len(intensity))
        self._step_seconds = step_seconds
        self._intensity = intensity
        self._cumulative = tuple(cumulative)

    @property
    def coverage_start(self) -> datetime:
        return self._start

    @property
    def coverage_end(self) -> datetime:
        return self._end

    def _integral_to(self, when: datetime) -> float:
        """Integral of gCO2e/kWh over time, from coverage start to ``when``."""

        offset = (when - self._start).total_seconds()
        index = min(int(offset // self._step_seconds), len(self._intensity) - 1)
        remainder = offset - index * self._step_seconds
        return self._cumulative[index] + remainder * self._intensity[index]

    def mean_intensity(self, start: datetime, end: datetime) -> float:
        """Time-weighted mean gCO2e/kWh over the half-open window."""

        start_utc = aware_utc(start, "start")
        end_utc = aware_utc(end, "end")
        if end_utc <= start_utc:
            raise ValueError("end must be later than start")
        if start_utc < self._start or end_utc > self._end:
            raise MissingCarbonIntensityError(
                f"window {start_utc.isoformat()} to {end_utc.isoformat()} leaves "
                f"the series covering {self._start.isoformat()} to "
                f"{self._end.isoformat()}"
            )

        span = (end_utc - start_utc).total_seconds()
        return (self._integral_to(end_utc) - self._integral_to(start_utc)) / span


def candidate_start_times(
    job: Job,
    *,
    max_delay: timedelta,
    granularity: timedelta,
) -> Iterator[datetime]:
    """Start times the policy is allowed to consider, earliest first.

    The job's own release instant is always a candidate, so a policy can decide
    to start immediately. The rest of the window is walked on the same grid the
    signal itself uses: a finer one cannot resolve anything a piecewise constant
    series distinguishes, and a coarser one is the sensitivity knob for how
    precisely a policy is allowed to aim.
    """

    deadline = job.release_time + max_delay
    yield job.release_time
    candidate = bucket_start(job.release_time, granularity) + granularity
    while candidate <= deadline:
        yield candidate
        candidate += granularity


def carbon_cost_gco2e(job: Job, start_time: datetime, signal: CarbonSignal) -> float:
    """Emissions the policy predicts for running ``job`` from ``start_time``.

    Built from the scheduling seam — the average power and duration a policy is
    allowed to read — so it is the constant-average-power model of §2.4. The
    evaluation still integrates the measured profile, which leaves a small
    modelling gap that :attr:`~hpc_sim.metrics.ScheduleMetrics.average_model_gap`
    reports on every run.
    """

    duration = job.scheduling_duration_seconds
    energy_kwh = energy_from_constant_power(
        job.scheduling_average_power_watts,
        duration,
    )
    return energy_kwh * signal.mean_intensity(start_time, start_time + seconds(duration))


def cheapest_start_time(
    job: Job,
    signal: CarbonSignal,
    *,
    max_delay: timedelta,
    granularity: timedelta,
) -> datetime:
    """The candidate start time with the lowest predicted emissions.

    Ties go to the earliest candidate, so the job is never held back without a
    strict carbon gain.
    """

    return min(
        candidate_start_times(job, max_delay=max_delay, granularity=granularity),
        key=lambda candidate: carbon_cost_gco2e(job, candidate, signal),
    )


class CarbonAwareScheduler(EASYBackfillScheduler):
    """EASY backfilling that holds each job for its cleanest start time.

    When a job becomes eligible the policy scores every candidate start in
    ``[release, release + max_delay]`` and picks the cheapest. Until that
    instant the job is simply not offered to the placement pass: it does not
    hold the head of the queue, so the jobs behind it move up and the machine
    keeps working. From that instant on it is an ordinary EASY candidate, back
    in its original queue position, which is what bounds the harm — a job that
    yields its place is at worst as delayed as the queue it re-enters.

    Because the schedule cannot change the signal, and the signal cannot change
    the job, the choice is fixed at release and never revisited. The policy asks
    the simulator for a wakeup at exactly that instant instead of subscribing to
    every bucket boundary, so the deferral costs one event per deferred job.

    Being withheld rather than refused also keeps a held job out of every
    reservation: EASY's promise is to the head of the *offered* queue, so a
    backfilled job may still be running when a held job comes back.

    ``max_delay`` is a budget on voluntary deferral, not a deadline. Once its
    target arrives, a job competes for nodes like any other and contention can
    still push it later; the delay attributable to the carbon decision is the
    part this parameter bounds. ``max_delay=0`` reproduces EASY exactly, which
    is what makes the trade-off sweep start from a known point.
    """

    name = "carbon-aware"

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        *,
        max_delay: timedelta,
        decision_granularity: timedelta | None = None,
        runtime_estimate: RuntimeEstimateSource = RuntimeEstimateSource.SCHEDULING,
        backfill_window: int | None = None,
    ) -> None:
        super().__init__(
            runtime_estimate=runtime_estimate,
            backfill_window=backfill_window,
        )
        if not isinstance(max_delay, timedelta):
            raise TypeError("max_delay must be a timedelta")
        if max_delay < timedelta(0):
            raise ValueError("max_delay cannot be negative")
        granularity = (
            provider.granularity if decision_granularity is None else decision_granularity
        )
        if not isinstance(granularity, timedelta):
            raise TypeError("decision_granularity must be a timedelta")
        if granularity <= timedelta(0):
            raise ValueError("decision_granularity must be greater than zero")

        self._signal = CarbonSignal(provider)
        self._max_delay = max_delay
        self._granularity = granularity
        self._targets: dict[object, datetime] = {}

    @property
    def max_delay(self) -> timedelta:
        return self._max_delay

    @property
    def target_start_times(self) -> Mapping[object, datetime]:
        """Instant each job was held for, for validation and inspection."""

        return MappingProxyType(self._targets)

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        target = cheapest_start_time(
            job,
            self._signal,
            max_delay=self._max_delay,
            granularity=self._granularity,
        )
        self._targets[job.job_id] = target
        if target > now:
            simulator.request_wakeup(target)

    def _ready(
        self,
        queue: PendingQueue,
        now: datetime,
        simulator: Simulator,
    ) -> Iterable[Job]:
        del simulator
        return (job for job in queue if self._targets[job.job_id] <= now)

