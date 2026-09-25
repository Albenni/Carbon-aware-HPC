"""Carbon-aware scheduling with perfect knowledge of the grid signal.

The policy here answers the question the baselines cannot: if a job may be held
back for a bounded amount of time, when should it run? Energy is invariant to
the schedule (§2.3 of the model formalization), so the only lever is *when* the
work meets the grid, and the only cost of pulling it is delay.

:class:`CarbonAwareScheduler` is that policy. Its ``forecast`` switch chooses
between the oracle, which reads the actual future intensity at decision time
and so measures how much carbon is available to save before any forecast error
is introduced (§2.7), and the realistic counterpart, which runs the identical
policy over the forecast the provider had already issued when each job was
released. The difference between those two runs is attributable to forecast
error and to nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil, isfinite
from time import perf_counter
from types import MappingProxyType

from carbon_accounting import energy_from_constant_power
from carbon_intensity import (
    CarbonIntensityProvider,
    CarbonIntensitySample,
    ForecastUnavailableError,
    MissingCarbonIntensityError,
    aware_utc,
    bucket_start,
)

from .engine import PendingQueue, SimulationError, Simulator
from .models import Job, seconds
from .schedulers import EASYBackfillScheduler, RuntimeEstimateSource, _PowerCapMixin


HOUR = timedelta(hours=1)
#: What a candidate start is scored by. ``carbon`` is the greedy minimum;
#: ``carbon-delay`` adds ``delay_price`` gCO2e per hour of additional delay;
#: ``benefit-rate`` takes the deferral saving the most gCO2e per hour waited.
OBJECTIVES = ("carbon", "carbon-delay", "benefit-rate")
CONGESTION_SCOPES = ("start", "interval")


def _non_negative(value: object, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _policy_name(*, forecast: bool, scaled: bool, bounded: bool) -> str:
    """The label a run is recorded under, from the three axes that define it."""

    if scaled:
        name = "carbon-forecast-scaled-delay" if forecast else "carbon-scaled-delay"
    else:
        name = "carbon-aware-forecast" if forecast else "carbon-aware"
    return name + "-bounded" if bounded else name


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

        self._load(
            provider.get_actual_range(coverage_start, coverage_end),
            provider.granularity,
        )

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[CarbonIntensitySample],
        granularity: timedelta,
    ) -> "CarbonSignal":
        """The same integral over an issued forecast rather than the actual series.

        A forecast trajectory is contiguous on the provider's own grid, so
        nothing here distinguishes it from an observed one: windows are scored
        identically and only the values being read change.
        """

        signal = cls.__new__(cls)
        signal._load(tuple(samples), granularity)
        return signal

    def _load(
        self,
        samples: tuple[CarbonIntensitySample, ...],
        granularity: timedelta,
    ) -> None:
        if not samples:
            raise ValueError("a carbon signal needs at least one bucket")
        step_seconds = granularity.total_seconds()
        start = aware_utc(samples[0].timestamp, "series start")

        intensity: list[float] = []
        cumulative = [0.0]
        for index, sample in enumerate(samples):
            if sample.timestamp != start + index * granularity:
                raise ValueError(
                    "a carbon signal needs a gap-free series on a single grid: "
                    f"{sample.timestamp.isoformat()} breaks it"
                )
            intensity.append(sample.intensity_gco2e_per_kwh)
            cumulative.append(cumulative[-1] + intensity[-1] * step_seconds)

        self._start = start
        self._end = start + timedelta(seconds=step_seconds * len(intensity))
        self._step_seconds = step_seconds
        self._intensity = tuple(intensity)
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
    earliest: datetime | None = None,
) -> Iterator[datetime]:
    """Start times the policy is allowed to consider, earliest first.

    The job's own release instant is always a candidate, so a policy can decide
    to start immediately. The rest of the window is walked on the same grid the
    signal itself uses: a finer one cannot resolve anything a piecewise constant
    series distinguishes, and a coarser one is the sensitivity knob for how
    precisely a policy is allowed to aim.

    ``earliest`` moves the first candidate forward without moving the deadline,
    which is what a decision retaken later in the job's life needs: the budget
    is spent from the release instant whoever revisits it.
    """

    deadline = job.release_time + max_delay
    first = job.release_time if earliest is None else max(earliest, job.release_time)
    yield first
    candidate = bucket_start(first, granularity) + granularity
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
    earliest: datetime | None = None,
) -> datetime:
    """The candidate start time with the lowest predicted emissions.

    Ties go to the earliest candidate, so the job is never held back without a
    strict carbon gain. This is :func:`choose_start` with the ``carbon``
    objective.
    """

    candidates = list(
        candidate_start_times(job, max_delay=max_delay, granularity=granularity, earliest=earliest)
    )
    costs = [carbon_cost_gco2e(job, candidate, signal) for candidate in candidates]
    return choose_start(candidates, costs).target


@dataclass(frozen=True, slots=True)
class StartChoice:
    """One decision, with everything the telemetry needs to explain it.

    ``best_rate`` is the largest positive saving per hour of delay any deferred
    candidate offered, in gCO2e per job-hour, whether or not it was taken.
    """

    target: datetime
    cost_now_gco2e: float
    cost_target_gco2e: float
    penalty_gco2e: float
    best_rate: float | None
    reason: str


def choose_start(
    candidates: Sequence[datetime],
    costs: Sequence[float],
    *,
    objective: str = "carbon",
    delay_price: float = 0.0,
    threshold: float = 0.0,
    penalties: Sequence[float] | None = None,
) -> StartChoice:
    """The one scoring rule every decision goes through.

    ``candidates[0]`` is the decision instant ``t`` and is always admissible:
    it means "offer the job to EASY now", not "start now". For a later
    candidate ``s``, ``D = (s - t)`` hours is the delay *added by this
    decision* and ``saving = C(t) - C(s)`` in gCO2e.

    ``carbon`` / ``carbon-delay``
        Minimise ``C(s) + delay_price * D + penalty(s)``. A positive
        ``threshold`` (gCO2e per job-hour) first drops every deferred candidate
        whose saving is not positive or whose ``saving / D`` does not exceed it;
        the penalty never counts as a saving. ``delay_price = threshold = 0``
        is the greedy minimum, bit for bit.
    ``benefit-rate``
        Among deferred candidates with a positive saving and ``saving / D >
        threshold``, take the largest ratio; with none, start now. A penalty in
        grams cannot be added to a ratio, so none is accepted.

    Ties go to the earliest candidate under every objective.
    """

    if not candidates or len(candidates) != len(costs):
        raise ValueError("every candidate needs exactly one cost")
    now = candidates[0]
    hours = [(candidate - now) / HOUR for candidate in candidates]
    rates: list[float | None] = [
        (costs[0] - cost) / delay if delay > 0.0 and costs[0] - cost > 0.0 else None
        for cost, delay in zip(costs, hours)
    ]
    best_rate = max((rate for rate in rates if rate is not None), default=None)

    if objective == "benefit-rate":
        if penalties is not None:
            raise ValueError("benefit-rate cannot take a penalty in gCO2e")
        worthwhile = [
            index for index, rate in enumerate(rates) if rate is not None and rate > threshold
        ]
        index = max(worthwhile, key=rates.__getitem__) if worthwhile else 0
    elif objective in ("carbon", "carbon-delay"):
        allowed = [
            index
            for index, rate in enumerate(rates)
            if index == 0 or threshold == 0.0 or (rate is not None and rate > threshold)
        ]
        if penalties is None:
            index = min(allowed, key=lambda i: costs[i] + delay_price * hours[i])
        else:
            index = min(allowed, key=lambda i: costs[i] + delay_price * hours[i] + penalties[i])
    else:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")

    if index:
        reason = "defer"
    elif best_rate is None:
        reason = "no-saving"
    else:
        reason = "not-worth-delay"
    return StartChoice(
        target=candidates[index],
        cost_now_gco2e=costs[0],
        cost_target_gco2e=costs[index],
        penalty_gco2e=0.0 if penalties is None else penalties[index],
        best_rate=best_rate,
        reason=reason,
    )


class CarbonAwareScheduler(EASYBackfillScheduler):
    """EASY backfilling that holds each job for its cleanest start time.

    When a job becomes eligible the policy scores every candidate start in its
    delay budget and picks the cheapest. Until that instant the job is simply
    not offered to the placement pass: it does not hold the head of the queue,
    so the jobs behind it move up and the machine keeps working. From that
    instant on it is an ordinary EASY candidate, back in its original queue
    position, which is what bounds the harm - a job that yields its place is at
    worst as delayed as the queue it re-enters.

    Because the schedule cannot change the signal, and the signal cannot change
    the job, the choice is fixed at release and never revisited. The policy asks
    the simulator for a wakeup at exactly that instant instead of subscribing to
    every bucket boundary, so the deferral costs one event per deferred job.

    Being withheld rather than refused also keeps a held job out of every
    reservation: EASY's promise is to the head of the *offered* queue, so a
    backfilled job may still be running when a held job comes back.

    A job without an accounting power profile is offered to EASY immediately.
    It participates in queueing and node contention, but the policy neither
    assigns it a carbon cost nor delays it for carbon reasons.

    Four parameters, and nothing else, separate the policies this class covers.

    ``max_delay`` or ``max_delay_fraction``
        The delay budget, either a fixed window or a per-job one worth
        ``scheduling_duration_seconds x max_delay_fraction``, optionally capped
        at ``max_delay_cap``. Exactly one is required. It is a budget on
        voluntary deferral, not a deadline: once a
        target arrives the job competes for nodes like any other and contention
        can still push it later. Zero of either reproduces EASY exactly, which
        is what makes the trade-off sweep start from a known point. The
        proportional form uses a duration prediction when one is present and
        never reads the actual duration by accident.

    ``objective``, ``delay_price_gco2e_per_hour``, ``min_carbon_saving_per_delay``
        How candidates are scored; see :func:`choose_start`. The defaults are
        the greedy carbon minimum.

    ``forecast``
        Where the signal comes from. ``False`` materialises the actual series
        once and reads the future perfectly: the oracle of section 2.7, which
        measures how much carbon is available to save before any forecast error
        is introduced. ``True`` asks the provider, at each release, for the
        forecast already issued at that instant, so every decision consumes
        exactly what a scheduler running at the time could have read and the
        actual series is never touched while deciding - the provider is free to
        refuse it outright. The policy is otherwise identical, so the emissions
        gap between the two runs is attributable to forecast error alone.

    ``reach``
        How far ahead any decision may look, when the signal is finite. Scoring
        a start needs the signal to the end of the job, so a job of duration
        ``d`` may only move within ``reach - d``; past that a forecast archive
        has nothing to offer and the provider rightly refuses. The oracle takes
        the same bound as its forecast counterpart even though it could read as
        far as it likes, otherwise the two runs would differ in delay budget as
        well as in signal. The bound bites hardest under a proportional budget,
        where the jobs earning the largest windows are the ones the horizon can
        least afford.
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        *,
        max_delay: timedelta | None = None,
        max_delay_fraction: float | None = None,
        max_delay_cap: timedelta | None = None,
        reach: timedelta | None = None,
        forecast: bool = False,
        decision_granularity: timedelta | None = None,
        objective: str = "carbon",
        delay_price_gco2e_per_hour: float = 0.0,
        min_carbon_saving_per_delay: float = 0.0,
        runtime_estimate: RuntimeEstimateSource = RuntimeEstimateSource.SCHEDULING,
        backfill_window: int | None = None,
    ) -> None:
        super().__init__(
            runtime_estimate=runtime_estimate,
            backfill_window=backfill_window,
        )
        if (max_delay is None) == (max_delay_fraction is None):
            raise ValueError("give exactly one of max_delay or max_delay_fraction")
        if max_delay is not None and max_delay < timedelta(0):
            raise ValueError("max_delay cannot be negative")
        if max_delay_fraction is not None:
            max_delay_fraction = _non_negative(max_delay_fraction, "max_delay_fraction")
        if max_delay_cap is not None:
            if max_delay_fraction is None:
                raise ValueError("max_delay_cap only bounds a duration-scaled budget")
            if max_delay_cap < timedelta(0):
                raise ValueError("max_delay_cap cannot be negative")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
        delay_price = _non_negative(delay_price_gco2e_per_hour, "delay_price_gco2e_per_hour")
        threshold = _non_negative(min_carbon_saving_per_delay, "min_carbon_saving_per_delay")
        if objective == "carbon" and (delay_price or threshold):
            raise ValueError("the carbon objective takes no delay price or threshold")
        if objective == "benefit-rate" and delay_price:
            raise ValueError("benefit-rate is a ratio; a delay price in grams cannot join it")

        granularity = (
            provider.granularity if decision_granularity is None else decision_granularity
        )
        if granularity <= timedelta(0):
            raise ValueError("decision_granularity must be greater than zero")

        self.name = _policy_name(
            forecast=forecast, scaled=max_delay_fraction is not None, bounded=reach is not None
        )
        if objective != "carbon":
            self.name += "-" + objective
        self._provider = provider
        self._forecast = forecast
        self._max_delay = max_delay
        self._max_delay_fraction = max_delay_fraction
        self._max_delay_cap = max_delay_cap
        self._reach = reach
        self._signal = None if forecast else CarbonSignal(provider)
        self._granularity = granularity
        self._objective = objective
        self._delay_price = delay_price
        self._threshold = threshold
        self._targets: dict[object, datetime] = {}
        self._issue_times: dict[object, datetime] = {}
        # One tuple per decision that set or moved a target; a revisit that
        # confirms the standing target adds nothing to the voluntary delay, so
        # it is counted, not stored.
        self._decisions: list[tuple] = []
        self.decision_count = 0
        self.decision_seconds = 0.0

    @property
    def reads_actual_series(self) -> bool:
        """Whether the policy may materialise the whole actual series up front."""

        return not self._forecast

    @property
    def target_start_times(self) -> Mapping[object, datetime]:
        """Instant each job was held for, for validation and inspection."""

        return MappingProxyType(self._targets)

    @property
    def forecast_issue_times(self) -> Mapping[object, datetime]:
        """Which issued forecast each decision read.

        Empty for an oracle, which reads none, and for jobs decided without
        consulting the signal at all.
        """

        return MappingProxyType(self._issue_times)

    def decision_records(self) -> list[dict[str, object]]:
        """Every decision that set or moved a target, oldest first.

        Costs are the policy's own predictions (gCO2e); the budget columns are
        seconds. Saving, delay and rate follow from the costs and instants.
        """

        fields = (
            "job_id", "decision_time", "forecast_issue_time", "previous_target", "target",
            "deadline", "budget_nominal_s", "budget_effective_s", "cost_now_gco2e",
            "cost_target_gco2e", "congestion_penalty_gco2e", "best_rate", "reason",
        )
        return [dict(zip(fields, record), policy=self.name) for record in self._decisions]

    def _budgets(self, job: Job) -> tuple[timedelta, timedelta]:
        """The nominal budget, and what is left of it inside the reach."""

        if self._max_delay is not None:
            budget = self._max_delay
        else:
            try:
                budget = seconds(
                    job.scheduling_duration_seconds * self._max_delay_fraction
                )
            except OverflowError as error:
                raise ValueError(
                    f"scaled delay for job {job.job_id} is too large"
                ) from error
            if self._max_delay_cap is not None:
                budget = min(budget, self._max_delay_cap)
        if self._reach is None:
            return budget, budget
        runtime = seconds(job.scheduling_duration_seconds)
        return budget, min(budget, max(self._reach - runtime, timedelta(0)))

    def _max_delay_for(self, job: Job) -> timedelta:
        return self._budgets(job)[1]

    def _signal_for(self, job: Job, now: datetime) -> CarbonSignal:
        """The signal this decision is scored against.

        A forecast is asked for what is left of the budget plus the predicted
        run, from the decision instant: a later look never re-requests the
        whole original window.
        """

        if not self._forecast:
            return self._signal
        remaining = max(job.release_time + self._max_delay_for(job) - now, timedelta(0))
        issued = self._provider.get_forecast(
            now, remaining + seconds(job.scheduling_duration_seconds)
        )
        self._note_issue(job, issued.issue_time, now)
        return CarbonSignal.from_samples(issued.samples, self._provider.granularity)

    def _note_issue(self, job: Job, issue: datetime, now: datetime) -> None:
        if issue > now:
            raise SimulationError(
                f"job {job.job_id} was decided at {now.isoformat()} on a forecast "
                f"issued at {issue.isoformat()}"
            )
        self._issue_times[job.job_id] = issue

    def _scored(
        self, job: Job, now: datetime, earliest: datetime | None
    ) -> tuple[list[datetime], list[float]]:
        signal = self._signal_for(job, now)
        candidates = list(
            candidate_start_times(
                job,
                max_delay=self._max_delay_for(job),
                granularity=self._granularity,
                earliest=earliest,
            )
        )
        return candidates, [carbon_cost_gco2e(job, start, signal) for start in candidates]

    def _choose(
        self,
        candidates: list[datetime],
        costs: list[float],
        penalties: list[float] | None = None,
    ) -> StartChoice:
        return choose_start(
            candidates,
            costs,
            objective=self._objective,
            delay_price=self._delay_price,
            threshold=self._threshold,
            penalties=penalties,
        )

    def _decide(
        self,
        job: Job,
        now: datetime,
        simulator: Simulator,
        earliest: datetime | None = None,
        penalize=None,
    ) -> None:
        """Pick and commit this job's target from ``earliest`` (release if None).

        ``penalize(candidates)``, when given, returns a gCO2e penalty per
        candidate that joins the score but never counts as a saving.
        """

        started = perf_counter()
        # An unscored job, or one with no delay budget, has a single admissible
        # start, so the signal is never consulted: that is what keeps a zero
        # budget identical to plain EASY whatever the signal could have said.
        budget = timedelta(0) if job.power is None else self._max_delay_for(job)
        if budget <= timedelta(0):
            self._commit(job, now, job.release_time, simulator, None,
                         "resource-only" if job.power is None else "no-budget")
        elif earliest is not None and job.release_time + budget <= now:
            return  # the budget is spent; there is nothing left to choose from
        else:
            try:
                candidates, costs = self._scored(job, now, earliest)
            except ForecastUnavailableError:
                # Nothing issued covers the window: offer the job to EASY
                # rather than plan on an observation it could not have had.
                self._issue_times.pop(job.job_id, None)
                self._commit(job, now, now, simulator, None, "forecast-unavailable")
            else:
                penalties = None if penalize is None else penalize(candidates)
                choice = self._choose(candidates, costs, penalties)
                self._commit(job, now, choice.target, simulator, choice)
        self.decision_seconds += perf_counter() - started

    def _commit(
        self,
        job: Job,
        now: datetime,
        target: datetime,
        simulator: Simulator,
        choice: StartChoice | None,
        reason: str | None = None,
    ) -> None:
        self.decision_count += 1
        previous = self._targets.get(job.job_id)
        self._targets[job.job_id] = target
        if target > now:
            simulator.request_wakeup(target)
        if previous == target:
            return
        nominal, effective = self._budgets(job) if job.power is not None else (timedelta(0),) * 2
        self._decisions.append((
            job.job_id, now, self._issue_times.get(job.job_id), previous, target,
            job.release_time + effective, nominal.total_seconds(), effective.total_seconds(),
            None if choice is None else choice.cost_now_gco2e,
            None if choice is None else choice.cost_target_gco2e,
            None if choice is None else choice.penalty_gco2e,
            None if choice is None else choice.best_rate,
            reason or choice.reason,
        ))

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        self._decide(job, now, simulator)

    def _ready(
        self,
        queue: PendingQueue,
        now: datetime,
        simulator: Simulator,
    ) -> Iterable[Job]:
        del simulator
        return (job for job in queue if self._targets[job.job_id] <= now)


class PowerCappedCarbonAwareScheduler(_PowerCapMixin, CarbonAwareScheduler):
    """Carbon-aware EASY constrained by an aggregate power budget.

    Jobs are held for the same carbon target as in :class:`CarbonAwareScheduler`.
    Once ready, they start only when both nodes and power headroom are available;
    EASY reservations protect a pivot on both resources. With ``max_delay=0``
    and the same runtime estimate this is exactly
    :class:`~hpc_sim.schedulers.PowerCappedEASYScheduler`.
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        power_cap_watts: float,
        **keywords,
    ) -> None:
        self._set_power_cap(power_cap_watts)
        super().__init__(provider, **keywords)
        self.name = "carbon-power-cap"


class RecedingHorizonCarbonAwareScheduler(CarbonAwareScheduler):
    """Carbon-aware EASY that revisits a held job when the forecast is reissued.

    :class:`CarbonAwareScheduler` decides once, at release, and lives with it.
    That is exactly right for an oracle - the actual series never changes, so a
    second look reads the same numbers - but a forecast run is stuck with the
    trajectory issued at the release instant for the whole delay budget, and the
    provider replaces that trajectory every cadence step with one that sees the
    intervening observations.

    This policy re-runs the identical choice whenever a newer snapshot exists,
    over the part of the budget that is still ahead. Everything that bounds the
    original policy still bounds this one:

    * the deadline stays ``release_time + max_delay``, recomputed from the
      release instant on every pass, so a revisit can only ever spend budget the
      job already had - never extend it;
    * candidates start at the current instant, so nothing is scheduled into the
      past, and ties still go to the earliest, so a job is only held for a
      strict predicted gain;
    * only the snapshot already issued at the decision instant is read, through
      the same provider call the original policy uses.

    With ``forecast=False`` the signal is immutable and every revisit returns
    the previous answer, so the oracle run is identical to its baseline by
    construction rather than by coincidence.

    ``redecide_ready``
        Whether a job whose target has already arrived - offered to EASY, but
        still queueing for nodes - may be pulled back for a cleaner window that
        a later snapshot revealed. ``False`` restricts revisits to jobs still
        being held.
    ``replan_on_events``
        Also revisit whenever the queue gains a job, the running set changes or
        a running job overruns its estimate, in the oracle as well. Boundary
        wakeups then stop once no queued job has budget left.
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        *,
        redecide_ready: bool = True,
        replan_on_events: bool = False,
        **keywords,
    ) -> None:
        super().__init__(provider, **keywords)
        if self._forecast and self._reach is None:
            raise ValueError("a receding-horizon forecast run needs a reach")
        self.name = "receding-" + self.name + ("-replan" if replan_on_events else "")
        self._redecide_ready = redecide_ready
        self._replan = bool(replan_on_events)
        self._decided_issue: datetime | None = None
        # One fetch per decision instant, shared by every job revisited at it:
        # the whole queue reads the same snapshot, and rebuilding the integral
        # per job is what would make a revisit expensive.
        self._cached: tuple[datetime, datetime | None, CarbonSignal | None] | None = None
        self._releases = 0
        self._seen: tuple | None = None
        self.revisit_count = 0

    def _signal_for(self, job: Job, now: datetime) -> CarbonSignal:
        if not self._forecast:
            return self._signal
        if self._cached is None or self._cached[0] != now:
            try:
                issued = self._provider.get_forecast(now, self._reach)
            except ForecastUnavailableError:
                # Near the end of the archive less than a whole reach is left,
                # so there is nothing to share: each job falls back to asking
                # for exactly the horizon it needs, as the baseline always does.
                self._cached = (now, None, None)
            else:
                self._cached = (
                    now,
                    issued.issue_time,
                    CarbonSignal.from_samples(issued.samples, self._provider.granularity),
                )
        if self._cached[2] is None:
            return super()._signal_for(job, now)
        self._note_issue(job, self._cached[1], now)
        return self._cached[2]

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        self._releases += 1
        super().on_release(job, now, simulator)

    def _retarget(self, job: Job, now: datetime, simulator: Simulator) -> None:
        """Re-pick this job's target over the budget it has left, or leave it."""

        if job.power is None:
            return
        if not self._redecide_ready and self._targets[job.job_id] <= now:
            return
        self._decide(job, now, simulator, earliest=now)

    def _deferrable(self, job: Job, now: datetime) -> bool:
        return job.power is not None and job.release_time + self._max_delay_for(job) > now

    def _ready(
        self,
        queue: PendingQueue,
        now: datetime,
        simulator: Simulator,
    ) -> Iterable[Job]:
        if len(queue):
            # Consulted on the decision grid for as long as anything is waiting,
            # so a reissued forecast is acted on at the boundary it arrives at
            # rather than at whatever completion happens to come next.
            if not self._replan or any(self._deferrable(job, now) for job in queue):
                simulator.request_wakeup(bucket_start(now, self._granularity) + self._granularity)
            changed = False
            if self._forecast:
                # One bucket is all a probe needs: the question is which
                # snapshot is current, not what it says.
                try:
                    issue = self._provider.get_forecast(now, self._granularity).issue_time
                except ForecastUnavailableError:
                    issue = self._decided_issue  # nothing newer to reconsider against
                if issue != self._decided_issue:
                    self._decided_issue = issue
                    changed = True
            if self._replan:
                running = simulator.running
                state = (
                    self._releases,
                    tuple(job.job_id for job, _ in running),
                    sum(start + seconds(self._estimate(job)) <= now for job, start in running),
                )
                if state != self._seen:
                    self._seen = state
                    changed = True
            if changed:
                self.revisit_count += 1
                self._revisit(queue, now, simulator)
        return [job for job in queue if self._targets[job.job_id] <= now]

    def _revisit(self, queue: PendingQueue, now: datetime, simulator: Simulator) -> None:
        """Retarget the waiting jobs against the snapshot just issued."""

        for job in tuple(queue):
            self._retarget(job, now, simulator)


class CongestionAwareRecedingScheduler(RecedingHorizonCarbonAwareScheduler):
    """Receding horizon that prices expected contention and rations clean windows.

    The receding policy sends every job to its own cleanest bucket as if the
    machine were empty. This one projects node occupancy over the budget from
    what a scheduler can know at the decision instant - running jobs until their
    *estimated* completion, plus the standing targets of released jobs not yet
    started - and charges a candidate start whose projected free nodes fall
    short of the request::

        congestion_price x energy_kWh x min(1, overflow_nodes / nodes_required)

    on top of its predicted emissions, in the same gCO2e units.

    When a newer snapshot is issued the waiting jobs are retargeted in order of
    predicted carbon opportunity - emissions starting now minus the cheapest
    candidate, under that snapshot - and each chosen target is claimed before
    the next job is scored. Where clean buckets are scarce, the jobs with most
    to lose keep them and the price moves the ones with least. Where capacity is
    plentiful the price never fires and the order is irrelevant.

    The budget is untouched: candidates come from the same deadline,
    ``release_time + max_delay``, and no future actual is read.
    ``congestion_price=0`` reproduces the receding policy exactly.

    Three options extend the coordination, all off by default:

    ``congestion_scope="interval"``
        Charge the overflow share averaged over the whole estimated run instead
        of at the start instant, so congestion that only builds up after the
        start is priced too.
    ``replan_on_events`` (see the receding policy) with ``planning_jobs``
        A new job gets its own claim-free decision at release; the first
        ``planning_jobs`` deferrable jobs in queue order are then coordinated
        whenever anything changes. The others keep their targets as fixed
        claims.
    ``overrun_to_horizon``
        A running job past its estimate holds its nodes to the end of the plan
        rather than for the current instant only.

    This is a penalised heuristic, not a capacity planner: claims may overlap,
    and EASY and the cluster still guarantee real capacity.
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        *,
        congestion_price: float = 1.0,
        congestion_scope: str = "start",
        planning_jobs: int | None = None,
        overrun_to_horizon: bool = False,
        **keywords,
    ) -> None:
        super().__init__(provider, **keywords)
        congestion_price = _non_negative(congestion_price, "congestion_price")
        if congestion_scope not in CONGESTION_SCOPES:
            raise ValueError(f"congestion_scope must be one of {', '.join(CONGESTION_SCOPES)}")
        if planning_jobs is not None and (
            isinstance(planning_jobs, bool) or not isinstance(planning_jobs, int) or planning_jobs <= 0
        ):
            raise ValueError("planning_jobs must be a positive integer")
        if self._objective == "benefit-rate" and congestion_price:
            raise ValueError("benefit-rate is a ratio; a congestion price in grams cannot join it")
        self.name = "congestion-" + self.name + ("-interval" if congestion_scope == "interval" else "")
        self._congestion_price = congestion_price
        self._scope = congestion_scope
        self._planning_jobs = planning_jobs
        self._overrun_to_horizon = bool(overrun_to_horizon)
        # Released and not yet started. Kept here rather than read off the queue
        # because jobs released at the same instant reach on_release before any
        # of them has been queued.
        self._pending: dict[object, Job] = {}

    def _slot(self, when: datetime, now: datetime) -> int:
        """How many points of the grid ``now, next boundary, ...`` precede ``when``."""

        if when <= now:
            return 0
        first = bucket_start(now, self._granularity) + self._granularity
        return 1 + max(0, ceil((when - first) / self._granularity))

    def _horizon(self, job: Job, candidates: list[datetime]) -> datetime:
        """The last instant a penalty for these candidates reads."""

        if self._scope == "start":
            return candidates[-1]
        return candidates[-1] + seconds(self._estimate(job))

    def _occupancy(
        self, now: datetime, horizon: datetime, simulator: Simulator, jobs: Iterable[Job]
    ) -> list[int]:
        use = [0] * (self._slot(horizon, now) + 1)
        claims = []
        for job, started in simulator.running:
            end = started + seconds(self._estimate(job))
            if self._overrun_to_horizon and end <= now:
                end = horizon + self._granularity
            # Otherwise an overrun has no honest end left; it holds its nodes now only.
            claims.append((started, max(end, now + timedelta(seconds=1)), job))
        for job in jobs:
            start = max(self._targets[job.job_id], now)
            claims.append((start, start + seconds(self._estimate(job)), job))
        for start, end, job in claims:
            self._claim(use, start, end, job.nodes_required, now)
        return use

    def _claim(self, use: list[int], start: datetime, end: datetime, nodes: int, now: datetime) -> None:
        for slot in range(self._slot(start, now), min(self._slot(end, now), len(use))):
            use[slot] += nodes

    def _penalties(
        self, job: Job, candidates: list[datetime], now: datetime, use: list[int], total_nodes: int
    ) -> list[float]:
        energy_kwh = energy_from_constant_power(
            job.scheduling_average_power_watts, job.scheduling_duration_seconds
        )
        weight = self._congestion_price * energy_kwh
        nodes = job.nodes_required
        if self._scope == "start":
            return [
                weight * min(1.0, max(0, nodes - (total_nodes - use[self._slot(start, now)])) / nodes)
                for start in candidates
            ]

        # ponytail: claims are snapped to the decision grid, so the calendar is
        # piecewise constant per bucket; exact claim breakpoints if sub-bucket
        # precision ever matters.
        first = (bucket_start(now, self._granularity) + self._granularity - now).total_seconds()
        step = self._granularity.total_seconds()
        shares = [min(1.0, max(0, nodes - (total_nodes - used)) / nodes) for used in use]
        area = [0.0]
        for index, share in enumerate(shares):
            area.append(area[-1] + share * (first if index == 0 else step))

        def integral(offset: float) -> float:
            """Overflow share integrated from ``now`` to ``now + offset`` seconds."""

            slot = 0 if offset < first else min(len(shares) - 1, 1 + int((offset - first) // step))
            opened = 0.0 if slot == 0 else first + (slot - 1) * step
            return area[slot] + shares[slot] * (offset - opened)

        reservation = self._estimate(job)
        penalties = []
        for start in candidates:
            offset = (start - now).total_seconds()
            share = (integral(offset + reservation) - integral(offset)) / reservation
            penalties.append(weight * share)
        return penalties

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        self._pending[job.job_id] = job
        if self._replan or not self._congestion_price:
            # Each new job gets its own decision, free of claims; with events
            # replanned, the batch is coordinated once every release is in.
            super().on_release(job, now, simulator)
            return
        self._releases += 1
        others = [other for other_id, other in self._pending.items() if other_id != job.job_id]
        self._decide(
            job,
            now,
            simulator,
            penalize=lambda candidates: self._penalties(
                job,
                candidates,
                now,
                self._occupancy(now, self._horizon(job, candidates), simulator, others),
                simulator.cluster.total_nodes,
            ),
        )

    def select(self, now, queue, cluster, simulator) -> tuple[Job, ...]:
        started = super().select(now, queue, cluster, simulator)
        for job in started:
            self._pending.pop(job.job_id, None)
        return started

    def _revisit(self, queue: PendingQueue, now: datetime, simulator: Simulator) -> None:
        started = perf_counter()
        fixed: list[Job] = []
        movable: list[tuple[float, Job, list[datetime], list[float]]] = []
        for job in queue:
            if (
                not self._deferrable(job, now)
                or (not self._redecide_ready and self._targets[job.job_id] <= now)
                or (self._planning_jobs is not None and len(movable) >= self._planning_jobs)
            ):
                fixed.append(job)
                continue
            try:
                candidates, costs = self._scored(job, now, now)
            except ForecastUnavailableError:
                self._issue_times.pop(job.job_id, None)
                self._commit(job, now, now, simulator, None, "forecast-unavailable")
                fixed.append(job)
                continue
            movable.append((costs[0] - min(costs), job, candidates, costs))
        if movable:
            # Largest predicted opportunity claims capacity first; the sort is
            # stable, so ties keep queue order: release, then job id.
            movable.sort(key=lambda entry: -entry[0])
            use = None
            if self._congestion_price > 0.0:
                horizon = max(self._horizon(job, candidates) for _, job, candidates, _ in movable)
                use = self._occupancy(now, horizon, simulator, fixed)
            for _, job, candidates, costs in movable:
                penalties = (
                    None
                    if use is None
                    else self._penalties(job, candidates, now, use, simulator.cluster.total_nodes)
                )
                choice = self._choose(candidates, costs, penalties)
                self._commit(job, now, choice.target, simulator, choice)
                if use is not None:
                    start = max(choice.target, now)
                    self._claim(
                        use, start, start + seconds(self._estimate(job)), job.nodes_required, now
                    )
        self.decision_seconds += perf_counter() - started
