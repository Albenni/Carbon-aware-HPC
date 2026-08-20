from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
import heapq
from itertools import count
from math import inf, isfinite
from types import MappingProxyType

from .cluster import Cluster
from .engine import PendingQueue, SimulationError, Simulator
from .models import Job, seconds


class Scheduler(ABC):
    """A placement policy consulted whenever the simulation clock moves.

    The simulator owns the clock, the queue, and the node budget; a scheduler
    only answers "which of these eligible jobs start now?". It may register
    interest in carbon-intensity boundaries and may ask for a wakeup through
    :meth:`Simulator.request_wakeup`.
    """

    name: str = "scheduler"
    wants_carbon_intensity_events: bool = False

    @abstractmethod
    def select(
        self,
        now: datetime,
        queue: PendingQueue,
        cluster: Cluster,
        simulator: Simulator,
    ) -> tuple[Job, ...]:
        """Return the jobs to start at ``now``, in start order.

        Every returned job must be in ``queue`` and must fit in the nodes free
        after the earlier jobs in the same tuple have been placed.
        """

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        """Called as a job becomes eligible, before it enters the queue."""

    def on_carbon_intensity_change(self, now: datetime, simulator: Simulator) -> None:
        """Called at a carbon-intensity bucket boundary, if opted in."""


class FCFSScheduler(Scheduler):
    """First-come, first-served with no backfilling.

    Jobs are considered in ``(release_time, job_id)`` order and the pass stops at
    the first job that does not fit. That halt is what makes the policy strict
    FCFS: a small job further back never overtakes a blocked large one, even
    when idle nodes could hold it. Relaxing exactly this rule, under a
    reservation for the blocked job, is EASY backfilling.
    """

    name = "fcfs"

    def select(
        self,
        now: datetime,
        queue: PendingQueue,
        cluster: Cluster,
        simulator: Simulator,
    ) -> tuple[Job, ...]:
        del now, simulator

        started: list[Job] = []
        free_nodes = cluster.free_nodes
        for job in queue:
            if job.nodes_required > free_nodes:
                break
            started.append(job)
            free_nodes -= job.nodes_required
        return tuple(started)


class TraceReplayScheduler(Scheduler):
    """Reproduces the start times recorded in the trace.

    This is the validation and reference policy rather than a policy under
    study: it re-runs the schedule the real system actually produced, which
    checks the engine against ground truth and provides the historical baseline
    that later carbon-aware runs are compared against.

    Because the real system used backfilling and priorities, replay is not
    FCFS-ordered and jobs start out of queue order.
    """

    name = "trace-replay"

    def __init__(self) -> None:
        # (trace_start_time, tie-break, job): a heap keyed by the recorded start
        # keeps each pass proportional to the jobs actually starting, instead of
        # rescanning a queue that can hold thousands of entries.
        self._due: list[tuple[datetime, int, Job]] = []
        self._sequence = count()

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        if job.trace_start_time is None:
            raise SimulationError(
                f"job {job.job_id} has no trace_start_time to replay"
            )
        heapq.heappush(self._due, (job.trace_start_time, next(self._sequence), job))
        if job.trace_start_time > now:
            simulator.request_wakeup(job.trace_start_time)

    def select(
        self,
        now: datetime,
        queue: PendingQueue,
        cluster: Cluster,
        simulator: Simulator,
    ) -> tuple[Job, ...]:
        del queue, simulator

        started: list[Job] = []
        free_nodes = cluster.free_nodes
        while self._due and self._due[0][0] <= now:
            _, _, job = heapq.heappop(self._due)
            if job.nodes_required > free_nodes:
                # The recorded schedule fitted the real machine. Exceeding the
                # configured capacity means the capacity is wrong, so surface it
                # instead of silently deferring and diverging from the trace.
                raise SimulationError(
                    f"replaying job {job.job_id} at {now.isoformat()} needs "
                    f"{job.nodes_required} nodes but only {free_nodes} are free; "
                    f"the configured capacity of {cluster.total_nodes} nodes is "
                    "smaller than the machine the trace came from"
                )
            started.append(job)
            free_nodes -= job.nodes_required
        return tuple(started)


class RuntimeEstimateSource(str, Enum):
    """Where a policy gets the runtime it plans with.

    A backfilling policy must guess when running jobs will end. Which guess it
    is allowed to use is a statement about the information available at
    decision time, so it is a parameter rather than a hard-coded choice.
    """

    #: Classic EASY, and the only estimate
    #: genuinely available at submission time in the PM100 trace.
    TIME_LIMIT = "time_limit"
    #: :attr:`Job.scheduling_duration_seconds` — a prediction once one
    #: exists, and the actual duration until then. This is
    #: information setting, and the source that puts the baselines on exactly
    #: the same footing as the carbon-aware policies.
    SCHEDULING = "scheduling"


def estimated_runtime_seconds(job: Job, source: RuntimeEstimateSource) -> float:
    """Runtime a policy may plan with, never the actual duration by accident.

    ``TIME_LIMIT`` falls back to the scheduling duration for a job with no
    recorded limit, so a synthetic workload does not have to invent one.
    """

    if source is RuntimeEstimateSource.TIME_LIMIT and job.time_limit_seconds is not None:
        return job.time_limit_seconds
    return job.scheduling_duration_seconds


class EASYBackfillScheduler(Scheduler):
    """FCFS with EASY backfilling: one reservation, for the blocked head job.

    Each pass starts the FCFS prefix that fits. The first job that does not fit
    becomes the *pivot* and is given a reservation: the earliest instant at
    which enough nodes are projected to be free, computed from the runtime
    estimates of the jobs that hold them. Jobs further back may then overtake
    the pivot, but only when doing so provably cannot push that reservation
    later — the job either ends before the reservation, or takes only nodes the
    pivot does not need at it.

    That single reservation is what separates EASY from unbounded backfilling:
    the head of the queue cannot starve, because it holds a promise that every
    subsequent decision has to respect.

    The promise is only as good as the estimates behind it. With
    :attr:`RuntimeEstimateSource.TIME_LIMIT`, PM100 users overshoot heavily
    (median actual runtime is 2.5% of the requested walltime), so reservations
    sit far in the future and are usually beaten comfortably. With
    :attr:`RuntimeEstimateSource.SCHEDULING` and no predictions loaded the
    estimates are exact, and the reservation becomes a genuine upper bound on
    the pivot's start.

    No wakeup is requested: the reservation is recomputed from scratch on every
    pass, and the engine already consults the scheduler at every completion,
    which is the only event that can bring a reservation forward.
    """

    name = "easy"

    def __init__(
        self,
        *,
        runtime_estimate: RuntimeEstimateSource = RuntimeEstimateSource.TIME_LIMIT,
        backfill_window: int | None = None,
    ) -> None:
        if not isinstance(runtime_estimate, RuntimeEstimateSource):
            runtime_estimate = RuntimeEstimateSource(runtime_estimate)
        if backfill_window is not None and backfill_window <= 0:
            raise ValueError("backfill_window must be greater than zero")

        self._runtime_estimate = runtime_estimate
        self._backfill_window = backfill_window
        # The first reservation each pivot received. Later passes can only pull
        # a reservation earlier, so keeping the first one gives the strongest
        # claim to check: with exact estimates, no job starts after it.
        self._first_reservations: dict[object, datetime] = {}
        self._backfilled: set[object] = set()

    @property
    def runtime_estimate(self) -> RuntimeEstimateSource:
        return self._runtime_estimate

    @property
    def first_reservations(self) -> Mapping[object, datetime]:
        """Reservation each pivot was first promised, for validation."""

        return MappingProxyType(self._first_reservations)

    @property
    def backfilled_job_ids(self) -> frozenset[object]:
        """Jobs that started by overtaking the pivot."""

        return frozenset(self._backfilled)

    def _estimate(self, job: Job) -> float:
        return estimated_runtime_seconds(job, self._runtime_estimate)

    def select(
        self,
        now: datetime,
        queue: PendingQueue,
        cluster: Cluster,
        simulator: Simulator,
    ) -> tuple[Job, ...]:
        free_nodes = cluster.free_nodes
        headroom = self._power_headroom(now, simulator)

        started: list[Job] = []
        pivot: Job | None = None
        candidates: list[Job] = []

        for job in self._ready(queue, now, simulator):
            if pivot is None:
                if self._admits(job, free_nodes, headroom):
                    started.append(job)
                    free_nodes -= job.nodes_required
                    headroom -= job.scheduling_average_power_watts
                    continue
                pivot = job
                continue
            candidates.append(job)
            if self._backfill_window is not None:
                if len(candidates) >= self._backfill_window:
                    break

        if pivot is None:
            return tuple(started)

        reservation, nodes_at, power_at = self._reserve(
            now, pivot, free_nodes, headroom, started, simulator
        )
        self._first_reservations.setdefault(pivot.job_id, reservation)

        # "Extra" capacity: what is free at the reservation that the pivot will
        # not claim. A backfilled job may hold that past the reservation;
        # anything else must be finished before it. Both budgets are tracked,
        # because a pivot blocked on power is starved just as effectively by a
        # job that takes its power as by one that takes its nodes.
        extra_nodes = max(0, nodes_at - pivot.nodes_required)
        extra_power = max(0.0, power_at - pivot.scheduling_average_power_watts)
        if reservation <= now:
            # The projection says the pivot should already fit, so it disagrees
            # with the cluster. Rather than backfill against a reservation that
            # cannot be trusted, fall back to strict FCFS for this pass.
            extra_nodes = 0
            extra_power = 0.0

        for job in candidates:
            if not self._admits(job, free_nodes, headroom):
                continue
            ends_before_reservation = now + seconds(self._estimate(job)) <= reservation
            if not ends_before_reservation:
                if job.nodes_required > extra_nodes:
                    continue
                if job.scheduling_average_power_watts > extra_power:
                    continue
            started.append(job)
            self._backfilled.add(job.job_id)
            free_nodes -= job.nodes_required
            headroom -= job.scheduling_average_power_watts
            if not ends_before_reservation:
                extra_nodes -= job.nodes_required
                extra_power -= job.scheduling_average_power_watts

        return tuple(started)

    def _ready(
        self,
        queue: PendingQueue,
        now: datetime,
        simulator: Simulator,
    ) -> Iterable[Job]:
        """Queue order restricted to the jobs this policy is willing to run.

        Plain EASY offers the whole queue. A policy that holds a job back for
        reasons of its own withholds it here instead of refusing it in
        :meth:`_admits`, because a withheld job must not become the pivot: it is
        waiting by choice, so it has nothing to be protected from.
        """

        del now, simulator
        return queue

    def _admits(self, job: Job, free_nodes: int, headroom_watts: float) -> bool:
        del headroom_watts  # nodes are the only budget for plain EASY
        return job.nodes_required <= free_nodes

    def _power_headroom(self, now: datetime, simulator: Simulator) -> float:
        del now, simulator
        return inf

    def _reserve(
        self,
        now: datetime,
        pivot: Job,
        free_nodes: int,
        headroom_watts: float,
        starting: list[Job],
        simulator: Simulator,
    ) -> tuple[datetime, int, float]:
        """Earliest projected instant the pivot fits, and the capacity free then.

        The projection covers the jobs already running and the ones this pass is
        about to start, each returning its nodes *and* its power at
        ``start + estimate``. An estimate already exhausted is clamped to
        ``now``: the job has overrun, and there is no honest guess left for when
        it will end.

        Power is carried through even for plain EASY, where the headroom is
        infinite and the power condition is therefore vacuous. That keeps one
        reservation rule for both policies instead of two that could drift.
        """

        releases: list[tuple[datetime, int, float]] = [
            (
                max(now, start_time + seconds(self._estimate(job))),
                job.nodes_required,
                job.scheduling_average_power_watts,
            )
            for job, start_time in simulator.running
        ]
        releases.extend(
            (
                now + seconds(self._estimate(job)),
                job.nodes_required,
                job.scheduling_average_power_watts,
            )
            for job in starting
        )
        releases.sort(key=lambda release: release[0])

        needed_power = pivot.scheduling_average_power_watts
        nodes = free_nodes
        power = headroom_watts
        for index, (when, released_nodes, released_power) in enumerate(releases):
            nodes += released_nodes
            power += released_power
            if nodes < pivot.nodes_required or power < needed_power:
                continue
            # Take every job releasing at the same instant, so the reported
            # capacity matches what the pivot would actually see there.
            for later_when, later_nodes, later_power in releases[index + 1 :]:
                if later_when != when:
                    break
                nodes += later_nodes
                power += later_power
            return when, nodes, power

        raise SimulationError(
            f"job {pivot.job_id} needs {pivot.nodes_required} nodes and "
            f"{needed_power:,.0f} W, which the schedule never frees at once; "
            f"capacity is {simulator.cluster.total_nodes} nodes"
        )


class PowerCappedEASYScheduler(EASYBackfillScheduler):
    """EASY under an aggregate power budget: the energy-aware baseline.

    In this model, shifting a job changes neither its duration nor its power, so
    total energy is invariant to the schedule (§2.3 of the model
    formalization) and no policy can reduce it. What a power-aware policy can
    change is *when* power is drawn, so the baseline here is the one used in
    practice on a capped machine: never let the sum of the running jobs'
    average power exceed ``power_cap_watts``.

    This is the contrast that makes the thesis question concrete. The cap moves
    jobs in time for a power reason while staying blind to the grid signal, so
    it lowers the power peak and leaves emissions essentially untouched —
    whereas a carbon-aware policy moves jobs for the opposite reason.

    The cap constrains starting decisions only; the pivot's reservation is
    still computed on nodes alone, which keeps it a lower bound on the instant
    the pivot can really start. Power is taken from
    :attr:`Job.scheduling_average_power_watts`, the same prediction seam the
    duration estimate uses, so the policy never reads a measured profile.
    """

    name = "power-cap"

    def __init__(
        self,
        power_cap_watts: float,
        *,
        runtime_estimate: RuntimeEstimateSource = RuntimeEstimateSource.TIME_LIMIT,
        backfill_window: int | None = None,
    ) -> None:
        super().__init__(
            runtime_estimate=runtime_estimate,
            backfill_window=backfill_window,
        )
        cap = float(power_cap_watts)
        if not isfinite(cap) or cap <= 0.0:
            raise ValueError("power_cap_watts must be a positive, finite number")
        self._power_cap_watts = cap

    @property
    def power_cap_watts(self) -> float:
        return self._power_cap_watts

    def on_release(self, job: Job, now: datetime, simulator: Simulator) -> None:
        # A job drawing more than the whole budget could never start, and would
        # stall the queue forever. That is a configuration error, so say so now
        # rather than through an exhausted event loop much later.
        if job.scheduling_average_power_watts > self._power_cap_watts:
            raise SimulationError(
                f"job {job.job_id} draws "
                f"{job.scheduling_average_power_watts:,.0f} W on average, above "
                f"the {self._power_cap_watts:,.0f} W cap, so it can never start"
            )

    def _admits(self, job: Job, free_nodes: int, headroom_watts: float) -> bool:
        if job.nodes_required > free_nodes:
            return False
        return job.scheduling_average_power_watts <= headroom_watts

    def _power_headroom(self, now: datetime, simulator: Simulator) -> float:
        del now
        drawn = sum(
            job.scheduling_average_power_watts for job, _ in simulator.running
        )
        return self._power_cap_watts - drawn
