from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import Enum
import heapq
from itertools import count

from .cluster import Cluster, reject_oversized_jobs
from .models import Job, JobRecord, SimulationResult, seconds


class EventKind(Enum):
    """Everything that can make the simulation clock move.

    A carbon-aware policy needs no bucket-boundary event of its own: it knows
    the signal it decides on, so it asks for a ``TIMER`` at exactly the instant
    it chose instead of being woken at every boundary.
    """

    COMPLETION = "completion"
    RELEASE = "release"
    TIMER = "timer"


class SimulationError(RuntimeError):
    """Raised when the simulation cannot make progress or breaks an invariant."""


class PendingQueue:
    """Jobs that are eligible but not yet running, in ``(release, id)`` order.

    A dict is the whole data structure: insertion order is queue order, and
    removal is O(1) without leaving a hole behind, so there is nothing to
    tombstone and nothing to compact.
    """

    __slots__ = ("_jobs",)

    def __init__(self) -> None:
        self._jobs: dict[object, Job] = {}

    def extend(self, jobs: Iterable[Job]) -> None:
        """Append a batch released at the same instant, ordered by job id."""

        for job in sorted(jobs, key=lambda released: str(released.job_id)):
            if job.job_id in self._jobs:
                raise SimulationError(f"job {job.job_id} was released twice")
            self._jobs[job.job_id] = job

    def remove(self, job: Job) -> None:
        if self._jobs.pop(job.job_id, None) is None:
            raise SimulationError(f"job {job.job_id} is not in the pending queue")

    def __iter__(self) -> Iterator[Job]:
        return iter(self._jobs.values())

    def __len__(self) -> int:
        return len(self._jobs)

    def peek(self) -> Job | None:
        return next(iter(self._jobs.values()), None)


class Simulator:
    """Discrete-event HPC simulator over continuous timestamps.

    There is no fixed tick. The clock jumps between event instants, and every
    event sharing an instant is drained before the scheduler is consulted, so a
    job completing at ``t`` frees nodes that another job can take at the same
    ``t`` without any ordering subtlety between the two events.
    """

    def __init__(
        self,
        jobs: Iterable[Job],
        cluster: Cluster,
        scheduler: "Scheduler",
    ) -> None:
        from .schedulers import Scheduler  # imported late to avoid a cycle

        if not isinstance(scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")

        self._jobs = tuple(jobs)
        if not self._jobs:
            raise ValueError("at least one job is required")
        identifiers = {job.job_id for job in self._jobs}
        if len(identifiers) != len(self._jobs):
            raise ValueError("job ids must be unique")

        self._cluster = cluster
        self._scheduler = scheduler
        reject_oversized_jobs(self._jobs, cluster)

        self._events: list[tuple[datetime, int, EventKind, object]] = []
        self._sequence = count()
        self._now: datetime | None = None
        self._queue = PendingQueue()
        self._running: dict[object, tuple[Job, datetime]] = {}
        self._records: list[JobRecord] = []

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise SimulationError("the simulation clock has not started")
        return self._now

    @property
    def cluster(self) -> Cluster:
        return self._cluster

    @property
    def running(self) -> tuple[tuple[Job, datetime], ...]:
        """The jobs currently executing, paired with the instant they started.

        A backfilling policy needs this to project when nodes come back: it
        combines each entry with its own runtime estimate. The engine exposes
        start times rather than end times on purpose, because the true end is
        exactly the information a scheduler is not allowed to read.
        """

        return tuple(self._running.values())

    def _push(self, when: datetime, kind: EventKind, payload: object = None) -> None:
        heapq.heappush(self._events, (when, next(self._sequence), kind, payload))

    def request_wakeup(self, when: datetime) -> None:
        """Ask to be consulted again at ``when``.

        Trace replay needs this to start a job at a recorded instant, and a
        carbon-aware policy needs it to reconsider a deferred job. A wakeup in
        the past is a policy bug, not a rounding artefact, so it raises.
        """

        if self._now is not None and when < self._now:
            raise SimulationError(
                f"cannot schedule a wakeup at {when.isoformat()}, which precedes "
                f"the current time {self._now.isoformat()}"
            )
        if self._now is not None and when == self._now:
            return  # already being served by the pass in progress
        self._push(when, EventKind.TIMER)

    def run(self) -> SimulationResult:
        for job in self._jobs:
            self._push(job.release_time, EventKind.RELEASE, job)

        first_event_time = self._events[0][0]
        while self._events:
            now = self._events[0][0]
            self._advance_clock_to(now)
            self._drain_events_at(now)
            self._start_selected_jobs(now)

        if self._queue:
            stalled = self._queue.peek()
            raise SimulationError(
                f"{len(self._queue)} job(s) never started; the queue stalled at "
                f"job {stalled.job_id if stalled else '?'}"
            )
        if self._running:
            raise SimulationError("the simulation ended with jobs still running")
        if len(self._records) != len(self._jobs):
            raise SimulationError(
                f"recorded {len(self._records)} executions for {len(self._jobs)} jobs"
            )

        return SimulationResult(
            scheduler_name=self._scheduler.name,
            total_nodes=self._cluster.total_nodes,
            records=tuple(self._records),
            first_event_time=first_event_time,
            last_event_time=self.now,
            busy_node_seconds=self._cluster.busy_node_seconds,
            peak_busy_nodes=self._cluster.peak_busy_nodes,
        )

    def _advance_clock_to(self, now: datetime) -> None:
        if self._now is None:
            self._now = now
            return
        if now < self._now:
            raise SimulationError("event queue produced a timestamp in the past")
        self._cluster.advance_to((now - self._now).total_seconds())
        self._now = now

    def _drain_events_at(self, now: datetime) -> None:
        completions: list[Job] = []
        releases: list[Job] = []

        while self._events and self._events[0][0] == now:
            _, _, kind, payload = heapq.heappop(self._events)
            if kind is EventKind.COMPLETION:
                completions.append(payload)  # type: ignore[arg-type]
            elif kind is EventKind.RELEASE:
                releases.append(payload)  # type: ignore[arg-type]
            # A TIMER only exists to bring us here; the scheduling pass follows.

        # Completions first: nodes freed at `now` must be reusable at `now`.
        for job in completions:
            self._complete(job, now)
        for job in releases:
            self._scheduler.on_release(job, now, self)
        self._queue.extend(releases)

    def _complete(self, job: Job, now: datetime) -> None:
        entry = self._running.pop(job.job_id, None)
        if entry is None:
            raise SimulationError(f"job {job.job_id} completed without running")
        _, start_time = entry
        self._cluster.release(job.nodes_required)
        self._records.append(
            JobRecord(
                job_id=job.job_id,
                nodes_required=job.nodes_required,
                submit_time=job.submit_time,
                release_time=job.release_time,
                start_time=start_time,
                end_time=now,
                trace_start_time=job.trace_start_time,
            )
        )

    def _start_selected_jobs(self, now: datetime) -> None:
        selected = self._scheduler.select(now, self._queue, self._cluster, self)
        for job in selected:
            if job.job_id in self._running:
                raise SimulationError(f"job {job.job_id} was started twice")
            # Allocation raises rather than clamps, so a policy that overcommits
            # fails loudly here instead of corrupting every downstream metric.
            self._cluster.allocate(job.nodes_required)
            self._queue.remove(job)
            self._running[job.job_id] = (job, now)
            self._push(
                now + seconds(job.actual_duration_seconds),
                EventKind.COMPLETION,
                job,
            )
