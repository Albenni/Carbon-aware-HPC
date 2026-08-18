from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import heapq
from itertools import count

from .cluster import Cluster
from .engine import PendingQueue, SimulationError, Simulator
from .models import Job


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
