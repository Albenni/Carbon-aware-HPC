from __future__ import annotations

from collections.abc import Iterable

from .models import Job


# Partition 1 of the PM100 trace touches 880 distinct node ids (20-979) across
# the raw COMPLETED jobs. Observed peak concurrency is lower (787 raw, 774 after
# the power-profile cleaning), so it is a lower bound on the real machine rather
# than a capacity; both remain useful for sensitivity analysis.
PM100_PARTITION_1_NODES = 880


class CapacityError(RuntimeError):
    """Raised when an allocation would break the cluster's node budget."""


class Cluster:
    """A homogeneous pool of nodes, tracked by count rather than identity.

    The model formalization treats the nodes of a partition as equivalent, so a
    job needs *some* ``nodes_required`` free nodes and never a specific set. The
    ``nodes`` id list in the trace stays validation-only.

    Every transition is checked against ``0 <= busy <= total``. A violation
    raises instead of clamping: silently over-committing would invalidate every
    downstream emission figure.
    """

    __slots__ = ("_total_nodes", "_busy_nodes", "_busy_node_seconds", "_peak_busy_nodes")

    def __init__(self, total_nodes: int = PM100_PARTITION_1_NODES) -> None:
        if isinstance(total_nodes, bool) or not isinstance(total_nodes, int):
            raise TypeError("total_nodes must be an integer")
        if total_nodes <= 0:
            raise ValueError("total_nodes must be greater than zero")

        self._total_nodes = total_nodes
        self._busy_nodes = 0
        self._busy_node_seconds = 0.0
        self._peak_busy_nodes = 0

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    @property
    def busy_nodes(self) -> int:
        return self._busy_nodes

    @property
    def free_nodes(self) -> int:
        return self._total_nodes - self._busy_nodes

    @property
    def peak_busy_nodes(self) -> int:
        return self._peak_busy_nodes

    @property
    def busy_node_seconds(self) -> float:
        return self._busy_node_seconds

    def can_fit(self, nodes: int) -> bool:
        return 0 < nodes <= self.free_nodes

    def accommodates(self, job: Job) -> bool:
        """Whether the job could ever run here, ignoring current occupancy."""

        return job.nodes_required <= self._total_nodes

    def allocate(self, nodes: int) -> None:
        if nodes <= 0:
            raise ValueError("allocation must request at least one node")
        if nodes > self.free_nodes:
            raise CapacityError(
                f"cannot allocate {nodes} nodes: {self.free_nodes} free of "
                f"{self._total_nodes}"
            )
        self._busy_nodes += nodes
        self._peak_busy_nodes = max(self._peak_busy_nodes, self._busy_nodes)

    def release(self, nodes: int) -> None:
        if nodes <= 0:
            raise ValueError("release must return at least one node")
        if nodes > self._busy_nodes:
            raise CapacityError(
                f"cannot release {nodes} nodes: only {self._busy_nodes} are busy"
            )
        self._busy_nodes -= nodes

    def advance_to(self, elapsed_seconds: float) -> None:
        """Integrate current occupancy over the time about to elapse."""

        if elapsed_seconds < 0.0:
            raise ValueError("simulation time cannot move backwards")
        self._busy_node_seconds += self._busy_nodes * elapsed_seconds


def reject_oversized_jobs(jobs: Iterable[Job], cluster: Cluster) -> None:
    """Fail fast on jobs the cluster can never run.

    Such a job would sit at the head of a strict FCFS queue forever and stall
    every job behind it, so it is a configuration error rather than a scheduling
    outcome. The largest PM100 partition-1 job needs 256 nodes.
    """

    oversized = [job for job in jobs if not cluster.accommodates(job)]
    if not oversized:
        return

    listed = ", ".join(
        f"{job.job_id} needs {job.nodes_required}" for job in oversized[:5]
    )
    suffix = "" if len(oversized) <= 5 else f", and {len(oversized) - 5} more"
    raise CapacityError(
        f"{len(oversized)} job(s) exceed the cluster's {cluster.total_nodes} "
        f"nodes: {listed}{suffix}"
    )
