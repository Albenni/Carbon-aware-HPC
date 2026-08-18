"""Discrete-event HPC simulator for carbon-aware scheduling experiments."""

from .cluster import PM100_PARTITION_1_NODES, CapacityError, Cluster
from .emissions import (
    CoverageError,
    account_schedule,
    check_coverage,
    total_emissions_gco2e,
    total_energy_kwh,
)
from .engine import EventKind, PendingQueue, SimulationError, Simulator
from .models import Job, JobRecord, SimulationResult
from .schedulers import FCFSScheduler, Scheduler, TraceReplayScheduler

__all__ = [
    "PM100_PARTITION_1_NODES",
    "CapacityError",
    "Cluster",
    "CoverageError",
    "EventKind",
    "FCFSScheduler",
    "Job",
    "JobRecord",
    "PendingQueue",
    "Scheduler",
    "SimulationError",
    "SimulationResult",
    "Simulator",
    "TraceReplayScheduler",
    "account_schedule",
    "check_coverage",
    "total_emissions_gco2e",
    "total_energy_kwh",
]
