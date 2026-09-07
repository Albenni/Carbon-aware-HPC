"""Discrete-event HPC simulator for carbon-aware scheduling experiments."""

from .carbon_aware import (
    CarbonAwareScheduler,
    CarbonSignal,
    PowerCappedCarbonAwareScheduler,
    carbon_cost_gco2e,
    cheapest_start_time,
)
from .cluster import PM100_PARTITION_1_NODES, CapacityError, Cluster
from .emissions import (
    CoverageError,
    account_schedule,
    total_emissions_gco2e,
    total_energy_kwh,
)
from .engine import SimulationError, Simulator
from .metrics import (
    BOUNDED_SLOWDOWN_THRESHOLD_SECONDS,
    Distribution,
    ScheduleMetrics,
    bounded_slowdown,
    format_metrics,
    peak_power_watts,
    schedule_metrics,
)
from .models import Job, SimulationResult
from .schedulers import (
    EASYBackfillScheduler,
    FCFSScheduler,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
    Scheduler,
    TraceReplayScheduler,
    estimated_runtime_seconds,
)

__all__ = [
    "BOUNDED_SLOWDOWN_THRESHOLD_SECONDS",
    "PM100_PARTITION_1_NODES",
    "CapacityError",
    "CarbonAwareScheduler",
    "CarbonSignal",
    "Cluster",
    "CoverageError",
    "Distribution",
    "EASYBackfillScheduler",
    "FCFSScheduler",
    "Job",
    "PowerCappedCarbonAwareScheduler",
    "PowerCappedEASYScheduler",
    "RuntimeEstimateSource",
    "ScheduleMetrics",
    "Scheduler",
    "SimulationError",
    "SimulationResult",
    "Simulator",
    "TraceReplayScheduler",
    "account_schedule",
    "bounded_slowdown",
    "carbon_cost_gco2e",
    "cheapest_start_time",
    "estimated_runtime_seconds",
    "format_metrics",
    "peak_power_watts",
    "schedule_metrics",
    "total_emissions_gco2e",
    "total_energy_kwh",
]
