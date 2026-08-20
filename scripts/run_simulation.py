"""Run one scheduling policy over a PM100 workload and record the outcome."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_intensity import CarbonIntensityProvider, TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    EASYBackfillScheduler,
    FCFSScheduler,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
    Scheduler,
    SimulationResult,
    Simulator,
    TraceReplayScheduler,
    account_schedule,
    bounded_slowdown,
    format_metrics,
    schedule_metrics,
)
from hpc_sim.workload import load_jobs


DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "processed" / "pm100_debug_5000.parquet"
DEFAULT_CARBON_CACHE = (
    PROJECT_ROOT
    / "data"
    / "carbon_intensity"
    / "electricity_maps_it_no_04_to_11_2020.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "simulations"

SCHEDULER_NAMES = ("fcfs", "easy", "power-cap", "carbon", "replay")

WATTS_PER_MEGAWATT = 1e6


def build_scheduler(
    arguments: argparse.Namespace,
    provider: CarbonIntensityProvider,
) -> Scheduler:
    """Instantiate the requested policy from the command line arguments."""

    estimate = RuntimeEstimateSource(arguments.runtime_estimate)
    if arguments.scheduler == "fcfs":
        return FCFSScheduler()
    if arguments.scheduler == "replay":
        return TraceReplayScheduler()
    if arguments.scheduler == "easy":
        return EASYBackfillScheduler(runtime_estimate=estimate)
    if arguments.scheduler == "power-cap":
        if arguments.power_cap_mw is None:
            raise SystemExit("--power-cap-mw is required for the power-cap scheduler")
        return PowerCappedEASYScheduler(
            arguments.power_cap_mw * WATTS_PER_MEGAWATT,
            runtime_estimate=estimate,
        )
    if arguments.scheduler == "carbon":
        return CarbonAwareScheduler(
            provider,
            max_delay=timedelta(hours=arguments.max_delay_hours),
            decision_granularity=(
                timedelta(minutes=arguments.decision_granularity_minutes)
                if arguments.decision_granularity_minutes is not None
                else None
            ),
            runtime_estimate=estimate,
        )
    raise SystemExit(f"unknown scheduler {arguments.scheduler}")


def parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a PM100 workload under one scheduling policy.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_NAMES,
        default="fcfs",
        help=(
            "fcfs = strict first-come first-served; easy = FCFS with EASY "
            "backfilling; power-cap = EASY under an aggregate power budget; "
            "carbon = EASY holding each job for its cleanest start within the "
            "delay budget; replay = the recorded schedule"
        ),
    )
    parser.add_argument(
        "--runtime-estimate",
        choices=tuple(source.value for source in RuntimeEstimateSource),
        default=RuntimeEstimateSource.TIME_LIMIT.value,
        help=(
            "runtime a backfilling policy may plan with: time_limit = the "
            "requested walltime (classic EASY); scheduling = the prediction "
            "seamlessly provided by the scheduler (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--power-cap-mw",
        type=float,
        default=None,
        help="aggregate power budget in MW, required by the power-cap scheduler",
    )
    parser.add_argument(
        "--max-delay-hours",
        type=float,
        default=6.0,
        help=(
            "how long the carbon scheduler may hold a job past its release "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--decision-granularity-minutes",
        type=float,
        default=None,
        help=(
            "spacing of the candidate start times the carbon scheduler "
            "considers (default: the carbon-intensity granularity)"
        ),
    )
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_CARBON_CACHE)
    parser.add_argument(
        "--nodes",
        type=int,
        default=PM100_PARTITION_1_NODES,
        help="cluster capacity in nodes (default: %(default)s)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--released-from", type=parse_timestamp, default=None)
    parser.add_argument("--released-before", type=parse_timestamp, default=None)
    parser.add_argument(
        "--average-power-source",
        choices=("weighted", "stored"),
        default="weighted",
        help="duration-weighted profile mean, or the stored arithmetic mean",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="parquet destination for the per-job records (default: auto-named)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="print the summary without writing a record table",
    )
    return parser


def display_path(path: Path) -> str:
    """Shorten a path against the project root when it lies inside it."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def default_output_path(scheduler_name: str, job_count: int, nodes: int) -> Path:
    return DEFAULT_OUTPUT_DIR / (
        f"simulation_{scheduler_name}_{job_count}jobs_{nodes}nodes.parquet"
    )


def write_records(result: SimulationResult, destination: Path) -> Path:
    import pyarrow
    import pyarrow.parquet as parquet

    records = sorted(result.records, key=lambda record: (record.start_time, str(record.job_id)))
    table = pyarrow.table(
        {
            "job_id": [record.job_id for record in records],
            "nodes_required": [record.nodes_required for record in records],
            "submit_time": [record.submit_time for record in records],
            "release_time": [record.release_time for record in records],
            "start_time": [record.start_time for record in records],
            "end_time": [record.end_time for record in records],
            "trace_start_time": [record.trace_start_time for record in records],
            "runtime_s": [record.runtime_seconds for record in records],
            "waiting_s": [record.waiting_seconds for record in records],
            "waiting_from_submit_s": [
                record.waiting_seconds_from_submit for record in records
            ],
            "turnaround_s": [record.turnaround_seconds for record in records],
            "turnaround_from_submit_s": [
                record.turnaround_seconds_from_submit for record in records
            ],
            "delay_vs_trace_s": [record.delay_vs_trace_seconds for record in records],
            "bounded_slowdown": [bounded_slowdown(record) for record in records],
            "energy_kwh": [record.energy_kwh for record in records],
            "emissions_gco2e": [record.emissions_gco2e for record in records],
            "energy_kwh_average_model": [
                record.energy_kwh_average_model for record in records
            ],
            "emissions_gco2e_average_model": [
                record.emissions_gco2e_average_model for record in records
            ],
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, destination)
    return destination


def print_summary(result: SimulationResult) -> None:
    """Report every metric for the run, from the shared scorer."""

    print(format_metrics(schedule_metrics(result)))


def main() -> int:
    arguments = build_parser().parse_args()

    jobs = load_jobs(
        arguments.workload,
        limit=arguments.limit,
        released_from=arguments.released_from,
        released_before=arguments.released_before,
        average_power_source=arguments.average_power_source,
    )
    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
    scheduler = build_scheduler(arguments, provider)

    result = Simulator(jobs, Cluster(arguments.nodes), scheduler).run()
    result = account_schedule(result, jobs, provider)

    print_summary(result)

    if not arguments.no_output:
        destination = arguments.output or default_output_path(
            result.scheduler_name,
            len(result.records),
            result.total_nodes,
        )
        written = write_records(result, destination)
        print()
        print(f"records written          {display_path(written)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
