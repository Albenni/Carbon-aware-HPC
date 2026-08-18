"""Run one scheduling policy over a PM100 workload and record the outcome."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    Cluster,
    FCFSScheduler,
    Scheduler,
    SimulationResult,
    Simulator,
    TraceReplayScheduler,
    account_schedule,
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

SCHEDULERS: dict[str, type[Scheduler]] = {
    "fcfs": FCFSScheduler,
    "replay": TraceReplayScheduler,
}


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
        choices=sorted(SCHEDULERS),
        default="fcfs",
        help="fcfs = strict first-come first-served; replay = the recorded schedule",
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
    records = result.records
    waits = sorted(record.waiting_seconds for record in records)
    energy_kwh = sum(record.energy_kwh or 0.0 for record in records)
    emissions_g = sum(record.emissions_gco2e or 0.0 for record in records)
    average_model_g = sum(
        record.emissions_gco2e_average_model or 0.0 for record in records
    )

    print(f"scheduler                {result.scheduler_name}")
    print(f"jobs                     {len(records):,}")
    print(f"cluster capacity         {result.total_nodes:,} nodes")
    print(f"peak nodes in use        {result.peak_busy_nodes:,}")
    print(f"schedule span            {result.schedule_start} -> {result.schedule_end}")
    print(f"makespan                 {result.makespan_seconds / 86_400:.2f} days")
    print(f"node utilisation         {result.utilisation:.1%}")
    print()
    print(f"waiting time mean        {fmean(waits):,.1f} s")
    print(f"waiting time median      {median(waits):,.1f} s")
    print(f"waiting time p99         {waits[int(len(waits) * 0.99)]:,.1f} s")
    print(f"waiting time max         {waits[-1]:,.1f} s")

    delays = [
        record.delay_vs_trace_seconds
        for record in records
        if record.delay_vs_trace_seconds is not None
    ]
    if delays:
        print(f"delay vs trace mean      {fmean(delays):,.1f} s")
        print(f"delay vs trace max       {max(delays):,.1f} s")
    print()
    print(f"total energy             {energy_kwh / 1_000:,.2f} MWh")
    print(f"total emissions          {emissions_g / 1e6:,.3f} tCO2e")
    if emissions_g:
        gap = (average_model_g - emissions_g) / emissions_g
        print(f"average-power model      {average_model_g / 1e6:,.3f} tCO2e ({gap:+.2%})")


def main() -> int:
    arguments = build_parser().parse_args()

    jobs = load_jobs(
        arguments.workload,
        limit=arguments.limit,
        released_from=arguments.released_from,
        released_before=arguments.released_before,
        average_power_source=arguments.average_power_source,
    )
    cluster = Cluster(arguments.nodes)
    scheduler = SCHEDULERS[arguments.scheduler]()

    result = Simulator(jobs, cluster, scheduler).run()

    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
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
        print(f"records written          {written.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
