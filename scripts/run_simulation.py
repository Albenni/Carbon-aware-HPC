"""Run one scheduling policy over a PM100 workload and record the outcome."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path


from common import (
    DEFAULT_CARBON_CACHE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKLOAD,
    display_path,
    parse_timestamp,
)

from carbon_intensity import CarbonIntensityProvider, TimeSeriesCarbonIntensityProvider
from carbon_intensity.scheduling_impact import archive_reach
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    EASYBackfillScheduler,
    FCFSScheduler,
    PowerCappedCarbonAwareScheduler,
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


#: Which of the three carbon-aware axes each policy name turns on. A forecast
#: policy also takes the archive's reach as its bound, which it needs anyway.
CARBON_POLICIES = {
    "carbon": {},
    "carbon-scaled-delay": {"scaled": True},
    "carbon-power-cap": {"capped": True},
    "carbon-forecast": {"forecast": True},
    "carbon-forecast-scaled-delay": {"forecast": True, "scaled": True},
}


SCHEDULER_NAMES = ("fcfs", "easy", "power-cap", *CARBON_POLICIES, "replay")

WATTS_PER_MEGAWATT = 1e6


def load_forecast_archive(arguments: argparse.Namespace) -> ForecastArchive:
    """Read the archive a forecast policy replays.

    The archive also bounds the delay budget: scoring a start time needs the
    signal to the end of the job, and a 24-hour trajectory issued hourly
    guarantees only 23 of them to a decision taken anywhere in the issue
    interval. Accounting stays on the actual provider either way.
    """

    if arguments.forecast_archive is None:
        raise SystemExit(f"--forecast-archive is required for {arguments.scheduler}")
    return ForecastArchive.load(arguments.forecast_archive)


def build_scheduler(
    arguments: argparse.Namespace,
    provider: CarbonIntensityProvider,
) -> Scheduler:
    """Instantiate the requested policy from the command line arguments."""

    estimate_name = arguments.runtime_estimate
    if estimate_name is None:
        estimate_name = (
            RuntimeEstimateSource.SCHEDULING.value
            if arguments.job_predictions is not None
            else RuntimeEstimateSource.TIME_LIMIT.value
        )
    estimate = RuntimeEstimateSource(estimate_name)
    if arguments.scheduler == "fcfs":
        return FCFSScheduler()
    if arguments.scheduler == "replay":
        return TraceReplayScheduler()
    if arguments.scheduler == "easy":
        return EASYBackfillScheduler(runtime_estimate=estimate)
    if arguments.scheduler == "power-cap":
        return PowerCappedEASYScheduler(
            power_cap_watts(arguments), runtime_estimate=estimate
        )

    axes = CARBON_POLICIES[arguments.scheduler]
    options = {
        "decision_granularity": (
            timedelta(minutes=arguments.decision_granularity_minutes)
            if arguments.decision_granularity_minutes is not None
            else None
        ),
        "runtime_estimate": estimate,
    }
    if axes.get("scaled"):
        options["max_delay_fraction"] = arguments.max_delay_fraction
    else:
        options["max_delay"] = timedelta(hours=arguments.max_delay_hours)
    if axes.get("forecast"):
        archive = load_forecast_archive(arguments)
        provider = ArchiveCarbonIntensityProvider(provider, archive)
        options |= {"forecast": True, "reach": archive_reach(archive)}
    if axes.get("capped"):
        return PowerCappedCarbonAwareScheduler(
            provider, power_cap_watts(arguments), **options
        )
    return CarbonAwareScheduler(provider, **options)


def power_cap_watts(arguments: argparse.Namespace) -> float:
    if arguments.power_cap_mw is None:
        raise SystemExit(f"--power-cap-mw is required for {arguments.scheduler}")
    return arguments.power_cap_mw * WATTS_PER_MEGAWATT


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
            "fixed delay budget; carbon-scaled-delay = the same policy with a "
            "per-job budget proportional to duration; carbon-power-cap = the "
            "fixed-delay policy under the aggregate power budget; "
            "carbon-forecast = the fixed-delay policy reading only the "
            "forecasts already issued at each decision; "
            "carbon-forecast-scaled-delay = the same restriction with the "
            "per-job budget; replay = the recorded schedule"
        ),
    )
    parser.add_argument(
        "--runtime-estimate",
        choices=tuple(source.value for source in RuntimeEstimateSource),
        default=None,
        help=(
            "runtime a backfilling policy may plan with: time_limit = the "
            "requested walltime (classic EASY); scheduling = the prediction "
            "seam exposed by each job (default: scheduling with a prediction "
            "file, otherwise time_limit)"
        ),
    )
    parser.add_argument(
        "--power-cap-mw",
        type=float,
        default=None,
        help="aggregate power budget in MW, required by either capped scheduler",
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
        "--max-delay-fraction",
        type=float,
        default=1.0,
        help=(
            "maximum voluntary delay as a fraction of each job's scheduling "
            "duration for carbon-scaled-delay (default: %(default)s)"
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
    parser.add_argument(
        "--job-predictions",
        type=Path,
        default=None,
        help=(
            "prediction parquet produced by train_job_models.py; its job ids "
            "also select the workload cohort"
        ),
    )
    parser.add_argument(
        "--forecast-archive",
        type=Path,
        default=None,
        help=(
            "forecast archive from carbon_intensity.snapshots, required by the "
            "carbon-forecast schedulers; pair it with the --carbon-cache it "
            "was generated against (data/carbon_intensity/actual/actual.json)"
        ),
    )
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

    if arguments.job_predictions is not None:
        from job_prediction import attach_predictions, load_prediction_file

    predictions = (
        load_prediction_file(arguments.job_predictions)
        if arguments.job_predictions is not None
        else None
    )
    if predictions is not None:
        if arguments.limit is not None and arguments.limit <= 0:
            raise ValueError("limit must be greater than zero")
        # Resolve the complete artifact cohort before applying optional debug
        # filters. This catches a wrong workload instead of silently simulating
        # the intersection, and makes --limit a chronological artifact prefix
        # even though the full PM100 parquet itself is unsorted.
        jobs = load_jobs(
            arguments.workload,
            average_power_source=arguments.average_power_source,
            job_ids=set(predictions),
        )
        loaded_ids = {job.job_id for job in jobs}
        missing_ids = set(predictions).difference(loaded_ids)
        if missing_ids:
            preview = ", ".join(
                str(job_id) for job_id in sorted(missing_ids, key=str)[:5]
            )
            suffix = "..." if len(missing_ids) > 5 else ""
            raise ValueError(
                f"prediction artifact contains job ids absent from the workload: "
                f"{preview}{suffix}"
            )
        artifact_order = {
            job_id: position for position, job_id in enumerate(predictions)
        }
        jobs = tuple(sorted(jobs, key=lambda job: artifact_order[job.job_id]))
        if arguments.released_from is not None:
            jobs = tuple(
                job for job in jobs if job.release_time >= arguments.released_from
            )
        if arguments.released_before is not None:
            jobs = tuple(
                job for job in jobs if job.release_time < arguments.released_before
            )
        if arguments.limit is not None:
            jobs = jobs[: arguments.limit]
        if not jobs:
            raise ValueError("no predicted jobs matched the requested filters")
        jobs = attach_predictions(jobs, predictions)
    else:
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
    if arguments.job_predictions is not None:
        print(f"job predictions          {display_path(arguments.job_predictions)}")
    if arguments.forecast_archive is not None:
        print(f"forecast archive         {display_path(arguments.forecast_archive)}")

    if not arguments.no_output:
        scheduler_label = (
            f"{result.scheduler_name}-predicted"
            if arguments.job_predictions is not None
            else result.scheduler_name
        )
        destination = arguments.output or default_output_path(
            scheduler_label,
            len(result.records),
            result.total_nodes,
        )
        written = write_records(result, destination)
        print()
        print(f"records written          {display_path(written)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
