"""Run comparable scheduling policies over one workload and tabulate the differences.

The point of the comparison is that all policies see the same jobs, the same
cluster, and the same runtime information, so a difference in the table is a
difference in the policy and nothing else. The historical replay is included
for reference only: its waiting times were produced under contention with jobs
the dataset preparation removed, so it is a fidelity anchor rather than a
performance baseline (see ``src/hpc_sim/README.md``). Optional delay arguments
add the fixed, power-capped and duration-scaled carbon-aware policies.
"""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    DurationScaledCarbonAwareScheduler,
    EASYBackfillScheduler,
    FCFSScheduler,
    PowerCappedCarbonAwareScheduler,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
    ScheduleMetrics,
    Simulator,
    TraceReplayScheduler,
    account_schedule,
    schedule_metrics,
)
from hpc_sim.workload import load_jobs

from run_simulation import (
    DEFAULT_CARBON_CACHE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKLOAD,
    display_path,
    parse_timestamp,
)


WATTS_PER_MEGAWATT = 1e6

#: Rows of the comparison table: label, accessor, format.
TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("total emissions (tCO2e)", "total_emissions_tco2e", ",.3f"),
    ("mean emissions/job (g)", "mean_emissions_gco2e", ",.1f"),
    ("total energy (MWh)", "total_energy_mwh", ",.2f"),
    ("peak power (MW)", "peak_power_mw", ",.3f"),
    ("waiting mean (s)", "waiting.mean", ",.1f"),
    ("waiting median (s)", "waiting.median", ",.1f"),
    ("waiting p95 (s)", "waiting.p95", ",.1f"),
    ("waiting p99 (s)", "waiting.p99", ",.1f"),
    ("waiting max (s)", "waiting.maximum", ",.1f"),
    ("turnaround mean (s)", "turnaround.mean", ",.1f"),
    ("bounded slowdown mean", "bounded_slowdown.mean", ",.2f"),
    ("bounded slowdown p95", "bounded_slowdown.p95", ",.2f"),
    ("bounded slowdown max", "bounded_slowdown.maximum", ",.2f"),
    ("node utilisation", "utilisation", ".1%"),
    ("peak nodes in use", "peak_busy_nodes", ",d"),
    ("makespan (days)", "makespan_days", ",.2f"),
    ("throughput (jobs/h)", "throughput_jobs_per_hour", ",.1f"),
)


def resolve(metrics: ScheduleMetrics, path: str) -> object:
    if path == "makespan_days":
        return metrics.makespan_seconds / 86_400.0
    value: object = metrics
    for attribute in path.split("."):
        value = getattr(value, attribute)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_CARBON_CACHE)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--released-from", type=parse_timestamp, default=None)
    parser.add_argument("--released-before", type=parse_timestamp, default=None)
    parser.add_argument(
        "--runtime-estimate",
        choices=tuple(source.value for source in RuntimeEstimateSource),
        default=RuntimeEstimateSource.TIME_LIMIT.value,
        help="runtime the backfilling policies may plan with (default: %(default)s)",
    )
    parser.add_argument(
        "--power-cap-mw",
        type=float,
        default=None,
        help="absolute power budget; overrides --power-cap-fraction",
    )
    parser.add_argument(
        "--power-cap-fraction",
        type=float,
        default=0.8,
        help=(
            "power budget as a fraction of the FCFS peak, so the cap always "
            "binds regardless of workload size (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-delay-hours",
        type=float,
        default=None,
        help=(
            "also compare carbon-aware scheduling, with and without the same "
            "power cap, using this delay budget"
        ),
    )
    parser.add_argument(
        "--decision-granularity-minutes",
        type=float,
        default=None,
        help="candidate spacing for carbon-aware schedulers",
    )
    parser.add_argument(
        "--max-delay-fraction",
        type=float,
        default=None,
        help=(
            "also compare carbon-aware scheduling with a per-job delay budget "
            "equal to this fraction of scheduling duration"
        ),
    )
    parser.add_argument("--no-replay", action="store_true", help="skip the trace replay")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV destination for the metric table (default: auto-named)",
    )
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    estimate = RuntimeEstimateSource(arguments.runtime_estimate)

    jobs = load_jobs(
        arguments.workload,
        limit=arguments.limit,
        released_from=arguments.released_from,
        released_before=arguments.released_before,
    )
    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)

    def score(scheduler) -> ScheduleMetrics:
        result = Simulator(jobs, Cluster(arguments.nodes), scheduler).run()
        return schedule_metrics(account_schedule(result, jobs, provider))

    scored: list[ScheduleMetrics] = []
    if not arguments.no_replay:
        scored.append(score(TraceReplayScheduler()))
    fcfs = score(FCFSScheduler())
    scored.append(fcfs)
    easy = score(EASYBackfillScheduler(runtime_estimate=estimate))
    scored.append(easy)

    # A relative cap cannot be lower than the largest job: that would make the
    # workload unschedulable instead of merely limiting concurrent power.
    if arguments.power_cap_mw is not None:
        cap_watts = arguments.power_cap_mw * WATTS_PER_MEGAWATT
    else:
        relative_cap_watts = arguments.power_cap_fraction * fcfs.peak_power_watts
        if relative_cap_watts <= 0.0:
            raise ValueError("--power-cap-fraction must be greater than zero")
        cap_watts = max(
            relative_cap_watts,
            max(job.scheduling_average_power_watts for job in jobs),
        )
    power_capped_easy = score(
        PowerCappedEASYScheduler(cap_watts, runtime_estimate=estimate)
    )
    scored.append(power_capped_easy)

    decision_granularity = (
        timedelta(minutes=arguments.decision_granularity_minutes)
        if arguments.decision_granularity_minutes is not None
        else None
    )
    carbon_saving_lost: float | None = None
    if arguments.max_delay_hours is not None:
        carbon_options = {
            "max_delay": timedelta(hours=arguments.max_delay_hours),
            "decision_granularity": decision_granularity,
            "runtime_estimate": estimate,
        }
        carbon = score(CarbonAwareScheduler(provider, **carbon_options))
        carbon_power_capped = score(
            PowerCappedCarbonAwareScheduler(
                provider,
                cap_watts,
                **carbon_options,
            )
        )
        scored.extend((carbon, carbon_power_capped))
        available_saving = easy.total_emissions_gco2e - carbon.total_emissions_gco2e
        if available_saving > 0.0:
            carbon_saving_lost = (
                carbon_power_capped.total_emissions_gco2e
                - carbon.total_emissions_gco2e
            ) / available_saving

    if arguments.max_delay_fraction is not None:
        scored.append(
            score(
                DurationScaledCarbonAwareScheduler(
                    provider,
                    max_delay_fraction=arguments.max_delay_fraction,
                    decision_granularity=decision_granularity,
                    runtime_estimate=estimate,
                )
            )
        )

    print(f"workload                 {arguments.workload.name}")
    print(f"jobs                     {fcfs.job_count:,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"runtime estimate         {estimate.value}")
    print(f"power cap                {cap_watts / WATTS_PER_MEGAWATT:,.3f} MW")
    print()

    width = max(len(label) for label, _, _ in TABLE_ROWS) + 2
    columns = [metrics.scheduler_name for metrics in scored]
    column_width = max(14, max(len(name) for name in columns) + 2)
    header = "".ljust(width) + "".join(name.rjust(column_width) for name in columns)
    print(header)
    print("-" * len(header))
    for label, path, spec in TABLE_ROWS:
        cells = "".join(
            format(resolve(metrics, path), spec).rjust(column_width)
            for metrics in scored
        )
        print(label.ljust(width) + cells)

    if arguments.max_delay_hours is not None:
        print()
        if carbon_saving_lost is None:
            print("carbon saving lost to cap  n/a (carbon-aware did not beat EASY)")
        else:
            print(f"carbon saving lost to cap  {carbon_saving_lost:.4%}")

    if arguments.no_output:
        return 0

    compares_carbon = (
        arguments.max_delay_hours is not None
        or arguments.max_delay_fraction is not None
    )
    comparison = "policy" if compares_carbon else "baseline"
    destination = arguments.output or DEFAULT_OUTPUT_DIR / (
        f"{comparison}_comparison_{fcfs.job_count}jobs_{arguments.nodes}nodes.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [metrics.as_row() for metrics in scored]
    for metrics, row in zip(scored, rows):
        capped = metrics.scheduler_name in {"power-cap", "carbon-power-cap"}
        fixed_delay = metrics.scheduler_name in {"carbon-aware", "carbon-power-cap"}
        scaled_delay = metrics.scheduler_name == "carbon-scaled-delay"
        carbon_aware = fixed_delay or scaled_delay
        row["power_cap_mw"] = cap_watts / WATTS_PER_MEGAWATT if capped else None
        row["emissions_saved_vs_easy"] = (
            (easy.total_emissions_gco2e - metrics.total_emissions_gco2e)
            / easy.total_emissions_gco2e
            if easy.total_emissions_gco2e
            else 0.0
        )
        if compares_carbon:
            row["max_delay_hours"] = (
                arguments.max_delay_hours if fixed_delay else None
            )
            row["max_delay_fraction"] = (
                arguments.max_delay_fraction if scaled_delay else None
            )
            row["decision_granularity_minutes"] = (
                (
                    arguments.decision_granularity_minutes
                    if arguments.decision_granularity_minutes is not None
                    else provider.granularity.total_seconds() / 60.0
                )
                if carbon_aware
                else None
            )
            if arguments.max_delay_hours is not None:
                row["carbon_saving_loss_fraction"] = (
                    carbon_saving_lost
                    if metrics.scheduler_name == "carbon-power-cap"
                    else None
                )
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"metrics written          {display_path(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
