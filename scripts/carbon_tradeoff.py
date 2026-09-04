"""Sweep the delay budget and trace the carbon / quality-of-service frontier.

Every run scores the same workload on the same cluster, so the only thing
changing along a row is how long the carbon-aware policy is allowed to hold a
job back. That is the whole experiment: emissions fall monotonically with the
budget, quality of service pays for it, and the table shows the exchange rate.

EASY at the same runtime information is the reference point, and a zero budget
reproduces it, so the first row doubles as a check that the sweep starts from
the baseline rather than from a different policy.
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
    EASYBackfillScheduler,
    FCFSScheduler,
    RuntimeEstimateSource,
    ScheduleMetrics,
    Simulator,
    account_schedule,
    schedule_metrics,
)
from hpc_sim.workload import load_contention_jobs, load_jobs

from run_simulation import (
    DEFAULT_CARBON_CACHE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKLOAD,
    display_path,
    parse_timestamp,
)


#: Columns of the printed table: header, accessor, format.
TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("delay (h)", "max_delay_hours", ",.2f"),
    ("grid (min)", "granularity_minutes", ",.0f"),
    ("tCO2e", "total_emissions_tco2e", ",.4f"),
    ("saved", "emissions_saved", ".2%"),
    ("wait mean", "waiting.mean", ",.1f"),
    ("wait p95", "waiting.p95", ",.1f"),
    ("wait max", "waiting.maximum", ",.1f"),
    ("bsld mean", "bounded_slowdown.mean", ",.2f"),
    ("bsld p95", "bounded_slowdown.p95", ",.2f"),
    ("peak MW", "peak_power_mw", ",.3f"),
    ("util", "utilisation", ".1%"),
)


class SweepPoint:
    """One configuration and what it scored, ready to print or export."""

    def __init__(
        self,
        metrics: ScheduleMetrics,
        *,
        max_delay: timedelta,
        granularity: timedelta,
        reference_emissions_gco2e: float,
    ) -> None:
        self.metrics = metrics
        self.max_delay_hours = max_delay.total_seconds() / 3_600.0
        self.granularity_minutes = granularity.total_seconds() / 60.0
        self.emissions_saved = (
            (reference_emissions_gco2e - metrics.total_emissions_gco2e)
            / reference_emissions_gco2e
            if reference_emissions_gco2e
            else 0.0
        )

    def value(self, path: str) -> object:
        if hasattr(self, path):
            return getattr(self, path)
        value: object = self.metrics
        for attribute in path.split("."):
            value = getattr(value, attribute)
        return value

    def as_row(self) -> dict[str, object]:
        return {
            "max_delay_hours": self.max_delay_hours,
            "decision_granularity_minutes": self.granularity_minutes,
            "emissions_saved_vs_easy": self.emissions_saved,
            **self.metrics.as_row(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument(
        "--contention-workload",
        type=Path,
        default=None,
        metavar="RAW_PARQUET",
        help=(
            "add valid terminal non-COMPLETED jobs from the raw PM100 trace "
            "as resource-only scheduler demand"
        ),
    )
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_CARBON_CACHE)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--released-from", type=parse_timestamp, default=None)
    parser.add_argument("--released-before", type=parse_timestamp, default=None)
    parser.add_argument(
        "--max-delay-hours",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 3.0, 6.0, 12.0, 24.0),
        help="delay budgets to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--decision-granularity-minutes",
        type=float,
        nargs="+",
        default=None,
        help=(
            "candidate start-time grids to sweep, one run per combination "
            "(default: the provider's own granularity)"
        ),
    )
    parser.add_argument(
        "--runtime-estimate",
        choices=tuple(source.value for source in RuntimeEstimateSource),
        default=RuntimeEstimateSource.SCHEDULING.value,
        help=(
            "runtime every policy plans with; the perfect-information benchmark "
            "uses %(default)s"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV destination for the sweep (default: auto-named)",
    )
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    estimate = RuntimeEstimateSource(arguments.runtime_estimate)

    evaluation_jobs = load_jobs(
        arguments.workload,
        limit=arguments.limit,
        released_from=arguments.released_from,
        released_before=arguments.released_before,
    )
    evaluation_ids = {job.job_id for job in evaluation_jobs}
    contention_jobs = ()
    if arguments.contention_workload is not None:
        window_start = arguments.released_from or min(
            job.release_time for job in evaluation_jobs
        )
        window_end = arguments.released_before or max(
            job.release_time for job in evaluation_jobs
        ) + timedelta(microseconds=1)
        contention_jobs = load_contention_jobs(
            arguments.contention_workload,
            window_start=window_start,
            window_end=window_end,
        )
    jobs = evaluation_jobs + contention_jobs
    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
    granularities = (
        [timedelta(minutes=minutes) for minutes in arguments.decision_granularity_minutes]
        if arguments.decision_granularity_minutes
        else [provider.granularity]
    )

    def score(scheduler) -> ScheduleMetrics:
        result = Simulator(jobs, Cluster(arguments.nodes), scheduler).run()
        evaluation = result.replace_records(
            tuple(record for record in result.records if record.job_id in evaluation_ids)
        )
        return schedule_metrics(
            account_schedule(evaluation, evaluation_jobs, provider)
        )

    # Both carbon-blind baselines are reported, because the saving only means
    # something if it does not depend on which of them it is measured against.
    # EASY is the reference for the sweep; strict FCFS reads no runtime
    # estimate at all, so its figures do not depend on that choice either.
    references = (
        score(FCFSScheduler()),
        score(EASYBackfillScheduler(runtime_estimate=estimate)),
    )
    reference = references[-1]

    points: list[SweepPoint] = []
    for granularity in granularities:
        for hours in arguments.max_delay_hours:
            max_delay = timedelta(hours=hours)
            metrics = score(
                CarbonAwareScheduler(
                    provider,
                    max_delay=max_delay,
                    decision_granularity=granularity,
                    runtime_estimate=estimate,
                )
            )
            points.append(
                SweepPoint(
                    metrics,
                    max_delay=max_delay,
                    granularity=granularity,
                    reference_emissions_gco2e=reference.total_emissions_gco2e,
                )
            )

    print(f"workload                 {arguments.workload.name}")
    if contention_jobs:
        print(f"contention workload      {arguments.contention_workload.name}")
        print(f"scheduled jobs           {len(jobs):,}")
        print(f"evaluation jobs          {reference.job_count:,} COMPLETED")
        print(f"contention-only jobs     {len(contention_jobs):,}")
        print("metric boundary          QoS/carbon: evaluation; nodes: all jobs")
    else:
        print(f"jobs                     {reference.job_count:,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"runtime estimate         {estimate.value}")
    for baseline in references:
        print(
            f"{baseline.scheduler_name:<25}{baseline.total_emissions_tco2e:,.4f} tCO2e, "
            f"waiting mean {baseline.waiting.mean:,.1f} s, "
            f"bounded slowdown {baseline.bounded_slowdown.mean:,.2f}"
        )
    print()

    widths = [max(len(header), 10) + 2 for header, _, _ in TABLE_COLUMNS]
    header = "".join(
        header.rjust(width) for (header, _, _), width in zip(TABLE_COLUMNS, widths)
    )
    print(header)
    print("-" * len(header))
    for point in points:
        print(
            "".join(
                format(point.value(path), spec).rjust(width)
                for (_, path, spec), width in zip(TABLE_COLUMNS, widths)
            )
        )

    print()
    if contention_jobs:
        print("evaluated-cohort energy is identical; only its timing changes.")
    else:
        print("energy is identical across every row; only the timing of it changes.")

    if arguments.no_output:
        return 0

    destination = arguments.output or DEFAULT_OUTPUT_DIR / (
        f"carbon_tradeoff_terminal_contention_{reference.job_count}evaluated_"
        f"{arguments.nodes}nodes.csv"
        if contention_jobs
        else f"carbon_tradeoff_{reference.job_count}jobs_{arguments.nodes}nodes.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [point.as_row() for point in points]
    if contention_jobs:
        rows = [
            {
                "scheduled_jobs": len(jobs),
                "contention_only_jobs": len(contention_jobs),
                **row,
            }
            for row in rows
        ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"sweep written            {display_path(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
