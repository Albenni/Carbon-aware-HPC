"""Sweep the delay budget and trace the carbon / quality-of-service frontier.

Every run scores the same workload on the same cluster, so the only thing
changing along a row is how long the carbon-aware policy is allowed to hold a
job back. That is the whole experiment: emissions fall monotonically with the
budget, quality of service pays for it, and the table shows the exchange rate.

EASY at the same runtime information is the reference point, and a zero budget
reproduces it, so the first row doubles as a check that the sweep starts from
the baseline rather than from a different policy.

With ``--job-predictions`` the same sweep is run twice over the artifact's
cohort, once planning with the actual durations and once with the model's, so
each budget reports how much of that budget's available carbon saving survives
imperfect information.
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

#: Only meaningful when a prediction artifact gives the sweep a
#: perfect-information twin to compare each budget against.
INPUTS_COLUMN = ("inputs", "inputs", "s")
RETAINED_COLUMN = ("retained", "retained", "s")


def table_columns(with_predictions: bool) -> tuple[tuple[str, str, str], ...]:
    """The printed columns, widened when both input cohorts are present."""

    if not with_predictions:
        return TABLE_COLUMNS
    after_saved = [header for header, _, _ in TABLE_COLUMNS].index("saved") + 1
    return (
        (INPUTS_COLUMN,)
        + TABLE_COLUMNS[:after_saved]
        + (RETAINED_COLUMN,)
        + TABLE_COLUMNS[after_saved:]
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
        inputs: str = "actual",
    ) -> None:
        self.metrics = metrics
        self.inputs = inputs
        #: Share of this budget's perfect-information saving, set once both
        #: cohorts have run; ``None`` whenever there is nothing to compare to.
        self.carbon_benefit_retained: float | None = None
        self.max_delay_hours = max_delay.total_seconds() / 3_600.0
        self.granularity_minutes = granularity.total_seconds() / 60.0
        self.emissions_saved = (
            (reference_emissions_gco2e - metrics.total_emissions_gco2e)
            / reference_emissions_gco2e
            if reference_emissions_gco2e
            else 0.0
        )

    @property
    def retained(self) -> str:
        value = self.carbon_benefit_retained
        return "-" if value is None else format(value, ".2%")

    def value(self, path: str) -> object:
        if hasattr(self, path):
            return getattr(self, path)
        value: object = self.metrics
        for attribute in path.split("."):
            value = getattr(value, attribute)
        return value

    def as_row(self) -> dict[str, object]:
        return {
            "inputs": self.inputs,
            "max_delay_hours": self.max_delay_hours,
            "decision_granularity_minutes": self.granularity_minutes,
            "emissions_saved_vs_easy": self.emissions_saved,
            "carbon_benefit_retained": self.carbon_benefit_retained,
            **self.metrics.as_row(),
        }


def assign_carbon_benefit_retained(points: list[SweepPoint]) -> None:
    """Record what each predicted row keeps of its own budget's saving.

    The comparison is against the actual-input row at the same budget and grid,
    never against the best row of the sweep: a model is not penalised for a
    budget the perfect-information policy also fails to exploit. Both savings
    share a denominator, so their ratio is the ratio of emissions avoided.
    """

    available = {
        (point.granularity_minutes, point.max_delay_hours): point.emissions_saved
        for point in points
        if point.inputs == "actual"
    }
    for point in points:
        if point.inputs == "actual":
            continue
        saving = available[(point.granularity_minutes, point.max_delay_hours)]
        point.carbon_benefit_retained = (
            point.emissions_saved / saving if saving else None
        )


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
    parser.add_argument(
        "--job-predictions",
        type=Path,
        default=None,
        help=(
            "prediction parquet from train_job_models.py; its job ids select "
            "the cohort, and every budget is then swept twice, once planning "
            "with actual durations and once with the model's"
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

    if arguments.job_predictions is not None:
        from job_prediction import load_prediction_cohort

        if arguments.released_from is not None or arguments.released_before is not None:
            raise SystemExit(
                "--job-predictions already fixes the cohort; drop the release window"
            )
        actual_jobs, predicted_jobs = load_prediction_cohort(
            arguments.workload, arguments.job_predictions
        )
        if arguments.limit is not None:
            actual_jobs = actual_jobs[: arguments.limit]
            predicted_jobs = predicted_jobs[: arguments.limit]
        cohorts = (("actual", actual_jobs), ("predicted", predicted_jobs))
    else:
        cohorts = (
            (
                "actual",
                load_jobs(
                    arguments.workload,
                    limit=arguments.limit,
                    released_from=arguments.released_from,
                    released_before=arguments.released_before,
                ),
            ),
        )

    # The cohorts hold the same jobs and differ only in the estimates a policy
    # may plan with, so the first one fixes the evaluation set for all of them.
    evaluation_jobs = cohorts[0][1]
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
    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
    granularities = (
        [timedelta(minutes=minutes) for minutes in arguments.decision_granularity_minutes]
        if arguments.decision_granularity_minutes
        else [provider.granularity]
    )

    def score(scheduler, cohort_jobs: tuple) -> ScheduleMetrics:
        jobs = cohort_jobs + contention_jobs
        result = Simulator(jobs, Cluster(arguments.nodes), scheduler).run()
        evaluation = result.replace_records(
            tuple(record for record in result.records if record.job_id in evaluation_ids)
        )
        return schedule_metrics(
            account_schedule(evaluation, cohort_jobs, provider)
        )

    # Both carbon-blind baselines are reported, because the saving only means
    # something if it does not depend on which of them it is measured against.
    # EASY on actual durations is the reference for every row of the sweep, so
    # that a predicted row and an actual row are savings against one number;
    # strict FCFS reads no runtime estimate at all, so it is scored once.
    references: list[tuple[str, ScheduleMetrics]] = [
        ("fcfs", score(FCFSScheduler(), evaluation_jobs))
    ]
    for label, cohort_jobs in cohorts:
        suffix = f" ({label} inputs)" if len(cohorts) > 1 else ""
        references.append(
            (
                f"easy{suffix}",
                score(EASYBackfillScheduler(runtime_estimate=estimate), cohort_jobs),
            )
        )
    reference = references[1][1]

    points: list[SweepPoint] = []
    for label, cohort_jobs in cohorts:
        for granularity in granularities:
            for hours in arguments.max_delay_hours:
                max_delay = timedelta(hours=hours)
                metrics = score(
                    CarbonAwareScheduler(
                        provider,
                        max_delay=max_delay,
                        decision_granularity=granularity,
                        runtime_estimate=estimate,
                    ),
                    cohort_jobs,
                )
                points.append(
                    SweepPoint(
                        metrics,
                        max_delay=max_delay,
                        granularity=granularity,
                        reference_emissions_gco2e=reference.total_emissions_gco2e,
                        inputs=label,
                    )
                )

    assign_carbon_benefit_retained(points)

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
    if arguments.job_predictions is not None:
        print(f"job predictions          {display_path(arguments.job_predictions)}")
    for name, baseline in references:
        print(
            f"{name:<25}{baseline.total_emissions_tco2e:,.4f} tCO2e, "
            f"waiting mean {baseline.waiting.mean:,.1f} s, "
            f"bounded slowdown {baseline.bounded_slowdown.mean:,.2f}"
        )
    print()

    columns = table_columns(len(cohorts) > 1)
    widths = [max(len(header), 10) + 2 for header, _, _ in columns]
    header = "".join(
        header.rjust(width) for (header, _, _), width in zip(columns, widths)
    )
    print(header)
    print("-" * len(header))
    for point in points:
        print(
            "".join(
                format(point.value(path), spec).rjust(width)
                for (_, path, spec), width in zip(columns, widths)
            )
        )

    print()
    if contention_jobs:
        print("evaluated-cohort energy is identical; only its timing changes.")
    else:
        print("energy is identical across every row; only the timing of it changes.")

    if arguments.no_output:
        return 0

    if contention_jobs:
        name = (
            f"carbon_tradeoff_terminal_contention_{reference.job_count}evaluated_"
            f"{arguments.nodes}nodes.csv"
        )
    elif arguments.job_predictions is not None:
        name = (
            f"carbon_tradeoff_predicted_{reference.job_count}jobs_"
            f"{arguments.nodes}nodes.csv"
        )
    else:
        name = f"carbon_tradeoff_{reference.job_count}jobs_{arguments.nodes}nodes.csv"
    destination = arguments.output or DEFAULT_OUTPUT_DIR / name
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
