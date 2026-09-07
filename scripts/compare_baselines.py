"""Tabulate every scheduling configuration over one cohort, into one CSV.

The comparison is a matrix, and each axis is one thing a production scheduler
would or would not know:

    job information   actual durations and power, or one arm per
                      ``--job-predictions`` artifact (``label=path``)
    carbon signal     the actual future intensity (the oracle), or one arm per
                      ``--forecast-archive``, replayed as it was issued
    delay policy      none (EASY, power-capped EASY), fixed delay, or
                      duration-scaled delay

Every cell sees the same jobs, the same cluster and the same power cap, so a
difference between two rows is a difference between those three axes and
nothing else. The identity columns name the cell a row came from; the rest of
the columns measure it.

Scoring is ex post and always on ground truth: the measured power profile of
each job, at the times the simulation produced, against the observed carbon
intensity. Predictions and forecasts enter the decisions only, never the score.

Nothing a decision reads postdates it. A prediction artifact is a held-out test
split, and a forecast archive answers each release with the snapshot already
issued at that instant. Because a snapshot only reaches so far, every archive
is run under one shared reach and the oracle it is compared against takes the
same bound, so ``oracle_recovery`` - the share of the oracle's saving a
forecast run recovers - compares two runs that differ in signal alone.

The historical replay is a fidelity anchor rather than a performance baseline:
its waiting times were produced under contention with jobs the dataset
preparation removed (see ``src/hpc_sim/README.md``).
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path


from carbon_intensity import TimeSeriesCarbonIntensityProvider
from carbon_intensity.scheduling_impact import archive_reach
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    EASYBackfillScheduler,
    FCFSScheduler,
    Job,
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

from common import (
    DEFAULT_CARBON_CACHE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKLOAD,
    display_path,
    parse_timestamp,
)


WATTS_PER_MEGAWATT = 1e6

#: Columns that name the configuration a row was produced by. Every row carries
#: all of them, empty on an axis that does not apply, so the CSV reads as a
#: relational table: these identify the run, the remaining columns measure it.
IDENTITY_FIELDS: tuple[str, ...] = (
    "scheduler_family",
    "delay_policy",
    "job_information",
    "job_model",
    "carbon_information",
    "forecast_model",
    "runtime_estimate",
    "max_delay_hours",
    "max_delay_fraction",
    "forecast_reach_hours",
    "power_cap_mw",
    "decision_granularity_minutes",
)

#: Each of these needs a second run to mean anything, so they are filled in
#: once the whole matrix has been scored.
DERIVED_FIELDS: tuple[str, ...] = (
    "emissions_saved_vs_easy",
    "oracle_recovery",
    "carbon_saving_loss_fraction",
)

#: Printed columns: header, row key, format. The CSV carries every metric; this
#: is the subset that fits on a terminal line, one line per configuration.
PRINT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("job info", "job_information", "s"),
    ("model", "job_model", "s"),
    ("carbon", "carbon_information", "s"),
    ("forecast", "forecast_model", "s"),
    ("delay", "delay_policy", "s"),
    ("policy", "scheduler", "s"),
    ("tCO2e", "total_emissions_tco2e", ",.3f"),
    ("vs easy", "emissions_saved_vs_easy", ".2%"),
    ("recovery", "oracle_recovery", ".2%"),
    ("wait mean", "waiting_mean_s", ",.0f"),
    ("wait p95", "waiting_p95_s", ",.0f"),
    ("bsld mean", "bounded_slowdown_mean", ",.2f"),
    ("bsld p95", "bounded_slowdown_p95", ",.2f"),
    ("peak MW", "peak_power_mw", ",.3f"),
    ("util", "utilisation", ".1%"),
    ("days", "makespan_days", ",.2f"),
)

#: ``(job_information, job_model, jobs)``. The arms hold the same cohort and
#: differ only in the estimates a policy may plan with.
Arm = tuple[str, str, tuple[Job, ...]]


def cell(value: object, spec: str) -> str:
    """An empty value means the axis does not apply to that row, not zero."""

    return "-" if value is None or value == "" else format(value, spec)


def archive_label(path: Path) -> str:
    """Name an archive by its model, dropping the partition prefix."""

    stem = path.stem
    return stem.split("_", 1)[1] if stem.startswith(("test_", "validation_")) else stem


def parse_labelled_path(value: str) -> tuple[str, Path]:
    """``label=path`` for a prediction artifact; a bare path names itself.

    The label becomes the ``job_model`` column, so it is how one model's rows
    are told from another's and is worth spelling out on the command line.
    """

    label, separator, path = value.partition("=")
    if not separator:
        return Path(label).parent.name or Path(label).stem, Path(label)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
    return label, Path(path)


def align_arms(
    cohorts: Mapping[str, tuple[tuple[Job, ...], tuple[Job, ...]]],
    limit: int | None = None,
) -> list[Arm]:
    """One cohort, seen with actual inputs and with each artifact's predictions.

    Only jobs every artifact predicts survive: scoring two models over two
    different cohorts would confound the model with the workload, so an id one
    of them is missing is dropped from all of them, the actual arm included.
    The arms therefore hold the same jobs in the same order, which is what makes
    a difference between two rows attributable to the information alone.
    """

    if not cohorts:
        raise ValueError("at least one prediction artifact is required")
    shared = set.intersection(
        *({job.job_id for job in predicted} for _, predicted in cohorts.values())
    )
    if not shared:
        raise ValueError("the prediction artifacts have no job in common")
    first_actual = next(iter(cohorts.values()))[0]
    order = [job.job_id for job in first_actual if job.job_id in shared]
    if limit is not None:
        order = order[:limit]

    def aligned(jobs: tuple[Job, ...]) -> tuple[Job, ...]:
        by_id = {job.job_id: job for job in jobs}
        return tuple(by_id[job_id] for job_id in order)

    return [("actual", "actual", aligned(first_actual))] + [
        ("predicted", label, aligned(predicted))
        for label, (_, predicted) in cohorts.items()
    ]


def compare(
    arguments: argparse.Namespace,
    arms: Sequence[Arm],
    provider: TimeSeriesCarbonIntensityProvider,
    archives: Mapping[str, ForecastArchive],
) -> list[dict[str, object]]:
    """Run every cell of the matrix and return one row per configuration.

    The first arm is the actual-input one; its jobs carry the measured power
    profiles that every run, whatever it planned with, is charged against.
    """

    estimate = RuntimeEstimateSource(arguments.runtime_estimate)
    actual_jobs = arms[0][2]
    rows: list[dict[str, object]] = []

    def add(scheduler, jobs: tuple[Job, ...], **identity) -> ScheduleMetrics:
        result = Simulator(jobs, Cluster(arguments.nodes), scheduler).run()
        # Ex post scoring: actual durations and power against actual intensity,
        # never the estimates or the forecast the run decided with.
        metrics = schedule_metrics(account_schedule(result, actual_jobs, provider))
        blank = dict.fromkeys(IDENTITY_FIELDS + DERIVED_FIELDS, "")
        rows.append(blank | metrics.as_row() | identity)
        return metrics

    # The replay and FCFS read no runtime estimate and no prediction, so they
    # are the same run whatever the arms are: scored once, on the actual cohort.
    carbon_blind = {
        "delay_policy": "none",
        "job_information": "actual",
        "job_model": "actual",
        "carbon_information": "none",
    }
    if not arguments.no_replay:
        add(TraceReplayScheduler(), actual_jobs, scheduler_family="trace-replay", **carbon_blind)
    fcfs = add(FCFSScheduler(), actual_jobs, scheduler_family="fcfs", **carbon_blind)

    # One cap for the whole table, and never below the largest job of any arm:
    # a cap a job cannot fit under makes the workload unschedulable instead of
    # merely limiting concurrent power.
    largest_watts = max(
        job.scheduling_average_power_watts for _, _, jobs in arms for job in jobs
    )
    if arguments.power_cap_mw is not None:
        cap_watts = arguments.power_cap_mw * WATTS_PER_MEGAWATT
    else:
        relative_cap_watts = arguments.power_cap_fraction * fcfs.peak_power_watts
        if relative_cap_watts <= 0.0:
            raise ValueError("--power-cap-fraction must be greater than zero")
        cap_watts = max(relative_cap_watts, largest_watts)
    cap_mw = cap_watts / WATTS_PER_MEGAWATT

    decision_granularity = (
        timedelta(minutes=arguments.decision_granularity_minutes)
        if arguments.decision_granularity_minutes is not None
        else None
    )
    granularity_minutes = (
        arguments.decision_granularity_minutes
        if arguments.decision_granularity_minutes is not None
        else provider.granularity.total_seconds() / 60.0
    )
    shared = {"decision_granularity": decision_granularity, "runtime_estimate": estimate}

    # Every archive is scored under one reach, and so is the oracle each one is
    # compared against: otherwise two runs would differ in their delay budget as
    # well as in their signal, and the gap would stop being forecast error.
    reach = min((archive_reach(archive) for archive in archives.values()), default=None)
    reach_hours = None if reach is None else reach / timedelta(hours=1)

    families: list[tuple[str, dict, dict]] = []
    if arguments.max_delay_hours is not None:
        families.append(
            (
                "fixed",
                {"max_delay": timedelta(hours=arguments.max_delay_hours)},
                {"max_delay_hours": arguments.max_delay_hours},
            )
        )
    if arguments.max_delay_fraction is not None:
        families.append(
            (
                "duration-scaled",
                {"max_delay_fraction": arguments.max_delay_fraction},
                {"max_delay_fraction": arguments.max_delay_fraction},
            )
        )

    easy_by_arm: dict[tuple[str, str], ScheduleMetrics] = {}
    for information, model, jobs in arms:
        arm = {
            "job_information": information,
            "job_model": model,
            "runtime_estimate": estimate.value,
        }
        # The carbon-blind reference for this arm: the same information, the
        # same cohort, no delay budget. Every saving on the arm is against it.
        easy = add(
            EASYBackfillScheduler(runtime_estimate=estimate),
            jobs,
            scheduler_family="easy",
            delay_policy="none",
            carbon_information="none",
            **arm,
        )
        easy_by_arm[(information, model)] = easy
        add(
            PowerCappedEASYScheduler(cap_watts, runtime_estimate=estimate),
            jobs,
            scheduler_family="power-cap-easy",
            delay_policy="none",
            carbon_information="none",
            power_cap_mw=cap_mw,
            **arm,
        )

        for delay_policy, options, recorded in families:
            budget = {
                **arm,
                "delay_policy": delay_policy,
                "decision_granularity_minutes": granularity_minutes,
                **recorded,
            }
            # Perfect knowledge of the future grid, unbounded: how much carbon
            # this delay budget can reach at all.
            oracle = add(
                CarbonAwareScheduler(provider, **shared, **options),
                jobs,
                scheduler_family="carbon",
                carbon_information="actual",
                **budget,
            )
            if delay_policy == "fixed":
                capped = add(
                    PowerCappedCarbonAwareScheduler(provider, cap_watts, **shared, **options),
                    jobs,
                    scheduler_family="power-cap-carbon",
                    carbon_information="actual",
                    power_cap_mw=cap_mw,
                    **budget,
                )
                # What the cap costs, as a share of the saving the same policy
                # reaches without it.
                uncapped_saving = easy.total_emissions_gco2e - oracle.total_emissions_gco2e
                if uncapped_saving > 0.0:
                    rows[-1]["carbon_saving_loss_fraction"] = (
                        capped.total_emissions_gco2e - oracle.total_emissions_gco2e
                    ) / uncapped_saving

            if reach is None:
                continue
            # The same oracle under the archives' own reach: the denominator a
            # forecast run is entitled to be measured against.
            bounded = add(
                CarbonAwareScheduler(provider, reach=reach, **shared, **options),
                jobs,
                scheduler_family="carbon",
                carbon_information="actual",
                forecast_reach_hours=reach_hours,
                **budget,
            )
            rows[-1]["oracle_recovery"] = 1.0
            available = easy.total_emissions_gco2e - bounded.total_emissions_gco2e
            for label, archive in archives.items():
                metrics = add(
                    CarbonAwareScheduler(
                        ArchiveCarbonIntensityProvider(provider, archive),
                        forecast=True,
                        reach=reach,
                        **shared,
                        **options,
                    ),
                    jobs,
                    scheduler_family="carbon",
                    carbon_information="forecast",
                    forecast_model=label,
                    forecast_reach_hours=reach_hours,
                    **budget,
                )
                if available > 0.0:
                    rows[-1]["oracle_recovery"] = (
                        easy.total_emissions_gco2e - metrics.total_emissions_gco2e
                    ) / available

    for row in rows:
        reference = easy_by_arm[(row["job_information"], row["job_model"])]
        row["emissions_saved_vs_easy"] = (
            (reference.total_emissions_tco2e - row["total_emissions_tco2e"])
            / reference.total_emissions_tco2e
            if reference.total_emissions_tco2e
            else 0.0
        )
    return rows


def print_table(rows: Sequence[Mapping[str, object]]) -> None:
    """One line per configuration, sized to what it actually contains."""

    table = [[cell(row.get(key, ""), spec) for _, key, spec in PRINT_COLUMNS] for row in rows]
    widths = [
        max(len(header), max(len(line[index]) for line in table)) + 2
        for index, (header, _, _) in enumerate(PRINT_COLUMNS)
    ]
    header = "".join(
        name.rjust(width) for (name, _, _), width in zip(PRINT_COLUMNS, widths, strict=True)
    )
    print(header)
    print("-" * len(header))
    for line in table:
        print("".join(value.rjust(width) for value, width in zip(line, widths, strict=True)))


def write_rows(rows: Sequence[Mapping[str, object]], destination: Path) -> Path:
    """One CSV row per configuration, identity columns first."""

    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument(
        "--job-predictions",
        type=parse_labelled_path,
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "prediction parquet from train_job_models.py, repeatable, e.g. "
            "gradient=data/job_predictions/test_predictions.parquet; each one "
            "adds a predicted-information arm labelled LABEL, and the cohort "
            "every artifact covers is also run once with actual durations and "
            "power as the perfect-information arm"
        ),
    )
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_CARBON_CACHE)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--released-from", type=parse_timestamp, default=None)
    parser.add_argument("--released-before", type=parse_timestamp, default=None)
    parser.add_argument(
        "--runtime-estimate",
        choices=tuple(source.value for source in RuntimeEstimateSource),
        default=RuntimeEstimateSource.TIME_LIMIT.value,
        help=(
            "runtime the backfilling policies may plan with (default: "
            "%(default)s, the only estimate PM100 users actually supply); "
            "'scheduling' hands them the job model of each arm instead, and on "
            "the actual arm that is the actual duration"
        ),
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
            "add the fixed-delay carbon-aware policy, with and without the same "
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
            "add the duration-scaled carbon-aware policy, with a per-job delay "
            "budget equal to this fraction of scheduling duration"
        ),
    )
    parser.add_argument(
        "--forecast-archive",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "also run the carbon-aware policies on this forecast archive, "
            "repeatable; every archive and the oracle they are compared against "
            "share the tightest reach of the set, so oracle_recovery is a "
            "like-for-like ratio. Pair it with the --carbon-cache it was "
            "generated against (data/carbon_intensity/actual/actual.json)"
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

    if arguments.job_predictions:
        from job_prediction import load_prediction_cohort

        if arguments.released_from is not None or arguments.released_before is not None:
            raise SystemExit(
                "--job-predictions already fixes the cohort; drop the release window"
            )
        labels = [label for label, _ in arguments.job_predictions]
        if len(set(labels)) != len(labels):
            raise SystemExit("--job-predictions labels must be unique")
        arms = align_arms(
            {
                label: load_prediction_cohort(arguments.workload, path)
                for label, path in arguments.job_predictions
            },
            limit=arguments.limit,
        )
    else:
        arms = [
            (
                "actual",
                "actual",
                load_jobs(
                    arguments.workload,
                    limit=arguments.limit,
                    released_from=arguments.released_from,
                    released_before=arguments.released_before,
                ),
            )
        ]

    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
    archives = {
        archive_label(path): ForecastArchive.load(path)
        for path in arguments.forecast_archive or ()
    }
    rows = compare(arguments, arms, provider, archives)

    print(f"workload                 {arguments.workload.name}")
    print(f"jobs                     {len(arms[0][2]):,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"runtime estimate         {arguments.runtime_estimate}")
    cap_mw = next(row["power_cap_mw"] for row in rows if row["power_cap_mw"] != "")
    print(f"power cap                {cap_mw:,.3f} MW")
    if len(arms) > 1:
        print(f"job models               {', '.join(model for _, model, _ in arms[1:])}")
    if archives:
        print(f"forecast archives        {', '.join(archives)}")
        reach = next(
            (row["forecast_reach_hours"] for row in rows if row["forecast_reach_hours"] != ""),
            None,
        )
        # No delay budget was asked for, so no run ever consulted an archive.
        print(f"forecast reach           {'unused' if reach is None else f'{reach:,.2f} h'}")
    print(f"configurations           {len(rows):,}")
    print()

    print_table(rows)
    print()
    print("energy is identical across every row; only the timing of it changes.")

    if arguments.no_output:
        return 0

    compares_carbon = (
        arguments.max_delay_hours is not None or arguments.max_delay_fraction is not None
    )
    if len(arms) > 1:
        kind = "matrix"
    else:
        kind = "policy" if compares_carbon else "baseline"
    destination = arguments.output or DEFAULT_OUTPUT_DIR / (
        f"{kind}_comparison_{len(arms[0][2])}jobs_{arguments.nodes}nodes.csv"
    )
    print()
    print(f"metrics written          {display_path(write_rows(rows, destination))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
