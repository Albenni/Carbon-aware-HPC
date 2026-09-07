"""Isolate the scheduling cost of carbon-intensity forecast error.

One cohort, one cluster, one delay budget, several schedules: the carbon-blind
EASY reference, the oracle reading the observations it is later scored against,
and one run per reconstructed forecast archive. Job durations and powers are the
real measured ones everywhere, so the only thing that changes between the oracle
and an archive is the signal the policy reads, and the emissions gap between
them is attributable to that archive's forecast error and to nothing else.

The headline is oracle recovery,

    (C_easy - C_forecast) / (C_easy - C_oracle),

the share of the saving available under perfect information that survives when
the future has to be predicted from what was published at decision time.

Run it with ``python -m carbon_intensity.scheduling_impact``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    Distribution,
    EASYBackfillScheduler,
    RuntimeEstimateSource,
    SimulationResult,
    Simulator,
    account_schedule,
    schedule_metrics,
)
from hpc_sim.models import Job, seconds
from hpc_sim.workload import load_jobs

from .series import TimeSeriesCarbonIntensityProvider
from .snapshots import ArchiveCarbonIntensityProvider, ForecastArchive


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = PROJECT_ROOT / "data/processed/pm100_clean.parquet"
DEFAULT_ACTUAL = PROJECT_ROOT / "data/carbon_intensity/actual/actual.json"
DEFAULT_SNAPSHOTS = PROJECT_ROOT / "data/carbon_intensity/snapshots"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/carbon_intensity/forecasts"


def archive_reach(archive: ForecastArchive) -> timedelta:
    """How far ahead a decision may look, whenever it is taken.

    A snapshot covers its own issue time plus the horizon, and a decision lands
    anywhere inside the cadence interval that follows, so in the worst case a
    whole cadence step of the trajectory is already in the past. What every
    decision can count on is therefore the horizon minus the cadence.
    """

    metadata = archive.metadata
    return timedelta(
        hours=metadata["horizon_hours"], minutes=-metadata["cadence_minutes"]
    )


def run(jobs, nodes: int, scheduler, actual) -> SimulationResult:
    """Simulate one policy and charge the outcome against the observations."""

    result = Simulator(jobs, Cluster(nodes), scheduler).run()
    return account_schedule(result, jobs, actual)


def _starts(result: SimulationResult) -> dict[object, datetime]:
    return {record.job_id: record.start_time for record in result.records}


def _deltas(subject, reference) -> dict[str, float]:
    """Distribution of the shift a run applies to the oracle's own choices."""

    shifts = [
        (subject[job_id] - instant).total_seconds()
        for job_id, instant in reference.items()
    ]
    absolute = Distribution.of([abs(value) for value in shifts])
    return {
        "changed_fraction": sum(value != 0.0 for value in shifts) / len(shifts),
        "signed_mean_s": sum(shifts) / len(shifts),
        "absolute_mean_s": absolute.mean,
        "absolute_median_s": absolute.median,
        "absolute_p95_s": absolute.p95,
        "absolute_max_s": absolute.maximum,
    }


def compare(
    jobs,
    actual: TimeSeriesCarbonIntensityProvider,
    archives: dict[str, ForecastArchive],
    *,
    nodes: int = PM100_PARTITION_1_NODES,
    max_delay: timedelta = timedelta(hours=6),
    granularity: timedelta | None = None,
) -> list[dict[str, object]]:
    """Score EASY, the oracle and every archive on the identical cohort."""

    # One bound for all of them, so the oracle is not handed a longer reach than
    # the archive it is being compared against.
    reach = min(archive_reach(archive) for archive in archives.values())
    estimate = RuntimeEstimateSource.SCHEDULING
    shared = {"decision_granularity": granularity, "runtime_estimate": estimate}

    easy = run(jobs, nodes, EASYBackfillScheduler(runtime_estimate=estimate), actual)
    oracle_scheduler = CarbonAwareScheduler(
        actual, reach=reach, max_delay=max_delay, **shared
    )
    oracle = run(jobs, nodes, oracle_scheduler, actual)

    easy_emissions = schedule_metrics(easy).total_emissions_gco2e
    available = easy_emissions - schedule_metrics(oracle).total_emissions_gco2e
    oracle_starts = _starts(oracle)
    oracle_targets = dict(oracle_scheduler.target_start_times)

    by_id = {job.job_id: job for job in jobs}
    rows: list[dict[str, object]] = []

    def record(label: str, result: SimulationResult, scheduler=None) -> None:
        metrics = schedule_metrics(result)
        saved = easy_emissions - metrics.total_emissions_gco2e
        row: dict[str, object] = {
            "configuration": label,
            "max_delay_hours": 0.0 if scheduler is None else max_delay / timedelta(hours=1),
            "forecast_reach_hours": 0.0 if scheduler is None else reach / timedelta(hours=1),
            "emissions_saved_vs_easy": saved / easy_emissions,
            "oracle_recovery": saved / available if available else float("nan"),
            **metrics.as_row(),
        }
        if scheduler is not None:
            row |= {
                f"start_{key}": value
                for key, value in _deltas(_starts(result), oracle_starts).items()
            }
            row |= {
                f"target_{key}": value
                for key, value in _deltas(
                    dict(scheduler.target_start_times), oracle_targets
                ).items()
            }
        rows.append(row)

    record("easy", easy)
    record("carbon_oracle", oracle, oracle_scheduler)
    for name, archive in archives.items():
        scheduler = CarbonAwareScheduler(
            ArchiveCarbonIntensityProvider(actual, archive),
            forecast=True,
            reach=reach,
            max_delay=max_delay,
            **shared,
        )
        result = run(jobs, nodes, scheduler, actual)
        # Every decision must have read a forecast published before the job was
        # released: the leak this whole experiment exists to rule out.
        late = [
            job_id
            for job_id, issue in scheduler.forecast_issue_times.items()
            if issue > by_id[job_id].release_time
        ]
        if late:
            raise AssertionError(f"{len(late)} decisions read a future forecast")
        record(f"carbon_forecast_{name}", result, scheduler)
    return rows


def load_archives(directory: Path, partition: str, names) -> dict[str, ForecastArchive]:
    """Load ``<partition>_<name>.json`` archives, or every one in the directory."""

    paths = (
        [directory / f"{partition}_{name}.json" for name in names]
        if names
        else sorted(directory.glob(f"{partition}_*.json"))
    )
    if not paths:
        raise ValueError(f"no {partition} forecast archives in {directory}")
    return {path.stem[len(partition) + 1 :]: ForecastArchive.load(path) for path in paths}


def _report(rows) -> None:
    header = f"{'configuration':<34}{'tCO2e':>11}{'saved':>9}{'recovery':>10}"
    print(f"{header}{'wait mean':>12}{'bsld mean':>11}{'start p95':>12}")
    print("-" * len(header + " " * 35))
    for row in rows:
        recovery = row["oracle_recovery"]
        print(
            f"{row['configuration']:<34}{row['total_emissions_tco2e']:>11,.4f}"
            f"{row['emissions_saved_vs_easy']:>9.2%}"
            f"{recovery:>10.2%}"
            f"{row['waiting_mean_s']:>12,.1f}{row['bounded_slowdown_mean']:>11,.2f}"
            f"{row.get('start_absolute_p95_s', 0.0):>12,.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--actual", type=Path, default=DEFAULT_ACTUAL)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--archive", action="append", metavar="NAME",
                        help="archive to include, repeatable; default is all of them")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--max-delay-hours", type=float, default=6.0)
    parser.add_argument("--decision-granularity-minutes", type=float)
    parser.add_argument("--limit", type=int, help="first N jobs, for a quick run")
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        jobs = load_jobs(arguments.workload, limit=arguments.limit)
        actual = TimeSeriesCarbonIntensityProvider.load(arguments.actual)
        archives = load_archives(
            arguments.snapshots, arguments.partition, arguments.archive
        )
        rows = compare(
            jobs,
            actual,
            archives,
            nodes=arguments.nodes,
            max_delay=timedelta(hours=arguments.max_delay_hours),
            granularity=(
                None
                if arguments.decision_granularity_minutes is None
                else timedelta(minutes=arguments.decision_granularity_minutes)
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, AssertionError) as error:
        print(f"Forecast scheduling comparison failed: {error}", file=sys.stderr)
        return 1

    print(f"jobs                     {len(jobs):,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"maximum carbon delay     {arguments.max_delay_hours:g} h")
    print()
    _report(rows)

    if arguments.no_output:
        return 0
    import pandas as pd

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / f"{arguments.partition}_scheduling_impact.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nmetrics written          {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
