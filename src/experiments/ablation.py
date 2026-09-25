"""Where the carbon saving is lost: job predictions, or the carbon forecast?

A carbon-aware scheduler running for real is wrong about two things at once. It
does not know how long a job will run or how much power it will draw, and it
does not know what the grid will do. Measured end to end, the two errors arrive
together and there is no way to tell which one cost what.

This is the ablation that separates them. The same cohort, cluster, delay
budget, decision granularity and runtime-estimate source are run four times,
changing exactly one information source at a time:

    job information x carbon information

    actual    + actual      the oracle: the whole saving that exists to be had
    actual    + forecast    only the grid is predicted
    predicted + actual      only the jobs are predicted
    predicted + forecast    the realistic scheduler, wrong about both

Every run is charged against the observed series afterwards, so the four
emissions totals are directly comparable and their differences are attributable
to the one input that moved. Writing ``C`` for total emissions and ``S`` for the
saving against the carbon-blind reference, the oracle's advantage decomposes as

    S(actual, actual) - S(predicted, forecast)
        = [C(actual, forecast) - C(actual, actual)]      carbon forecast error
        + [C(predicted, actual) - C(actual, actual)]     job model error
        + interaction

reported as shares of the saving the oracle had available. The denominator
cancels out of every difference, so the split does not depend on which
carbon-blind run is used as the reference.

Two EASY runs are included for provenance rather than for the split: job
predictions also feed the backfill runtime estimate, so the carbon-blind
baseline is not the same schedule under the two cohorts, and the CSV should say
so rather than leave it to be assumed.

Run it with ``python -m experiments.ablation``.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from math import isfinite
from pathlib import Path
import sys

from carbon_intensity.scheduling_impact import archive_reach, load_archives
from carbon_intensity.series import TimeSeriesCarbonIntensityProvider
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    EASYBackfillScheduler,
    Job,
    RuntimeEstimateSource,
    SimulationResult,
    Simulator,
    account_schedule,
    schedule_metrics,
)
from job_prediction.integration import load_prediction_cohort


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = PROJECT_ROOT / "data/processed/pm100_clean.parquet"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data/job_predictions/test_predictions.parquet"
DEFAULT_ACTUAL = PROJECT_ROOT / "data/carbon_intensity/actual/actual.json"
DEFAULT_SNAPSHOTS = PROJECT_ROOT / "data/carbon_intensity/snapshots"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/experiments"
DEFAULT_ARCHIVE = "boosted_ridge"
DEFAULT_DELAYS = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0)
INFORMATION_CONFIGURATIONS = {
    ("actual", "actual"): "A_perfect_information",
    ("predicted", "actual"): "B_job_model_error_only",
    ("actual", "forecast"): "C_forecast_error_only",
    ("predicted", "forecast"): "D_realistic",
}

#: The cells, in the order the report reads best: best information first.
CELLS = (("actual", "actual"), ("actual", "forecast"), ("predicted", "actual"),
         ("predicted", "forecast"))
#: ``schedule_metrics`` only moves these when the time origin changes, so this
#: is what the submit-time view has to repeat and nothing else.
TIMING = ("waiting_", "turnaround_", "bounded_slowdown_")


def _run(jobs: tuple[Job, ...], nodes: int, scheduler, actual) -> SimulationResult:
    """Simulate one policy and charge the outcome against the observations."""

    return account_schedule(Simulator(jobs, Cluster(nodes), scheduler).run(), jobs, actual)


def _check_causal(scheduler: CarbonAwareScheduler) -> None:
    """Every decision must have read a forecast issued by its own decision time.

    A policy that replans decides after release too, so the release is the
    wrong reference. Every consultation is also refused as it happens when the
    snapshot postdates it; this re-checks the recorded decisions after the run.
    """

    late = [
        record["job_id"]
        for record in scheduler.decision_records()
        if record["forecast_issue_time"] is not None
        and record["forecast_issue_time"] > record["decision_time"]
    ]
    if late:
        raise AssertionError(f"{len(late)} decisions read a forecast issued after they were taken")


def compare(
    actual_jobs: tuple[Job, ...],
    predicted_jobs: tuple[Job, ...],
    actual: TimeSeriesCarbonIntensityProvider,
    archive: ForecastArchive,
    *,
    nodes: int = PM100_PARTITION_1_NODES,
    max_delay: timedelta = timedelta(hours=6),
    granularity: timedelta = timedelta(minutes=15),
    easy_runs: dict[tuple[str, str], SimulationResult] | None = None,
) -> list[dict[str, object]]:
    """Score the four cells, plus a carbon-blind run per cohort, on one protocol."""

    if {job.job_id for job in actual_jobs} != {job.job_id for job in predicted_jobs}:
        raise ValueError("the two cohorts must hold the same jobs")

    # The archive's reach bounds the oracle too. Otherwise the cells would
    # differ in delay budget as well as in signal, and the split would be
    # measuring the bound rather than the forecast.
    reach = archive_reach(archive)
    estimate = RuntimeEstimateSource.SCHEDULING
    cohorts = {"actual": actual_jobs, "predicted": predicted_jobs}
    providers = {"actual": actual, "forecast": ArchiveCarbonIntensityProvider(actual, archive)}

    runs = dict(easy_runs) if easy_runs is not None else {
        (information, "none"): _run(
            jobs, nodes, EASYBackfillScheduler(runtime_estimate=estimate), actual
        )
        for information, jobs in cohorts.items()
    }
    for job_information, carbon_information in CELLS:
        scheduler = CarbonAwareScheduler(
            providers[carbon_information],
            forecast=carbon_information == "forecast",
            reach=reach,
            max_delay=max_delay,
            decision_granularity=granularity,
            runtime_estimate=estimate,
        )
        result = _run(cohorts[job_information], nodes, scheduler, actual)
        if carbon_information == "forecast":
            _check_causal(scheduler)
        runs[job_information, carbon_information] = result

    metrics = {cell: schedule_metrics(result) for cell, result in runs.items()}
    total = {cell: scored.total_emissions_gco2e for cell, scored in metrics.items()}
    reference = total["actual", "none"]
    available = reference - total["actual", "actual"]

    def lost(cell: tuple[str, str]) -> float:
        """Saving this cell gives up against the oracle, as a share of the oracle's."""

        return (total[cell] - total["actual", "actual"]) / available if available else float("nan")

    carbon_loss, job_loss = lost(("actual", "forecast")), lost(("predicted", "actual"))
    combined = lost(("predicted", "forecast"))
    attribution = {
        "carbon_forecast_loss_share": carbon_loss,
        "job_model_loss_share": job_loss,
        "interaction_loss_share": combined - carbon_loss - job_loss,
        "combined_loss_share": combined,
    }

    rows: list[dict[str, object]] = []
    for cell, result in runs.items():
        job_information, carbon_information = cell
        carbon_aware = carbon_information != "none"
        saved = reference - total[cell]
        row: dict[str, object] = {
            "configuration": f"{'carbon' if carbon_aware else 'easy'}_"
            f"{job_information}_jobs_{carbon_information}_carbon",
            "job_information": job_information,
            "carbon_information": carbon_information,
            "runtime_estimate": estimate.value,
            "nodes": nodes,
            "max_delay_hours": max_delay / timedelta(hours=1) if carbon_aware else 0.0,
            "forecast_reach_hours": reach / timedelta(hours=1) if carbon_aware else 0.0,
            "decision_granularity_minutes": (
                granularity / timedelta(minutes=1) if carbon_aware else 0.0
            ),
            "easy_reference_emissions_tco2e": reference / 1e6,
            "emissions_saved_tco2e": saved / 1e6,
            "emissions_saved_vs_easy": saved / reference if reference else float("nan"),
            "oracle_recovery": saved / available if available else float("nan"),
            "carbon_benefit_retained": saved / available if available else float("nan"),
            **metrics[cell].as_row(),
            **{
                f"submit_{key}": value
                for key, value in schedule_metrics(result, reference="submit").as_row().items()
                if key.startswith(TIMING)
            },
        }
        rows.append(row | attribution if cell == ("predicted", "forecast") else row)
    return rows


def delay_sweep(
    actual_jobs, predicted_jobs, actual, archive, *,
    delay_hours=DEFAULT_DELAYS, nodes=PM100_PARTITION_1_NODES,
    granularity=timedelta(minutes=15),
):
    """Yield the four information configurations per budget, sharing one EASY reference.

    The two carbon-blind simulations are reused across budgets. As in compare(),
    every saving uses actual-job EASY, including the predicted-job scenarios.
    """

    if not delay_hours or any(not isfinite(hours) or hours < 0 for hours in delay_hours):
        raise ValueError("delay budgets must be finite, nonnegative, and nonempty")
    easy_runs = {
        (information, "none"): _run(jobs, nodes, EASYBackfillScheduler(
            runtime_estimate=RuntimeEstimateSource.SCHEDULING,
        ), actual)
        for information, jobs in (("actual", actual_jobs), ("predicted", predicted_jobs))
    }
    for hours in delay_hours:
        rows = compare(actual_jobs, predicted_jobs, actual, archive, nodes=nodes,
                       max_delay=timedelta(hours=hours), granularity=granularity,
                       easy_runs=easy_runs)
        for row in rows:
            key = row["job_information"], row["carbon_information"]
            if key in INFORMATION_CONFIGURATIONS:
                yield {"information_configuration": INFORMATION_CONFIGURATIONS[key], **row}


def _report(rows: list[dict[str, object]]) -> None:
    header = f"{'configuration':<42}{'tCO2e':>11}{'saved':>9}{'recovery':>10}"
    print(f"{header}{'wait mean':>12}{'bsld mean':>11}{'submit wait':>14}")
    print("-" * (len(header) + 37))
    for row in rows:
        print(
            f"{row['configuration']:<42}{row['total_emissions_tco2e']:>11,.4f}"
            f"{row['emissions_saved_vs_easy']:>9.2%}{row['oracle_recovery']:>10.2%}"
            f"{row['waiting_mean_s']:>12,.1f}{row['bounded_slowdown_mean']:>11,.2f}"
            f"{row['submit_waiting_mean_s']:>14,.1f}"
        )

    realistic = next(row for row in rows if "combined_loss_share" in row)
    print()
    print("share of the oracle's saving given up by the realistic scheduler")
    for label, key in (
        ("carbon forecast error", "carbon_forecast_loss_share"),
        ("job model error", "job_model_loss_share"),
        ("interaction", "interaction_loss_share"),
        ("combined", "combined_loss_share"),
    ):
        print(f"  {label:<24}{realistic[key]:>9.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--actual", type=Path, default=DEFAULT_ACTUAL)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE,
                        help="forecast archive to replay; the model selected in "
                             "results/forecast_results.md, best on validation and on test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--max-delay-hours", type=float, default=6.0)
    parser.add_argument("--decision-granularity-minutes", type=float, default=15.0)
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        actual_jobs, predicted_jobs = load_prediction_cohort(
            arguments.workload, arguments.predictions
        )
        actual = TimeSeriesCarbonIntensityProvider.load(arguments.actual)
        archive = load_archives(
            arguments.snapshots, arguments.partition, [arguments.archive]
        )[arguments.archive]
        rows = compare(
            actual_jobs,
            predicted_jobs,
            actual,
            archive,
            nodes=arguments.nodes,
            max_delay=timedelta(hours=arguments.max_delay_hours),
            granularity=timedelta(minutes=arguments.decision_granularity_minutes),
        )
    except (OSError, KeyError, TypeError, ValueError, AssertionError) as error:
        print(f"Information ablation failed: {error}", file=sys.stderr)
        return 1

    print(f"jobs                     {len(actual_jobs):,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"maximum carbon delay     {arguments.max_delay_hours:g} h")
    print(f"decision granularity     {arguments.decision_granularity_minutes:g} min")
    print(f"forecast archive         {archive.metadata['model_name']}")
    print()
    _report(rows)

    if arguments.no_output:
        return 0
    import pandas as pd

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / f"{arguments.partition}_ablation_{arguments.archive}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nmetrics written          {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
