"""Measure how job predictions change carbon-aware scheduling decisions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from math import log1p
from pathlib import Path
from statistics import correlation

import pandas as pd

from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    Cluster,
    Distribution,
    EASYBackfillScheduler,
    Job,
    RuntimeEstimateSource,
    SimulationResult,
    Simulator,
    account_schedule,
    bounded_slowdown,
    schedule_metrics,
)
from hpc_sim.workload import load_jobs

from .integration import attach_predictions, load_prediction_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "processed" / "pm100_clean.parquet"
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "data" / "job_predictions" / "test_predictions.parquet"
)
DEFAULT_CARBON_CACHE = (
    PROJECT_ROOT
    / "data"
    / "carbon_intensity"
    / "electricity_maps_it_no_04_to_11_2020.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "job_predictions"

_DURATION_BANDS = (
    ("<10 s", 0.0, 10.0),
    ("10-60 s", 10.0, 60.0),
    ("1-10 min", 60.0, 600.0),
    ("10-60 min", 600.0, 3_600.0),
    (">=1 h", 3_600.0, float("inf")),
)


def decision_table(
    actual_jobs: tuple[Job, ...],
    predicted_jobs: tuple[Job, ...],
    actual_result: SimulationResult,
    predicted_result: SimulationResult,
    actual_targets: Mapping[object, datetime],
    predicted_targets: Mapping[object, datetime],
) -> pd.DataFrame:
    """Return one row per job with prediction errors and schedule changes."""

    expected_ids = {job.job_id for job in actual_jobs}
    predicted_by_id = {job.job_id: job for job in predicted_jobs}
    actual_records = {record.job_id: record for record in actual_result.records}
    predicted_records = {record.job_id: record for record in predicted_result.records}
    mappings = (
        predicted_by_id,
        actual_records,
        predicted_records,
        actual_targets,
        predicted_targets,
    )
    if any(set(mapping) != expected_ids for mapping in mappings):
        raise ValueError("jobs, records, and target start times must contain the same ids")

    rows: list[dict[str, object]] = []
    for job in actual_jobs:
        if job.power is None:
            raise ValueError(f"job {job.job_id} has no measured power")
        predicted_job = predicted_by_id[job.job_id]
        predicted_duration = predicted_job.predicted_duration_seconds
        predicted_power = predicted_job.predicted_average_power_watts
        if predicted_duration is None or predicted_power is None:
            raise ValueError(f"job {job.job_id} has no scheduling prediction")

        actual_record = actual_records[job.job_id]
        predicted_record = predicted_records[job.job_id]
        target_delta = (
            predicted_targets[job.job_id] - actual_targets[job.job_id]
        ).total_seconds()
        start_delta = (
            predicted_record.start_time - actual_record.start_time
        ).total_seconds()
        duration_error = predicted_duration / job.actual_duration_seconds - 1.0
        power_error = predicted_power / job.power.average_power_watts - 1.0
        rows.append(
            {
                "job_id": job.job_id,
                "actual_duration_s": job.actual_duration_seconds,
                "predicted_duration_s": predicted_duration,
                "duration_relative_error": duration_error,
                "duration_absolute_relative_error": abs(duration_error),
                "actual_average_power_w": job.power.average_power_watts,
                "predicted_average_power_w": predicted_power,
                "power_relative_error": power_error,
                "power_absolute_relative_error": abs(power_error),
                "actual_target_start_time": actual_targets[job.job_id],
                "predicted_target_start_time": predicted_targets[job.job_id],
                "target_start_delta_s": target_delta,
                "absolute_target_start_delta_s": abs(target_delta),
                "actual_start_time": actual_record.start_time,
                "predicted_start_time": predicted_record.start_time,
                "simulated_start_delta_s": start_delta,
                "absolute_simulated_start_delta_s": abs(start_delta),
                "actual_waiting_s": actual_record.waiting_seconds,
                "predicted_waiting_s": predicted_record.waiting_seconds,
                "actual_bounded_slowdown": bounded_slowdown(actual_record),
                "predicted_bounded_slowdown": bounded_slowdown(predicted_record),
                "actual_emissions_gco2e": actual_record.emissions_gco2e,
                "predicted_emissions_gco2e": predicted_record.emissions_gco2e,
            }
        )
    return pd.DataFrame(rows)


def _log_error_correlation(frame: pd.DataFrame, error_column: str) -> float:
    errors = [log1p(value) for value in frame[error_column]]
    shifts = [log1p(value) for value in frame["absolute_simulated_start_delta_s"]]
    if len(set(errors)) < 2 or len(set(shifts)) < 2:
        return 0.0
    return correlation(errors, shifts)


def impact_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    """Summarise target and realised start-time differences."""

    target = Distribution.of(frame["absolute_target_start_delta_s"].tolist())
    start = Distribution.of(frame["absolute_simulated_start_delta_s"].tolist())
    return {
        "changed_target_jobs": int((frame["target_start_delta_s"] != 0.0).sum()),
        "changed_target_fraction": float((frame["target_start_delta_s"] != 0.0).mean()),
        "absolute_target_delta_mean_s": target.mean,
        "target_delta_mean_s": float(frame["target_start_delta_s"].mean()),
        "absolute_target_delta_median_s": target.median,
        "absolute_target_delta_p95_s": target.p95,
        "absolute_target_delta_max_s": target.maximum,
        "changed_start_jobs": int((frame["simulated_start_delta_s"] != 0.0).sum()),
        "changed_start_fraction": float((frame["simulated_start_delta_s"] != 0.0).mean()),
        "absolute_start_delta_mean_s": start.mean,
        "start_delta_mean_s": float(frame["simulated_start_delta_s"].mean()),
        "absolute_start_delta_median_s": start.median,
        "absolute_start_delta_p95_s": start.p95,
        "absolute_start_delta_max_s": start.maximum,
        "duration_error_start_delta_correlation": _log_error_correlation(
            frame, "duration_absolute_relative_error"
        ),
        "power_error_start_delta_correlation": _log_error_correlation(
            frame, "power_absolute_relative_error"
        ),
    }


def duration_band_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Show whether the large relative duration errors belong to impactful jobs."""

    rows = []
    for label, lower, upper in _DURATION_BANDS:
        selected = frame[
            (frame["actual_duration_s"] >= lower)
            & (frame["actual_duration_s"] < upper)
        ]
        if selected.empty:
            continue
        shifts = Distribution.of(selected["absolute_simulated_start_delta_s"].tolist())
        rows.append(
            {
                "actual_duration": label,
                "jobs": len(selected),
                "median_duration_absolute_relative_error": selected[
                    "duration_absolute_relative_error"
                ].median(),
                "changed_target_fraction": (
                    selected["target_start_delta_s"] != 0.0
                ).mean(),
                "changed_start_fraction": (
                    selected["simulated_start_delta_s"] != 0.0
                ).mean(),
                "absolute_start_delta_mean_s": shifts.mean,
                "absolute_start_delta_median_s": shifts.median,
                "absolute_start_delta_p95_s": shifts.p95,
            }
        )
    return pd.DataFrame(rows)


def _metric_table(
    baseline_result: SimulationResult,
    actual_result: SimulationResult,
    predicted_result: SimulationResult,
    *,
    max_delay: timedelta,
    granularity: timedelta,
    summary: dict[str, int | float],
) -> pd.DataFrame:
    scenarios = (
        ("easy_actual_jobs", schedule_metrics(baseline_result)),
        ("carbon_actual_jobs", schedule_metrics(actual_result)),
        ("carbon_predicted_jobs", schedule_metrics(predicted_result)),
    )
    baseline_emissions = scenarios[0][1].total_emissions_gco2e
    actual_emissions = scenarios[1][1].total_emissions_gco2e
    available_saving = baseline_emissions - actual_emissions
    rows = []
    for scenario, metrics in scenarios:
        row = {
            "scenario": scenario,
            "max_delay_hours": (
                0.0 if scenario == "easy_actual_jobs" else max_delay.total_seconds() / 3_600.0
            ),
            "decision_granularity_minutes": granularity.total_seconds() / 60.0,
            "emissions_saved_vs_easy": (
                (baseline_emissions - metrics.total_emissions_gco2e) / baseline_emissions
            ),
            "carbon_benefit_retained": (
                (baseline_emissions - metrics.total_emissions_gco2e) / available_saving
                if available_saving
                else 0.0
            ),
            **metrics.as_row(),
        }
        if scenario == "carbon_predicted_jobs":
            row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_test_cohort(
    workload: Path,
    prediction_path: Path,
) -> tuple[tuple[Job, ...], tuple[Job, ...]]:
    predictions = load_prediction_file(prediction_path)
    loaded = load_jobs(workload, job_ids=set(predictions))
    jobs_by_id = {job.job_id: job for job in loaded}
    missing = set(predictions).difference(jobs_by_id)
    if missing:
        preview = ", ".join(str(job_id) for job_id in sorted(missing, key=str)[:5])
        raise ValueError(f"prediction ids absent from the workload: {preview}")
    actual_jobs = tuple(jobs_by_id[job_id] for job_id in predictions)
    return actual_jobs, attach_predictions(actual_jobs, predictions)


def _print_report(
    metrics: pd.DataFrame,
    summary: dict[str, int | float],
    bands: pd.DataFrame,
) -> None:
    print(
        f"{'scenario':<24}{'tCO2e':>12}{'saved':>11}"
        f"{'wait mean':>14}{'bsld mean':>14}"
    )
    print("-" * 75)
    for row in metrics.to_dict("records"):
        print(
            f"{row['scenario']:<24}{row['total_emissions_tco2e']:>12,.4f}"
            f"{row['emissions_saved_vs_easy']:>10.2%}"
            f"{row['waiting_mean_s']:>14,.1f}{row['bounded_slowdown_mean']:>14,.2f}"
        )

    actual = metrics.iloc[1]
    predicted = metrics.iloc[2]
    print()
    print(
        "predicted vs actual      "
        f"emissions {predicted.total_emissions_tco2e / actual.total_emissions_tco2e - 1:+.2%}, "
        f"waiting {predicted.waiting_mean_s / actual.waiting_mean_s - 1:+.2%}, "
        "bounded slowdown "
        f"{predicted.bounded_slowdown_mean / actual.bounded_slowdown_mean - 1:+.2%}"
    )
    print(f"carbon benefit retained  {predicted.carbon_benefit_retained:.2%}")
    print(
        "target starts changed    "
        f"{summary['changed_target_jobs']:,} ({summary['changed_target_fraction']:.2%})"
    )
    print(
        "simulated starts changed "
        f"{summary['changed_start_jobs']:,} ({summary['changed_start_fraction']:.2%})"
    )
    print(
        "absolute start delta     "
        f"mean {summary['absolute_start_delta_mean_s']:,.1f} s, "
        f"median {summary['absolute_start_delta_median_s']:,.1f} s, "
        f"p95 {summary['absolute_start_delta_p95_s']:,.1f} s, "
        f"max {summary['absolute_start_delta_max_s']:,.1f} s"
    )
    print(f"signed start delta       mean {summary['start_delta_mean_s']:+,.1f} s")
    print(
        "log-error correlation    "
        f"duration {summary['duration_error_start_delta_correlation']:+.3f}, "
        f"power {summary['power_error_start_delta_correlation']:+.3f}"
    )
    print("power counterfactual     identical targets and starts")
    print()
    print(
        bands.to_string(
            index=False,
            formatters={
                "median_duration_absolute_relative_error": "{:.2%}".format,
                "changed_target_fraction": "{:.2%}".format,
                "changed_start_fraction": "{:.2%}".format,
                "absolute_start_delta_mean_s": "{:,.1f}".format,
                "absolute_start_delta_median_s": "{:,.1f}".format,
                "absolute_start_delta_p95_s": "{:,.1f}".format,
            },
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_CARBON_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--max-delay-hours", type=float, default=6.0)
    parser.add_argument("--decision-granularity-minutes", type=float, default=15.0)
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    max_delay = timedelta(hours=arguments.max_delay_hours)
    granularity = timedelta(minutes=arguments.decision_granularity_minutes)
    actual_jobs, predicted_jobs = _load_test_cohort(
        arguments.workload, arguments.predictions
    )
    provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)

    baseline_result = account_schedule(
        Simulator(
            actual_jobs,
            Cluster(arguments.nodes),
            EASYBackfillScheduler(runtime_estimate=RuntimeEstimateSource.SCHEDULING),
        ).run(),
        actual_jobs,
        provider,
    )
    actual_scheduler = CarbonAwareScheduler(
        provider,
        max_delay=max_delay,
        decision_granularity=granularity,
        runtime_estimate=RuntimeEstimateSource.SCHEDULING,
    )
    actual_result = account_schedule(
        Simulator(actual_jobs, Cluster(arguments.nodes), actual_scheduler).run(),
        actual_jobs,
        provider,
    )
    predicted_scheduler = CarbonAwareScheduler(
        provider,
        max_delay=max_delay,
        decision_granularity=granularity,
        runtime_estimate=RuntimeEstimateSource.SCHEDULING,
    )
    predicted_result = account_schedule(
        Simulator(predicted_jobs, Cluster(arguments.nodes), predicted_scheduler).run(),
        predicted_jobs,
        provider,
    )

    # Power scales every candidate cost equally in this scheduler. This
    # counterfactual makes that invariance executable rather than assumed.
    duration_only_jobs = tuple(
        replace(job, predicted_average_power_watts=None) for job in predicted_jobs
    )
    duration_only_scheduler = CarbonAwareScheduler(
        provider,
        max_delay=max_delay,
        decision_granularity=granularity,
        runtime_estimate=RuntimeEstimateSource.SCHEDULING,
    )
    duration_only_result = Simulator(
        duration_only_jobs, Cluster(arguments.nodes), duration_only_scheduler
    ).run()
    predicted_starts = {
        record.job_id: record.start_time for record in predicted_result.records
    }
    duration_only_starts = {
        record.job_id: record.start_time for record in duration_only_result.records
    }
    if (
        predicted_starts != duration_only_starts
        or predicted_scheduler.target_start_times
        != duration_only_scheduler.target_start_times
    ):
        raise AssertionError("power predictions unexpectedly changed scheduling")

    decisions = decision_table(
        actual_jobs,
        predicted_jobs,
        actual_result,
        predicted_result,
        actual_scheduler.target_start_times,
        predicted_scheduler.target_start_times,
    )
    summary = impact_summary(decisions)
    summary["power_counterfactual_changed_target_jobs"] = 0
    summary["power_counterfactual_changed_start_jobs"] = 0
    bands = duration_band_table(decisions)
    metrics = _metric_table(
        baseline_result,
        actual_result,
        predicted_result,
        max_delay=max_delay,
        granularity=granularity,
        summary=summary,
    )

    print(f"jobs                     {len(actual_jobs):,}")
    print(f"cluster capacity         {arguments.nodes:,} nodes")
    print(f"maximum carbon delay     {arguments.max_delay_hours:g} h")
    print(f"decision granularity     {arguments.decision_granularity_minutes:g} min")
    print()
    _print_report(metrics, summary, bands)

    if arguments.no_output:
        return 0
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.output_dir / "scheduling_impact_metrics.csv"
    jobs_path = arguments.output_dir / "scheduling_impact_jobs.parquet"
    bands_path = arguments.output_dir / "scheduling_impact_duration_bands.csv"
    metrics.to_csv(metrics_path, index=False)
    decisions.to_parquet(jobs_path, index=False)
    bands.to_csv(bands_path, index=False)
    print()
    print(f"metrics written          {metrics_path}")
    print(f"job comparison written   {jobs_path}")
    print(f"duration bands written   {bands_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
