"""Compare simple predictors and saved models on the existing temporal test sets."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd

from carbon_intensity.baselines import BASELINE_PERIODS, BaselineCarbonIntensityProvider
from carbon_intensity.evaluate import (
    archived_forecasters,
    evaluate_forecast,
    load_inputs,
    provenance,
    run_entry_point,
)
from job_prediction import (
    AVERAGE_POWER_WATTS,
    DURATION_SECONDS,
    JOB_ID,
    load_job_data,
    load_prediction_file,
    temporal_split,
)
from job_prediction.data import completed_by
from job_prediction.evaluation import baseline_estimates, job_metric_row, runtime_band_metrics
from experiments.ablation import DEFAULT_ARCHIVE, DEFAULT_DELAYS, delay_sweep
from hpc_sim import PM100_PARTITION_1_NODES
from common import PROJECT_ROOT


JOB_DIR = PROJECT_ROOT / "data/job_predictions"
CARBON_DIR = PROJECT_ROOT / "data/carbon_intensity"


def job_comparison(split, runs: dict[str, Path], *, power_by_nodes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align by job ID and reject different cohorts or temporal split metadata."""

    expected_split = {
        "train_count": len(split.train),
        "validation_count": len(split.validation),
        "test_count": len(split.test),
        "validation_start": split.train_until.isoformat(),
        "test_start": split.validation_until.isoformat(),
    }
    rows, bands = [], []

    def record(name, estimate):
        rows.append({"model": name, **job_metric_row(split.test, *estimate)})
        if estimate[1] == DURATION_SECONDS:
            bands.extend({"model": name, "unit": "s", **row} for row in runtime_band_metrics(
                split.test[DURATION_SECONDS], estimate[3],
            ).to_dict("records"))

    for estimate in baseline_estimates(split, runtime_by_nodes=True, power_by_nodes=power_by_nodes):
        record(estimate[0], estimate)
    job_ids = split.test[JOB_ID].tolist()
    for name, directory in runs.items():
        metadata = json.loads((directory / "job_models.json").read_text(encoding="utf-8"))["metadata"]
        if metadata["split"] != expected_split:
            raise ValueError(f"{name}: saved model uses a different temporal split")
        predictions = load_prediction_file(directory / "test_predictions.parquet")
        if set(predictions) != set(job_ids):
            raise ValueError(f"{name}: predictions must contain exactly the temporal test job IDs")
        for target, unit in ((DURATION_SECONDS, "s"), (AVERAGE_POWER_WATTS, "W")):
            predicted = [getattr(predictions[job_id], target) for job_id in job_ids]
            record(name, ("saved prediction", target, unit, predicted))
    return (
        pd.DataFrame(rows).assign(
            partition="test", test_start=split.validation_until.isoformat(),
            test_last_submit=split.test.submit_time.max().isoformat(),
        ),
        pd.DataFrame(bands).assign(partition="test"),
    )


def carbon_comparison(actual, protocol, actual_hash, snapshots, *, horizon, cadence) -> list[dict]:
    """Use one issue/target grid and the existing evaluator for every forecast."""

    rows = []
    for method in BASELINE_PERIODS:
        baseline = BaselineCarbonIntensityProvider(actual, protocol, method)
        rows.append({"model": method, "training_cutoff": None, **evaluate_forecast(
            actual, protocol, baseline.get_forecast,
            partition="test", horizon=horizon, cadence=cadence,
        )})
    for name, get_forecast, cutoff in archived_forecasters(
        snapshots, "test", protocol=protocol, actual_hash=actual_hash,
    ):
        if name in BASELINE_PERIODS:
            continue  # Recompute causal baselines even when their archives are absent.
        rows.append({"model": name, "training_cutoff": cutoff, **evaluate_forecast(
            actual, protocol, get_forecast,
            partition="test", horizon=horizon, cadence=cadence,
        )})
    return rows


def comparison_summary(jobs, carbon, horizons, scheduler=None) -> str:
    """Report test results without selecting or changing any serving model."""

    lines = []
    for target, name, label in ((DURATION_SECONDS, "gradient", "Gradient runtime"),
                                (AVERAGE_POWER_WATTS, "gradient", "Gradient power"),
                                (AVERAGE_POWER_WATTS, "ridge", "Ridge power")):
        table = jobs.loc[jobs.target == target].set_index("model")
        if name not in table.index:
            lines.append(f"{label}: not evaluated.")
            continue
        baselines = table.loc[table.estimate != "saved prediction"]
        wins = [f"{label_} {'yes' if (table.loc[name, metric] < baselines[metric]).all() else 'no'}"
                for label_, metric in (("MAE", "mae"), ("RMSE", "rmse"),
                                       ("WARE", "weighted_absolute_relative_error"))]
        lines.append(f"{label} beats every simple baseline: {', '.join(wins)}.")
    for label, table in [("overall", carbon), *list(horizons.groupby("horizon_band", sort=False))]:
        winners = []
        for title, metric in (("MAE", "mae_gco2e_per_kwh"), ("RMSE", "rmse_gco2e_per_kwh")):
            row = table.loc[table[metric].idxmin()]
            winners.append(f"{title} {row['model']} ({row[metric]:.3f})")
        lines.append(f"CI lowest error, {label}: {', '.join(winners)} gCO2e/kWh.")
    if scheduler is not None:
        realistic = scheduler.loc[scheduler.information_configuration == "D_realistic"]
        retained = [f"{row.max_delay_hours:g} h: " + (
            "n/a (zero perfect-information saving)" if pd.isna(row.carbon_benefit_retained)
            else f"{row.carbon_benefit_retained:.1%}"
        ) for row in realistic.itertuples()]
        lines.append("Realistic carbon benefit retained: " + "; ".join(retained) + ".")
    else:
        lines.append("Scheduler sweep not run; add --scheduler-sweep to evaluate retention.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=PROJECT_ROOT / "data/processed/pm100_clean.parquet")
    parser.add_argument("--job-run", action="append", metavar="LABEL=DIRECTORY",
                        help="saved job model and test predictions; repeatable (default: ridge and gradient)")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--power-by-nodes", action=argparse.BooleanOptionalAction, default=True,
                        help="also compare power medians grouped by num_nodes_alloc (default: enabled)")
    parser.add_argument("--actual", type=Path, default=CARBON_DIR / "actual/actual.json")
    parser.add_argument("--protocol", type=Path, default=CARBON_DIR / "actual/protocol.json")
    parser.add_argument("--snapshots", type=Path, default=CARBON_DIR / "snapshots")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/prediction_baselines")
    parser.add_argument("--scheduler-sweep", action="store_true",
                        help="also run the four information configurations at each delay budget")
    parser.add_argument("--scheduler-job-model", default="gradient", help="job-run label fixed before evaluation")
    parser.add_argument("--scheduler-archive", default=DEFAULT_ARCHIVE, help="forecast fixed before evaluation")
    parser.add_argument("--max-delay-hours", nargs="+", type=float, default=DEFAULT_DELAYS)
    parser.add_argument("--nodes", type=int, default=PM100_PARTITION_1_NODES)
    parser.add_argument("--decision-granularity-minutes", type=float, default=15.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    def work() -> None:
        runs = {"ridge": JOB_DIR / "ridge_baseline", "gradient": JOB_DIR}
        if args.job_run:
            runs = {}
            for item in args.job_run:
                name, separator, directory = item.partition("=")
                if not separator or not name.strip() or not directory or name in runs:
                    raise ValueError("--job-run requires unique LABEL=DIRECTORY entries")
                runs[name] = Path(directory)
        data = load_job_data(args.workload)
        split = temporal_split(data, train_fraction=args.train_fraction,
                               validation_fraction=args.validation_fraction)
        jobs, bands = job_comparison(split, runs, power_by_nodes=args.power_by_nodes)
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        carbon = carbon_comparison(
            actual, protocol, actual_hash, args.snapshots,
            horizon=timedelta(hours=args.horizon_hours),
            cadence=timedelta(minutes=args.cadence_minutes),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target, name in ((DURATION_SECONDS, "runtime"), (AVERAGE_POWER_WATTS, "power")):
            table = jobs.loc[jobs.target == target]
            table.to_csv(args.output_dir / f"test_{name}_comparison.csv", index=False)
            print(table[["model", "count", "mae", "rmse", "weighted_absolute_relative_error",
                         "bias", "relative_bias"]].to_string(index=False), flush=True)
        bands.to_csv(args.output_dir / "test_runtime_band_comparison.csv", index=False)
        carbon_table = pd.DataFrame([
            {key: value for key, value in row.items() if not key.startswith("by_")}
            for row in carbon
        ])
        carbon_table.to_csv(args.output_dir / "test_carbon_intensity_comparison.csv", index=False)
        horizons = pd.DataFrame([
            {"model": row["model"], "partition": "test", **band}
            for row in carbon for band in row["by_horizon_band"]
        ])
        horizons.to_csv(args.output_dir / "test_carbon_intensity_horizon_comparison.csv", index=False)
        scheduler = None
        if args.scheduler_sweep:
            from carbon_intensity.snapshots import ForecastArchive
            from job_prediction.integration import load_prediction_cohort

            predictions = runs[args.scheduler_job_model] / "test_predictions.parquet"
            archive = ForecastArchive.load(args.snapshots / f"test_{args.scheduler_archive}.json")
            actual_jobs, predicted_jobs = load_prediction_cohort(args.workload, predictions)
            print(f"Scheduler sweep: {len(actual_jobs):,} jobs, {args.nodes} nodes, "
                  f"{args.scheduler_job_model} jobs, {args.scheduler_archive} CI", flush=True)
            scheduler_rows = []
            for row in delay_sweep(actual_jobs, predicted_jobs, actual, archive,
                                   nodes=args.nodes, delay_hours=args.max_delay_hours,
                                   granularity=timedelta(minutes=args.decision_granularity_minutes)):
                scheduler_rows.append({"job_model": args.scheduler_job_model,
                                       "forecast_model": args.scheduler_archive, **row})
                print(f"  {row['max_delay_hours']:g} h {row['information_configuration']}: "
                      f"{row['total_emissions_tco2e']:.6f} tCO2e", flush=True)
            scheduler = pd.DataFrame(scheduler_rows).sort_values(
                ["max_delay_hours", "information_configuration"], ignore_index=True,
            )
            scheduler.to_csv(args.output_dir / "test_scheduler_information_comparison.csv", index=False)
        train = completed_by(split.train, split.train_until, period_name="training")
        metadata = {
            "workload": str(args.workload),
            "job_runs": {name: str(directory) for name, directory in runs.items()},
            "job_split": {
                "counts": split.counts, "train_until": split.train_until.isoformat(),
                "validation_until": split.validation_until.isoformat(),
            },
            "power_baseline": {
                "fit_partition": "train", "fit_count": len(train),
                "completion_cutoff": split.train_until.isoformat(),
                "median_watts": float(train[AVERAGE_POWER_WATTS].median()),
                "group_by": "num_nodes_alloc" if args.power_by_nodes else None,
                "fallback": "training global median",
            },
            "runtime_baseline": {
                "fit_partition": "train", "fit_count": len(train),
                "completion_cutoff": split.train_until.isoformat(),
                "median_seconds": float(train[DURATION_SECONDS].median()),
                "group_by": "num_nodes_alloc", "fallback": "training global median",
            },
            "bias_definition": "mean(prediction - actual); relative bias divides the signed sum by actual sum",
            "runtime_bands": "actual seconds only: [0,300), [300,1800), [1800,3600), [3600,14400], (14400,infinity)",
            "scheduler": None if scheduler is None else {
                "job_model": args.scheduler_job_model, "forecast_model": args.scheduler_archive,
                "delay_hours": list(args.max_delay_hours), "nodes": args.nodes,
                "decision_granularity_minutes": args.decision_granularity_minutes,
                "common_forecast_reach_hours": float(scheduler.forecast_reach_hours.iloc[0]),
                "easy_reference": "actual-job EASY, scheduling runtime estimates",
                "retention": "(EASY emissions - scenario emissions) / (EASY emissions - A emissions), per delay; null if zero",
                "accounting": "measured job profiles and actual CI for every run",
                "selection": "existing fixed job model and forecast; no selection from these test results",
            },
            "carbon_intensity": {**provenance(args, protocol, actual_hash), "evaluation": carbon},
        }
        (args.output_dir / "comparison_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        summary = comparison_summary(jobs, carbon_table, horizons, scheduler)
        (args.output_dir / "comparison_summary.txt").write_text(summary + "\n", encoding="utf-8")
        print(summary)
        print(f"Saved comparison tables to {args.output_dir}")

    return run_entry_point("Predictive baseline comparison", work)


if __name__ == "__main__":
    raise SystemExit(main())
