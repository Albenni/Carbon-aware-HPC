"""Put two prediction models side by side on the same held-out cohort.

``job_prediction.scheduling_impact`` compares one prediction file against EASY
and against the actual-value carbon schedule. Running it once per model leaves
two directories of artifacts; this reads them back and prints the four-way
table, including the jobs of at least one hour, which carry almost all of the
workload's runtime seconds and almost all of the carbon decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "data" / "job_predictions"
LONG_JOB_SECONDS = 3_600.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL=DIRECTORY",
        help="a scheduling-impact output directory to include, repeatable",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "model_comparison.csv")
    parser.add_argument("--no-output", action="store_true")
    return parser


def _runs(arguments) -> dict[str, Path]:
    if not arguments.run:
        return {
            "ridge baseline": DEFAULT_DIR / "ridge_baseline",
            "gradient boosting": DEFAULT_DIR,
        }
    runs: dict[str, Path] = {}
    for item in arguments.run:
        label, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--run expects LABEL=DIRECTORY, got {item!r}")
        runs[label] = Path(directory)
    return runs


def _agreement(jobs: pd.DataFrame) -> dict[str, float]:
    """How often the predicted schedule reproduces the actual-value one."""

    absolute = jobs["absolute_simulated_start_delta_s"]
    return {
        "jobs": len(jobs),
        "duration_wape": (
            (jobs.predicted_duration_s - jobs.actual_duration_s).abs().sum()
            / jobs.actual_duration_s.sum()
        ),
        "target_agreement": float((jobs.target_start_delta_s == 0.0).mean()),
        "start_agreement": float((jobs.simulated_start_delta_s == 0.0).mean()),
        "start_delta_mean_s": float(absolute.mean()),
        "start_delta_median_s": float(absolute.median()),
        "start_delta_p95_s": float(absolute.quantile(0.95)),
        "start_delta_p99_s": float(absolute.quantile(0.99)),
        "start_delta_max_s": float(absolute.max()),
        "waiting_mean_s": float(jobs.predicted_waiting_s.mean()),
        "waiting_mean_actual_s": float(jobs.actual_waiting_s.mean()),
        "bounded_slowdown_mean": float(jobs.predicted_bounded_slowdown.mean()),
        "bounded_slowdown_mean_actual": float(jobs.actual_bounded_slowdown.mean()),
        "emissions_tco2e": float(jobs.predicted_emissions_gco2e.sum() / 1e6),
        "emissions_tco2e_actual": float(jobs.actual_emissions_gco2e.sum() / 1e6),
    }


def main() -> int:
    arguments = build_parser().parse_args()
    runs = _runs(arguments)

    scenarios: list[dict[str, object]] = []
    agreements: list[dict[str, object]] = []
    for label, directory in runs.items():
        metrics = pd.read_csv(directory / "scheduling_impact_metrics.csv")
        jobs = pd.read_parquet(directory / "scheduling_impact_jobs.parquet")
        if not scenarios:
            for row in metrics.to_dict("records")[:2]:
                scenarios.append({"configuration": row["scenario"], **row})
        predicted = metrics.to_dict("records")[2]
        scenarios.append({"configuration": f"carbon_predicted ({label})", **predicted})
        long_jobs = jobs[jobs.actual_duration_s >= LONG_JOB_SECONDS]
        agreements.append({"model": label, "cohort": "all", **_agreement(jobs)})
        agreements.append({"model": label, "cohort": ">=1 h", **_agreement(long_jobs)})

    table = pd.DataFrame(scenarios)
    columns = [
        "configuration",
        "total_emissions_tco2e",
        "emissions_saved_vs_easy",
        "carbon_benefit_retained",
        "waiting_mean_s",
        "bounded_slowdown_mean",
    ]
    print("=== scheduling outcome on the held-out cohort ===")
    print(
        table[[column for column in columns if column in table]].to_string(
            index=False, float_format=lambda value: f"{value:,.4f}"
        )
    )

    agreement = pd.DataFrame(agreements)
    print("\n=== prediction agreement with the actual-value carbon schedule ===")
    print(agreement.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    if arguments.no_output:
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output, index=False)
    agreement.to_csv(
        arguments.output.with_name(arguments.output.stem + "_agreement.csv"), index=False
    )
    print(f"\ncomparison written       {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
