"""Evaluate carbon-intensity forecasts on the saved temporal protocol."""

from collections.abc import Callable
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import sys

from .baselines import BASELINE_PERIODS, BaselineCarbonIntensityProvider
from .forecasting import RidgeCarbonIntensityForecaster
from .protocol import TemporalProtocol
from .series import (
    FIFTEEN_MINUTES,
    CarbonIntensityForecast,
    TimeSeriesCarbonIntensityProvider,
)


REPORTED_HORIZONS = (1.0, 3.0, 6.0, 12.0, 24.0)
# Baselines first, then fitted models, so tables read in order of sophistication.
MODEL_ORDER = ("persistence", "seasonal_daily", "seasonal_weekly", "ridge_direct")


def error_metrics(totals: list) -> dict:
    count, absolute_error, squared_error, signed_error = totals
    return {
        "predicted_buckets": count,
        "mae_gco2e_per_kwh": absolute_error / count,
        "rmse_gco2e_per_kwh": sqrt(squared_error / count),
        "bias_gco2e_per_kwh": signed_error / count,
    }


def evaluate_forecast(
    actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
    get_forecast: Callable[[datetime, timedelta], CarbonIntensityForecast], *,
    partition: str = "validation", horizon: timedelta = timedelta(hours=24),
    cadence: timedelta = timedelta(hours=1),
) -> dict:
    """Score complete, equally weighted trajectories; never select on test.

    The callable uses the provider API, so subsequent models can use the same
    evaluator. Actual targets are read only after generating each forecast.
    Overlapping trajectories count separately, once per issue/bucket pair.
    Lead time is measured to the target bucket end; months use UTC issue time.
    Hours use the UTC hour of the target bucket, which is the hour a scheduler
    would be placing work into. Positive bias means overprediction.
    """
    if partition not in ("validation", "test"):
        raise ValueError("evaluation partition must be validation or test")
    for name, value in (("horizon", horizon), ("cadence", cadence)):
        if not isinstance(value, timedelta) or value <= timedelta(0) or value % FIFTEEN_MINUTES:
            raise ValueError(f"{name} must be a positive multiple of 15 minutes")
    issue_time, end = protocol.intervals[partition]
    label_delay = protocol.observation_delay if partition == "validation" else timedelta(0)
    first_issue = issue_time
    totals = [0, 0.0, 0.0, 0.0]
    by_horizon = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    by_month = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    by_hour = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    forecasts = 0
    while issue_time + horizon + label_delay <= end:
        forecast = get_forecast(issue_time, horizon)
        example = protocol.example(actual, issue_time, FIFTEEN_MINUTES, horizon, partition=partition)
        if forecast.issue_time != issue_time or tuple(s.timestamp for s in forecast.samples) != tuple(
            s.timestamp for s in example.targets
        ):
            raise ValueError("forecast must match the requested issue time and complete target grid")
        monthly = by_month[issue_time.strftime("%Y-%m")]
        for predicted, observed in zip(forecast.samples, example.targets, strict=True):
            error = predicted.intensity_gco2e_per_kwh - observed.intensity_gco2e_per_kwh
            lead_hours = (observed.timestamp + FIFTEEN_MINUTES - issue_time).total_seconds() / 3600
            for group in (totals, by_horizon[lead_hours], monthly, by_hour[observed.timestamp.hour]):
                group[0] += 1
                group[1] += abs(error)
                group[2] += error * error
                group[3] += error
        forecasts += 1
        issue_time += cadence
    if not forecasts:
        raise ValueError("partition contains no complete forecast windows")
    return {
        "partition": partition, "forecasts": forecasts,
        "first_issue_time": first_issue.isoformat(),
        "last_issue_time": (issue_time - cadence).isoformat(),
        "horizon_hours": horizon.total_seconds() / 3600,
        "cadence_minutes": cadence.total_seconds() / 60,
        **error_metrics(totals),
        "by_horizon": [{"lead_hours": lead, **error_metrics(group)} for lead, group in sorted(by_horizon.items())],
        "by_month": [{"issue_month": month, **error_metrics(group)} for month, group in sorted(by_month.items())],
        "by_hour": [{"target_hour_utc": hour, **error_metrics(group)} for hour, group in sorted(by_hour.items())],
    }


def forecasters(
    actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol, model_path: Path,
):
    """Yield every model behind the same provider call, so the formats stay comparable.

    Baselines need no fit. The ridge is fitted on train once and reused
    afterwards; delete the saved model to refit it.
    """
    for method in BASELINE_PERIODS:
        yield method, BaselineCarbonIntensityProvider(actual, protocol, method).get_forecast, None
    if model_path.exists():
        ridge = RidgeCarbonIntensityForecaster.load(model_path, actual, protocol)
    else:
        ridge = RidgeCarbonIntensityForecaster.fit(actual, protocol, model_path, **selected(model_path))
        ridge.save(model_path)
    yield ridge.metadata["model_name"], ridge.get_forecast, ridge.metadata["training_cutoff"]


def archived_forecasters(directory: Path, partition: str):
    """Yield each saved archive, so the scored forecasts are the replayed ones.

    On the shared hourly grid an archive answers ``get_forecast`` exactly as the
    model that produced it, so scoring the files removes any doubt that the
    evaluation and the scheduler see different numbers.
    """
    from .snapshots import ForecastArchive

    paths = sorted(directory.glob(f"{partition}_*.json"), key=lambda path: (
        MODEL_ORDER.index(name) if (name := path.stem[len(partition) + 1:]) in MODEL_ORDER
        else len(MODEL_ORDER), name,
    ))
    if not paths:
        raise ValueError(f"{directory} holds no {partition} forecast archives")
    for path in paths:
        archive = ForecastArchive.load(path)
        yield archive.metadata["model_name"], archive.get_forecast, archive.metadata["training_cutoff"]


def selected(model_path: Path) -> dict:
    """Fit the configuration frozen on validation, not an ad-hoc default."""
    path = model_path.parent / "selected_model.json"
    if not path.exists():
        return {}
    choice = json.loads(path.read_text(encoding="utf-8"))
    span = choice["history_span_days"]
    return {
        "alpha": choice["alpha"],
        "history_span": None if span is None else timedelta(days=span),
    }


def load_inputs(actual_path: Path, protocol_path: Path) -> tuple:
    protocol = TemporalProtocol.load(protocol_path)
    manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
    actual_hash = sha256(actual_path.read_bytes()).hexdigest()
    if manifest.get("actual_sha256") != actual_hash:
        raise ValueError("actual cache does not match the saved protocol SHA-256")
    return TimeSeriesCarbonIntensityProvider.load(actual_path), protocol, actual_hash


def write_evaluation(output_dir: Path, partition: str, rows: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {"horizon_metrics": "by_horizon", "monthly_metrics": "by_month", "hour_metrics": "by_hour"}
    tables = {
        "metrics": [{key: value for key, value in row.items() if key not in groups.values()} for row in rows],
        **{name: [{"model": row["model"], "partition": partition, **group}
                  for row in rows for group in row.get(key, [])]
           for name, key in groups.items()},
    }
    for name, table in tables.items():
        if not table:  # a comparison may group by horizon only
            continue
        with (output_dir / f"{partition}_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (output_dir / f"{partition}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )


def base_parser(description: str) -> argparse.ArgumentParser:
    """The three paths every evaluation entry point reads and writes."""

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--actual", type=Path,
                        default=root / "data/carbon_intensity/actual/actual.json")
    parser.add_argument("--protocol", type=Path,
                        default=root / "data/carbon_intensity/actual/protocol.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/carbon_intensity/forecasts")
    return parser


def report(name: str, row: dict, suffix: str = "") -> None:
    """Print one model's headline metrics and its MAE at the named horizons."""

    print(f"{name}: MAE={row['mae_gco2e_per_kwh']:.4f}, "
          f"RMSE={row['rmse_gco2e_per_kwh']:.4f}, "
          f"bias={row['bias_gco2e_per_kwh']:+.4f}{suffix}")
    leads = {group["lead_hours"]: group["mae_gco2e_per_kwh"] for group in row["by_horizon"]}
    reported = [f"+{hours:g}h={leads[hours]:.4f}" for hours in REPORTED_HORIZONS if hours in leads]
    if reported:
        print("  MAE by horizon: " + ", ".join(reported))


def provenance(args: argparse.Namespace, protocol: TemporalProtocol, actual_hash: str) -> dict:
    """The metadata every evaluation records: inputs, splits, metric conventions."""

    return {
        **protocol.metadata(),
        "actual_path": str(args.actual),
        "actual_sha256": actual_hash,
        "protocol_path": str(args.protocol),
        "metric_weighting": "each issue_time/target bucket pair has equal weight",
        "lead_time_reference": "target bucket end minus issue_time",
        "bias_definition": "prediction minus actual",
    }


def run_entry_point(label: str, work: Callable[[], None]) -> int:
    """Run one entry point, turning a bad input into a message and status 1."""

    try:
        work()
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        print(f"{label} failed: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    parser.add_argument("--snapshots", type=Path, default=None,
                        help="score saved forecast archives instead of live models")
    args = parser.parse_args()
    model_path = args.model or args.output_dir / "ridge_direct.json"

    def work() -> None:
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        models = (archived_forecasters(args.snapshots, args.partition) if args.snapshots
                  else forecasters(actual, protocol, model_path))
        for name, get_forecast, cutoff in models:
            row = {"model": name, "training_cutoff": cutoff, **evaluate_forecast(
                actual, protocol, get_forecast, partition=args.partition,
                horizon=timedelta(hours=args.horizon_hours),
                cadence=timedelta(minutes=args.cadence_minutes),
            )}
            rows.append(row)
            report(name, row)
        write_evaluation(args.output_dir, args.partition, rows, {
            **provenance(args, protocol, actual_hash), "evaluation": rows,
            "model_path": None if args.snapshots else str(model_path),
            "snapshot_dir": str(args.snapshots) if args.snapshots else None,
            "seasonal_alignment": "UTC; repeat the last observable day/week",
            "monthly_grouping": "UTC issue month",
            "hourly_grouping": "UTC hour of the target bucket",
        })
        print(f"Saved {args.partition} results to {args.output_dir}")

    return run_entry_point("Forecast evaluation", work)


if __name__ == "__main__":
    raise SystemExit(main())
