"""Train and evaluate submission-time PM100 job prediction models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from job_prediction import (
    AVERAGE_POWER_WATTS,
    DEFAULT_RIDGE_ALPHAS,
    DURATION_SECONDS,
    ENERGY_KWH,
    JOB_ID,
    PREDICTED_AVERAGE_POWER_WATTS,
    PREDICTED_DURATION_SECONDS,
    PREDICTED_ENERGY_KWH,
    PredictionComposition,
    SUBMISSION_FEATURES,
    fit_job_predictor,
    load_job_data,
    regression_metrics,
    temporal_split,
)


DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "processed" / "pm100_clean.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "job_predictions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=DEFAULT_RIDGE_ALPHAS,
        help="regularisation values selected on the validation period",
    )
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="evaluate without writing the model, metrics, or test predictions",
    )
    return parser


def _metric_rows(frame, predictor) -> list[dict[str, object]]:
    components = predictor.predict_components(frame)
    scheduling = predictor.predict(frame)
    estimates: list[tuple[str, str, str, object]] = [
        ("duration model", DURATION_SECONDS, "s", components[DURATION_SECONDS]),
        (
            "average-power model",
            AVERAGE_POWER_WATTS,
            "W",
            components[AVERAGE_POWER_WATTS],
        ),
        ("energy model", ENERGY_KWH, "kWh", components[ENERGY_KWH]),
    ]
    if predictor.composition is PredictionComposition.DURATION_POWER:
        estimates.append(
            (
                "composed energy",
                ENERGY_KWH,
                "kWh",
                scheduling[PREDICTED_ENERGY_KWH],
            )
        )
    else:
        estimates.append(
            (
                "composed average power",
                AVERAGE_POWER_WATTS,
                "W",
                scheduling[PREDICTED_AVERAGE_POWER_WATTS],
            )
        )

    rows: list[dict[str, object]] = []
    for estimate, target, unit, predicted in estimates:
        metrics = regression_metrics(frame[target], predicted)
        rows.append(
            {
                "estimate": estimate,
                "target": target,
                "unit": unit,
                **metrics.as_row(),
            }
        )
    return rows


def _print_report(arguments, split, training, rows) -> None:
    train_count, validation_count, test_count = split.counts
    print(f"workload                 {arguments.workload.name}")
    print(
        "chronological split       "
        f"{train_count:,} train / {validation_count:,} validation / "
        f"{test_count:,} test"
    )
    print(f"validation starts        {split.train_until.isoformat()}")
    print(f"test starts              {split.validation_until.isoformat()}")
    print(
        "targets available         "
        f"{training.fit_counts['train']:,} train / "
        f"{training.fit_counts['validation']:,} validation / "
        f"{training.fit_counts['development']:,} final refit"
    )
    print(f"prediction composition   {training.predictor.composition.value}")
    print(
        "selected ridge alphas     "
        + ", ".join(
            f"{target}={alpha:g}"
            for target, alpha in training.selected_alphas.items()
        )
    )
    print("validation energy MAE    " + ", ".join(
        f"{composition.value}={mae:,.4f} kWh"
        for composition, mae in training.validation_energy_mae.items()
    ))
    print()
    print(
        f"{'test estimate':<27}{'MAE':>14}{'RMSE':>14}"
        f"{'WAPE':>11}{'mean rel.':>13}{'median rel.':>14}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{row['estimate']:<27}"
            f"{row['mae']:>11,.3f} {row['unit']:<3}"
            f"{row['rmse']:>11,.3f} {row['unit']:<3}"
            f"{row['weighted_absolute_relative_error']:>10.2%}"
            f"{row['mean_absolute_relative_error']:>12.2%}"
            f"{row['median_absolute_relative_error']:>14.2%}"
        )


def _write_outputs(arguments, split, training, rows) -> tuple[Path, Path, Path]:
    import pyarrow
    import pyarrow.parquet as parquet

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "job_models.json"
    predictions_path = output_dir / "test_predictions.parquet"
    metrics_path = output_dir / "test_metrics.csv"

    metadata = {
        "source": str(arguments.workload),
        "submission_features": list(SUBMISSION_FEATURES),
        "split": {
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "test_count": len(split.test),
            "validation_start": split.train_until.isoformat(),
            "test_start": split.validation_until.isoformat(),
        },
        "selected_alphas": dict(training.selected_alphas),
        "fit_counts": dict(training.fit_counts),
        "validation_energy_mae_kwh": {
            composition.value: value
            for composition, value in training.validation_energy_mae.items()
        },
        "target_definitions": {
            DURATION_SECONDS: "run_time",
            AVERAGE_POWER_WATTS: "duration-weighted node power profile mean",
            ENERGY_KWH: "integrated measured node power profile",
        },
    }
    training.predictor.save(model_path, metadata=metadata)

    predictions = training.predictor.predict(split.test)
    # Test artifacts intentionally contain no actual targets. Loading this file
    # cannot expose an outcome to a scheduler by accident.
    prediction_table = pyarrow.Table.from_pandas(
        predictions[
            [
                JOB_ID,
                PREDICTED_DURATION_SECONDS,
                PREDICTED_AVERAGE_POWER_WATTS,
                PREDICTED_ENERGY_KWH,
            ]
        ],
        preserve_index=False,
    )
    parquet.write_table(prediction_table, predictions_path)

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return model_path, predictions_path, metrics_path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    arguments = build_parser().parse_args()
    data = load_job_data(arguments.workload, batch_size=arguments.batch_size)
    split = temporal_split(
        data,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
    )
    training = fit_job_predictor(split, alphas=tuple(arguments.ridge_alphas))
    rows = _metric_rows(split.test, training.predictor)
    _print_report(arguments, split, training, rows)

    if not arguments.no_output:
        model, predictions, metrics = _write_outputs(
            arguments,
            split,
            training,
            rows,
        )
        print()
        print(f"model written            {_display(model)}")
        print(f"test predictions written {_display(predictions)}")
        print(f"metrics written          {_display(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
