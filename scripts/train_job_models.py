"""Train and evaluate submission-time PM100 job prediction models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


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
    banded_metrics,
    fit_job_predictor,
    load_job_data,
    long_job_metrics,
    regression_metrics,
    temporal_split,
)
from job_prediction.features import FeatureSpec
from job_prediction.gradient import TargetConfig, fit_gradient_predictor
from common import PROJECT_ROOT


# Frozen on the development period only; see src/job_prediction/README.md for
# the validation evidence behind every field. Nothing here was chosen by
# looking at the test partition.
SELECTED_FEATURE_SPEC = FeatureSpec(
    groups=("base", "derived", "user_hist", "signature_hist"),
    name="base+derived+user_hist+signature_hist",
)
SELECTED_CONFIGS = {
    DURATION_SECONDS: TargetConfig(
        model="hist_gbr",
        transform="log1p",
        loss="absolute_error",
        weight="log_duration",
    ),
    AVERAGE_POWER_WATTS: TargetConfig(
        model="hist_gbr",
        transform="log1p",
        loss="absolute_error",
        weight="none",
    ),
    # Kept only as the benchmark arm of the composition comparison: a direct
    # energy model loses badly to duration x power on validation.
    ENERGY_KWH: TargetConfig(
        model="hist_gbr",
        transform="log1p",
        loss="absolute_error",
        weight="none",
    ),
}


DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "processed" / "pm100_clean.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "job_predictions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("ridge", "gradient"),
        default="gradient",
        help="ridge reproduces the original baseline; gradient is the selected model",
    )
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


def _metric_rows(frame, components, scheduling, composition) -> list[dict[str, object]]:
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
    if composition is PredictionComposition.DURATION_POWER:
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

    durations = frame[DURATION_SECONDS].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for estimate, target, unit, predicted in estimates:
        actual = frame[target].to_numpy(dtype=float)
        metrics = regression_metrics(actual, predicted)
        long_jobs = long_job_metrics(actual, predicted, durations=durations)
        rows.append(
            {
                "estimate": estimate,
                "target": target,
                "unit": unit,
                **metrics.as_row(),
                "long_job_count": long_jobs.count,
                "long_job_mae": long_jobs.mae,
                "long_job_wape": long_jobs.weighted_absolute_relative_error,
                "long_job_bias": long_jobs.bias,
            }
        )
    return rows


def _band_rows(frame, components) -> "list[dict[str, object]]":
    """Per-band duration accuracy, which is where WAPE is actually decided."""

    durations = frame[DURATION_SECONDS].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for target, unit, predicted in (
        (DURATION_SECONDS, "s", components[DURATION_SECONDS]),
        (AVERAGE_POWER_WATTS, "W", components[AVERAGE_POWER_WATTS]),
    ):
        table = banded_metrics(
            frame[target].to_numpy(dtype=float), predicted, by=durations
        )
        table.insert(0, "unit", unit)
        table.insert(0, "target", target)
        rows.extend(table.to_dict("records"))
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
    if arguments.model == "ridge":
        print(
            "selected ridge alphas     "
            + ", ".join(
                f"{target}={alpha:g}"
                for target, alpha in training.selected_alphas.items()
            )
        )
    else:
        print(f"feature groups           {', '.join(training.feature_groups)}")
        for target, config in training.configs.items():
            row = config.as_row()
            print(
                f"  {target:<22} {row['model']}, {row['transform']}, "
                f"{row['loss']}, weight={row['weight']}"
            )
    print("validation energy MAE    " + ", ".join(
        f"{composition.value}={mae:,.4f} kWh"
        for composition, mae in training.validation_energy_mae.items()
    ))
    print()
    print(
        f"{'test estimate':<25}{'MAE':>13}{'RMSE':>13}{'WAPE':>9}"
        f"{'median rel.':>13}{'bias':>13}{'p95 abs':>13}{'>=1h WAPE':>11}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['estimate']:<25}"
            f"{row['mae']:>10,.2f} {row['unit']:<2}"
            f"{row['rmse']:>10,.2f} {row['unit']:<2}"
            f"{row['weighted_absolute_relative_error']:>8.2%}"
            f"{row['median_absolute_relative_error']:>13.2%}"
            f"{row['bias']:>10,.2f} {row['unit']:<2}"
            f"{row['p95_absolute_error']:>10,.2f} {row['unit']:<2}"
            f"{row['long_job_wape']:>11.2%}"
        )


def _write_outputs(arguments, split, training, rows, bands, predictions):
    import pyarrow
    import pyarrow.parquet as parquet

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    is_ridge = arguments.model == "ridge"
    model_path = output_dir / ("job_models.json" if is_ridge else "job_models.joblib")
    predictions_path = output_dir / "test_predictions.parquet"
    metrics_path = output_dir / "test_metrics.csv"
    bands_path = output_dir / "test_duration_bands.csv"

    metadata = {
        "source": str(arguments.workload),
        "model": arguments.model,
        "submission_features": list(SUBMISSION_FEATURES),
        "split": {
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "test_count": len(split.test),
            "validation_start": split.train_until.isoformat(),
            "test_start": split.validation_until.isoformat(),
        },
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
    if is_ridge:
        metadata["selected_alphas"] = dict(training.selected_alphas)
    else:
        metadata["feature_groups"] = list(training.feature_groups)
        metadata["target_configs"] = {
            target: config.as_row() for target, config in training.configs.items()
        }
    training.predictor.save(model_path, metadata=metadata)

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

    for path, table in ((metrics_path, rows), (bands_path, bands)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return model_path, predictions_path, metrics_path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _test_estimates(arguments, data, split, training):
    """Predict the frozen test rows exactly once, for either model family."""

    if arguments.model == "ridge":
        return (
            training.predictor.predict_components(split.test),
            training.predictor.predict(split.test),
        )

    # The boosted predictor needs the whole trace: a test job's causal history
    # is made of the earlier jobs, which live in the development partitions.
    import numpy as np

    position = pd.Series(np.arange(len(data)), index=data[JOB_ID].to_numpy())
    rows = position.loc[split.test[JOB_ID].to_numpy()].to_numpy()
    components = training.predictor.predict_components(data)
    scheduling = training.predictor.predict(data)
    return (
        {target: values[rows] for target, values in components.items()},
        scheduling.iloc[rows].reset_index(drop=True),
    )


def main() -> int:
    arguments = build_parser().parse_args()
    data = load_job_data(arguments.workload, batch_size=arguments.batch_size)
    split = temporal_split(
        data,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
    )
    if arguments.model == "ridge":
        training = fit_job_predictor(split, alphas=tuple(arguments.ridge_alphas))
    else:
        training = fit_gradient_predictor(
            data,
            split,
            spec=SELECTED_FEATURE_SPEC,
            configs=SELECTED_CONFIGS,
        )
    components, scheduling = _test_estimates(arguments, data, split, training)
    rows = _metric_rows(
        split.test, components, scheduling, training.predictor.composition
    )
    bands = _band_rows(split.test, components)
    _print_report(arguments, split, training, rows)
    print()
    print("duration accuracy by actual duration band")
    print(
        pd.DataFrame(bands)
        .query("target == @DURATION_SECONDS")[
            [
                "band",
                "count",
                "mae",
                "weighted_absolute_relative_error",
                "median_absolute_relative_error",
                "bias",
                "p95_absolute_error",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:,.3f}")
    )

    if not arguments.no_output:
        model, predictions, metrics = _write_outputs(
            arguments, split, training, rows, bands, scheduling
        )
        print()
        print(f"model written            {_display(model)}")
        print(f"test predictions written {_display(predictions)}")
        print(f"metrics written          {_display(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
