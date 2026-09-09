"""Figures for the forecast comparison, drawn from the saved snapshot archives.

Every panel is computed from the same archives the evaluation tables score, so a
figure can never disagree with the numbers next to it. Nothing here selects or
fits anything: it reads what the replayed scheduler would have been given and
shows where those forecasts are wrong.

The five families answer separate questions: what a trajectory looks like beside
the observation, how the error grows with lead time, how it is distributed, which
hours of the day are hardest, and whether letting the model keep learning during
the replay changes any of it.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

from .evaluate import MODEL_ORDER, load_inputs
from .series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider
from .snapshots import ForecastArchive


LABELS = {
    "persistence": "Persistence",
    "seasonal_daily": "Daily seasonal",
    "seasonal_weekly": "Weekly seasonal",
    "ridge_direct": "Ridge, frozen 2019",
    "ridge_refit_once": "Ridge, frozen pre-test",
    "boosted_ridge": "Boosted ridge (selected)",
}
PREFIX = "ridge_refit_"


def style(name: str) -> dict:
    """Dash the update variants: on test they sit exactly on the frozen curve.

    Solid lines would hide two of the three models behind the third, which is the
    result the figures exist to show.
    """
    if not name.startswith(PREFIX):
        return {}
    return {"linestyle": (0, (6, 2)) if name.endswith("once") else (0, (1.5, 1.5))}


def label(name: str) -> str:
    if name not in LABELS and name.startswith(PREFIX) and name.endswith("d"):
        return f"Ridge, refit every {name[len(PREFIX):-1]} days"
    return LABELS.get(name, name)


def load_archives(directory: Path, partition: str) -> dict[str, ForecastArchive]:
    """Read every archive for a partition, ordered from baseline to fitted model."""
    archives = {}
    for path in directory.glob(f"{partition}_*.json"):
        archive = ForecastArchive.load(path)
        archives[archive.metadata["model_name"]] = archive
    if not archives:
        raise ValueError(f"{directory} holds no {partition} forecast archives")
    return dict(sorted(archives.items(), key=lambda item: (
        MODEL_ORDER.index(item[0]) if item[0] in MODEL_ORDER else len(MODEL_ORDER), item[0],
    )))


def scored_grid(
    actual: TimeSeriesCarbonIntensityProvider, bounds: tuple[datetime, datetime],
    horizon: timedelta, cadence: timedelta, label_delay: timedelta = timedelta(0),
) -> tuple[list[datetime], np.ndarray]:
    """Issue times with a complete label window, and the observations they face.

    This repeats the evaluator's own origin rule, ``label_delay`` included, rather
    than trusting the archive length: a snapshot may reach past the end of the
    partition, an origin may not.
    """
    start, end = bounds
    observed = np.array([
        sample.intensity_gco2e_per_kwh for sample in actual.get_actual_range(start, end)
    ])
    step = cadence // STEP
    buckets = horizon // STEP
    issues = []
    while start + cadence * len(issues) + horizon + label_delay <= end:
        issues.append(start + cadence * len(issues))
    targets = np.array([observed[index * step:index * step + buckets] for index in range(len(issues))])
    return issues, targets


def predictions(archive: ForecastArchive, issues: list[datetime], buckets: int) -> np.ndarray:
    return np.array([
        [sample.intensity_gco2e_per_kwh for sample in archive.get_forecast(issue).samples[:buckets]]
        for issue in issues
    ])


def trajectory_figure(plt, issues, targets, predicted, example: datetime, path: Path) -> None:
    """One day-ahead trajectory beside the observation, then a week at day-ahead lead.

    The upper panel is a single decision: what each model told a scheduler
    issuing at that instant. The lower one is the same forecast quality sustained
    over a week, reading each bucket from the trajectory issued 24 hours earlier,
    which is the hardest lead the horizon offers.
    """
    index = issues.index(example)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7.5))
    hours = np.arange(targets.shape[1]) * STEP.total_seconds() / 3600
    top.plot(hours, targets[index], color="black", linewidth=2.2, label="Actual", zorder=3)
    for name, values in predicted.items():
        top.plot(hours, values[index], linewidth=1.3, label=label(name), **style(name))
    top.set(xlabel=f"Hours after {example:%Y-%m-%d %H:%M} UTC", ylabel="gCO2e/kWh",
            title="One issued 24-hour trajectory against the observation")
    top.legend(fontsize=8, ncol=3)

    week = slice(index, index + 7 * 24)
    stamps = [issues[position] + timedelta(hours=24) for position in range(*week.indices(len(issues)))]
    bottom.plot(stamps, targets[week][:, -1], color="black", linewidth=2.0, label="Actual", zorder=3)
    for name, values in predicted.items():
        bottom.plot(stamps, values[week][:, -1], linewidth=1.1, label=label(name), **style(name))
    bottom.set(xlabel="UTC", ylabel="gCO2e/kWh",
               title="One week of the same bucket predicted 24 hours ahead")
    bottom.tick_params(axis="x", rotation=20)
    bottom.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def horizon_figure(plt, errors: dict[str, np.ndarray], path: Path) -> None:
    """MAE against lead time; the seasonal baselines are flat here by construction."""
    figure, axes = plt.subplots(figsize=(9, 5))
    leads = (np.arange(next(iter(errors.values())).shape[1]) + 1) * STEP.total_seconds() / 3600
    for name, error in errors.items():
        axes.plot(leads, np.abs(error).mean(axis=0), linewidth=1.6, label=label(name), **style(name))
    axes.set(xlabel="Lead time (hours to the end of the target bucket)", ylabel="MAE (gCO2e/kWh)",
             title="Forecast error against prediction horizon", xlim=(0, leads[-1]))
    axes.grid(alpha=0.3)
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def distribution_figure(plt, errors: dict[str, np.ndarray], path: Path) -> None:
    """Signed error densities and their spread; a shifted centre is a bias."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    limit = float(np.percentile(np.abs(np.concatenate([e.ravel() for e in errors.values()])), 99))
    bins = np.linspace(-limit, limit, 81)
    for name, error in errors.items():
        left.hist(error.ravel(), bins=bins, histtype="step", density=True, linewidth=1.5,
                  label=label(name), **style(name))
    left.axvline(0, color="black", linewidth=1.0, linestyle="--")
    left.set(xlabel="Prediction minus actual (gCO2e/kWh)", ylabel="Density",
             title="Signed error distribution")
    left.legend(fontsize=8)
    right.boxplot([error.ravel() for error in errors.values()],
                  tick_labels=[label(name) for name in errors], showfliers=False)
    right.axhline(0, color="black", linewidth=1.0, linestyle="--")
    right.set(ylabel="Prediction minus actual (gCO2e/kWh)", title="Error spread, outliers hidden")
    right.tick_params(axis="x", labelsize=8, rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def hour_figure(plt, issues, errors: dict[str, np.ndarray], targets: np.ndarray, path: Path) -> None:
    """MAE by the UTC hour of the target bucket, the hour a scheduler places work in."""
    hours = np.array([
        (issue + bucket * STEP).hour
        for issue in issues for bucket in range(targets.shape[1])
    ]).reshape(targets.shape)
    figure, axes = plt.subplots(figsize=(9, 5))
    for name, error in errors.items():
        axes.plot(range(24), [np.abs(error[hours == hour]).mean() for hour in range(24)],
                  marker="o", markersize=3, linewidth=1.5, label=label(name), **style(name))
    twin = axes.twinx()
    twin.plot(range(24), [targets[hours == hour].mean() for hour in range(24)],
              color="black", linestyle=":", linewidth=1.4, label="Mean actual")
    twin.set_ylabel("Mean actual carbon intensity (gCO2e/kWh)")
    axes.set(xlabel="UTC hour of the target bucket", ylabel="MAE (gCO2e/kWh)",
             title="Forecast error across the hours of the day", xticks=range(0, 24, 2))
    axes.grid(alpha=0.3)
    axes.legend(fontsize=9, loc="upper left")
    twin.legend(fontsize=9, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def update_figure(plt, issues, errors: dict[str, np.ndarray], path: Path) -> None:
    """Frozen against walk-forward: monthly MAE and the running mean over the replay."""
    tracked = {name: error for name, error in errors.items() if name.startswith("ridge_")}
    months = np.array([issue.strftime("%Y-%m") for issue in issues])
    order = sorted(set(months))
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.8 / len(tracked)
    for position, (name, error) in enumerate(tracked.items()):
        monthly = [np.abs(error[months == month]).mean() for month in order]
        left.bar(np.arange(len(order)) + position * width, monthly, width, label=label(name))
        right.plot(issues, np.abs(error).mean(axis=1).cumsum() / (np.arange(len(issues)) + 1),
                   linewidth=1.6, label=label(name), **style(name))
    left.set(xlabel="Issue month (UTC)", ylabel="MAE (gCO2e/kWh)",
             title="Monthly error, frozen against walk-forward",
             xticks=np.arange(len(order)) + (len(tracked) - 1) * width / 2, xticklabels=order)
    left.legend(fontsize=8)
    right.set(xlabel="UTC", ylabel="Running MAE (gCO2e/kWh)",
              title="Error accumulated over the replay")
    right.tick_params(axis="x", rotation=20)
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=root / "data/carbon_intensity/actual/actual.json")
    parser.add_argument("--protocol", type=Path, default=root / "data/carbon_intensity/actual/protocol.json")
    parser.add_argument("--snapshots", type=Path, default=root / "data/carbon_intensity/snapshots")
    parser.add_argument("--output-dir", type=Path, default=root / "data/carbon_intensity/figures")
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    parser.add_argument("--example-issue", type=datetime.fromisoformat, default=None,
                        help="issue time of the single plotted trajectory; default the first midnight")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        actual, protocol, _ = load_inputs(args.actual, args.protocol)
        horizon, cadence = timedelta(hours=args.horizon_hours), timedelta(minutes=args.cadence_minutes)
        issues, targets = scored_grid(
            actual, protocol.intervals[args.partition], horizon, cadence,
            timedelta(0) if args.partition == "test" else protocol.observation_delay,
        )
        archives = load_archives(args.snapshots, args.partition)
        predicted = {
            name: predictions(archive, issues, targets.shape[1]) for name, archive in archives.items()
        }
        errors = {name: values - targets for name, values in predicted.items()}
        example = args.example_issue or next(issue for issue in issues if issue.hour == 0)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            "actual_vs_forecast": lambda path: trajectory_figure(
                plt, issues, targets, predicted, example, path),
            "error_by_horizon": lambda path: horizon_figure(plt, errors, path),
            "error_distribution": lambda path: distribution_figure(plt, errors, path),
            "error_by_hour": lambda path: hour_figure(plt, issues, errors, targets, path),
            "frozen_vs_walkforward": lambda path: update_figure(plt, issues, errors, path),
        }
        for name, draw in figures.items():
            path = args.output_dir / f"{args.partition}_{name}.png"
            draw(path)
            print(f"wrote {path.name}")
    except (OSError, ImportError, KeyError, IndexError, TypeError, ValueError) as error:
        print(f"Figure generation failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {args.partition} figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
