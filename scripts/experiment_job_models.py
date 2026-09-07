"""Validation-only search behind the frozen job-prediction configuration.

This script never reads the test partition. It splits the trace with the same
frozen boundaries as the trainer, throws the test rows away, and scores every
candidate on expanding temporal folds inside the development period. The
leaderboard it writes is the evidence for the recipe hard-coded in
``scripts/train_job_models.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd


from job_prediction import (
    AVERAGE_POWER_WATTS,
    DURATION_SECONDS,
    ENERGY_KWH,
    load_job_data,
    regression_metrics,
    temporal_split,
)
from job_prediction.evaluation import LONG_JOB_SECONDS
from job_prediction.experiment import (
    Candidate,
    fit_predict,
    leaderboard,
    rolling_folds,
    score_candidate,
)
from job_prediction.features import FeatureSpec, build_features
from common import PROJECT_ROOT


DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "processed" / "pm100_clean.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "job_predictions"

# The ablation ladder: each rung adds exactly one group, so any improvement is
# attributable to that group and nothing else.
FEATURE_LADDER = {
    "A base": ("base",),
    "B +derived": ("base", "derived"),
    "C +user_id": ("base", "derived", "user_id"),
    "D +user_hist": ("base", "derived", "user_hist"),
    "E +signature_hist": ("base", "derived", "user_hist", "signature_hist"),
    "F +fingerprint_hist": (
        "base",
        "derived",
        "user_hist",
        "signature_hist",
        "fingerprint_hist",
    ),
    "G +global_hist": (
        "base",
        "derived",
        "user_hist",
        "signature_hist",
        "fingerprint_hist",
        "global_hist",
    ),
    "H +queue": (
        "base",
        "derived",
        "user_hist",
        "signature_hist",
        "fingerprint_hist",
        "global_hist",
        "queue",
    ),
}
CHOSEN_GROUPS = FEATURE_LADDER["E +signature_hist"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("features", "models", "tuning", "targets", "curve", "all"),
        default=["all"],
        nargs="+",
    )
    parser.add_argument("--no-output", action="store_true")
    return parser


def _development(arguments) -> pd.DataFrame:
    data = load_job_data(arguments.workload)
    split = temporal_split(
        data,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
    )
    # The test frame is discarded here and never referenced again.
    return pd.concat([split.train, split.validation], ignore_index=True)


class FeatureCache:
    """Build one feature frame at a time.

    Every group combination over 133k rows is a few hundred megabytes, and the
    ladder alone asks for eight of them. Candidates are grouped by feature set,
    so keeping a single frame resident costs one rebuild per group and bounds
    the memory instead of holding the whole ladder at once.
    """

    def __init__(self, development: pd.DataFrame) -> None:
        self._development = development
        self._name: str | None = None
        self._frame: pd.DataFrame | None = None

    def get(self, spec: FeatureSpec) -> pd.DataFrame:
        if self._name != spec.name:
            self._frame = None
            self._frame = build_features(self._development, spec)
            self._name = spec.name
        return self._frame


def _run(candidates, folds, cache) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for candidate in candidates:
        started = time.perf_counter()
        result = score_candidate(
            candidate, folds, features=cache.get(candidate.spec)
        )
        results.append(result)
        print(
            f"{candidate.label:<72} "
            f"WAPE mean {result['wape_mean_folds']:>7.2%} "
            f"last {result['wape']:>7.2%} "
            f">=1h {result['long_wape']:>7.2%} "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    return results


def stage_models(folds, cache, spec) -> list[dict[str, object]]:
    print("\n=== model families on the chosen feature set ===")
    grid = [
        ("median", "raw", "squared_error"),
        ("ridge", "log1p", "squared_error"),
        ("decision_tree", "log1p", "squared_error"),
        ("decision_tree", "log1p", "absolute_error"),
        ("random_forest", "log1p", "squared_error"),
        ("extra_trees", "log1p", "squared_error"),
        ("hist_gbr", "log1p", "squared_error"),
        ("hist_gbr", "log1p", "absolute_error"),
        ("hist_gbr", "log1p", "poisson"),
        ("hist_gbr", "raw", "squared_error"),
        ("hist_gbr", "raw", "absolute_error"),
        ("segmented", "raw", "absolute_error"),
        ("segmented_hard", "raw", "absolute_error"),
    ]
    return _run(
        [Candidate(model, spec, transform, loss) for model, transform, loss in grid],
        folds,
        cache,
    )


def stage_tuning(folds, cache, spec) -> list[dict[str, object]]:
    print("\n=== target weighting and capacity on the chosen model ===")
    candidates = [
        Candidate("hist_gbr", spec, "log1p", "absolute_error", weight=weight)
        for weight in ("none", "log_duration", "log_duration_squared", "quartic_duration")
    ]
    candidates += [
        Candidate(
            "hist_gbr",
            spec,
            "log1p",
            "absolute_error",
            weight="log_duration",
            overrides=overrides,
        )
        for overrides in (
            (("max_iter", 800),),
            (("max_iter", 800), ("learning_rate", 0.03)),
            (("max_leaf_nodes", 127),),
            (("max_leaf_nodes", 31), ("max_iter", 800)),
            (("min_samples_leaf", 100),),
            (("l2_regularization", 10.0),),
        )
    ]
    candidates.append(
        Candidate(
            "segmented_hard", spec, "raw", "absolute_error", weight="log_duration"
        )
    )
    return _run(candidates, folds, cache)


def stage_targets(folds, cache, spec) -> list[dict[str, object]]:
    print("\n=== power and energy ===")
    candidates = []
    for target in (AVERAGE_POWER_WATTS, ENERGY_KWH):
        for model, transform, loss, weight in (
            ("ridge", "log1p", "squared_error", "none"),
            ("hist_gbr", "log1p", "squared_error", "none"),
            ("hist_gbr", "log1p", "absolute_error", "none"),
            ("hist_gbr", "raw", "absolute_error", "none"),
            ("hist_gbr", "log1p", "absolute_error", "log_duration"),
            ("random_forest", "log1p", "squared_error", "none"),
            ("extra_trees", "log1p", "squared_error", "none"),
        ):
            candidates.append(
                Candidate(
                    model, spec, transform, loss, target=target, weight=weight
                )
            )
    return _run(candidates, folds, cache)


def stage_curve(folds, cache, spec) -> pd.DataFrame:
    """Separate data-limited from feature-, model-limited, and shifted."""

    print("\n=== learning curves on the last fold ===")
    fold = folds[-1]
    actual = fold.validation[DURATION_SECONDS].to_numpy(dtype=float)
    long_jobs = actual >= LONG_JOB_SECONDS
    rows: list[dict[str, object]] = []
    for label, groups in (("base", ("base",)), ("chosen", CHOSEN_GROUPS)):
        candidate = Candidate(
            "hist_gbr",
            FeatureSpec(groups=groups, name=label),
            "log1p",
            "absolute_error",
            weight="log_duration",
        )
        for mode in ("recent", "random"):
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
                train = fold.train
                if fraction < 1.0:
                    train = (
                        train.iloc[-int(len(train) * fraction) :]
                        if mode == "recent"
                        else train.sample(frac=fraction, random_state=0).sort_index()
                    )
                predicted, _, _, _ = fit_predict(
                    candidate, train, fold.validation, features=cache.get(candidate.spec)
                )
                metrics = regression_metrics(actual, predicted)
                long_metrics = regression_metrics(
                    actual[long_jobs], predicted[long_jobs]
                )
                rows.append(
                    {
                        "features": label,
                        "mode": mode,
                        "fraction": fraction,
                        "rows": len(train),
                        "wape": metrics.weighted_absolute_relative_error,
                        "long_wape": long_metrics.weighted_absolute_relative_error,
                        "mae": metrics.mae,
                        "rmse": metrics.rmse,
                    }
                )
                print(
                    f"{label:<8}{mode:<8}{fraction:>5.0%} n={len(train):>7,} "
                    f"WAPE {metrics.weighted_absolute_relative_error:>7.2%} "
                    f">=1h {long_metrics.weighted_absolute_relative_error:>7.2%}",
                    flush=True,
                )
    # An oracle that had this window's own labels bounds what the features can
    # express when there is no shift at all. The distance from it to the
    # out-of-time score is the price of the model being out of date.
    candidate = Candidate(
        "hist_gbr",
        FeatureSpec(groups=CHOSEN_GROUPS, name="chosen"),
        "log1p",
        "absolute_error",
        weight="log_duration",
    )
    predicted, _, _, _ = fit_predict(
        candidate, fold.validation, fold.validation, features=cache.get(candidate.spec)
    )
    metrics = regression_metrics(actual, predicted)
    rows.append(
        {
            "features": "chosen",
            "mode": "in_sample_floor",
            "fraction": 1.0,
            "rows": len(fold.validation),
            "wape": metrics.weighted_absolute_relative_error,
            "long_wape": regression_metrics(
                actual[long_jobs], predicted[long_jobs]
            ).weighted_absolute_relative_error,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
        }
    )
    print(
        f"{'chosen':<8}{'in-sample':<8}{1.0:>5.0%} n={len(fold.validation):>7,} "
        f"WAPE {metrics.weighted_absolute_relative_error:>7.2%}",
        flush=True,
    )
    return pd.DataFrame(rows)


def main() -> int:
    arguments = build_parser().parse_args()
    stages = set(arguments.stage)
    if "all" in stages:
        stages = {"features", "models", "tuning", "targets", "curve"}

    development = _development(arguments)
    folds = rolling_folds(development, folds=arguments.folds)
    print(f"development jobs         {len(development):,}")
    for fold in folds:
        share = (fold.validation[DURATION_SECONDS] < 10.0).mean()
        print(
            f"  {fold.name} cutoff {fold.cutoff:%Y-%m-%d %H:%M} "
            f"train {len(fold.train):>7,} validation {len(fold.validation):>7,} "
            f"share <10 s {share:>6.1%}"
        )

    chosen = FeatureSpec(groups=CHOSEN_GROUPS, name="E +signature_hist")
    cache = FeatureCache(development)

    results: list[dict[str, object]] = []
    if "features" in stages:
        print("\n=== feature ablation (hist_gbr, log1p, absolute error) ===")
        results += _run(
            [
                Candidate(
                    "hist_gbr",
                    FeatureSpec(groups=groups, name=name),
                    "log1p",
                    "absolute_error",
                )
                for name, groups in FEATURE_LADDER.items()
            ],
            folds,
            cache,
        )
    if "models" in stages:
        results += stage_models(folds, cache, chosen)
    if "tuning" in stages:
        results += stage_tuning(folds, cache, chosen)
    if "targets" in stages:
        results += stage_targets(folds, cache, chosen)
    curve = stage_curve(folds, cache, chosen) if "curve" in stages else None

    if results:
        table = leaderboard(results)
        print("\n=== validation leaderboard (duration unless stated) ===")
        print(table.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    if arguments.no_output:
        return 0

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if results:
        path = arguments.output_dir / "validation_leaderboard.csv"
        table.to_csv(path, index=False)
        print(f"\nleaderboard written      {path}")
    if curve is not None:
        path = arguments.output_dir / "validation_learning_curve.csv"
        curve.to_csv(path, index=False)
        print(f"learning curve written   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
