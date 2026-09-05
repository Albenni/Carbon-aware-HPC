"""Validation-only model search for the submission-time job models.

Nothing in this module may read the test partition. It builds rolling temporal
folds inside the development period, fits candidates on the labels that were
already observable at each fold cutoff, and scores them on the following
window. The winning configuration is then frozen and handed to
``job_prediction.model`` for a single test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
import pandas as pd

from .data import (
    AVERAGE_POWER_WATTS,
    COMPLETION_TIME,
    DURATION_SECONDS,
    ENERGY_KWH,
    SUBMIT_TIME,
)
from .evaluation import LONG_JOB_SECONDS, regression_metrics
from .features import FeatureSpec


TRANSFORMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "raw": (lambda values: values, lambda values: values),
    "log1p": (np.log1p, lambda values: np.expm1(np.clip(values, -50.0, 50.0))),
}


@dataclass(frozen=True, slots=True)
class Fold:
    """One expanding-window fold: fit before ``cutoff``, score after it."""

    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    cutoff: pd.Timestamp


def _completed_by(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    completion = pd.to_datetime(frame[COMPLETION_TIME], utc=True)
    return frame.loc[completion <= cutoff]


def rolling_folds(
    development: pd.DataFrame,
    *,
    folds: int = 4,
    validation_fraction: float = 0.15,
) -> list[Fold]:
    """Expanding-window folds over the development period.

    The last fold reproduces the frozen protocol's own train/validation
    boundary, so a candidate's headline score is directly comparable with the
    published ridge baseline. The earlier folds only exist to show whether that
    ranking is stable as the workload drifts.
    """

    ordered = development.sort_values([SUBMIT_TIME, "job_id"], kind="stable")
    timestamps = pd.to_datetime(ordered[SUBMIT_TIME], utc=True).reset_index(drop=True)
    total = len(ordered)
    window = max(1, int(total * validation_fraction))

    result: list[Fold] = []
    for index in range(folds):
        end = total - index * window
        start = end - window
        if start <= window:
            break
        cutoff = timestamps.iloc[start]
        limit = timestamps.iloc[end - 1]
        selector = pd.to_datetime(ordered[SUBMIT_TIME], utc=True)
        train = _completed_by(ordered.loc[selector < cutoff], cutoff)
        validation = _completed_by(
            ordered.loc[(selector >= cutoff) & (selector <= limit)],
            timestamps.iloc[end] if end < total else limit,
        )
        if train.empty or validation.empty:
            break
        # The source index is preserved so a precomputed feature frame can be
        # sliced by it; recomputing per fold would give identical values but
        # would repeat the history scan for every candidate.
        result.append(
            Fold(name=f"fold{index}", train=train, validation=validation, cutoff=cutoff)
        )
    result.reverse()
    return result


class SegmentedDurationRegressor:
    """Classify long jobs, then let a specialist regressor handle each side.

    The band report shows why: a model trained on ``log1p`` duration is well
    calibrated on the short jobs that dominate the job count, but under-predicts
    the long jobs that dominate total runtime, and therefore WAPE. A model
    trained on raw seconds does the opposite. One classifier plus two
    specialists lets each regime be fitted in the space that suits it.

    ``blend`` mixes the two branches by the classifier's probability, which is
    the mean-optimal combination; ``hard`` routes each job to one branch, which
    is the median-optimal one. Both are offered because the choice is decided on
    validation, not asserted here.
    """

    def __init__(
        self,
        *,
        threshold: float = 3_600.0,
        blend: bool = True,
        seed: int = 0,
        long_loss: str = "absolute_error",
        short_loss: str = "absolute_error",
    ) -> None:
        self.threshold = threshold
        self.blend = blend
        self.seed = seed
        self.long_loss = long_loss
        self.short_loss = short_loss

    def _booster(self, loss: str):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss=loss,
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        )

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SegmentedDurationRegressor":
        from sklearn.ensemble import HistGradientBoostingClassifier

        target = np.asarray(target, dtype=float)
        weights = None if sample_weight is None else np.asarray(sample_weight, float)
        is_long = target >= self.threshold
        if is_long.all() or not is_long.any():
            self.classifier_ = None
            self.constant_long_ = bool(is_long.all())
        else:
            self.classifier_ = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=63,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed,
            ).fit(features, is_long, sample_weight=weights)
            self.constant_long_ = False

        # The short branch works in log space, where its targets span four
        # decades; the long branch works in seconds, where absolute error is
        # exactly the quantity WAPE sums.
        self.short_ = (
            self._booster(self.short_loss).fit(
                features[~is_long],
                np.log1p(target[~is_long]),
                sample_weight=None if weights is None else weights[~is_long],
            )
            if (~is_long).any()
            else None
        )
        self.long_ = (
            self._booster(self.long_loss).fit(
                features[is_long],
                target[is_long],
                sample_weight=None if weights is None else weights[is_long],
            )
            if is_long.any()
            else None
        )
        self.short_fallback_ = float(np.median(target[~is_long])) if (~is_long).any() else 1.0
        self.long_fallback_ = float(np.median(target[is_long])) if is_long.any() else 1.0
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        rows = len(features)
        short = (
            np.expm1(np.clip(self.short_.predict(features), -50.0, 50.0))
            if self.short_ is not None
            else np.full(rows, self.short_fallback_)
        )
        long = (
            self.long_.predict(features)
            if self.long_ is not None
            else np.full(rows, self.long_fallback_)
        )
        if self.classifier_ is None:
            return long if self.constant_long_ else short
        probability = self.classifier_.predict_proba(features)[:, 1]
        if self.blend:
            return (1.0 - probability) * short + probability * long
        return np.where(probability >= 0.5, long, short)


def build_estimator(name: str, loss: str, *, seed: int = 0, **overrides):
    """Instantiate one candidate. Hyperparameters stay deliberately plain."""

    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    if name == "median":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "decision_tree":
        return DecisionTreeRegressor(
            criterion=loss,
            min_samples_leaf=50,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            criterion=loss,
            min_samples_leaf=5,
            max_features=0.5,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            criterion=loss,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    if name in {"segmented", "segmented_hard"}:
        return SegmentedDurationRegressor(
            blend=name == "segmented",
            seed=seed,
            long_loss=loss,
        )
    if name == "hist_gbr":
        settings = {
            "max_iter": 400,
            # Median regression on a right-skewed conditional distribution
            # under-predicts the long jobs that carry the runtime seconds; an
            # upper quantile is the standard correction, and which quantile is
            # a validation question.
            **({"quantile": 0.5} if loss == "quantile" else {}),
            "learning_rate": 0.06,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
        }
        settings.update(overrides)
        return HistGradientBoostingRegressor(
            loss=loss, early_stopping=False, random_state=seed, **settings
        )
    raise ValueError(f"unknown estimator {name!r}")


@dataclass(frozen=True, slots=True)
class Candidate:
    model: str
    spec: FeatureSpec
    transform: str = "log1p"
    loss: str = "squared_error"
    target: str = DURATION_SECONDS
    weight: str = "none"
    overrides: tuple[tuple[str, object], ...] = ()
    seed: int = 0
    label: str = field(default="")

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        if not self.label:
            object.__setattr__(
                self,
                "label",
                f"{self.model}|{self.spec.name}|{self.transform}|{self.loss}"
                + ("" if self.weight == "none" else f"|w={self.weight}")
                + "".join(f"|{key}={value}" for key, value in self.overrides)
                # Power and energy candidates otherwise collide with the
                # duration ones, which share every other field.
                + ("" if self.target == DURATION_SECONDS else f"|{self.target}"),
            )


def fit_predict(
    candidate: Candidate,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    features: pd.DataFrame,
) -> tuple[np.ndarray, float, object, tuple[str, ...]]:
    """Fit on ``train`` and predict ``evaluation``; also return the fitted model.

    ``features`` is the feature frame built once over the whole development
    period and indexed like it. Rows are selected by index rather than rebuilt
    per split, so a job's causal history is identical whichever fold it lands
    in, and the history scan runs once instead of once per candidate.
    """

    train_features = features.loc[train.index]
    evaluation_features = features.loc[evaluation.index]

    forward, inverse = TRANSFORMS[candidate.transform]
    target = pd.to_numeric(train[candidate.target], errors="coerce").to_numpy(float)
    estimator = build_estimator(
        candidate.model,
        candidate.loss,
        seed=candidate.seed,
        **dict(candidate.overrides),
    )

    matrix = train_features.to_numpy(dtype=float)
    evaluation_matrix = evaluation_features.to_numpy(dtype=float)
    if candidate.model in {"ridge", "random_forest", "extra_trees", "decision_tree"}:
        # Only the histogram booster handles missing values natively.
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        matrix = np.where(np.isnan(matrix), medians, matrix)
        evaluation_matrix = np.where(
            np.isnan(evaluation_matrix), medians, evaluation_matrix
        )
    if candidate.model == "ridge":
        centers = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales < 1e-12] = 1.0
        matrix = (matrix - centers) / scales
        evaluation_matrix = (evaluation_matrix - centers) / scales

    weights = {
        "none": None,
        # Weighting a log-space fit by the target makes the optimiser care
        # about the long jobs that carry the workload's runtime seconds, which
        # is what WAPE sums, without giving up log-space calibration.
        "duration": target,
        "sqrt_duration": np.sqrt(target),
        "quartic_duration": np.power(target, 0.25),
        "log_duration": np.log1p(target),
        # A floor on the weight keeps the very short jobs fitted: log1p alone
        # gives a one-second job a weight of 0.7 against 11 for a day-long one,
        # and their systematic over-prediction then leaks into composed energy.
        "log_duration_plus_2": np.log1p(target) + 2.0,
        "log_duration_plus_5": np.log1p(target) + 5.0,
        "log_duration_squared": np.square(np.log1p(target)),
    }[candidate.weight]

    started = time.perf_counter()
    if weights is None:
        estimator.fit(matrix, forward(target))
    else:
        estimator.fit(matrix, forward(target), sample_weight=weights)
    elapsed = time.perf_counter() - started
    predicted = inverse(np.asarray(estimator.predict(evaluation_matrix), dtype=float))
    # Predictions feed a scheduler and an energy identity: they must be positive
    # and must not extrapolate past what the fitting period actually contained.
    predicted = np.clip(predicted, 1.0, float(np.max(target)))
    return predicted, elapsed, estimator, tuple(train_features.columns)


def score_candidate(
    candidate: Candidate,
    folds: list[Fold],
    *,
    features: pd.DataFrame,
) -> dict[str, object]:
    """Score one candidate across every fold; the last fold is the headline."""

    rows: list[dict[str, object]] = []
    for fold in folds:
        predicted, elapsed, _, columns = fit_predict(
            candidate, fold.train, fold.validation, features=features
        )
        actual = pd.to_numeric(
            fold.validation[candidate.target], errors="coerce"
        ).to_numpy(float)
        durations = pd.to_numeric(
            fold.validation[DURATION_SECONDS], errors="coerce"
        ).to_numpy(float)
        metrics = regression_metrics(actual, predicted)
        long_jobs = durations >= LONG_JOB_SECONDS
        long_metrics = regression_metrics(actual[long_jobs], predicted[long_jobs])
        rows.append(
            {
                "fold": fold.name,
                "wape": metrics.weighted_absolute_relative_error,
                "long_wape": long_metrics.weighted_absolute_relative_error,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "median_relative": metrics.median_absolute_relative_error,
                "bias": metrics.bias,
                "p95_absolute_error": metrics.p95_absolute_error,
                "train_seconds": elapsed,
                "features": len(columns),
            }
        )

    frame = pd.DataFrame(rows)
    headline = frame.iloc[-1]
    return {
        "label": candidate.label,
        "model": candidate.model,
        "features": candidate.spec.name,
        "transform": candidate.transform,
        "loss": candidate.loss,
        "target": candidate.target,
        "n_features": int(headline["features"]),
        "wape": float(headline["wape"]),
        "long_wape": float(headline["long_wape"]),
        "mae": float(headline["mae"]),
        "rmse": float(headline["rmse"]),
        "median_relative": float(headline["median_relative"]),
        "bias": float(headline["bias"]),
        "p95_absolute_error": float(headline["p95_absolute_error"]),
        "wape_mean_folds": float(frame["wape"].mean()),
        "wape_std_folds": float(frame["wape"].std(ddof=0)),
        "long_wape_mean_folds": float(frame["long_wape"].mean()),
        "train_seconds": float(frame["train_seconds"].mean()),
        "folds": frame,
    }


def leaderboard(results: list[dict[str, object]]) -> pd.DataFrame:
    """Rank candidates by validation WAPE, then by the long-job WAPE."""

    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "folds"} for row in results]
    )
    # Stages overlap deliberately, so the same recipe can be scored twice; the
    # scores are identical and only one row belongs on the board.
    frame = frame.drop_duplicates(subset=["label", "target"], keep="first")
    return frame.sort_values(["wape", "long_wape"], kind="stable").reset_index(drop=True)


__all__ = [
    "Candidate",
    "Fold",
    "SegmentedDurationRegressor",
    "build_estimator",
    "fit_predict",
    "leaderboard",
    "rolling_folds",
    "score_candidate",
]
