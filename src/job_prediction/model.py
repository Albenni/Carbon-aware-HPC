"""Log-ridge models for duration, power, and energy.

The simpler of the two predictor families and the baseline the boosted models
in :mod:`job_prediction.gradient` are measured against. Both read the same
column vocabulary and temporal-cutoff helpers from :mod:`job_prediction.data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import (
    AVERAGE_POWER_WATTS,
    CATEGORICAL_FEATURES,
    COMPLETION_TIME,
    DURATION_SECONDS,
    ENERGY_KWH,
    JOB_ID,
    MODEL_FORMAT_VERSION,
    NUMERIC_FEATURES,
    PREDICTED_AVERAGE_POWER_WATTS,
    PREDICTED_DURATION_SECONDS,
    PREDICTED_ENERGY_KWH,
    SUBMISSION_FEATURES,
    SUBMIT_TIME,
    TARGETS,
    WATT_SECONDS_PER_KILOWATT_HOUR,
    PredictionComposition,
    TemporalSplit,
    completed_by,
    require_columns,
)


DEFAULT_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
_MISSING_CATEGORY = "<missing>"


def _category_values(series: pd.Series) -> np.ndarray:
    return series.map(
        lambda value: _MISSING_CATEGORY if pd.isna(value) else str(value)
    ).to_numpy()


@dataclass(frozen=True, slots=True)
class SubmissionFeatureEncoder:
    """Train-fitted imputation, calendar expansion, and standardisation."""

    numeric_medians: Mapping[str, float]
    missing_indicators: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    feature_names: tuple[str, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> SubmissionFeatureEncoder:
        require_columns(frame, SUBMISSION_FEATURES)

        medians: dict[str, float] = {}
        missing_indicators: list[str] = []
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isinf(values)):
                raise ValueError(f"{column} contains an infinite value")
            present = values[~np.isnan(values)]
            if not present.size:
                raise ValueError(f"{column} has no training values")
            if np.any(present < 0.0):
                raise ValueError(f"{column} cannot contain negative values")
            medians[column] = float(np.median(present))
            if np.any(np.isnan(values)):
                missing_indicators.append(column)

        # The first value is the reference category, so no redundant one-hot
        # column is carried into the regression.
        categories = {
            column: tuple(sorted(set(_category_values(frame[column])))[1:])
            for column in CATEGORICAL_FEATURES
        }
        raw, names = _raw_features(
            frame,
            numeric_medians=medians,
            missing_indicators=tuple(missing_indicators),
            categories=categories,
        )
        centers = np.mean(raw, axis=0)
        scales = np.std(raw, axis=0)
        scales[scales < 1e-12] = 1.0
        return cls(
            numeric_medians=medians,
            missing_indicators=tuple(missing_indicators),
            categories=categories,
            feature_names=names,
            centers=tuple(float(value) for value in centers),
            scales=tuple(float(value) for value in scales),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        require_columns(frame, SUBMISSION_FEATURES)
        raw, names = _raw_features(
            frame,
            numeric_medians=self.numeric_medians,
            missing_indicators=self.missing_indicators,
            categories=self.categories,
        )
        if names != self.feature_names:
            raise ValueError("encoded feature layout does not match the fitted model")
        return (
            raw - np.asarray(self.centers, dtype=float)
        ) / np.asarray(self.scales, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "numeric_medians": dict(self.numeric_medians),
            "missing_indicators": list(self.missing_indicators),
            "categories": {
                name: list(values) for name, values in self.categories.items()
            },
            "feature_names": list(self.feature_names),
            "centers": list(self.centers),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SubmissionFeatureEncoder:
        categories = values["categories"]
        if not isinstance(categories, Mapping):
            raise ValueError("invalid feature categories in model artifact")
        medians = values["numeric_medians"]
        if not isinstance(medians, Mapping):
            raise ValueError("invalid numeric medians in model artifact")
        return cls(
            numeric_medians={str(key): float(value) for key, value in medians.items()},
            missing_indicators=tuple(values["missing_indicators"]),  # type: ignore[arg-type]
            categories={
                str(key): tuple(str(item) for item in value)  # type: ignore[union-attr]
                for key, value in categories.items()
            },
            feature_names=tuple(values["feature_names"]),  # type: ignore[arg-type]
            centers=tuple(float(value) for value in values["centers"]),  # type: ignore[arg-type]
            scales=tuple(float(value) for value in values["scales"]),  # type: ignore[arg-type]
        )


def _raw_features(
    frame: pd.DataFrame,
    *,
    numeric_medians: Mapping[str, float],
    missing_indicators: tuple[str, ...],
    categories: Mapping[str, tuple[str, ...]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []

    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isinf(values)):
            raise ValueError(f"{name} contains an infinite value")
        missing = np.isnan(values)
        filled = np.where(missing, numeric_medians[name], values)
        if np.any(filled < 0.0):
            raise ValueError(f"{name} cannot contain negative values")
        columns.append(np.log1p(filled))
        names.append(f"log1p({name})")
        if name in missing_indicators:
            columns.append(missing.astype(float))
            names.append(f"{name}_missing")

    timestamps = pd.to_datetime(frame[SUBMIT_TIME], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"{SUBMIT_TIME} cannot be missing")
    hour = (
        timestamps.dt.hour.to_numpy(dtype=float)
        + timestamps.dt.minute.to_numpy(dtype=float) / 60.0
        + timestamps.dt.second.to_numpy(dtype=float) / 3_600.0
    )
    weekday = timestamps.dt.dayofweek.to_numpy(dtype=float)
    for name, values in (
        ("submit_hour_sin", np.sin(2.0 * np.pi * hour / 24.0)),
        ("submit_hour_cos", np.cos(2.0 * np.pi * hour / 24.0)),
        ("submit_weekday_sin", np.sin(2.0 * np.pi * weekday / 7.0)),
        ("submit_weekday_cos", np.cos(2.0 * np.pi * weekday / 7.0)),
    ):
        columns.append(values)
        names.append(name)

    for name in CATEGORICAL_FEATURES:
        values = _category_values(frame[name])
        for category in categories[name]:
            columns.append((values == category).astype(float))
            names.append(f"{name}={category}")

    return np.column_stack(columns), tuple(names)


@dataclass(frozen=True, slots=True)
class LogRidgeRegressor:
    """Ridge regression in log-target space, implemented with NumPy.

    Estimates are bounded by the development period's observed target range.
    This uses no future information and prevents an extrapolated log prediction
    from becoming physically absurd.
    """

    target: str
    alpha: float
    intercept: float
    coefficients: tuple[float, ...]
    minimum_log_prediction: float
    maximum_log_prediction: float

    def __post_init__(self) -> None:
        parameters = (
            self.alpha,
            self.intercept,
            self.minimum_log_prediction,
            self.maximum_log_prediction,
            *self.coefficients,
        )
        if not all(np.isfinite(value) for value in parameters):
            raise ValueError("regressor parameters must be finite")
        if self.alpha <= 0.0:
            raise ValueError("ridge alpha must be greater than zero")
        if self.minimum_log_prediction > self.maximum_log_prediction:
            raise ValueError("prediction bounds are reversed")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        target: object,
        *,
        target_name: str,
        alpha: float,
    ) -> LogRidgeRegressor:
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("ridge alpha must be greater than zero")
        matrix = np.asarray(features, dtype=float)
        values = np.asarray(target, dtype=float)
        if matrix.ndim != 2 or values.ndim != 1 or len(matrix) != len(values):
            raise ValueError("features and target have incompatible shapes")
        if not len(values):
            raise ValueError("at least one training row is required")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must be finite")
        if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
            raise ValueError("training targets must be finite and greater than zero")

        logged = np.log(values)
        feature_means = np.mean(matrix, axis=0)
        centered = matrix - feature_means
        target_mean = float(np.mean(logged))
        observations = float(len(values))
        gram = centered.T @ centered / observations
        right_hand_side = centered.T @ (logged - target_mean) / observations
        coefficients = np.linalg.solve(
            gram + alpha * np.eye(matrix.shape[1]),
            right_hand_side,
        )
        intercept = target_mean - float(feature_means @ coefficients)
        return cls(
            target=target_name,
            alpha=float(alpha),
            intercept=intercept,
            coefficients=tuple(float(value) for value in coefficients),
            minimum_log_prediction=float(np.min(logged)),
            maximum_log_prediction=float(np.max(logged)),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coefficients):
            raise ValueError("features do not match the fitted regressor")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must be finite")
        logged = self.intercept + matrix @ np.asarray(self.coefficients, dtype=float)
        return np.exp(
            np.clip(
                logged,
                self.minimum_log_prediction,
                self.maximum_log_prediction,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "alpha": self.alpha,
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "minimum_log_prediction": self.minimum_log_prediction,
            "maximum_log_prediction": self.maximum_log_prediction,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> LogRidgeRegressor:
        return cls(
            target=str(values["target"]),
            alpha=float(values["alpha"]),
            intercept=float(values["intercept"]),
            coefficients=tuple(float(value) for value in values["coefficients"]),  # type: ignore[arg-type]
            minimum_log_prediction=float(values["minimum_log_prediction"]),
            maximum_log_prediction=float(values["maximum_log_prediction"]),
        )


@dataclass(frozen=True, slots=True)
class JobPredictor:
    """Three learned targets combined into consistent scheduling inputs."""

    encoder: SubmissionFeatureEncoder
    duration_model: LogRidgeRegressor
    power_model: LogRidgeRegressor
    energy_model: LogRidgeRegressor
    composition: PredictionComposition

    def predict_components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        features = self.encoder.transform(frame)
        return {
            DURATION_SECONDS: self.duration_model.predict(features),
            AVERAGE_POWER_WATTS: self.power_model.predict(features),
            ENERGY_KWH: self.energy_model.predict(features),
        }

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict a positive, physically consistent scheduling description."""

        components = self.predict_components(frame)
        duration = components[DURATION_SECONDS]
        if self.composition is PredictionComposition.DURATION_POWER:
            power = components[AVERAGE_POWER_WATTS]
            energy = power * duration / WATT_SECONDS_PER_KILOWATT_HOUR
        else:
            energy = components[ENERGY_KWH]
            power = energy * WATT_SECONDS_PER_KILOWATT_HOUR / duration

        result = pd.DataFrame(
            {
                PREDICTED_DURATION_SECONDS: duration,
                PREDICTED_AVERAGE_POWER_WATTS: power,
                PREDICTED_ENERGY_KWH: energy,
            },
            index=frame.index,
        )
        if JOB_ID in frame:
            result.insert(0, JOB_ID, frame[JOB_ID].to_numpy())
        return result.reset_index(drop=True)

    def to_dict(self, *, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "metadata": dict(metadata or {}),
            "composition": self.composition.value,
            "encoder": self.encoder.to_dict(),
            "models": {
                DURATION_SECONDS: self.duration_model.to_dict(),
                AVERAGE_POWER_WATTS: self.power_model.to_dict(),
                ENERGY_KWH: self.energy_model.to_dict(),
            },
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(metadata=metadata), handle, indent=2)
            handle.write("\n")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> JobPredictor:
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if values.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("unsupported job-model artifact version")
        models = values["models"]
        return cls(
            encoder=SubmissionFeatureEncoder.from_dict(values["encoder"]),
            duration_model=LogRidgeRegressor.from_dict(models[DURATION_SECONDS]),
            power_model=LogRidgeRegressor.from_dict(models[AVERAGE_POWER_WATTS]),
            energy_model=LogRidgeRegressor.from_dict(models[ENERGY_KWH]),
            composition=PredictionComposition(values["composition"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    predictor: JobPredictor
    selected_alphas: Mapping[str, float]
    validation_energy_mae: Mapping[PredictionComposition, float]
    fit_counts: Mapping[str, int]


def _positive_target(frame: pd.DataFrame, name: str) -> np.ndarray:
    require_columns(frame, (name,))
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise ValueError(f"{name} must be finite and greater than zero")
    return values


def _select_regressor(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    target_name: str,
    alphas: tuple[float, ...],
) -> LogRidgeRegressor:
    candidates = [
        LogRidgeRegressor.fit(
            train_features,
            train_target,
            target_name=target_name,
            alpha=alpha,
        )
        for alpha in alphas
    ]
    return min(
        candidates,
        key=lambda model: float(
            np.mean(
                np.square(
                    np.log(model.predict(validation_features))
                    - np.log(validation_target)
                )
            )
        ),
    )


def fit_job_predictor(
    split: TemporalSplit,
    *,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
) -> TrainingResult:
    """Tune on validation, then refit the selected models on train+validation."""

    normalized_alphas = tuple(sorted(set(float(alpha) for alpha in alphas)))
    if not normalized_alphas or any(
        not np.isfinite(alpha) or alpha <= 0.0 for alpha in normalized_alphas
    ):
        raise ValueError("alphas must contain positive finite values")

    train = completed_by(
        split.train,
        split.train_until,
        period_name="training",
    )
    validation = completed_by(
        split.validation,
        split.validation_until,
        period_name="validation",
    )
    tuning_encoder = SubmissionFeatureEncoder.fit(train)
    train_features = tuning_encoder.transform(train)
    validation_features = tuning_encoder.transform(validation)
    tuned = {
        target: _select_regressor(
            train_features,
            _positive_target(train, target),
            validation_features,
            _positive_target(validation, target),
            target_name=target,
            alphas=normalized_alphas,
        )
        for target in TARGETS
    }

    validation_duration = tuned[DURATION_SECONDS].predict(validation_features)
    validation_power = tuned[AVERAGE_POWER_WATTS].predict(validation_features)
    validation_energy = tuned[ENERGY_KWH].predict(validation_features)
    actual_energy = _positive_target(validation, ENERGY_KWH)
    energy_candidates = {
        PredictionComposition.DURATION_POWER: (
            validation_duration
            * validation_power
            / WATT_SECONDS_PER_KILOWATT_HOUR
        ),
        PredictionComposition.DURATION_ENERGY: validation_energy,
    }
    validation_mae = {
        composition: float(np.mean(np.abs(predicted - actual_energy)))
        for composition, predicted in energy_candidates.items()
    }
    composition = min(validation_mae, key=validation_mae.__getitem__)

    development = completed_by(
        pd.concat([split.train, split.validation], ignore_index=True),
        split.validation_until,
        period_name="development",
    )
    encoder = SubmissionFeatureEncoder.fit(development)
    development_features = encoder.transform(development)
    final_models = {
        target: LogRidgeRegressor.fit(
            development_features,
            _positive_target(development, target),
            target_name=target,
            alpha=tuned[target].alpha,
        )
        for target in TARGETS
    }
    predictor = JobPredictor(
        encoder=encoder,
        duration_model=final_models[DURATION_SECONDS],
        power_model=final_models[AVERAGE_POWER_WATTS],
        energy_model=final_models[ENERGY_KWH],
        composition=composition,
    )
    return TrainingResult(
        predictor=predictor,
        selected_alphas={target: tuned[target].alpha for target in TARGETS},
        validation_energy_mae=validation_mae,
        fit_counts={
            "train": len(train),
            "validation": len(validation),
            "development": len(development),
        },
    )
