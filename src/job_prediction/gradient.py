"""Gradient-boosted duration, power, and energy models.

The second of the two predictor families. It shares the column vocabulary and
the temporal-cutoff helpers in :mod:`job_prediction.data` with the log-ridge
baseline in :mod:`job_prediction.model`, and nothing else: the features come
from :mod:`job_prediction.features` and the estimators from
:mod:`job_prediction.experiment`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import (
    AVERAGE_POWER_WATTS,
    COMPLETION_TIME,
    DURATION_SECONDS,
    ENERGY_KWH,
    JOB_ID,
    MODEL_FORMAT_VERSION,
    PREDICTED_AVERAGE_POWER_WATTS,
    PREDICTED_DURATION_SECONDS,
    PREDICTED_ENERGY_KWH,
    SUBMIT_TIME,
    TARGETS,
    WATT_SECONDS_PER_KILOWATT_HOUR,
    PredictionComposition,
    TemporalSplit,
    completed_by,
    require_columns,
)


@dataclass(frozen=True, slots=True)
class TargetModel:
    """One fitted estimator plus the transform and bounds it was chosen with."""

    target: str
    transform: str
    weight: str
    estimator: object
    minimum: float
    maximum: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.estimator.predict(features), dtype=float)
        if self.transform == "log1p":
            raw = np.expm1(np.clip(raw, -50.0, 50.0))
        elif self.transform != "raw":
            raise ValueError(f"unknown transform {self.transform!r}")
        # Same guard as the ridge baseline: never leave the range of targets the
        # fitting period actually contained, and never emit a non-positive value
        # into an energy identity.
        return np.clip(raw, self.minimum, self.maximum)


@dataclass(frozen=True, slots=True)
class GradientJobPredictor:
    """Boosted duration, power, and energy models over causal-history features.

    ``predict`` deliberately takes a whole trace rather than an arbitrary slice:
    the history features of a job are computed from the jobs that precede it, so
    handing this a partial frame would silently change them.
    """

    spec: object
    feature_names: tuple[str, ...]
    duration_model: TargetModel
    power_model: TargetModel
    energy_model: TargetModel
    composition: PredictionComposition

    def feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        from .features import build_features

        built = build_features(data, self.spec).reindex(columns=self.feature_names)
        return built.to_numpy(dtype=float)

    def predict_components(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        features = self.feature_matrix(data)
        return {
            DURATION_SECONDS: self.duration_model.predict(features),
            AVERAGE_POWER_WATTS: self.power_model.predict(features),
            ENERGY_KWH: self.energy_model.predict(features),
        }

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict a positive, physically consistent description of every row."""

        components = self.predict_components(data)
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
            index=data.index,
        )
        if JOB_ID in data:
            result.insert(0, JOB_ID, data[JOB_ID].to_numpy())
        return result.reset_index(drop=True)

    def save(self, path: str | Path, *, metadata: Mapping[str, object] | None = None) -> Path:
        """Persist the fitted trees, and the audit trail beside them as JSON.

        Boosted trees have no compact honest text form, so the estimator itself
        is pickled; everything a reader needs to judge the protocol - features,
        transforms, split boundaries, hyperparameters - stays readable.
        """

        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        sidecar = destination.with_suffix(".json")
        with sidecar.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format_version": MODEL_FORMAT_VERSION,
                    "metadata": dict(metadata or {}),
                    "composition": self.composition.value,
                    "feature_groups": list(self.spec.groups),
                    "feature_names": list(self.feature_names),
                    "models": {
                        model.target: {
                            "transform": model.transform,
                            "weight": model.weight,
                            "minimum": model.minimum,
                            "maximum": model.maximum,
                            "estimator": type(model.estimator).__name__,
                            "parameters": {
                                key: value
                                for key, value in getattr(
                                    model.estimator, "get_params", dict
                                )().items()
                                if isinstance(value, (int, float, str, bool, type(None)))
                            },
                        }
                        for model in (
                            self.duration_model,
                            self.power_model,
                            self.energy_model,
                        )
                    },
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "GradientJobPredictor":
        import joblib

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise ValueError("artifact does not contain a gradient job predictor")
        return loaded


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """A frozen per-target recipe: everything selected on validation."""

    model: str = "hist_gbr"
    transform: str = "log1p"
    loss: str = "absolute_error"
    weight: str = "none"
    overrides: tuple[tuple[str, object], ...] = ()

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "transform": self.transform,
            "loss": self.loss,
            "weight": self.weight,
            "hyperparameters": dict(self.overrides),
        }


@dataclass(frozen=True, slots=True)
class GradientTrainingResult:
    predictor: GradientJobPredictor
    configs: Mapping[str, TargetConfig]
    validation_energy_mae: Mapping[PredictionComposition, float]
    fit_counts: Mapping[str, int]
    feature_groups: tuple[str, ...]


def _fit_target(
    features: np.ndarray,
    target: np.ndarray,
    *,
    target_name: str,
    config: TargetConfig,
    seed: int = 0,
) -> TargetModel:
    from .experiment import TRANSFORMS, build_estimator

    forward, _ = TRANSFORMS[config.transform]
    weights = {
        "none": None,
        "sqrt_duration": np.sqrt(target),
        "quartic_duration": np.power(target, 0.25),
        "log_duration": np.log1p(target),
        "log_duration_squared": np.square(np.log1p(target)),
    }[config.weight]
    estimator = build_estimator(
        config.model, config.loss, seed=seed, **dict(config.overrides)
    )
    if weights is None:
        estimator.fit(features, forward(target))
    else:
        estimator.fit(features, forward(target), sample_weight=weights)
    return TargetModel(
        target=target_name,
        transform=config.transform,
        weight=config.weight,
        estimator=estimator,
        minimum=max(float(np.min(target)), 1e-9),
        maximum=float(np.max(target)),
    )


def fit_gradient_predictor(
    data: pd.DataFrame,
    split: TemporalSplit,
    *,
    spec: object,
    configs: Mapping[str, TargetConfig],
    seed: int = 0,
) -> GradientTrainingResult:
    """Fit the frozen recipes, choose the composition on validation, refit.

    ``data`` must be the complete chronological trace: the causal history
    features of a validation or test job are built from the jobs that precede
    it, which live in earlier partitions. Only labels are partitioned, and only
    labels observable at each cutoff are ever fitted.
    """

    from .features import build_features

    features = build_features(data, spec)
    names = tuple(features.columns)
    matrix = features.to_numpy(dtype=float)
    position = pd.Series(np.arange(len(data)), index=data[JOB_ID].to_numpy())

    def rows_of(frame: pd.DataFrame, cutoff: object, period: str) -> np.ndarray:
        available = completed_by(frame, cutoff, period_name=period)
        return position.loc[available[JOB_ID].to_numpy()].to_numpy()

    train_rows = rows_of(split.train, split.train_until, "training")
    validation_rows = rows_of(split.validation, split.validation_until, "validation")
    development_rows = rows_of(
        pd.concat([split.train, split.validation], ignore_index=True),
        split.validation_until,
        "development",
    )

    targets = {
        target: pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        for target in TARGETS
    }
    tuned = {
        target: _fit_target(
            matrix[train_rows],
            targets[target][train_rows],
            target_name=target,
            config=configs[target],
            seed=seed,
        )
        for target in TARGETS
    }

    validation_features = matrix[validation_rows]
    predicted_duration = tuned[DURATION_SECONDS].predict(validation_features)
    predicted_power = tuned[AVERAGE_POWER_WATTS].predict(validation_features)
    predicted_energy = tuned[ENERGY_KWH].predict(validation_features)
    actual_energy = targets[ENERGY_KWH][validation_rows]
    validation_mae = {
        PredictionComposition.DURATION_POWER: float(
            np.mean(
                np.abs(
                    predicted_duration
                    * predicted_power
                    / WATT_SECONDS_PER_KILOWATT_HOUR
                    - actual_energy
                )
            )
        ),
        PredictionComposition.DURATION_ENERGY: float(
            np.mean(np.abs(predicted_energy - actual_energy))
        ),
    }
    composition = min(validation_mae, key=validation_mae.__getitem__)

    final = {
        target: _fit_target(
            matrix[development_rows],
            targets[target][development_rows],
            target_name=target,
            config=configs[target],
            seed=seed,
        )
        for target in TARGETS
    }
    predictor = GradientJobPredictor(
        spec=spec,
        feature_names=names,
        duration_model=final[DURATION_SECONDS],
        power_model=final[AVERAGE_POWER_WATTS],
        energy_model=final[ENERGY_KWH],
        composition=composition,
    )
    return GradientTrainingResult(
        predictor=predictor,
        configs=dict(configs),
        validation_energy_mae=validation_mae,
        fit_counts={
            "train": int(len(train_rows)),
            "validation": int(len(validation_rows)),
            "development": int(len(development_rows)),
        },
        feature_groups=tuple(spec.groups),
    )
