"""Ridge equations corrected by a gradient-boosted residual, on the same grid.

``forecasting.RidgeCarbonIntensityForecaster`` answers every target bucket with
one linear equation. That is interpretable and, on this zone, already strong,
but it leaves two things on the table. Its features describe the issue time far
better than the target bucket, and the relationship it fits is linear in a
signal whose daily shape is not.

This model keeps the direct per-bucket ridge as the *level carrier* and fits a
gradient-boosted correction to its residual. Two properties make that safe:

* Every feature the booster sees is a **deviation** from the last observed
  value, never an absolute intensity. IT-NO fell from a 392 gCO2eq/kWh yearly
  mean in 2016 to 289 in 2020, and a tree trained on the old levels cannot
  extrapolate to the new ones. The ridge carries the level; the booster only
  ever says *how far from it*, which is the part that transfers across years.
* Residual scale grows steeply with lead time, so one pooled booster would
  spend its whole capacity on the far horizons and damage the near ones. The
  leads are therefore split into bands, each with its own correction.

Causality is unchanged from the rest of the package: a feature at issue time
``t`` reads only buckets that ended at or before ``t``, and every fit is
restricted to rows whose complete 24-hour label window was already published at
the refit cutoff. ``TemporalProtocol`` is asked to confirm both.

Run ``python -m carbon_intensity.boosted`` to score it against the baselines and
the incumbent ridge on validation, then on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

import numpy as np

from .protocol import TemporalProtocol
from .series import (
    FIFTEEN_MINUTES as STEP,
    CarbonIntensityForecast,
    CarbonIntensitySample,
    TimeSeriesCarbonIntensityProvider,
    aware_utc,
    bucket_start,
)

HORIZON = timedelta(hours=24)
CADENCE = timedelta(hours=1)
BUCKETS = HORIZON // STEP
DAY, WEEK = 96, 672
# The deepest feature reads eight weeks back from the *target*, so an origin
# needs that much closed history behind it.
LOOKBACK = 8 * WEEK
MIN_HISTORY = LOOKBACK + 1
MODEL_VERSION = 1

LAGS = (1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672)
ROLLS = (4, 12, 24, 96, 672, 2880)
EWMS = (2, 8, 48)
DIFF_LAGS = (2, 4, 24, 96, 672)
DIFF_ROLLS = (24, 96, 672, 2880)
DAY_BACK = (1, 2, 3, 7)
WEEK_BACK = (1, 2)
DAY_PROFILES = (3, 7)
WEEK_PROFILES = (2, 4, 8)
# Near, mid and far bands: 0-1 h, 1-6 h and 6-24 h ahead.
LEAD_BANDS = (4, 24)


def _cyclical(values: np.ndarray, period: float) -> list[np.ndarray]:
    angle = 2 * np.pi * np.asarray(values, dtype=float) / period
    return [np.sin(angle), np.cos(angle)]


def calendar_columns(stamps) -> tuple[np.ndarray, list[str]]:
    """UTC calendar of a bucket; deterministic, so never a source of leakage."""
    hour = np.asarray(stamps.hour + stamps.minute / 60, dtype=float)
    weekday = stamps.dayofweek.to_numpy().astype(float)
    columns, names = [], []
    for values, period, name in (
        (hour, 24, "hour"), (weekday, 7, "wday"), (stamps.dayofyear.to_numpy(), 365.25, "doy"),
    ):
        columns += _cyclical(values, period)
        names += [f"t_{name}_sin", f"t_{name}_cos"]
    columns.append((weekday >= 5).astype(float))
    names.append("t_weekend")
    return np.column_stack(columns).astype(np.float32), names


def issue_columns(values: np.ndarray, stamps) -> tuple[np.ndarray, list[str]]:
    """Everything readable at issue index ``i``, built from the series shifted by one.

    Shifting first is what makes the bucket in progress unreachable: column ``i``
    of every rolling window ends at ``i - 1``, the last bucket that has closed.
    """
    import pandas as pd

    series = pd.Series(values)
    past = series.shift(1)
    columns, names = [], []
    for lag in LAGS:
        columns.append(series.shift(lag).to_numpy()); names.append(f"lag{lag}")
    for window in ROLLS:
        rolling = past.rolling(window)
        for stat in ("mean", "min", "max", "std"):
            columns.append(getattr(rolling, stat)().to_numpy()); names.append(f"roll{window}_{stat}")
    for halflife in EWMS:
        columns.append(past.ewm(halflife=halflife).mean().to_numpy()); names.append(f"ewm{halflife}")
    last = past.to_numpy()
    for lag in DIFF_LAGS:
        columns.append(last - series.shift(lag).to_numpy()); names.append(f"d_lag{lag}")
    for window in DIFF_ROLLS:
        columns.append(last - past.rolling(window).mean().to_numpy()); names.append(f"d_roll{window}")
    for values_, period, name in (
        (np.asarray(stamps.hour + stamps.minute / 60, dtype=float), 24, "hour"),
        (stamps.dayofweek.to_numpy().astype(float), 7, "wday"),
    ):
        columns += _cyclical(values_, period)
        names += [f"i_{name}_sin", f"i_{name}_cos"]
    return np.column_stack(columns).astype(np.float32), names


def target_columns(
    values: np.ndarray, calendar: np.ndarray, origins: np.ndarray, leads: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Columns that depend on the target bucket as well as the issue time.

    Every source index is at most ``i - 1``: a bucket one day before a target at
    most 24 hours ahead is, at the latest, the bucket before the issue time.
    That is what lets a direct equation see the seasonal profile *at its own
    target hour* without seeing the future.
    """
    target = origins[:, None] + leads[None, :]
    last = origins[:, None] - 1
    columns, names = [], []
    for back in DAY_BACK:
        columns.append(values[target - back * DAY]); names.append(f"y_d{back}")
        columns.append(values[target - back * DAY] - values[last - back * DAY])
        names.append(f"shape_d{back}")
    for back in WEEK_BACK:
        columns.append(values[target - back * WEEK]); names.append(f"y_w{back}")
        columns.append(values[target - back * WEEK] - values[last - back * WEEK])
        names.append(f"shape_w{back}")
    for span in DAY_PROFILES:
        columns.append(np.mean([values[target - k * DAY] for k in range(1, span + 1)], axis=0))
        names.append(f"dprof{span}")
    for span in WEEK_PROFILES:
        columns.append(np.mean([values[target - k * WEEK] for k in range(1, span + 1)], axis=0))
        names.append(f"wprof{span}")
    columns.append(np.broadcast_to((leads + 1) * 0.25, target.shape)); names.append("lead_h")
    stacked = np.stack([np.asarray(column, dtype=np.float32) for column in columns], axis=-1)
    return np.concatenate([stacked, calendar[target]], axis=-1), names


LEVEL_PREFIXES = ("lag", "ewm", "y_d", "y_w", "dprof", "wprof")


def level_mask(names: list[str]) -> np.ndarray:
    """Columns carrying an absolute intensity, which the booster sees as deviations."""
    def absolute(name: str) -> bool:
        if name.startswith("roll"):
            return not name.endswith("_std")
        return name.startswith(LEVEL_PREFIXES)
    return np.array([absolute(name) for name in names])


def ridge_solve(design: np.ndarray, labels: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Standardise, penalise, then return the weights in original feature units."""
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    standardised = (design - mean) / scale
    centre = labels.mean()
    weights = np.linalg.solve(
        standardised.T @ standardised + alpha * np.eye(standardised.shape[1]),
        standardised.T @ (labels - centre),
    )
    coefficients = weights / scale
    return coefficients, centre - coefficients @ mean


def booster(seed: int = 0):
    """The residual learner: shallow, L1, and small enough not to chase noise.

    Absolute error is the loss because MAE is the reported metric, and a
    15-minute carbon-intensity series has heavy enough tails that the
    conditional mean is the wrong target for it.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


@dataclass(frozen=True, slots=True)
class Fit:
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_models: tuple
    rows: int
    latest_label: int


@dataclass
class BoostedCarbonIntensityForecaster:
    """Direct ridge per bucket plus a banded residual booster, refit on schedule."""

    actual: TimeSeriesCarbonIntensityProvider
    protocol: TemporalProtocol
    values: np.ndarray = field(repr=False)
    start: datetime
    shared: np.ndarray = field(repr=False)
    calendar: np.ndarray = field(repr=False)
    names: list = field(repr=False)
    fits: dict = field(repr=False)
    period: timedelta | None
    metadata: dict = field(repr=False)
    level_columns: np.ndarray = field(default=None, repr=False)

    # ------------------------------------------------------------- setup ----
    @staticmethod
    def grid(actual: TimeSeriesCarbonIntensityProvider) -> tuple[np.ndarray, datetime, object]:
        import pandas as pd

        if actual.granularity != STEP:
            raise ValueError("boosted forecasts require 15-minute actuals")
        samples = actual.samples
        values = np.array([s.intensity_gco2e_per_kwh for s in samples], dtype=float)
        start = samples[0].timestamp
        stamps = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
        if stamps[-1] != samples[-1].timestamp:
            raise ValueError("the actual series must be gap-free on the 15-minute grid")
        return values, start, stamps

    def index(self, moment: datetime) -> int:
        moment = aware_utc(moment, "timestamp")
        if moment != bucket_start(moment):
            raise ValueError("timestamps must align with the 15-minute UTC grid")
        return (moment - self.start) // STEP

    def moment(self, index: int) -> datetime:
        return self.start + int(index) * STEP

    def training_origins(self, cutoff: int, history: timedelta | None = None) -> np.ndarray:
        """Hourly origins whose entire label window was published before ``cutoff``.

        This is the only place the walk-forward rule lives: a row is admitted
        once its last label bucket has closed, never earlier.
        """
        earliest = self.moment(self.index(self.protocol.train_start) + MIN_HISTORY)
        aligned = bucket_start(earliest, CADENCE)
        first = self.index(aligned if aligned == earliest else aligned + CADENCE)
        origins = np.arange(first, cutoff - BUCKETS + 1, CADENCE // STEP)
        if history is not None:
            origins = origins[origins >= cutoff - history // STEP]
        if not origins.size:
            raise ValueError(f"no training rows are observable at {self.moment(cutoff).isoformat()}")
        return origins

    # ------------------------------------------------------------ design ----
    def design(self, origins: np.ndarray, leads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        targets, target_names = target_columns(self.values, self.calendar, origins, leads)
        shared = np.broadcast_to(
            self.shared[origins][:, None, :], (len(origins), len(leads), self.shared.shape[1]),
        )
        return np.concatenate([shared, targets], axis=-1), target_names

    def deviations(self, origins: np.ndarray, leads: np.ndarray) -> np.ndarray:
        """The booster's view: every level column re-expressed against the last observation."""
        block, _ = self.design(origins, leads)   # design already returns a fresh array
        block[..., self.level_columns] -= self.values[origins - 1][:, None, None].astype(np.float32)
        return block.reshape(-1, block.shape[-1])

    # --------------------------------------------------------------- fit ----
    def fit_at(self, cutoff: int, alpha: float, history: timedelta | None) -> Fit:
        origins = self.training_origins(cutoff, history)
        every = np.arange(BUCKETS)
        labels = self.values[origins[:, None] + every[None, :]]
        coefficients = np.empty((BUCKETS, len(self.names)))
        intercept = np.empty(BUCKETS)
        fitted = np.empty_like(labels)
        # One bucket at a time: the full (origin, lead, feature) block would be
        # gigabytes, and each direct equation only ever needs its own slice.
        for bucket in range(BUCKETS):
            slice_, _ = self.design(origins, every[bucket:bucket + 1])
            design_ = slice_[:, 0, :].astype(float)
            coefficients[bucket], intercept[bucket] = ridge_solve(design_, labels[:, bucket], alpha)
            fitted[:, bucket] = design_ @ coefficients[bucket] + intercept[bucket]
        sample = origins[::2]
        models = []
        for band in np.split(every, LEAD_BANDS):
            leads = band[::2]
            residual = (labels[::2][:, leads] - fitted[::2][:, leads]).ravel()
            models.append((band, booster().fit(self.deviations(sample, leads), residual)))
        return Fit(coefficients, intercept, tuple(models), len(origins),
                   int(origins[-1] + BUCKETS - 1))

    @classmethod
    def build(
        cls, actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol, *,
        replay: str = "validation", period: timedelta | None = None, alpha: float = 1.0,
        history: timedelta | None = None,
    ) -> "BoostedCarbonIntensityForecaster":
        """Fit one model per scheduled cutoff before any forecast is issued."""
        values, start, stamps = cls.grid(actual)
        shared, shared_names = issue_columns(values, stamps)
        calendar, calendar_names = calendar_columns(stamps)
        model = cls(actual, protocol, values, start, shared, calendar, [], {}, period, {})
        _, target_names = target_columns(values, calendar, np.array([MIN_HISTORY]), np.array([0]))
        model.names = shared_names + target_names + calendar_names
        model.level_columns = level_mask(model.names)

        begin, end = protocol.intervals[replay]
        cutoffs = [begin]
        if period is not None:
            while cutoffs[-1] + period < end:
                cutoffs.append(cutoffs[-1] + period)
        schedule = []
        for cutoff in cutoffs:
            fit = model.fit_at(model.index(cutoff), alpha, history)
            model.fits[cutoff] = fit
            latest = model.moment(fit.latest_label)
            schedule.append({
                "refit_at": cutoff.isoformat(), "training_cutoff": latest.isoformat(),
                "training_available_at": (latest + STEP + protocol.observation_delay).isoformat(),
                "training_examples": fit.rows,
            })
        model.metadata = {
            **protocol.metadata(),
            "model_name": "boosted_ridge" if period is None else f"boosted_ridge_{period.days}d",
            "model_version": MODEL_VERSION, "alpha": alpha,
            "history_span_days": None if history is None else history.days,
            "refit_days": None if period is None else period.days,
            "replay_partition": replay, "refits": schedule,
            "training_cutoff": schedule[0]["training_cutoff"],
            "training_available_at": schedule[0]["training_available_at"],
            "training_examples": schedule[0]["training_examples"],
            "feature_names": model.names, "lead_bands": list(LEAD_BANDS),
            "residual_loss": "absolute_error", "prediction_floor": 0,
            "strategy": "direct ridge per bucket, gradient-boosted residual per lead band",
            "residual_features": "level columns expressed as deviations from the last observation",
            "update_rule": "a training row is used only once its whole label window is observable",
        }
        return model

    # ----------------------------------------------------------- forecast ---
    def refit_at(self, issue_time: datetime) -> datetime:
        """The most recent scheduled cutoff at or before ``issue_time``."""
        first = min(self.fits)
        if issue_time < first:
            raise ValueError("the replay had not started at issue_time")
        if self.period is None:
            return first
        return min(first + (issue_time - first) // self.period * self.period, max(self.fits))

    def provenance(self, issue_time: datetime) -> dict:
        """Which fit serves this issue time, so a snapshot records its own cutoff."""
        refit = self.refit_at(issue_time).isoformat()
        return next(e for e in self.metadata["refits"] if e["refit_at"] == refit)

    def get_forecast(self, issue_time: datetime, horizon: timedelta) -> CarbonIntensityForecast:
        issue_time = aware_utc(issue_time, "issue_time")
        if not isinstance(horizon, timedelta) or not timedelta(0) < horizon <= HORIZON:
            raise ValueError("horizon must be positive and at most 24 hours")
        if issue_time != bucket_start(issue_time):
            raise ValueError("issue_time must align with the 15-minute UTC grid")
        fit = self.fits[self.refit_at(issue_time)]
        if issue_time < datetime.fromisoformat(self.provenance(issue_time)["training_available_at"]):
            raise ValueError("the serving fit was not available at issue_time")
        # The protocol re-checks, on the actual samples themselves, that every
        # bucket this forecast reads had already closed at issue_time.
        self.protocol.history(self.actual, issue_time, LOOKBACK * STEP)
        origin = np.array([self.index(issue_time)])
        count = (horizon + STEP - timedelta(microseconds=1)) // STEP
        leads = np.arange(count)
        block, _ = self.design(origin, leads)
        prediction = (
            np.einsum("nbf,bf->nb", block.astype(float), fit.coefficients[:count])
            + fit.intercept[:count]
        )
        for band, model in fit.residual_models:
            band = band[band < count]
            if band.size:
                prediction[:, band] += model.predict(self.deviations(origin, band))
        values = np.maximum(prediction[0], 0)
        return CarbonIntensityForecast(issue_time, tuple(
            CarbonIntensitySample(issue_time + i * STEP, float(v)) for i, v in enumerate(values)
        ))


def main() -> int:
    from .evaluate import (
        base_parser, evaluate_forecast, forecasters, load_inputs, provenance, report,
        run_entry_point, write_evaluation,
    )

    parser = base_parser(__doc__)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--refit-days", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--baselines", action="store_true",
                        help="also score the baselines and the incumbent ridge")
    args = parser.parse_args()

    def work() -> None:
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        rows = []
        if args.baselines:
            for name, get_forecast, cutoff in forecasters(
                actual, protocol, args.output_dir / "ridge_direct.json",
            ):
                row = {"model": name, "training_cutoff": cutoff, **evaluate_forecast(
                    actual, protocol, get_forecast, partition=args.partition,
                )}
                rows.append(row)
                report(name, row)
        model = BoostedCarbonIntensityForecaster.build(
            actual, protocol, replay=args.partition, alpha=args.alpha,
            period=None if args.refit_days is None else timedelta(days=args.refit_days),
        )
        row = {"model": model.metadata["model_name"],
               "training_cutoff": model.metadata["training_cutoff"],
               **evaluate_forecast(actual, protocol, model.get_forecast, partition=args.partition)}
        rows.append(row)
        report(row["model"], row, f", rows={model.metadata['training_examples']}")
        write_evaluation(args.output_dir, f"boosted_{args.partition}", rows, {
            **provenance(args, protocol, actual_hash), "evaluation": rows,
            "model": model.metadata,
        })
        print(f"Saved {args.partition} results to {args.output_dir}")

    return run_entry_point("Boosted forecast evaluation", work)


if __name__ == "__main__":
    raise SystemExit(main())
