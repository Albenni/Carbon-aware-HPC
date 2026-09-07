"""Historical archive of the forecasts a scheduler could have known in 2020.

Electricity Maps supplies observations, not an archive of the forecasts it
published at the time, so the forecasts a replayed scheduler consumes have to be
reconstructed. For each issue time on a fixed cadence, a model builds its
features from observations complete at that instant and predicts the following
day; the resulting trajectory is stored together with the metadata needed to
reproduce it. A snapshot's ``issue_time`` is the moment from which the
scheduler may use it, never the moment its targets occur.

Generation is deterministic: the same actual cache, protocol and model produce
a byte-identical archive, so nothing records a wall-clock generation time.
"""

import argparse
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys

from .baselines import BASELINE_PERIODS, BaselineCarbonIntensityProvider
from .evaluate import load_inputs, selected
from .forecasting import CADENCE, HORIZON, RidgeCarbonIntensityForecaster
from .series import CarbonIntensityForecast, CarbonIntensitySample
from .protocol import TemporalProtocol
from .series import (
    FIFTEEN_MINUTES as STEP,
    CarbonIntensityProvider,
    ForecastUnavailableError,
    TimeSeriesCarbonIntensityProvider,
    aware_utc,
    bucket_start,
)
from .walkforward import WalkForwardForecaster, pooled_design


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ForecastArchive:
    """Snapshots in issue order, with the provenance needed to replay them."""

    metadata: dict
    snapshots: tuple[CarbonIntensityForecast, ...]

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("a forecast archive must contain at least one snapshot")
        issued = [snapshot.issue_time for snapshot in self.snapshots]
        if issued != sorted(set(issued)):
            raise ValueError("archive snapshots must be chronological and unique")

    @classmethod
    def generate(
        cls, model_name: str, get_forecast: Callable[[datetime, timedelta], CarbonIntensityForecast],
        protocol: TemporalProtocol, *, partition: str = "test", horizon: timedelta = HORIZON,
        cadence: timedelta = CADENCE, metadata: dict | None = None,
    ) -> "ForecastArchive":
        """Issue one forecast per cadence step across the partition.

        Unlike an evaluation origin, a snapshot needs no labels: its trajectory
        may reach past the end of the partition, which is exactly what a
        scheduler deciding on the final day requires. Causality is enforced
        upstream, by the protocol that hands each model its observations.
        """
        for name, value in (("horizon", horizon), ("cadence", cadence)):
            if not isinstance(value, timedelta) or value <= timedelta(0) or value % STEP:
                raise ValueError(f"{name} must be a positive multiple of 15 minutes")
        start, end = protocol.intervals[partition]
        if start != bucket_start(start, cadence):
            raise ValueError("the partition must start on the cadence grid")
        snapshots, issue = [], start
        while issue < end:
            snapshots.append(get_forecast(issue, horizon))
            issue += cadence
        return cls({
            "schema_version": SCHEMA_VERSION, "model_name": model_name, "partition": partition,
            "granularity_seconds": int(STEP.total_seconds()),
            "horizon_hours": horizon.total_seconds() / 3600,
            "cadence_minutes": cadence.total_seconds() / 60,
            "snapshots": len(snapshots),
            "first_issue_time": snapshots[0].issue_time.isoformat(),
            "last_issue_time": snapshots[-1].issue_time.isoformat(),
            "issue_time_meaning": "the instant from which a scheduler may use the snapshot",
            **(metadata or {}),
        }, tuple(snapshots))

    def verify(self) -> int:
        """Recheck that every snapshot could have existed at its own issue time.

        The model must already have been trained, the trajectory must start at
        the issue time and cover a complete 15-minute grid, and issue times must
        follow the declared cadence without a gap. A walk-forward archive also
        records its refit schedule, and each snapshot is then checked against the
        fit that actually served it, so ``training_cutoff <= issue_time`` holds
        per snapshot and not only for the first fit.
        """
        cadence = timedelta(minutes=self.metadata["cadence_minutes"])
        buckets = int(self.metadata["horizon_hours"] * 3600 // STEP.total_seconds())
        available = self.metadata.get("training_available_at")
        available = aware_utc(datetime.fromisoformat(available)) if available else None
        schedule = [
            tuple(aware_utc(datetime.fromisoformat(entry[key]))
                  for key in ("refit_at", "training_available_at"))
            for entry in self.metadata.get("refits", ())
        ]
        if schedule != sorted(schedule):
            raise ValueError("the refit schedule must be chronological")
        previous = None
        for snapshot in self.snapshots:
            issue = snapshot.issue_time
            if previous is not None and issue != previous + cadence:
                raise ValueError(f"snapshot {issue.isoformat()} breaks the declared cadence")
            if available is not None and available > issue:
                raise ValueError(f"snapshot {issue.isoformat()} predates its own training data")
            if schedule:
                serving = bisect_right([refit for refit, _ in schedule], issue) - 1
                if serving < 0 or schedule[serving][1] > issue:
                    raise ValueError(f"snapshot {issue.isoformat()} predates the fit serving it")
            if len(snapshot.samples) != buckets or any(
                sample.timestamp != issue + index * STEP
                for index, sample in enumerate(snapshot.samples)
            ):
                raise ValueError(f"snapshot {issue.isoformat()} is not a complete trajectory")
            previous = issue
        return len(self.snapshots)

    def get_forecast(
        self, issue_time: datetime, horizon: timedelta | None = None,
    ) -> CarbonIntensityForecast:
        """Return the most recent snapshot already issued at ``issue_time``.

        ``horizon`` truncates the trajectory, measured from the snapshot's own
        issue time. On the cadence grid the two coincide, so an archive answers
        the provider call exactly as the model that produced it did.
        """
        issue_time = aware_utc(issue_time, "issue_time")
        index = bisect_right([snapshot.issue_time for snapshot in self.snapshots], issue_time) - 1
        if index < 0:
            raise ForecastUnavailableError(
                f"no forecast had been issued at {issue_time.isoformat()}"
            )
        snapshot = self.snapshots[index]
        if horizon is None:
            return snapshot
        if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
            raise ValueError("horizon must be a positive timedelta")
        count = (horizon + STEP - timedelta(microseconds=1)) // STEP
        if count > len(snapshot.samples):
            raise ValueError("the archive does not reach the requested horizon")
        return CarbonIntensityForecast(snapshot.issue_time, snapshot.samples[:count])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "metadata": self.metadata,
            "snapshots": [{
                "issue_time": snapshot.issue_time.isoformat(),
                "start": snapshot.samples[0].timestamp.isoformat(),
                "end": (snapshot.samples[-1].timestamp + STEP).isoformat(),
                "values_gco2e_per_kwh": [
                    sample.intensity_gco2e_per_kwh for sample in snapshot.samples
                ],
            } for snapshot in self.snapshots],
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ForecastArchive":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported forecast archive schema")
        if metadata["granularity_seconds"] != STEP.total_seconds():
            raise ValueError("forecast archives require a 15-minute grid")
        snapshots = []
        for record in payload["snapshots"]:
            start = aware_utc(datetime.fromisoformat(record["start"]))
            samples = tuple(
                CarbonIntensitySample(start + index * STEP, value)
                for index, value in enumerate(record["values_gco2e_per_kwh"])
            )
            if aware_utc(datetime.fromisoformat(record["end"])) != samples[-1].timestamp + STEP:
                raise ValueError(f"snapshot {record['issue_time']} declares an inconsistent end")
            snapshots.append(CarbonIntensityForecast(
                aware_utc(datetime.fromisoformat(record["issue_time"])), samples,
            ))
        archive = cls(metadata, tuple(snapshots))
        archive.verify()
        return archive


class ArchiveCarbonIntensityProvider(CarbonIntensityProvider):
    """Serve a replayed decision only what the archive had already published.

    This is the seam between the reconstructed archive and the simulator. A
    decision at 07:04 gets the 07:00 snapshot, and the coverage it needs is
    measured from the decision instant rather than from the issue time: fifty
    minutes into an hourly issue there are fifty fewer minutes of trajectory
    left, and asking for more than remains is an error, not a shorter answer.

    Actuals stay reachable through the same object, because the oracle and the
    ex-post accounting both need them, but they never stand in for a forecast.
    Nothing here falls back to the observation it is supposed to be predicting,
    so a window the archive cannot cover raises instead of leaking the answer.
    """

    def __init__(
        self, actual: TimeSeriesCarbonIntensityProvider, archive: ForecastArchive,
    ) -> None:
        if actual.granularity != STEP:
            raise ValueError("forecast archives require 15-minute actuals")
        self.actual = actual
        self.archive = archive

    @property
    def granularity(self) -> timedelta:
        return self.actual.granularity

    def get_actual(self, timestamp: datetime) -> float:
        return self.actual.get_actual(timestamp)

    def get_actual_range(self, start: datetime, end: datetime) -> tuple[CarbonIntensitySample, ...]:
        return self.actual.get_actual_range(start, end)

    def get_forecast(self, as_of: datetime, horizon: timedelta) -> CarbonIntensityForecast:
        """The latest forecast issued at or before ``as_of``, covering the window."""
        as_of = aware_utc(as_of, "as_of")
        if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
            raise ValueError("horizon must be a positive timedelta")
        snapshot = self.archive.get_forecast(as_of)
        first, end = bucket_start(as_of, STEP), as_of + horizon
        samples = tuple(
            sample for sample in snapshot.samples if first <= sample.timestamp < end
        )
        if (
            not samples
            or samples[0].timestamp != first
            or samples[-1].timestamp + STEP < end
        ):
            raise ForecastUnavailableError(
                f"the forecast issued at {snapshot.issue_time.isoformat()} does not cover "
                f"{first.isoformat()} to {end.isoformat()}"
            )
        return CarbonIntensityForecast(snapshot.issue_time, samples)


def update_period(model_path: Path, fallback: timedelta) -> timedelta:
    """The retraining period the replay archives, taken from the validation decision.

    When a walk-forward schedule was selected, that is the period. When the frozen
    model won, the archive still carries the schedule that scored best on
    validation, so the comparison on the PM100 period faces the strongest arm.
    """
    path = model_path.parent / "selected_update.json"
    if path.exists():
        decision = json.loads(path.read_text(encoding="utf-8"))
        days = decision["refit_days"] or decision.get("comparison_refit_days")
        if days:
            return timedelta(days=days)
    return fallback


def archive_models(
    actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol, model_path: Path,
    partition: str = "test", refit: timedelta = timedelta(days=30),
):
    """Yield each model behind the same provider call, with its provenance.

    The ridge selected on validation is loaded rather than refitted, so the
    archive replays those exact weights. Two further archives answer the update
    question on the replay itself: ``ridge_refit_once`` is the same configuration
    trained on everything published before the period starts, and
    ``ridge_refit_Nd`` keeps learning during it. Both share one set of design
    matrices, which is the expensive part.
    """
    for method in BASELINE_PERIODS:
        yield method, BaselineCarbonIntensityProvider(actual, protocol, method).get_forecast, {
            "model_version": None, "training_cutoff": None,
            "training_note": "unfitted baseline; no training data is involved",
            "baseline_period_hours": BASELINE_PERIODS[method].total_seconds() / 3600,
        }
    if not model_path.exists():
        raise ValueError(f"{model_path} is missing; run carbon_intensity.evaluate to fit it")
    ridge = RidgeCarbonIntensityForecaster.load(model_path, actual, protocol)
    yield ridge.metadata["model_name"], ridge.get_forecast, {
        key: ridge.metadata[key] for key in (
            "model_version", "alpha", "feature_names", "history_span_days", "lookback_hours",
            "strategy", "training_cutoff", "training_available_at", "training_examples",
            "training_window_start", "numpy_version", "sklearn_version",
        )
    } | {"model_path": str(model_path), "model_sha256": sha256(model_path.read_bytes()).hexdigest()}
    configuration = {"alpha": ridge.metadata["alpha"], "history_span": None} | selected(model_path)
    shared = {key: ridge.metadata[key] for key in (
        "feature_names", "lookback_hours", "numpy_version", "sklearn_version",
    )}
    rows = pooled_design(actual, protocol)
    for period in (None, update_period(model_path, refit)):
        forecaster = WalkForwardForecaster.build(
            actual, protocol, rows, replay=partition, period=period, **configuration,
        )
        yield forecaster.metadata["model_name"], forecaster.get_forecast, forecaster.metadata | shared


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=root / "data/carbon_intensity/actual/actual.json")
    parser.add_argument("--protocol", type=Path, default=root / "data/carbon_intensity/actual/protocol.json")
    parser.add_argument("--output-dir", type=Path, default=root / "data/carbon_intensity/snapshots")
    parser.add_argument("--model", type=Path, default=(
        root / "data/carbon_intensity/forecasts/ridge_direct.json"
    ))
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    parser.add_argument("--refit-days", type=int, default=30,
                        help="walk-forward period when validation kept the frozen model")
    args = parser.parse_args()
    try:
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        for name, get_forecast, provenance in archive_models(
            actual, protocol, args.model, args.partition, timedelta(days=args.refit_days),
        ):
            archive = ForecastArchive.generate(
                name, get_forecast, protocol, partition=args.partition,
                horizon=timedelta(hours=args.horizon_hours),
                cadence=timedelta(minutes=args.cadence_minutes),
                metadata={
                    **protocol.metadata(), **provenance,
                    "actual_path": str(args.actual), "actual_sha256": actual_hash,
                    "protocol_path": str(args.protocol),
                },
            )
            count = archive.verify()
            path = archive.save(args.output_dir / f"{args.partition}_{name}.json")
            print(f"{name}: {count} snapshots "
                  f"{archive.metadata['first_issue_time']} .. {archive.metadata['last_issue_time']} "
                  f"-> {path.name}")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        print(f"Snapshot generation failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {args.partition} forecast archives to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
