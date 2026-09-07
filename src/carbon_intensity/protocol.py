"""Temporal partitions and observation boundaries for historical forecasting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import chain
import json
from pathlib import Path

from .series import CarbonIntensitySample
from .series import FIFTEEN_MINUTES, TimeSeriesCarbonIntensityProvider, aware_utc, bucket_start


@dataclass(frozen=True, slots=True)
class ForecastExample:
    issue_time: datetime
    features: tuple[CarbonIntensitySample, ...]
    targets: tuple[CarbonIntensitySample, ...]


@dataclass(frozen=True, slots=True)
class TemporalProtocol:
    """Half-open UTC splits; a bucket is observable only after it ends."""

    test_start: datetime
    test_end: datetime
    train_start: datetime = datetime(2016, 1, 1, tzinfo=timezone.utc)
    validation_start: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)
    observation_delay: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        for name in ("train_start", "validation_start", "test_start", "test_end"):
            value = aware_utc(getattr(self, name), name)
            if value != bucket_start(value):
                raise ValueError(f"{name} must align with the 15-minute UTC grid")
            object.__setattr__(self, name, value)
        if not self.train_start < self.validation_start < self.test_start < self.test_end:
            raise ValueError("train, validation and test boundaries must be strictly ordered")
        if not isinstance(self.observation_delay, timedelta) or self.observation_delay < timedelta(0):
            raise ValueError("observation_delay must be a nonnegative timedelta")

    @classmethod
    def from_workload(
        cls, path: str | Path, *, observation_delay: timedelta = timedelta(0),
    ) -> TemporalProtocol:
        """Reserve every bucket touched between the first release and last completion."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["release_time", "end_time"])
        if not table.num_rows or any(column.null_count for column in table.columns):
            raise ValueError("workload release/completion timestamps must be present")
        start = aware_utc(pc.min(table["release_time"]).as_py())
        end = aware_utc(pc.max(table["end_time"]).as_py())
        if start >= end:
            raise ValueError("workload must end after its first release")
        end_bucket = bucket_start(end)
        return cls(
            test_start=bucket_start(start),
            test_end=end_bucket if end == end_bucket else end_bucket + FIFTEEN_MINUTES,
            observation_delay=observation_delay,
        )

    @classmethod
    def load(cls, path: str | Path) -> TemporalProtocol:
        """Restore the saved split boundaries and publication delay."""
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        if metadata["granularity_seconds"] != FIFTEEN_MINUTES.total_seconds():
            raise ValueError("the temporal protocol requires 15-minute actuals")
        intervals = {
            name: tuple(datetime.fromisoformat(value) for value in metadata["intervals"][name])
            for name in ("train", "validation", "test")
        }
        if any(len(bounds) != 2 for bounds in intervals.values()):
            raise ValueError("each partition must have two boundaries")
        train, validation, test = (intervals[name] for name in ("train", "validation", "test"))
        if train[1] != validation[0] or validation[1] != test[0]:
            raise ValueError("temporal partitions must be contiguous")
        return cls(
            test_start=test[0], test_end=test[1], train_start=train[0],
            validation_start=validation[0],
            observation_delay=timedelta(seconds=metadata["observation_delay_seconds"]),
        )

    @property
    def intervals(self) -> dict[str, tuple[datetime, datetime]]:
        return {
            "train": (self.train_start, self.validation_start),
            "validation": (self.validation_start, self.test_start),
            "test": (self.test_start, self.test_end),
        }

    def split(self, actual: TimeSeriesCarbonIntensityProvider) -> dict[str, tuple[CarbonIntensitySample, ...]]:
        if actual.granularity != FIFTEEN_MINUTES:
            raise ValueError("the temporal protocol requires 15-minute actuals")
        return {name: actual.get_actual_range(*bounds) for name, bounds in self.intervals.items()}

    def validate_features(self, samples: Iterable[CarbonIntensitySample], issue_time: datetime) -> None:
        """Also validate externally built features using their source observations."""
        issue_time = aware_utc(issue_time, "issue_time")
        for sample in samples:
            if sample.timestamp < self.train_start or sample.timestamp != bucket_start(sample.timestamp):
                raise ValueError("feature observations must lie on the historical UTC grid")
            if sample.timestamp + FIFTEEN_MINUTES + self.observation_delay > issue_time:
                raise ValueError("feature observation is unavailable at issue_time")

    def history(
        self, actual: TimeSeriesCarbonIntensityProvider, issue_time: datetime, lookback: timedelta,
    ) -> tuple[CarbonIntensitySample, ...]:
        if actual.granularity != FIFTEEN_MINUTES:
            raise ValueError("the temporal protocol requires 15-minute actuals")
        if lookback <= timedelta(0) or lookback % FIFTEEN_MINUTES:
            raise ValueError("lookback must be a positive multiple of 15 minutes")
        # ponytail: lag models publication timing; replaying revisions needs publication vintages.
        end = bucket_start(aware_utc(issue_time) - self.observation_delay)
        samples = actual.get_actual_range(end - lookback, end)
        self.validate_features(samples, issue_time)
        return samples

    def _target_window(self, issue_time: datetime, end: datetime, partition: str) -> datetime:
        if partition not in self.intervals:
            raise ValueError("partition must be train, validation or test")
        lower, upper = self.intervals[partition]
        start = bucket_start(aware_utc(issue_time, "issue_time"))
        if not lower <= start < end <= upper:
            raise ValueError(f"the complete target window must belong to {partition}")
        if partition != "test" and end + self.observation_delay > upper:
            raise ValueError("target observations become available after the partition ends")
        return start

    def example(
        self, actual: TimeSeriesCarbonIntensityProvider, issue_time: datetime,
        lookback: timedelta, horizon: timedelta, *, partition: str,
    ) -> ForecastExample:
        """Return past observations and separate labels, checking the entire horizon.

        Targets may include the unfinished bucket containing issue_time; that
        bucket is never a feature. Invalid target windows fail before any lookup.
        """
        issue_time = aware_utc(issue_time, "issue_time")
        if horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        end = issue_time + horizon
        end_bucket = bucket_start(end)
        end = end_bucket if end == end_bucket else end_bucket + FIFTEEN_MINUTES
        start = self._target_window(issue_time, end, partition)
        return ForecastExample(
            issue_time, self.history(actual, issue_time, lookback),
            actual.get_actual_range(start, end),
        )

    def validate_example(self, example: ForecastExample, partition: str) -> None:
        """Recheck manually assembled examples before recording a fitted model."""
        if not example.features or not example.targets:
            raise ValueError("forecast examples require observations and targets")
        start = self._target_window(
            example.issue_time, example.targets[-1].timestamp + FIFTEEN_MINUTES, partition,
        )
        if any(sample.timestamp != start + i * FIFTEEN_MINUTES for i, sample in enumerate(example.targets)):
            raise ValueError("forecast targets must form a complete chronological trajectory")
        self.validate_features(example.features, example.issue_time)

    def metadata(self) -> dict:
        return {
            "intervals": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in self.intervals.items()},
            "granularity_seconds": int(FIFTEEN_MINUTES.total_seconds()),
            "observation_delay_seconds": self.observation_delay.total_seconds(),
            "observation_availability": "bucket end + observation delay <= issue_time",
            "selection_partition": "validation",
            "availability_assumption": "historical revised actuals, not an as-published archive",
        }

    def save_training_metadata(
        self, path: str | Path, model_name: str, examples: Iterable[ForecastExample],
    ) -> dict:
        """Persist the latest observation/label actually used, never the split limit."""
        if not model_name.strip():
            raise ValueError("model_name must be nonempty")
        cutoff = None
        count = 0
        for example in examples:
            self.validate_example(example, "train")
            latest = max(sample.timestamp for sample in chain(example.features, example.targets))
            cutoff = latest if cutoff is None else max(cutoff, latest)
            count += 1
        if cutoff is None:
            raise ValueError("training examples cannot be empty")
        metadata = {
            **self.metadata(), "model_name": model_name, "training_examples": count,
            "training_cutoff": cutoff.isoformat(),
            "training_available_at": (cutoff + FIFTEEN_MINUTES + self.observation_delay).isoformat(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
