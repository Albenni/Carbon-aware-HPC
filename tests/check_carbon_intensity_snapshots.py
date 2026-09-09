"""Small offline check: python tests/check_carbon_intensity_snapshots.py."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from carbon_intensity.baselines import BaselineCarbonIntensityProvider as Baseline
from check_carbon_intensity_history import rejects
from carbon_intensity.forecasting import BUCKETS, HORIZON, design
from carbon_intensity.series import CarbonIntensitySample as Sample
from carbon_intensity.protocol import TemporalProtocol
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
from carbon_intensity.selection import Candidate, predict_validation, select
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive


def main():
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)
    actual = Series(tuple(Sample(start + i * STEP, 100 + i % 96 + (i // 96) % 5) for i in range(2880)))
    day = timedelta(days=1)
    protocol = TemporalProtocol(start + 22 * day, start + 24 * day, start, start + 20 * day)
    candidates = (Candidate("full"), Candidate("short", history_span=8 * day),
                  Candidate("refit", refit=day))

    # Every candidate reads the same matrices; only the visible rows differ.
    train, validation = design(actual, protocol, "train"), design(actual, protocol, "validation")
    pooled = tuple(np.concatenate([left, right]) for left, right in zip(train, validation))
    counts = {}
    for candidate in candidates:
        predicted, used = predict_validation(candidate, protocol, pooled, validation)
        assert predicted.shape == (len(validation[3]), BUCKETS) and (predicted >= 0).all()
        counts[candidate.name] = used
    assert counts["full"][0] == len(train[3]) and counts["short"][0] < counts["full"][0]
    # A refit may add the validation rows whose labels have already become observable.
    assert len(counts["refit"]) == 2 and counts["refit"][1] > counts["refit"][0]

    scores = [10.0, 9.95, 8.0]
    rows = [{"model": item.name, "mae_gco2e_per_kwh": value}
            for item, value in zip(candidates, scores, strict=True)]
    assert select(rows[:2], candidates[:2])["model"] == "full"  # 0.5% never buys the extra history
    chosen = select(rows, candidates)
    assert chosen["model"] == "refit" and chosen["frozen"] is False

    forecast = Baseline(actual, protocol, "seasonal_daily").get_forecast
    archive = ForecastArchive.generate("seasonal_daily", forecast, protocol, metadata={
        "training_available_at": protocol.test_start.isoformat(),
    })
    assert archive.verify() == 48 and len(archive.snapshots[0].samples) == BUCKETS
    assert archive.metadata["first_issue_time"] == protocol.test_start.isoformat()
    # A decision between snapshots uses the one already issued, never a later one.
    assert archive.get_forecast(protocol.test_start + timedelta(minutes=40)) == archive.snapshots[0]
    assert archive.get_forecast(protocol.test_start + timedelta(hours=1)) == archive.snapshots[1]
    assert archive.get_forecast(protocol.test_start, STEP).samples == archive.snapshots[0].samples[:1]
    rejects(lambda: archive.get_forecast(protocol.test_start - STEP))
    rejects(lambda: archive.get_forecast(protocol.test_start, HORIZON + STEP))

    provider = ArchiveCarbonIntensityProvider(actual, archive)
    late = protocol.test_start + timedelta(minutes=40)
    hour = timedelta(hours=1)
    served = provider.get_forecast(late, hour)
    # The decision reads the snapshot already published, from its own bucket on.
    assert served.issue_time == archive.snapshots[0].issue_time
    assert served.samples[0].timestamp == protocol.test_start + timedelta(minutes=30)
    assert served.samples[-1].timestamp + STEP >= late + hour
    # Coverage counts from the decision, so 40 minutes in the full horizon is gone.
    rejects(lambda: provider.get_forecast(late, HORIZON))
    rejects(lambda: provider.get_forecast(protocol.test_start - STEP, hour))
    assert provider.get_actual(late) == actual.get_actual(late)  # still separate, never a fallback

    with TemporaryDirectory() as directory:
        path = archive.save(Path(directory) / "archive.json")
        assert ForecastArchive.load(path).snapshots == archive.snapshots
        assert archive.save(Path(directory) / "again.json").read_bytes() == path.read_bytes()
        late = dict(archive.metadata, training_available_at=(protocol.test_start + day).isoformat())
        rejects(ForecastArchive(late, archive.snapshots).verify)  # never predates its own model
    print("History selection, snapshot archive and provider availability checks passed.")


if __name__ == "__main__":
    main()
