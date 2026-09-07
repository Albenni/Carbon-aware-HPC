"""Small offline check: python -m carbon_intensity.check_walkforward."""

from datetime import datetime, timedelta, timezone

from .check_history import rejects
from .forecasting import BUCKETS, HORIZON
from .series import CarbonIntensitySample as Sample
from .protocol import TemporalProtocol
from .series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
from .snapshots import ForecastArchive
from .walkforward import WalkForwardForecaster, pooled_design, refit_cutoffs


def main():
    day = timedelta(days=1)
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)
    actual = Series(tuple(Sample(start + i * STEP, 100 + i % 96 + (i // 96) % 5) for i in range(2880)))
    protocol = TemporalProtocol(start + 22 * day, start + 24 * day, start, start + 20 * day)
    rows = pooled_design(actual, protocol)

    assert refit_cutoffs(start, start + 3 * day, None) == (start,)
    assert refit_cutoffs(start, start + 3 * day, day) == (start, start + day, start + 2 * day)
    rejects(lambda: refit_cutoffs(start, start, day))

    frozen = WalkForwardForecaster.build(actual, protocol, rows, period=None, alpha=0.01)
    rolling = WalkForwardForecaster.build(actual, protocol, rows, period=day, alpha=0.01)
    assert len(frozen.metadata["refits"]) == 1 and len(rolling.metadata["refits"]) == 2
    # Keeping the model learning can only add rows, and never unpublished ones.
    assert rolling.metadata["refits"][1]["training_examples"] > frozen.metadata["training_examples"]
    for entry in rolling.metadata["refits"]:
        assert entry["training_available_at"] <= entry["refit_at"]

    issue = protocol.test_start + timedelta(hours=5)
    assert frozen.refit_at(issue) == protocol.test_start
    assert rolling.refit_at(issue) == protocol.test_start
    assert rolling.refit_at(issue + day) == protocol.test_start + day
    rejects(lambda: rolling.refit_at(protocol.test_start - STEP))
    forecast = rolling.get_forecast(issue, HORIZON)
    assert len(forecast.samples) == BUCKETS and all(s.intensity_gco2e_per_kwh >= 0 for s in forecast.samples)
    rejects(lambda: rolling.get_forecast(issue, HORIZON + STEP))

    archive = ForecastArchive.generate(
        rolling.metadata["model_name"], rolling.get_forecast, protocol, metadata=rolling.metadata,
    )
    assert archive.verify() == 48
    # Every snapshot is checked against the fit that served it, not only the first.
    late = dict(archive.metadata, refits=[
        dict(entry, training_available_at=(protocol.test_end).isoformat())
        for entry in rolling.metadata["refits"]
    ])
    rejects(ForecastArchive(late, archive.snapshots).verify)
    print("Walk-forward schedule, leak-free refits and per-snapshot cutoff checks passed.")


if __name__ == "__main__":
    main()
