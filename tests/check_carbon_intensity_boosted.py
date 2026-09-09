"""Small offline check: python tests/check_carbon_intensity_boosted.py."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from carbon_intensity.boosted import (
    BUCKETS,
    HORIZON,
    MIN_HISTORY,
    BoostedCarbonIntensityForecaster,
    calendar_columns,
    issue_columns,
    level_mask,
)
from check_carbon_intensity_history import rejects
from carbon_intensity.series import CarbonIntensitySample as Sample
from carbon_intensity.protocol import TemporalProtocol
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series


def main():
    day = timedelta(days=1)
    start = datetime(2019, 1, 7, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    # A daily shape, a weekly offset and noise: enough structure for both stages.
    grid = np.arange(80 * 96)
    values = (300 + 40 * np.sin(2 * np.pi * grid / 96) + 10 * (grid // 96 % 7)
              + rng.normal(0, 5, grid.size))
    actual = Series(tuple(Sample(start + int(i) * STEP, float(v)) for i, v in zip(grid, values)))
    protocol = TemporalProtocol(start + 75 * day, start + 80 * day, start, start + 70 * day)

    model = BoostedCarbonIntensityForecaster.build(actual, protocol, replay="validation", alpha=1.0)
    assert model.metadata["training_examples"] > len(model.names)
    entry = model.metadata["refits"][0]
    assert entry["training_available_at"] <= entry["refit_at"], "a fit was used before it existed"

    issue = protocol.validation_start + 3 * timedelta(hours=1)
    forecast = model.get_forecast(issue, HORIZON)
    assert len(forecast.samples) == BUCKETS and forecast.issue_time == issue
    assert all(s.intensity_gco2e_per_kwh >= 0 for s in forecast.samples)
    assert forecast.samples[-1].timestamp == issue + HORIZON - STEP
    short = model.get_forecast(issue, timedelta(hours=3))
    assert len(short.samples) == 12
    # A shorter horizon must not change the buckets both trajectories cover.
    assert [s.intensity_gco2e_per_kwh for s in short.samples] == [
        s.intensity_gco2e_per_kwh for s in forecast.samples[:12]
    ]
    rejects(lambda: model.get_forecast(issue, HORIZON + STEP))
    rejects(lambda: model.get_forecast(issue, timedelta(0)))
    rejects(lambda: model.get_forecast(issue + timedelta(minutes=5), HORIZON))
    rejects(lambda: model.get_forecast(protocol.train_start, HORIZON))

    # Nothing the model reads may come from the issue instant onwards: rebuild
    # every feature on a series whose future has been replaced by noise and
    # require the same trajectory, byte for byte.
    import pandas as pd

    cut = model.index(issue)
    poisoned = model.values.copy()
    poisoned[cut:] = rng.uniform(0, 900, poisoned.size - cut)
    stamps = pd.date_range(model.start, periods=poisoned.size, freq="15min", tz="UTC")
    shared, _ = issue_columns(poisoned, stamps)
    calendar, _ = calendar_columns(stamps)
    twin = replace(model, values=poisoned, shared=shared, calendar=calendar,
                   actual=Series(tuple(Sample(model.moment(i), float(v))
                                       for i, v in enumerate(poisoned))))
    assert [s.intensity_gco2e_per_kwh for s in twin.get_forecast(issue, HORIZON).samples] == [
        s.intensity_gco2e_per_kwh for s in forecast.samples
    ], "the forecast moved when only the unobservable future changed"

    assert level_mask(["lag4", "roll96_mean", "roll96_std", "d_lag4", "lead_h"]).tolist() == [
        True, True, False, False, False,
    ]
    rolling = BoostedCarbonIntensityForecaster.build(
        actual, protocol, replay="validation", alpha=1.0, period=2 * day,
    )
    assert len(rolling.metadata["refits"]) > 1
    assert all(e["training_available_at"] <= e["refit_at"] for e in rolling.metadata["refits"])
    # Later refits may only gain rows, and only published ones.
    counts = [e["training_examples"] for e in rolling.metadata["refits"]]
    assert counts == sorted(counts) and counts[-1] > counts[0]
    assert rolling.provenance(protocol.validation_start + 3 * day)["refit_at"] == (
        protocol.validation_start + 2 * day
    ).isoformat()
    print("Boosted forecaster: causality, horizons, refit schedule and future-poisoning checks passed.")


if __name__ == "__main__":
    main()
