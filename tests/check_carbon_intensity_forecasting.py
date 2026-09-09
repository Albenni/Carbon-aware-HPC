"""Small offline check: python tests/check_carbon_intensity_forecasting.py."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from carbon_intensity.baselines import BASELINE_PERIODS, BaselineCarbonIntensityProvider as Baseline
from check_carbon_intensity_history import rejects
from carbon_intensity.evaluate import REPORTED_HORIZONS, evaluate_forecast
from carbon_intensity.features import FEATURE_NAMES, LOOKBACK, forecast_features, target_features
from carbon_intensity.forecasting import BUCKETS, FEATURE_LAYOUT, HORIZON, RidgeCarbonIntensityForecaster as Ridge
from carbon_intensity.series import CarbonIntensitySample as Sample
from carbon_intensity.protocol import TemporalProtocol
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series


def main():
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)  # a Monday, so the calendar angles are zero
    points = tuple(Sample(start + i * STEP, i) for i in range(1152))
    actual = Series(points)
    validation_start = start + timedelta(days=9)
    protocol = TemporalProtocol(validation_start + timedelta(days=2), start + timedelta(days=12),
                                start, validation_start)
    issue = start + LOOKBACK
    history = actual.get_actual_range(start, issue)
    values = dict(zip(FEATURE_NAMES, forecast_features(history, issue, protocol), strict=True))
    assert [values[f"lag_{m}min"] for m in (15, 60, 1440, 10080)] == [671, 668, 576, 0]
    assert [values[f"rolling_1h_{s}"] for s in ("mean", "min", "max")] == [669.5, 668, 671]
    assert [values[f"rolling_24h_{s}"] for s in ("mean", "min", "max")] == [623.5, 576, 671]
    assert [values[f"{u}_cos"] for u in ("hour", "weekday", "month")] == [1, 1, 1]
    assert values["weekend"] == 0
    delayed = TemporalProtocol(protocol.test_start, protocol.test_end, start, validation_start, STEP)
    shifted = forecast_features(history, issue + STEP, delayed)
    assert shifted[FEATURE_NAMES.index("lag_15min")] == 671  # the delay moves the lag reference back
    rejects(lambda: forecast_features(actual.get_actual_range(start + STEP, issue + STEP), issue, protocol))
    rejects(lambda: forecast_features(history[:-1], issue, protocol))
    rejects(lambda: forecast_features(history, issue + STEP / 2, protocol))

    # Each target reads its own bucket one day and one week earlier, not the issue hour's.
    seasonal = target_features(history, issue, protocol, BUCKETS)
    assert seasonal.shape == (BUCKETS, 2)
    assert list(seasonal[:, 0]) == [values["lag_1440min"] + i for i in range(BUCKETS)]
    assert list(seasonal[:, 1]) == [values["lag_10080min"] + i for i in range(BUCKETS)]
    shifted_seasonal = target_features(history, issue + STEP, delayed, BUCKETS)
    assert shifted_seasonal[0, 0] == seasonal[1, 0]  # the reference follows the target bucket
    assert shifted_seasonal[95, 0] == seasonal[0, 0]  # unobservable offsets wrap to the last day
    rejects(lambda: target_features(history, issue, protocol, BUCKETS + 1000))

    with TemporaryDirectory() as directory:
        path = Path(directory) / "ridge.json"
        model = Ridge.fit(actual, protocol, path)
        model.save(path)
        assert model.metadata["training_cutoff"] == (validation_start - STEP).isoformat()
        assert model.coefficients.shape == (BUCKETS, len(FEATURE_LAYOUT))
        assert (Ridge.load(path, actual, protocol).coefficients == model.coefficients).all()
        grid = [s.timestamp for s in model.get_forecast(validation_start, HORIZON).samples]
        assert len(grid) == 96
        for method in BASELINE_PERIODS:  # every model answers get_forecast on the same grid
            baseline = Baseline(actual, protocol, method).get_forecast(validation_start, HORIZON)
            assert [s.timestamp for s in baseline.samples] == grid
        assert all(s.intensity_gco2e_per_kwh >= 0 for s in model.get_forecast(validation_start, HORIZON).samples)
        rejects(lambda: model.get_forecast(validation_start, HORIZON + STEP))
        rejects(lambda: model.get_forecast(validation_start - STEP, HORIZON))
        result = evaluate_forecast(actual, protocol, model.get_forecast)
        assert set(REPORTED_HORIZONS) <= {row["lead_hours"] for row in result["by_horizon"]}
    print("Feature causality, ridge forecasts and horizon reporting checks passed.")


if __name__ == "__main__":
    main()
