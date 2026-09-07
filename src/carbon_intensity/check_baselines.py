"""Small offline check: python -m carbon_intensity.check_baselines."""

from dataclasses import replace
import csv
from datetime import datetime, timedelta, timezone
import json
from math import isclose, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from .baselines import BaselineCarbonIntensityProvider as Baseline
from .check_history import rejects
from .evaluate import evaluate_forecast, write_evaluation
from .series import CarbonIntensitySample as Sample
from .protocol import TemporalProtocol
from .series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series


def main():
    start = datetime(2020, 1, 23, tzinfo=timezone.utc)
    issue = start + 768 * STEP
    points = tuple(Sample(start + i * STEP, i) for i in range(960))
    actual = Series(points)
    protocol = TemporalProtocol(issue + 4 * STEP, issue + 8 * STEP, start, issue)
    for method, expected in (
        ("persistence", [767, 767]), ("seasonal_daily", [672, 673]),
        ("seasonal_weekly", [96, 97]),
    ):
        model = Baseline(actual, protocol, method)
        forecast = model.get_forecast(issue, 2 * STEP)
        assert forecast.issue_time == issue
        assert [s.timestamp for s in forecast.samples] == [issue, issue + STEP]
        assert [s.intensity_gco2e_per_kwh for s in forecast.samples] == expected
        past = Baseline(Series(points[:768]), protocol, method)
        assert past.get_forecast(issue, 100 * STEP) == model.get_forecast(issue, 100 * STEP)
    delayed = replace(protocol, observation_delay=STEP)
    model = Baseline(actual, delayed, "seasonal_daily")
    forecast = model.get_forecast(issue + STEP / 2, timedelta(days=1))
    assert [s.intensity_gco2e_per_kwh for s in forecast.samples] == list(range(672, 767)) + [671, 672]
    past = Baseline(Series(points[:767]), delayed, "seasonal_daily")
    assert past.get_forecast(issue + STEP / 2, timedelta(days=1)) == forecast
    for rules, count, mae, mse in ((protocol, 3, 1.5, 2.5), (delayed, 2, 2.5, 6.5)):
        model = Baseline(actual, rules)
        result = evaluate_forecast(actual, rules, model.get_forecast, horizon=2 * STEP, cadence=STEP)
        assert result["forecasts"] == count and result["predicted_buckets"] == 2 * count
        assert result["last_issue_time"] == (issue + (count - 1) * STEP).isoformat()
        assert result["mae_gco2e_per_kwh"] == mae and isclose(result["rmse_gco2e_per_kwh"], sqrt(mse))
        assert result["bias_gco2e_per_kwh"] == -mae
        assert [row["lead_hours"] for row in result["by_horizon"]] == [0.25, 0.5]
        assert [row["bias_gco2e_per_kwh"] for row in result["by_horizon"]] == [-mae + 0.5, -mae - 0.5]
        assert result["by_month"][0]["predicted_buckets"] == 2 * count
    truncated = lambda when, horizon: replace(model.get_forecast(when, horizon), samples=forecast.samples[:1])
    rejects(lambda: evaluate_forecast(actual, delayed, truncated, horizon=2 * STEP))
    rejects(lambda: model.get_forecast(issue, timedelta(0)))
    rejects(lambda: evaluate_forecast(actual, protocol, model.get_forecast, horizon=5 * STEP))
    wide = replace(protocol, test_start=issue + 192 * STEP, test_end=issue + 288 * STEP)
    result = evaluate_forecast(actual, wide, Baseline(actual, wide).get_forecast, cadence=96 * STEP)
    assert len(result["by_horizon"]) == 96
    assert {1, 3, 6, 12, 24} <= {row["lead_hours"] for row in result["by_horizon"]}
    assert result["by_horizon"][-1]["bias_gco2e_per_kwh"] == -96
    assert [row["issue_month"] for row in result["by_month"]] == ["2020-01", "2020-02"]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(delayed.metadata()), encoding="utf-8")
        assert TemporalProtocol.load(path) == delayed
        write_evaluation(Path(directory), "validation", [{"model": "persistence", **result}], {})
        with (Path(directory) / "validation_metrics.csv").open() as stream:
            assert "by_horizon" not in next(csv.DictReader(stream))
    print("Baseline trajectories, temporal leakage and evaluation checks passed.")


if __name__ == "__main__":
    main()
