"""Persistence forecasts using only observations available at issue time."""

from datetime import datetime, timedelta

from .series import CarbonIntensityForecast, CarbonIntensitySample
from .protocol import TemporalProtocol
from .series import (
    FIFTEEN_MINUTES,
    CarbonIntensityProvider,
    TimeSeriesCarbonIntensityProvider,
    aware_utc,
    bucket_start,
)


BASELINE_PERIODS = {
    "persistence": FIFTEEN_MINUTES,
    "seasonal_daily": timedelta(days=1),
    "seasonal_weekly": timedelta(days=7),
}


class BaselineCarbonIntensityProvider(CarbonIntensityProvider):
    """Expose actuals separately from a causal, unfitted baseline forecast.

    Seasonal forecasts repeat the last observable UTC day or week. This also
    handles horizons longer than a season and delayed publication without
    borrowing future actuals. Missing history raises instead of falling back.
    """

    def __init__(
        self, actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
        method: str = "persistence",
    ) -> None:
        if method not in BASELINE_PERIODS:
            raise ValueError(f"method must be one of {tuple(BASELINE_PERIODS)}")
        if actual.granularity != FIFTEEN_MINUTES:
            raise ValueError("baseline forecasts require 15-minute actuals")
        self.actual = actual
        self.protocol = protocol
        self.method = method

    @property
    def granularity(self) -> timedelta:
        return self.actual.granularity

    def get_actual(self, timestamp: datetime) -> float:
        return self.actual.get_actual(timestamp)

    def get_actual_range(self, start: datetime, end: datetime) -> tuple[CarbonIntensitySample, ...]:
        return self.actual.get_actual_range(start, end)

    def get_forecast(self, issue_time: datetime, horizon: timedelta) -> CarbonIntensityForecast:
        issue_time = aware_utc(issue_time, "issue_time")
        if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
            raise ValueError("horizon must be a positive timedelta")
        history = self.protocol.history(self.actual, issue_time, BASELINE_PERIODS[self.method])
        samples = []
        timestamp = bucket_start(issue_time)
        while timestamp < issue_time + horizon:
            index = ((timestamp - history[0].timestamp) // self.granularity) % len(history)
            samples.append(CarbonIntensitySample(timestamp, history[index].intensity_gco2e_per_kwh))
            timestamp += self.granularity
        return CarbonIntensityForecast(issue_time, tuple(samples))
