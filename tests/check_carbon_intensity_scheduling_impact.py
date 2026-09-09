"""Small offline check: python tests/check_carbon_intensity_scheduling_impact.py."""

from datetime import datetime, timedelta, timezone

from carbon_accounting import JobPowerProfile
from hpc_sim import CarbonAwareScheduler
from hpc_sim.models import Job

from carbon_intensity.series import CarbonIntensityForecast, CarbonIntensitySample as Sample
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
from carbon_intensity.scheduling_impact import archive_reach, compare
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive


BASE = datetime(2020, 5, 6, tzinfo=timezone.utc)
DAY = 96  # 15-minute buckets


def intensity(index: int) -> float:
    """Dirty for the first two hours of every day, clean afterwards."""

    return 400.0 if index % DAY < 8 else 100.0


def archive(values) -> ForecastArchive:
    """Hourly snapshots of one day each, taken from ``values``."""

    snapshots = tuple(
        CarbonIntensityForecast(
            BASE + timedelta(hours=hour),
            tuple(
                Sample(BASE + timedelta(hours=hour) + index * STEP, values(hour * 4 + index))
                for index in range(DAY)
            ),
        )
        for hour in range(48)
    )
    return ForecastArchive(
        {"schema_version": 1, "horizon_hours": 24.0, "cadence_minutes": 60.0}, snapshots
    )


def job(job_id: str, *, duration_hours: float) -> Job:
    duration = duration_hours * 3_600.0
    return Job(
        job_id=job_id, submit_time=BASE, release_time=BASE, nodes_required=1,
        actual_duration_seconds=duration,
        power=JobPowerProfile(
            job_id=job_id, duration_seconds=duration, average_power_watts=1_000.0
        ),
    )


def main() -> None:
    actual = Series(tuple(Sample(BASE + index * STEP, intensity(index)) for index in range(3 * DAY)))
    perfect, flat = archive(intensity), archive(lambda index: 200.0)
    assert archive_reach(perfect) == timedelta(hours=23)

    jobs = (job("short", duration_hours=0.25), job("long", duration_hours=23.5))
    # A job that outlasts the reach cannot be placed with a forecast at all.
    bounded = CarbonAwareScheduler(
        actual, reach=timedelta(hours=23), max_delay=timedelta(hours=6)
    )
    assert bounded._max_delay_for(jobs[0]) == timedelta(hours=6)
    assert bounded._max_delay_for(jobs[1]) == timedelta(0)

    # The per-job budget composes with the forecast signal: same clamp, and
    # still no route to the observations.
    scaled = CarbonAwareScheduler(
        ArchiveCarbonIntensityProvider(actual, perfect),
        forecast=True,
        reach=timedelta(hours=23),
        max_delay_fraction=1.0,
    )
    assert scaled.reads_actual_series is False
    assert scaled._max_delay_for(jobs[0]) == timedelta(minutes=15)
    assert scaled._max_delay_for(jobs[1]) == timedelta(0)

    rows = {
        row["configuration"]: row
        for row in compare(
            jobs, actual, {"perfect": perfect, "flat": flat},
            nodes=2, max_delay=timedelta(hours=6),
        )
    }
    assert rows["easy"]["emissions_saved_vs_easy"] == 0.0
    assert rows["carbon_oracle"]["emissions_saved_vs_easy"] > 0.0
    # A forecast equal to the observations recovers all of it; a flat one has no
    # slope to follow, never defers, and recovers none.
    assert rows["carbon_forecast_perfect"]["oracle_recovery"] == 1.0
    assert rows["carbon_forecast_flat"]["oracle_recovery"] == 0.0
    assert rows["carbon_forecast_flat"]["start_changed_fraction"] > 0.0

    print("Forecast-driven scheduling impact checks passed.")


if __name__ == "__main__":
    main()
