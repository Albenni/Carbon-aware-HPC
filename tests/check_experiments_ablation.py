"""Small offline check: python tests/check_experiments_ablation.py."""

import contextlib
import io
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from carbon_accounting import JobPowerProfile
from carbon_intensity.series import CarbonIntensityForecast, CarbonIntensitySample as Sample
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
from carbon_intensity.snapshots import ForecastArchive
from experiments.ablation import compare


BASE = datetime(2020, 5, 6, tzinfo=timezone.utc)
DAY = 96  # 15-minute buckets


def intensity(index: int) -> float:
    """Dirty for the first two hours of every day, clean afterwards."""

    return 400.0 if index % DAY < 8 else 100.0


def job(job_id: str, hours: float) -> object:
    from hpc_sim.models import Job

    duration = hours * 3_600.0
    return Job(
        job_id=job_id, submit_time=BASE, release_time=BASE, nodes_required=1,
        actual_duration_seconds=duration,
        power=JobPowerProfile(
            job_id=job_id, duration_seconds=duration, average_power_watts=1_000.0
        ),
    )


def main() -> None:
    actual = Series(tuple(Sample(BASE + index * STEP, intensity(index)) for index in range(3 * DAY)))
    perfect = ForecastArchive(
        {"schema_version": 1, "horizon_hours": 24.0, "cadence_minutes": 60.0},
        tuple(
            CarbonIntensityForecast(
                BASE + timedelta(hours=hour),
                tuple(
                    Sample(BASE + timedelta(hours=hour) + index * STEP, intensity(hour * 4 + index))
                    for index in range(DAY)
                ),
            )
            for hour in range(48)
        ),
    )

    jobs = (job("a", 0.25), job("b", 0.5))
    # Predictions that halve every duration: same execution, worse planning.
    predicted = tuple(
        replace(item, predicted_duration_seconds=item.actual_duration_seconds / 2,
                predicted_average_power_watts=1_000.0)
        for item in jobs
    )
    rows = {row["configuration"]: row for row in compare(
        jobs, predicted, actual, perfect, nodes=2, max_delay=timedelta(hours=6)
    )}

    assert len(rows) == 6, rows.keys()
    oracle = rows["carbon_actual_jobs_actual_carbon"]
    assert oracle["oracle_recovery"] == 1.0
    assert oracle["emissions_saved_vs_easy"] > 0.0
    assert rows["easy_actual_jobs_none_carbon"]["emissions_saved_vs_easy"] == 0.0
    # An archive equal to the observations costs nothing, so the whole gap
    # between the realistic cell and the oracle belongs to the job models.
    realistic = rows["carbon_predicted_jobs_forecast_carbon"]
    assert realistic["carbon_forecast_loss_share"] == 0.0
    assert realistic["combined_loss_share"] == realistic["job_model_loss_share"]
    assert realistic["interaction_loss_share"] == 0.0
    # The submit-time view is reported alongside the release-time one and,
    # with release == submit here, has to agree with it.
    assert realistic["submit_waiting_mean_s"] == realistic["waiting_mean_s"]

    print("Information ablation checks passed.")


class InformationAblationChecks(unittest.TestCase):
    """Make ``unittest discover`` run the script above, which prints its summary."""

    def test_checks_pass(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main()


if __name__ == "__main__":
    main()
