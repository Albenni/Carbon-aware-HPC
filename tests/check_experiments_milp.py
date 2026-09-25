"""Small offline check: python tests/check_experiments_milp.py."""

import contextlib
import io
import unittest
from datetime import datetime, timedelta, timezone

from carbon_accounting import JobPowerProfile
from carbon_intensity.series import CarbonIntensitySample as Sample
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
from experiments.milp import compare
from hpc_sim import Job


BASE = datetime(2020, 5, 6, tzinfo=timezone.utc)


def job(job_id: str, watts: float) -> Job:
    return Job(
        job_id=job_id, submit_time=BASE, release_time=BASE, nodes_required=1,
        actual_duration_seconds=3_600.0,
        power=JobPowerProfile(job_id=job_id, duration_seconds=3_600.0, average_power_watts=watts),
    )


def main() -> None:
    # One clean hour, [2 h, 3 h), and one node: only one job can have it.
    provider = Series(tuple(
        Sample(BASE + index * STEP, 100.0 if 8 <= index < 12 else 400.0) for index in range(96)
    ))
    rows = {row["configuration"]: row for row in compare(
        (job("a", 1_000.0), job("b", 10_000.0)), provider,
        nodes=1, max_delay=timedelta(hours=3), granularity=timedelta(hours=1),
    )}

    grams = {name: round(row["model_emissions_tco2e"] * 1e6) for name, row in rows.items()}
    # The greedy hands the clean hour to whoever comes first; the optimum gives
    # it to the job that draws ten times the power.
    assert grams == {"easy": 4_400, "carbon_greedy": 4_100, "milp_optimum": 1_400}, grams
    assert rows["milp_optimum"]["proven_optimal"]
    assert rows["milp_optimum"]["jobs_past_budget"] == 0
    assert abs(rows["carbon_greedy"]["optimal_saving_captured"] - 0.1) < 1e-9

    print("MILP optimum checks passed.")


class MILPOptimumChecks(unittest.TestCase):
    """Make ``unittest discover`` run the script above, which prints its summary."""

    def test_checks_pass(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main()


if __name__ == "__main__":
    main()
