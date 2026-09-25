"""Deterministic checks for the receding-horizon carbon-aware scheduler.

The archive here is deliberately wrong at first and corrected an hour later, so
the one thing the policy adds over its baseline — acting on the correction —
is the only thing that can produce the expected start times.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from carbon_intensity import CarbonIntensityForecast, CarbonIntensitySample
from carbon_intensity.snapshots import ForecastArchive
from hpc_sim import (
    CarbonAwareScheduler,
    RecedingHorizonCarbonAwareScheduler,
    RuntimeEstimateSource,
)

from check_hpc_sim_carbon_aware import (
    BASE,
    CLEAN,
    CLEAN_FROM,
    DIRTY,
    QUARTER,
    SealedProvider,
    make_job,
    provider,
    run,
    start_times,
)


HOUR = timedelta(hours=1)
REACH = timedelta(hours=23)


def revised_archive(first_clean_from, later_clean_from) -> SealedProvider:
    """Hourly snapshots: the 00:00 issue is wrong, every later one is right."""

    def trajectory(issue, clean_from):
        return tuple(
            CarbonIntensitySample(
                issue + index * QUARTER,
                DIRTY if issue + index * QUARTER < clean_from else CLEAN,
            )
            for index in range(96)
        )

    snapshots = tuple(
        CarbonIntensityForecast(
            BASE + hour * HOUR,
            trajectory(BASE + hour * HOUR, first_clean_from if hour == 0 else later_clean_from),
        )
        for hour in range(12)
    )
    return SealedProvider(provider(), ForecastArchive({"schema_version": 1}, snapshots))


class RecedingHorizonTest(unittest.TestCase):
    def test_a_reissued_forecast_moves_a_job_that_has_not_started(self) -> None:
        # At release the archive says the clean window opens at 05:00; from the
        # 01:00 issue on it says 03:00. Only a policy that looks again can land
        # on 03:00, and its baseline must still sit on the stale 05:00.
        stale, revised = BASE + 5 * HOUR, BASE + 3 * HOUR
        jobs = [make_job("j")]
        options = dict(forecast=True, reach=REACH, max_delay=timedelta(hours=6))

        baseline = CarbonAwareScheduler(revised_archive(stale, revised), **options)
        receding = RecedingHorizonCarbonAwareScheduler(revised_archive(stale, revised), **options)

        self.assertEqual(start_times(run(jobs, baseline))["j"], stale)
        self.assertEqual(start_times(run(jobs, receding))["j"], revised)

    def test_rescheduling_never_extends_the_budget_or_reaches_backwards(self) -> None:
        # The revision moves the clean window past the deadline, so the best a
        # revisit may do is the cheapest start still inside the original budget.
        budget = timedelta(hours=2)
        deadline = BASE + budget
        jobs = [make_job("j")]
        receding = RecedingHorizonCarbonAwareScheduler(
            revised_archive(CLEAN_FROM, BASE + 9 * HOUR),
            forecast=True,
            reach=REACH,
            max_delay=budget,
        )
        started = start_times(run(jobs, receding))["j"]

        self.assertLessEqual(started, deadline)
        self.assertGreaterEqual(started, BASE)

    def test_an_immutable_signal_reproduces_the_baseline_exactly(self) -> None:
        # Against the actual series there is nothing to revise, so every revisit
        # returns the previous answer and the two policies must agree job for job.
        jobs = [make_job(index, release_seconds=index * 900, nodes=2) for index in range(8)]
        options = dict(
            max_delay=timedelta(hours=6),
            decision_granularity=QUARTER,
            runtime_estimate=RuntimeEstimateSource.SCHEDULING,
        )

        self.assertEqual(
            start_times(run(jobs, RecedingHorizonCarbonAwareScheduler(provider(), **options))),
            start_times(run(jobs, CarbonAwareScheduler(provider(), **options))),
        )

    def test_a_forecast_run_without_a_reach_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            RecedingHorizonCarbonAwareScheduler(
                revised_archive(CLEAN_FROM, CLEAN_FROM), forecast=True, max_delay=HOUR
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
