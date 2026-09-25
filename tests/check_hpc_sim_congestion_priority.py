"""Deterministic checks for the congestion-aware, opportunity-priority receding scheduler.

The cluster has one node and the signal turns clean at 02:00, so every job wants
the same bucket and only one can have it: which one does is the whole policy.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from hpc_sim import CongestionAwareRecedingScheduler, RecedingHorizonCarbonAwareScheduler

from check_hpc_sim_carbon_aware import BASE, CLEAN_FROM, EXACT, make_job, run, start_times
from check_hpc_sim_receding_horizon import REACH, revised_archive


BUDGET = timedelta(hours=6)
OPTIONS = dict(forecast=True, reach=REACH, max_delay=BUDGET, runtime_estimate=EXACT)


def schedule(jobs, **keywords):
    scheduler = CongestionAwareRecedingScheduler(
        revised_archive(CLEAN_FROM, CLEAN_FROM), **OPTIONS, **keywords
    )
    return start_times(run(jobs, scheduler, total_nodes=1))


class CongestionPriorityTest(unittest.TestCase):
    def test_price_zero_is_the_receding_policy(self) -> None:
        jobs = [make_job(i, release_seconds=i * 600, duration_seconds=1_800) for i in range(6)]
        receding = RecedingHorizonCarbonAwareScheduler(revised_archive(CLEAN_FROM, CLEAN_FROM), **OPTIONS)
        self.assertEqual(
            schedule(jobs, congestion_price=0.0),
            start_times(run(jobs, receding, total_nodes=1)),
        )

    def test_the_larger_opportunity_keeps_the_clean_window(self) -> None:
        # Both jobs are 90 minutes: starting at 00:30 is 60 dirty minutes out
        # of 90, but "big" draws ten times the power, so it has ten times the
        # carbon to save. The 01:00 reissue retargets both; big claims 02:00 and
        # small is priced off to 03:30, when big is projected to be done.
        big = make_job("big", release_seconds=1_800, duration_seconds=5_400, average_power_watts=10_000.0)
        small = make_job("small", release_seconds=1_800, duration_seconds=5_400)
        started = schedule([small, big], congestion_price=1e6)
        self.assertEqual(started["big"], CLEAN_FROM)
        self.assertEqual(started["small"], CLEAN_FROM + timedelta(minutes=90))

    def test_targets_stay_inside_the_budget_from_release(self) -> None:
        jobs = [make_job(i, release_seconds=i * 300, duration_seconds=3_600) for i in range(8)]
        scheduler = CongestionAwareRecedingScheduler(
            revised_archive(CLEAN_FROM, CLEAN_FROM), **OPTIONS, congestion_price=1e6
        )
        run(jobs, scheduler, total_nodes=2)
        for job in jobs:
            self.assertLessEqual(scheduler.target_start_times[job.job_id], job.release_time + BUDGET)
            self.assertGreaterEqual(scheduler.target_start_times[job.job_id], BASE)

    def test_a_negative_price_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CongestionAwareRecedingScheduler(
                revised_archive(CLEAN_FROM, CLEAN_FROM), **OPTIONS, congestion_price=-1.0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
