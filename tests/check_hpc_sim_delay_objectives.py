"""Deterministic checks for delay-priced objectives, adaptive budgets and replanning.

The grid signal is the one in ``check_hpc_sim_carbon_aware``: 400 gCO2e/kWh
until 02:00, 100 afterwards. A 10-minute, 1 kW job released at 00:00 saves
50 g by waiting two hours, 25 g per hour of delay.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import random
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from carbon_intensity import CarbonIntensityForecast, CarbonIntensitySample, TimeSeriesCarbonIntensityProvider
from carbon_intensity.snapshots import ForecastArchive
from hpc_sim import (
    CarbonAwareScheduler,
    Cluster,
    CongestionAwareRecedingScheduler,
    EASYBackfillScheduler,
    RecedingHorizonCarbonAwareScheduler,
    SimulationError,
    Simulator,
    account_schedule,
    schedule_metrics,
)
from hpc_sim.carbon_aware import choose_start

from check_hpc_sim_carbon_aware import (
    BASE, CLEAN, CLEAN_FROM, EXACT, QUARTER, SealedProvider, make_job, provider, run,
    sealed_archive, start_times,
)
from check_hpc_sim_receding_horizon import REACH


HOUR = timedelta(hours=1)
SIX = dict(max_delay=6 * HOUR, decision_granularity=QUARTER, runtime_estimate=EXACT)


def target(scheduler, job):
    run([job], scheduler)
    return scheduler.target_start_times[job.job_id]


class ChooseStartTest(unittest.TestCase):
    candidates = [BASE, BASE + HOUR, BASE + 2 * HOUR]

    def test_objectives_disagree_where_they_should(self) -> None:
        # 60 g saved after one hour, 80 g after two: greedy takes the cheapest,
        # a 30 g/h price stops at one hour, and the rate prefers 60 g/h to 40.
        costs = [100.0, 40.0, 20.0]
        self.assertEqual(choose_start(self.candidates, costs).target, BASE + 2 * HOUR)
        self.assertEqual(
            choose_start(self.candidates, costs, objective="carbon-delay", delay_price=30.0).target,
            BASE + HOUR,
        )
        self.assertEqual(choose_start(self.candidates, costs, objective="benefit-rate").target, BASE + HOUR)
        # Above every rate nothing is worth waiting for; ties go to the earliest.
        self.assertEqual(
            choose_start(self.candidates, costs, objective="benefit-rate", threshold=60.0).target, BASE
        )
        self.assertEqual(choose_start(self.candidates, [5.0, 5.0, 5.0]).reason, "no-saving")
        tie = choose_start(self.candidates, [100.0, 50.0, 0.0], objective="benefit-rate")
        self.assertEqual(tie.target, BASE + HOUR)

    def test_a_threshold_keeps_now_and_ignores_the_penalty(self) -> None:
        # The deferral beats now only through a penalty on now; the threshold
        # asks for a carbon saving, so now survives.
        choice = choose_start(
            self.candidates, [100.0, 100.0, 100.0], objective="carbon-delay",
            threshold=1.0, penalties=[500.0, 0.0, 0.0],
        )
        self.assertEqual(choice.target, BASE)


class DelayObjectiveSchedulerTest(unittest.TestCase):
    def test_flat_signal_useful_and_negligible_savings(self) -> None:
        flat = TimeSeriesCarbonIntensityProvider(
            [CarbonIntensitySample(BASE + i * QUARTER, CLEAN) for i in range(96)]
        )
        job = make_job("j")
        for options in ({}, {"objective": "benefit-rate"}, {"objective": "carbon-delay", "delay_price_gco2e_per_hour": 1.0}):
            with self.subTest(**options):
                self.assertEqual(target(CarbonAwareScheduler(flat, **SIX, **options), job), BASE)
        # 25 g/h is on offer: a 10 g/h bar defers, a 30 g/h bar does not.
        rate = dict(objective="benefit-rate", **SIX)
        self.assertEqual(target(CarbonAwareScheduler(provider(), min_carbon_saving_per_delay=10.0, **rate), job), CLEAN_FROM)
        self.assertEqual(target(CarbonAwareScheduler(provider(), min_carbon_saving_per_delay=30.0, **rate), job), BASE)
        # Predicted power scales the saving, so the same bar flips with it.
        heavy = replace(job, predicted_average_power_watts=2_000.0)
        self.assertEqual(target(CarbonAwareScheduler(provider(), min_carbon_saving_per_delay=30.0, **rate), heavy), CLEAN_FROM)

    def test_zero_price_and_threshold_is_the_greedy_and_zero_budget_is_easy(self) -> None:
        jobs = [make_job(i, release_seconds=i * 400, duration_seconds=900 + 300 * i, nodes=1 + i % 3) for i in range(12)]
        greedy = start_times(run(jobs, CarbonAwareScheduler(provider(), **SIX)))
        scalarized = CarbonAwareScheduler(provider(), objective="carbon-delay", **SIX)
        self.assertEqual(start_times(run(jobs, scalarized)), greedy)

        easy = run(jobs, EASYBackfillScheduler(runtime_estimate=EXACT))
        for scheduler in (
            CarbonAwareScheduler(provider(), objective="benefit-rate", max_delay=timedelta(0), runtime_estimate=EXACT),
            CongestionAwareRecedingScheduler(
                provider(), max_delay=timedelta(0), runtime_estimate=EXACT, replan_on_events=True,
                planning_jobs=32, congestion_scope="interval", overrun_to_horizon=True,
            ),
        ):
            result = run(jobs, scheduler)
            self.assertEqual(result.records, easy.records)  # starts, ends, energy, emissions
            self.assertEqual(schedule_metrics(result).as_row() | {"scheduler": "easy"}, schedule_metrics(easy).as_row())

    def test_adaptive_budget_uses_the_predicted_duration_and_its_cap(self) -> None:
        scheduler = CarbonAwareScheduler(provider(), max_delay_fraction=2.0, max_delay_cap=3 * HOUR, reach=REACH)
        short = replace(make_job("s", duration_seconds=3_600), predicted_duration_seconds=600.0)
        long = make_job("l", duration_seconds=7_200)
        self.assertEqual(scheduler._budgets(short), (20 * timedelta(minutes=1), 20 * timedelta(minutes=1)))
        self.assertEqual(scheduler._budgets(long), (3 * HOUR, 3 * HOUR))
        huge = make_job("h", duration_seconds=22 * 3_600)
        self.assertEqual(scheduler._budgets(huge), (3 * HOUR, HOUR))  # nominal, then the reach

    def test_bad_parameters_are_refused(self) -> None:
        for options in (
            {"objective": "cheapest"},
            {"delay_price_gco2e_per_hour": 1.0},  # carbon takes no price
            {"objective": "benefit-rate", "delay_price_gco2e_per_hour": 1.0},
            {"objective": "carbon-delay", "min_carbon_saving_per_delay": -1.0},
            {"objective": "carbon-delay", "delay_price_gco2e_per_hour": float("nan")},
            {"max_delay_cap": HOUR},  # a cap with a fixed budget means nothing
        ):
            with self.subTest(**options), self.assertRaises(ValueError):
                CarbonAwareScheduler(provider(), **SIX, **options)
        with self.assertRaises(ValueError):
            CongestionAwareRecedingScheduler(provider(), objective="benefit-rate", **SIX)


class ReplanningTest(unittest.TestCase):
    def test_a_forecast_newer_than_the_decision_is_refused(self) -> None:
        future = CarbonIntensityForecast(BASE + HOUR, tuple(
            CarbonIntensitySample(BASE + i * QUARTER, CLEAN) for i in range(96)))

        class Leaky(SealedProvider):
            def get_forecast(self, as_of, horizon):
                return future

        leaky = Leaky(provider(), ForecastArchive({"schema_version": 1}, (future,)))
        with self.assertRaises(SimulationError):
            run([make_job("j")], CarbonAwareScheduler(leaky, forecast=True, **SIX))

    def test_price_zero_matches_the_receding_policy_under_the_same_triggers(self) -> None:
        jobs = [make_job(i, release_seconds=i * 700, duration_seconds=1_800, nodes=2) for i in range(10)]
        options = dict(forecast=True, reach=REACH, replan_on_events=True, **SIX)
        receding = RecedingHorizonCarbonAwareScheduler(sealed_archive(CLEAN_FROM), **options)
        congestion = CongestionAwareRecedingScheduler(sealed_archive(CLEAN_FROM), congestion_price=0.0, **options)
        self.assertEqual(start_times(run(jobs, congestion, total_nodes=2)), start_times(run(jobs, receding, total_nodes=2)))

    def test_interval_scope_sees_congestion_that_builds_after_the_start(self) -> None:
        # "held" is parked for 02:30 on the only node. At 02:00 the node is free,
        # so a start-only price lets the 1-hour job start there; the interval
        # price sees the overlap and moves it past held's projected end.
        held = make_job("held", duration_seconds=3_600)
        mover = make_job("mover", release_seconds=60, duration_seconds=3_600, average_power_watts=5_000.0)
        chosen = {}
        for scope in ("start", "interval"):
            scheduler = CongestionAwareRecedingScheduler(
                provider(), congestion_price=1e6, congestion_scope=scope, **SIX
            )
            simulator = Simulator([held, mover], Cluster(1), scheduler)
            scheduler._targets["held"] = BASE + timedelta(minutes=150)
            scheduler._pending["held"] = held
            scheduler.on_release(mover, mover.release_time, simulator)
            chosen[scope] = scheduler.target_start_times["mover"]
        self.assertEqual(chosen["start"], CLEAN_FROM)
        self.assertGreaterEqual(chosen["interval"], BASE + timedelta(minutes=210))

    def test_replanning_is_deterministic_never_overcommits_and_ends_on_time(self) -> None:
        jobs = [make_job(i, release_seconds=(i // 3) * 600, duration_seconds=600 + 450 * (i % 5), nodes=1 + i % 4) for i in range(30)]
        # Two jobs overrun their estimate, so the overrun claim is exercised.
        jobs = [replace(job, predicted_duration_seconds=60.0) if i in (4, 9) else job for i, job in enumerate(jobs)]

        def schedule(ordered):
            scheduler = CongestionAwareRecedingScheduler(
                provider(), replan_on_events=True, planning_jobs=4, congestion_scope="interval",
                overrun_to_horizon=True, **SIX,
            )
            result = run(ordered, scheduler, total_nodes=4)  # the cluster raises on overcommit
            return result, scheduler

        result, scheduler = schedule(jobs)
        shuffled = list(jobs)
        random.Random(0).shuffle(shuffled)
        self.assertEqual(start_times(schedule(shuffled)[0]), start_times(result))
        self.assertGreater(scheduler.revisit_count, 1)  # the oracle replans on events too
        # The simulation stops at the last completion, not at a leftover wakeup.
        self.assertEqual(result.last_event_time, max(record.end_time for record in result.records))
        for record in scheduler.decision_records():
            self.assertLessEqual(record["target"], record["deadline"])
            self.assertGreaterEqual(record["target"], record["decision_time"])

    def test_the_accounting_ignores_what_the_policy_was_told(self) -> None:
        jobs = [make_job(i, release_seconds=i * 900) for i in range(4)]
        told = [replace(job, predicted_duration_seconds=5_000.0, predicted_average_power_watts=9.0) for job in jobs]
        result = Simulator(told, Cluster(4), CarbonAwareScheduler(provider(), **SIX)).run()
        self.assertEqual(account_schedule(result, told, provider()).records,
                         account_schedule(result, jobs, provider()).records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
