"""Deterministic checks for the carbon-aware oracle scheduler.

The synthetic checks use a hand-made grid signal — dirty for two hours, clean
afterwards — so every decision can be predicted by reading the series. The
trace checks read the committed 5,000-job debug subset and are skipped when it,
or pyarrow, is absent.

What is being verified is the policy's contract, not its outcomes: a job is
never held past its delay budget, a zero budget reproduces EASY exactly, and
holding a job moves emissions without moving energy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_accounting import JobPowerProfile, carbon_emissions
from carbon_intensity import (
    CarbonIntensitySample,
    MissingCarbonIntensityError,
    TimeSeriesCarbonIntensityProvider,
)
from hpc_sim import (
    CarbonAwareScheduler,
    CarbonSignal,
    Cluster,
    EASYBackfillScheduler,
    FCFSScheduler,
    Job,
    RuntimeEstimateSource,
    Simulator,
    account_schedule,
    carbon_cost_gco2e,
    cheapest_start_time,
    schedule_metrics,
    total_emissions_gco2e,
    total_energy_kwh,
)


UTC = timezone.utc
BASE = datetime(2020, 5, 6, 0, 0, tzinfo=UTC)
QUARTER = timedelta(minutes=15)
DIRTY = 400.0
CLEAN = 100.0
#: The signal turns clean two hours in and stays clean for the rest of the day.
CLEAN_FROM = BASE + timedelta(hours=2)
EXACT = RuntimeEstimateSource.SCHEDULING

DEBUG_TRACE = PROJECT_ROOT / "data" / "processed" / "pm100_debug_5000.parquet"
CARBON_CACHE = (
    PROJECT_ROOT
    / "data"
    / "carbon_intensity"
    / "electricity_maps_it_no_04_to_11_2020.json"
)


def provider() -> TimeSeriesCarbonIntensityProvider:
    return TimeSeriesCarbonIntensityProvider(
        [
            CarbonIntensitySample(
                BASE + index * QUARTER,
                DIRTY if BASE + index * QUARTER < CLEAN_FROM else CLEAN,
            )
            for index in range(96)  # one day of 15-minute buckets
        ]
    )


def make_job(
    job_id: object,
    *,
    release_seconds: float = 0.0,
    duration_seconds: float = 600.0,
    nodes: int = 1,
    average_power_watts: float = 1_000.0,
) -> Job:
    release = BASE + timedelta(seconds=release_seconds)
    return Job(
        job_id=job_id,
        submit_time=release,
        release_time=release,
        nodes_required=nodes,
        actual_duration_seconds=duration_seconds,
        power=JobPowerProfile(
            job_id=job_id,
            duration_seconds=duration_seconds,
            average_power_watts=average_power_watts,
        ),
    )


def run(jobs, scheduler, *, total_nodes: int = 4, source=None):
    source = source or provider()
    result = Simulator(jobs, Cluster(total_nodes), scheduler).run()
    return account_schedule(result, jobs, source)


def start_times(result) -> dict[object, datetime]:
    return {record.job_id: record.start_time for record in result.records}


class CarbonSignalTest(unittest.TestCase):
    """The window integral, against the accounting it has to agree with."""

    def test_a_window_spanning_the_transition_is_time_weighted(self) -> None:
        signal = CarbonSignal(provider())
        # One hour ending one hour into the clean period: half dirty, half clean.
        mean = signal.mean_intensity(CLEAN_FROM - timedelta(hours=1), CLEAN_FROM + timedelta(hours=1))
        self.assertAlmostEqual(mean, (DIRTY + CLEAN) / 2)

    def test_the_predicted_cost_matches_the_accounting(self) -> None:
        job = make_job("j", duration_seconds=3_600, average_power_watts=2_000.0)
        signal = CarbonSignal(provider())
        for offset_minutes in (0, 90, 300):
            start = BASE + timedelta(minutes=offset_minutes)
            with self.subTest(offset_minutes=offset_minutes):
                self.assertAlmostEqual(
                    carbon_cost_gco2e(job, start, signal),
                    carbon_emissions(job.power, start, provider().get_actual),
                    places=9,
                )

    def test_a_window_outside_the_series_is_refused(self) -> None:
        signal = CarbonSignal(provider())
        with self.assertRaises(MissingCarbonIntensityError):
            signal.mean_intensity(BASE + timedelta(days=1), BASE + timedelta(days=2))


class CheapestStartTest(unittest.TestCase):
    """The greedy choice itself, before any scheduling gets involved."""

    def test_a_job_is_moved_into_the_clean_window(self) -> None:
        job = make_job("j")
        self.assertEqual(
            cheapest_start_time(
                job, CarbonSignal(provider()), max_delay=timedelta(hours=6), granularity=QUARTER
            ),
            CLEAN_FROM,
        )

    def test_a_tie_and_a_zero_budget_both_keep_the_release_time(self) -> None:
        signal = CarbonSignal(provider())
        # Released inside the clean window, every candidate costs the same, so
        # the earliest must win: the policy never defers without a strict gain.
        released_clean = make_job("clean", release_seconds=timedelta(hours=3).total_seconds())
        self.assertEqual(
            cheapest_start_time(released_clean, signal, max_delay=timedelta(hours=6), granularity=QUARTER),
            released_clean.release_time,
        )
        dirty = make_job("dirty")
        self.assertEqual(
            cheapest_start_time(dirty, signal, max_delay=timedelta(0), granularity=QUARTER),
            dirty.release_time,
        )

    def test_a_budget_too_short_to_reach_the_clean_window_buys_the_best_overlap(self) -> None:
        # The budget ends at 01:55, before the signal turns clean at 02:00, so
        # the best reachable start is the last candidate on the grid: this
        # half-hour job is the only one whose window reaches into clean time.
        job = make_job("j", duration_seconds=1_800)
        self.assertEqual(
            cheapest_start_time(
                job,
                CarbonSignal(provider()),
                max_delay=timedelta(minutes=115),
                granularity=QUARTER,
            ),
            BASE + timedelta(minutes=105),
        )


class CarbonAwareSchedulerTest(unittest.TestCase):
    def test_resource_only_job_contends_but_is_not_scored(self) -> None:
        background = Job("background", BASE, BASE, 1, 600, power=None)
        evaluated = make_job("evaluated", release_seconds=1, duration_seconds=600)
        carbon = CarbonAwareScheduler(provider(), max_delay=timedelta(0))

        for scheduler in (
            FCFSScheduler(),
            EASYBackfillScheduler(runtime_estimate=EXACT),
            carbon,
        ):
            result = Simulator((background, evaluated), Cluster(1), scheduler).run()
            by_id = {record.job_id: record for record in result.records}
            self.assertEqual(by_id["evaluated"].start_time, BASE + timedelta(seconds=600))
            with self.assertRaisesRegex(ValueError, "select the evaluation"):
                account_schedule(result, (background, evaluated), provider())

            cohort = result.replace_records((by_id["evaluated"],))
            metrics = schedule_metrics(account_schedule(cohort, (evaluated,), provider()))
            self.assertEqual(metrics.job_count, 1)
            self.assertEqual(metrics.peak_busy_nodes, 1)
            self.assertEqual(metrics.utilisation, 1.0)

        self.assertEqual(carbon.target_start_times["background"], BASE)

    def test_a_zero_budget_reproduces_easy_exactly(self) -> None:
        jobs = [make_job(index, release_seconds=index * 30, nodes=2) for index in range(8)]
        easy = run(jobs, EASYBackfillScheduler(runtime_estimate=EXACT))
        carbon = run(jobs, CarbonAwareScheduler(provider(), max_delay=timedelta(0)))
        self.assertEqual(start_times(easy), start_times(carbon))

    def test_holding_a_job_cuts_emissions_and_leaves_energy_untouched(self) -> None:
        jobs = [make_job(index, release_seconds=index * 60) for index in range(3)]
        easy = run(jobs, EASYBackfillScheduler(runtime_estimate=EXACT))
        carbon = run(jobs, CarbonAwareScheduler(provider(), max_delay=timedelta(hours=6)))

        self.assertAlmostEqual(total_energy_kwh(carbon), total_energy_kwh(easy), places=9)
        self.assertAlmostEqual(
            total_emissions_gco2e(carbon),
            total_emissions_gco2e(easy) * CLEAN / DIRTY,
            places=6,
        )

    def test_a_held_job_yields_its_place_instead_of_blocking_the_queue(self) -> None:
        # `held` is released first and would occupy the only node, but it is
        # waiting for the clean window, so `prompt` — released inside it — must
        # not be stuck behind it.
        held = make_job("held")
        prompt = make_job("prompt", release_seconds=timedelta(hours=3).total_seconds())
        result = run(
            [held, prompt],
            CarbonAwareScheduler(provider(), max_delay=timedelta(hours=6)),
            total_nodes=1,
        )
        started = start_times(result)
        self.assertEqual(started["held"], CLEAN_FROM)
        self.assertEqual(started["prompt"], prompt.release_time)

    def test_the_budget_is_validated(self) -> None:
        for invalid in (timedelta(seconds=-1), "6h"):
            with self.subTest(max_delay=invalid), self.assertRaises((TypeError, ValueError)):
                CarbonAwareScheduler(provider(), max_delay=invalid)
        with self.assertRaises(ValueError):
            CarbonAwareScheduler(
                provider(), max_delay=timedelta(hours=1), decision_granularity=timedelta(0)
            )


@unittest.skipUnless(DEBUG_TRACE.exists(), f"{DEBUG_TRACE.name} is not present")
@unittest.skipUnless(CARBON_CACHE.exists(), "carbon-intensity cache is not present")
class PM100CarbonAwareTest(unittest.TestCase):
    """The policy on the committed PM100 debug subset."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from hpc_sim.workload import load_jobs
        except ImportError as error:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"pyarrow is unavailable: {error}") from error
        cls.jobs = load_jobs(DEBUG_TRACE, limit=2_000)
        cls.provider = TimeSeriesCarbonIntensityProvider.load(CARBON_CACHE)

    def test_no_job_is_held_beyond_its_delay_budget(self) -> None:
        max_delay = timedelta(hours=6)
        scheduler = CarbonAwareScheduler(self.provider, max_delay=max_delay)
        result = Simulator(self.jobs, Cluster(880), scheduler).run()

        self.assertEqual(len(result.records), len(self.jobs))
        self.assertLessEqual(result.peak_busy_nodes, 880)
        targets = scheduler.target_start_times
        by_id = {job.job_id: job for job in self.jobs}
        held = 0
        for record in result.records:
            target = targets[record.job_id]
            self.assertLessEqual(target, by_id[record.job_id].release_time + max_delay)
            self.assertGreaterEqual(record.start_time, target)
            held += target > record.release_time
        self.assertGreater(held, 0, "no job was ever held; the test is vacuous")

    def test_a_larger_budget_saves_more_carbon_at_the_same_energy(self) -> None:
        def emissions_and_energy(max_delay: timedelta) -> tuple[float, float]:
            scheduler = CarbonAwareScheduler(self.provider, max_delay=max_delay)
            accounted = account_schedule(
                Simulator(self.jobs, Cluster(880), scheduler).run(), self.jobs, self.provider
            )
            return total_emissions_gco2e(accounted), total_energy_kwh(accounted)

        baseline, baseline_energy = emissions_and_energy(timedelta(0))
        short, short_energy = emissions_and_energy(timedelta(hours=3))
        long, long_energy = emissions_and_energy(timedelta(hours=12))

        self.assertLess(long, short)
        self.assertLess(short, baseline)
        for energy in (short_energy, long_energy):
            self.assertAlmostEqual(energy, baseline_energy, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
