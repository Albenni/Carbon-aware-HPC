"""Deterministic checks for the baselines and evaluation metrics.

The synthetic checks need neither the PM100 tables nor a network connection.
The trace checks read the committed 5,000-job debug subset and are skipped when
it, or pyarrow, is absent.

What is actually being verified about EASY is its one guarantee: the job at the
head of the queue holds a reservation, and no backfilled job is allowed to push
it later. Everything else about backfilling is a heuristic; that promise is not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_accounting import JobPowerProfile
from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    BOUNDED_SLOWDOWN_THRESHOLD_SECONDS,
    Cluster,
    Distribution,
    EASYBackfillScheduler,
    FCFSScheduler,
    Job,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
    SimulationError,
    Simulator,
    account_schedule,
    bounded_slowdown,
    estimated_runtime_seconds,
    peak_power_watts,
    schedule_metrics,
    total_emissions_gco2e,
    total_energy_kwh,
)


UTC = timezone.utc
BASE = datetime(2020, 5, 6, 0, 0, tzinfo=UTC)
DEBUG_TRACE = PROJECT_ROOT / "data" / "processed" / "pm100_debug_5000.parquet"
CARBON_CACHE = (
    PROJECT_ROOT
    / "data"
    / "carbon_intensity"
    / "electricity_maps_it_no_04_to_11_2020.json"
)
EXACT = RuntimeEstimateSource.SCHEDULING


def make_job(
    job_id: object,
    *,
    release_seconds: float = 0.0,
    duration_seconds: float = 60.0,
    nodes: int = 1,
    average_power_watts: float = 1_000.0,
    time_limit_seconds: float | None = None,
    submit_seconds: float | None = None,
) -> Job:
    submit = BASE + timedelta(
        seconds=submit_seconds if submit_seconds is not None else release_seconds
    )
    return Job(
        job_id=job_id,
        submit_time=submit,
        release_time=BASE + timedelta(seconds=release_seconds),
        nodes_required=nodes,
        actual_duration_seconds=duration_seconds,
        power=JobPowerProfile(
            job_id=job_id,
            duration_seconds=duration_seconds,
            average_power_watts=average_power_watts,
        ),
        time_limit_seconds=time_limit_seconds,
    )


def at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def random_workload(seed: int, count: int, total_nodes: int) -> list[Job]:
    generator = random.Random(seed)
    return [
        make_job(
            index,
            release_seconds=generator.randrange(0, 4_000),
            duration_seconds=generator.randrange(10, 900),
            nodes=generator.randrange(1, total_nodes + 1),
            average_power_watts=generator.randrange(200, 2_000),
        )
        for index in range(count)
    ]


def peak_concurrency(records) -> int:
    """Recompute peak node occupancy from the output, independently of Cluster."""

    deltas: list[tuple[datetime, int]] = []
    for record in records:
        deltas.append((record.start_time, record.nodes_required))
        deltas.append((record.end_time, -record.nodes_required))
    deltas.sort(key=lambda item: (item[0], item[1]))

    busy = 0
    peak = 0
    for _, delta in deltas:
        busy += delta
        peak = max(peak, busy)
    return peak


def peak_scheduled_power(records, jobs) -> float:
    """Peak aggregate power from the jobs' scheduling-time average power.

    Deliberately independent of :func:`hpc_sim.metrics.peak_power_watts`, which
    recovers power from accounted energy instead, so a power cap is checked
    against the quantity the policy itself reasoned about.
    """

    by_id = {job.job_id: job for job in jobs}
    deltas: list[tuple[datetime, int, float]] = []
    for record in records:
        watts = by_id[record.job_id].scheduling_average_power_watts
        deltas.append((record.end_time, 0, -watts))
        deltas.append((record.start_time, 1, watts))
    deltas.sort(key=lambda item: (item[0], item[1]))

    drawn = 0.0
    peak = 0.0
    for _, _, delta in deltas:
        drawn += delta
        peak = max(peak, drawn)
    return peak


class EASYHandComputedTest(unittest.TestCase):
    """Backfilling decisions small enough to verify by hand."""

    def test_a_short_job_overtakes_the_blocked_head(self) -> None:
        # `holder` keeps 3 of 4 nodes until t=100, so `big` cannot start and is
        # reserved for t=100. `small` fits in the idle node and ends at t=12,
        # well before that reservation, so it may overtake.
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3),
            make_job("big", release_seconds=1, duration_seconds=10, nodes=4),
            make_job("small", release_seconds=2, duration_seconds=10, nodes=1),
        ]
        scheduler = EASYBackfillScheduler(runtime_estimate=EXACT)
        result = Simulator(jobs, Cluster(4), scheduler).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["small"].start_time, at(2))
        self.assertEqual(by_id["big"].start_time, at(100))
        self.assertEqual(scheduler.first_reservations["big"], at(100))
        self.assertEqual(scheduler.backfilled_job_ids, frozenset({"small"}))

    def test_a_job_that_would_break_the_reservation_waits(self) -> None:
        # Same shape, but `long` would still hold its node at t=100, and the
        # pivot needs every node then. It must not overtake.
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3),
            make_job("big", release_seconds=1, duration_seconds=10, nodes=4),
            make_job("long", release_seconds=2, duration_seconds=500, nodes=1),
        ]
        scheduler = EASYBackfillScheduler(runtime_estimate=EXACT)
        result = Simulator(jobs, Cluster(4), scheduler).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["big"].start_time, at(100))
        self.assertEqual(by_id["long"].start_time, at(110))
        self.assertEqual(scheduler.backfilled_job_ids, frozenset())

    def test_a_long_job_may_use_nodes_the_pivot_does_not_need(self) -> None:
        # `holder` frees 3 nodes at t=100 but the pivot needs only 2, so one
        # node is spare at the reservation. `long` fits in that spare node and
        # may overtake despite running well past t=100.
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3),
            make_job("pivot", release_seconds=1, duration_seconds=10, nodes=4),
            make_job("long", release_seconds=2, duration_seconds=500, nodes=1),
        ]
        scheduler = EASYBackfillScheduler(runtime_estimate=EXACT)
        result = Simulator(jobs, Cluster(5), scheduler).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(scheduler.first_reservations["pivot"], at(100))
        self.assertEqual(by_id["long"].start_time, at(2))
        self.assertEqual(by_id["pivot"].start_time, at(100))
        self.assertEqual(scheduler.backfilled_job_ids, frozenset({"long"}))

    def test_easy_matches_fcfs_when_nothing_can_backfill(self) -> None:
        # Every job needs the whole cluster, so backfilling has no opening and
        # EASY must degenerate to exactly the FCFS schedule.
        jobs = [
            make_job(index, release_seconds=index, duration_seconds=50, nodes=4)
            for index in range(6)
        ]
        fcfs = Simulator(jobs, Cluster(4), FCFSScheduler()).run()
        easy = Simulator(jobs, Cluster(4), EASYBackfillScheduler(runtime_estimate=EXACT)).run()

        self.assertEqual(
            {record.job_id: record.start_time for record in fcfs.records},
            {record.job_id: record.start_time for record in easy.records},
        )

    def test_easy_fills_a_hole_that_fcfs_leaves_idle(self) -> None:
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3),
            make_job("big", release_seconds=1, duration_seconds=10, nodes=4),
            make_job("small", release_seconds=2, duration_seconds=10, nodes=1),
        ]
        fcfs = Simulator(jobs, Cluster(4), FCFSScheduler()).run()
        easy = Simulator(jobs, Cluster(4), EASYBackfillScheduler(runtime_estimate=EXACT)).run()

        self.assertGreater(easy.utilisation, fcfs.utilisation)
        self.assertLess(easy.makespan_seconds, fcfs.makespan_seconds)


class RuntimeEstimateTest(unittest.TestCase):
    """The estimate source is an information choice, and it must bite."""

    def test_time_limit_is_preferred_when_present(self) -> None:
        job = make_job("j", duration_seconds=60, time_limit_seconds=3_600)
        self.assertEqual(
            estimated_runtime_seconds(job, RuntimeEstimateSource.TIME_LIMIT),
            3_600,
        )
        self.assertEqual(estimated_runtime_seconds(job, EXACT), 60)

    def test_time_limit_falls_back_when_no_limit_is_recorded(self) -> None:
        job = make_job("j", duration_seconds=60)
        self.assertEqual(
            estimated_runtime_seconds(job, RuntimeEstimateSource.TIME_LIMIT),
            60,
        )

    def test_an_overstated_limit_suppresses_a_legal_backfill(self) -> None:
        # `small` really ends at t=12, before the pivot's reservation at t=100,
        # but its requested walltime says t=902. Classic EASY must believe the
        # request and refuse the backfill that perfect information allows.
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3, time_limit_seconds=100),
            make_job("big", release_seconds=1, duration_seconds=10, nodes=4, time_limit_seconds=10),
            make_job(
                "small",
                release_seconds=2,
                duration_seconds=10,
                nodes=1,
                time_limit_seconds=900,
            ),
        ]
        pessimistic = EASYBackfillScheduler(
            runtime_estimate=RuntimeEstimateSource.TIME_LIMIT
        )
        Simulator(jobs, Cluster(4), pessimistic).run()
        self.assertEqual(pessimistic.backfilled_job_ids, frozenset())

        optimistic = EASYBackfillScheduler(runtime_estimate=EXACT)
        Simulator(jobs, Cluster(4), optimistic).run()
        self.assertEqual(optimistic.backfilled_job_ids, frozenset({"small"}))


class EASYReservationInvariantTest(unittest.TestCase):
    """The no-starvation promise, checked on randomized workloads."""

    def test_no_job_starts_after_the_reservation_it_was_promised(self) -> None:
        for seed in (1, 20200506, 77):
            with self.subTest(seed=seed):
                jobs = random_workload(seed, 300, 32)
                scheduler = EASYBackfillScheduler(runtime_estimate=EXACT)
                result = Simulator(jobs, Cluster(32), scheduler).run()

                reservations = scheduler.first_reservations
                self.assertTrue(reservations, "no job was ever blocked; test is vacuous")
                for record in result.records:
                    promised = reservations.get(record.job_id)
                    if promised is None:
                        continue
                    self.assertLessEqual(
                        record.start_time,
                        promised,
                        f"job {record.job_id} started after its reservation",
                    )

    def test_capacity_holds_and_every_job_runs_once(self) -> None:
        for total_nodes in (4, 16, 64):
            with self.subTest(total_nodes=total_nodes):
                jobs = random_workload(11 + total_nodes, 250, total_nodes)
                result = Simulator(
                    jobs, Cluster(total_nodes), EASYBackfillScheduler(runtime_estimate=EXACT)
                ).run()

                self.assertEqual(len(result.records), len(jobs))
                self.assertLessEqual(result.peak_busy_nodes, total_nodes)
                self.assertEqual(peak_concurrency(result.records), result.peak_busy_nodes)
                by_id = {job.job_id: job for job in jobs}
                for record in result.records:
                    self.assertGreaterEqual(record.start_time, record.release_time)
                    self.assertAlmostEqual(
                        record.runtime_seconds,
                        by_id[record.job_id].actual_duration_seconds,
                    )

    def test_backfilling_never_loses_ground_to_fcfs(self) -> None:
        # Not a theorem, but the reason the policy exists: on contended
        # workloads filling the holes must not cost makespan or utilisation.
        for seed in (3, 5, 8):
            with self.subTest(seed=seed):
                jobs = random_workload(seed, 400, 16)
                fcfs = Simulator(jobs, Cluster(16), FCFSScheduler()).run()
                easy = Simulator(
                    jobs, Cluster(16), EASYBackfillScheduler(runtime_estimate=EXACT)
                ).run()

                self.assertLessEqual(easy.makespan_seconds, fcfs.makespan_seconds)
                easy_waits = [record.waiting_seconds for record in easy.records]
                fcfs_waits = [record.waiting_seconds for record in fcfs.records]
                self.assertLessEqual(sum(easy_waits), sum(fcfs_waits))

    def test_a_narrow_backfill_window_stays_correct(self) -> None:
        jobs = random_workload(42, 200, 16)
        scheduler = EASYBackfillScheduler(runtime_estimate=EXACT, backfill_window=5)
        result = Simulator(jobs, Cluster(16), scheduler).run()

        self.assertEqual(len(result.records), len(jobs))
        self.assertLessEqual(result.peak_busy_nodes, 16)
        for record in result.records:
            promised = scheduler.first_reservations.get(record.job_id)
            if promised is not None:
                self.assertLessEqual(record.start_time, promised)

    def test_a_rejected_window_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            EASYBackfillScheduler(backfill_window=0)


class PowerCapTest(unittest.TestCase):
    """The energy-aware baseline: a budget on power, blind to the grid."""

    def test_the_cap_is_never_exceeded(self) -> None:
        for seed in (2, 4, 6):
            with self.subTest(seed=seed):
                jobs = random_workload(seed, 250, 16)
                cap = 6_000.0
                result = Simulator(
                    jobs, Cluster(16), PowerCappedEASYScheduler(cap, runtime_estimate=EXACT)
                ).run()

                self.assertLessEqual(peak_scheduled_power(result.records, jobs), cap)
                self.assertEqual(len(result.records), len(jobs))

    def test_the_cap_actually_binds(self) -> None:
        jobs = random_workload(9, 250, 16)
        uncapped = Simulator(
            jobs, Cluster(16), EASYBackfillScheduler(runtime_estimate=EXACT)
        ).run()
        cap = 0.5 * peak_scheduled_power(uncapped.records, jobs)
        capped = Simulator(
            jobs, Cluster(16), PowerCappedEASYScheduler(cap, runtime_estimate=EXACT)
        ).run()

        self.assertLess(
            peak_scheduled_power(capped.records, jobs),
            peak_scheduled_power(uncapped.records, jobs),
        )
        # Deferring for power costs time; it cannot cost energy, because the
        # jobs are unchanged. That contrast is the point of the baseline.
        self.assertGreater(
            sum(record.waiting_seconds for record in capped.records),
            sum(record.waiting_seconds for record in uncapped.records),
        )

    def test_a_job_above_the_cap_is_rejected_at_release(self) -> None:
        jobs = [make_job("hog", average_power_watts=9_000.0)]
        with self.assertRaises(SimulationError):
            Simulator(jobs, Cluster(4), PowerCappedEASYScheduler(2_500.0)).run()

    def test_the_cap_must_be_positive_and_finite(self) -> None:
        for invalid in (0.0, -1.0, float("inf")):
            with self.subTest(cap=invalid), self.assertRaises(ValueError):
                PowerCappedEASYScheduler(invalid)

    def test_the_cap_serialises_jobs_that_together_exceed_it(self) -> None:
        jobs = [make_job(index, duration_seconds=60, average_power_watts=1_000.0) for index in range(4)]
        result = Simulator(
            jobs, Cluster(4), PowerCappedEASYScheduler(2_500.0, runtime_estimate=EXACT)
        ).run()

        self.assertEqual(result.peak_busy_nodes, 2)
        self.assertEqual(peak_scheduled_power(result.records, jobs), 2_000.0)


class MetricsTest(unittest.TestCase):
    def test_bounded_slowdown_is_computed_from_wait_and_runtime(self) -> None:
        jobs = [
            make_job("a-hog", duration_seconds=100, nodes=1),
            make_job("waiter", release_seconds=0, duration_seconds=100, nodes=1),
        ]
        result = Simulator(jobs, Cluster(1), FCFSScheduler()).run()
        by_id = {record.job_id: record for record in result.records}

        # `a-hog` starts immediately; `waiter` waits 100 s and runs 100 s.
        self.assertEqual(bounded_slowdown(by_id["a-hog"]), 1.0)
        self.assertEqual(bounded_slowdown(by_id["waiter"]), 2.0)

    def test_short_jobs_are_floored_by_the_threshold(self) -> None:
        # Ids order the queue at equal release times, so "a-hog" runs first and
        # "z-blip" is the one left waiting.
        jobs = [
            make_job("a-hog", duration_seconds=100, nodes=1),
            make_job("z-blip", release_seconds=0, duration_seconds=1, nodes=1),
        ]
        result = Simulator(jobs, Cluster(1), FCFSScheduler()).run()
        blip = next(record for record in result.records if record.job_id == "z-blip")

        # Waited 100 s and ran 1 s: unbounded slowdown would be 101, the
        # threshold caps the denominator's collapse at 10 s.
        self.assertEqual(blip.waiting_seconds, 100.0)
        self.assertAlmostEqual(
            bounded_slowdown(blip),
            101.0 / BOUNDED_SLOWDOWN_THRESHOLD_SECONDS,
        )
        self.assertGreater(bounded_slowdown(blip, threshold_seconds=1.0), bounded_slowdown(blip))

    def test_distribution_uses_nearest_rank_percentiles(self) -> None:
        distribution = Distribution.of(list(range(1, 101)))

        self.assertEqual(distribution.median, 50.5)
        self.assertEqual(distribution.p95, 95)
        self.assertEqual(distribution.p99, 99)
        self.assertEqual(distribution.maximum, 100)
        self.assertEqual(Distribution.of([7.0]).p99, 7.0)

    def test_peak_power_follows_the_overlap(self) -> None:
        # Two 1 kW jobs on one node each, overlapping for 50 s of their runs.
        jobs = [
            make_job("a", duration_seconds=100, average_power_watts=1_000.0),
            make_job("b", release_seconds=50, duration_seconds=100, average_power_watts=1_000.0),
        ]
        result = Simulator(jobs, Cluster(2), FCFSScheduler()).run()
        self.assertAlmostEqual(peak_scheduled_power(result.records, jobs), 2_000.0)

    def test_metrics_switch_reference_point(self) -> None:
        jobs = [make_job("j", submit_seconds=0, release_seconds=30, duration_seconds=10)]
        result = Simulator(jobs, Cluster(1), FCFSScheduler()).run()

        from_release = schedule_metrics(result, reference="release")
        from_submit = schedule_metrics(result, reference="submit")

        self.assertEqual(from_release.waiting.mean, 0.0)
        self.assertEqual(from_submit.waiting.mean, 30.0)
        self.assertEqual(from_release.turnaround.mean, 10.0)
        self.assertEqual(from_submit.turnaround.mean, 40.0)
        with self.assertRaises(ValueError):
            schedule_metrics(result, reference="whenever")

    def test_throughput_and_utilisation_are_consistent(self) -> None:
        jobs = [
            make_job("a", duration_seconds=1_800, nodes=2),
            make_job("b", release_seconds=1_800, duration_seconds=1_800, nodes=2),
        ]
        result = Simulator(jobs, Cluster(2), FCFSScheduler()).run()
        metrics = schedule_metrics(result)

        self.assertEqual(metrics.makespan_seconds, 3_600.0)
        self.assertAlmostEqual(metrics.throughput_jobs_per_hour, 2.0)
        self.assertAlmostEqual(metrics.utilisation, 1.0)


@unittest.skipUnless(DEBUG_TRACE.exists(), f"{DEBUG_TRACE.name} is not present")
class PM100BaselineTest(unittest.TestCase):
    """The baselines on the committed PM100 debug subset."""

    jobs: tuple[Job, ...]

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from hpc_sim.workload import load_jobs
        except ImportError as error:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"pyarrow is unavailable: {error}") from error
        cls.jobs = load_jobs(DEBUG_TRACE, limit=5_000)

    def test_easy_respects_capacity_and_reservations_on_the_real_trace(self) -> None:
        scheduler = EASYBackfillScheduler(runtime_estimate=EXACT)
        result = Simulator(self.jobs, Cluster(880), scheduler).run()

        self.assertEqual(len(result.records), len(self.jobs))
        self.assertLessEqual(result.peak_busy_nodes, 880)
        self.assertEqual(peak_concurrency(result.records), result.peak_busy_nodes)
        self.assertTrue(scheduler.backfilled_job_ids, "no job ever backfilled")
        for record in result.records:
            promised = scheduler.first_reservations.get(record.job_id)
            if promised is not None:
                self.assertLessEqual(record.start_time, promised)

    def test_easy_shortens_the_slowdown_tail_against_fcfs(self) -> None:
        fcfs = schedule_metrics(Simulator(self.jobs, Cluster(880), FCFSScheduler()).run())
        easy = schedule_metrics(
            Simulator(
                self.jobs, Cluster(880), EASYBackfillScheduler(runtime_estimate=EXACT)
            ).run()
        )

        self.assertLess(easy.bounded_slowdown.maximum, fcfs.bounded_slowdown.maximum)
        self.assertLessEqual(easy.waiting.mean, fcfs.waiting.mean)

    def test_the_power_cap_binds_on_the_real_trace(self) -> None:
        uncapped = Simulator(
            self.jobs, Cluster(880), EASYBackfillScheduler(runtime_estimate=EXACT)
        ).run()
        cap = 0.8 * peak_scheduled_power(uncapped.records, self.jobs)
        capped = Simulator(
            self.jobs, Cluster(880), PowerCappedEASYScheduler(cap, runtime_estimate=EXACT)
        ).run()

        self.assertLessEqual(peak_scheduled_power(capped.records, self.jobs), cap)
        self.assertGreater(
            sum(record.waiting_seconds for record in capped.records),
            sum(record.waiting_seconds for record in uncapped.records),
        )

    @unittest.skipUnless(CARBON_CACHE.exists(), "carbon-intensity cache is not present")
    def test_every_baseline_consumes_the_same_energy(self) -> None:
        provider = TimeSeriesCarbonIntensityProvider.load(CARBON_CACHE)
        schedulers = (
            FCFSScheduler(),
            EASYBackfillScheduler(runtime_estimate=EXACT),
            PowerCappedEASYScheduler(400_000.0, runtime_estimate=EXACT),
        )
        accounted = [
            account_schedule(
                Simulator(self.jobs, Cluster(880), scheduler).run(), self.jobs, provider
            )
            for scheduler in schedulers
        ]

        energies = [total_energy_kwh(result) for result in accounted]
        for energy in energies[1:]:
            self.assertAlmostEqual(
                energy,
                energies[0],
                places=6,
                msg="a schedule cannot change the energy the same jobs consume",
            )

        # Emissions may move, because the jobs meet a different signal. That
        # separation is the whole reason a carbon-aware policy can exist.
        emissions = [total_emissions_gco2e(result) for result in accounted]
        self.assertNotAlmostEqual(emissions[0], emissions[-1], places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
