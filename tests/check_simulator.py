"""Deterministic checks for the discrete-event simulator.

The synthetic checks need neither the PM100 tables nor a network connection.
The trace checks read the committed 5,000-job debug subset and are skipped when
it, or pyarrow, is absent.
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
from carbon_intensity import (
    FIFTEEN_MINUTES,
    CarbonIntensitySample,
    TimeSeriesCarbonIntensityProvider,
)
from hpc_sim import (
    CapacityError,
    Cluster,
    CoverageError,
    FCFSScheduler,
    Job,
    Scheduler,
    SimulationError,
    Simulator,
    TraceReplayScheduler,
    account_schedule,
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


def make_job(
    job_id: object,
    *,
    release_seconds: float = 0.0,
    duration_seconds: float = 60.0,
    nodes: int = 1,
    average_power_watts: float = 1_000.0,
    profile: tuple[float, ...] | None = None,
    trace_start_seconds: float | None = None,
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
            power_profile_watts=profile,
        ),
        trace_start_time=(
            None
            if trace_start_seconds is None
            else BASE + timedelta(seconds=trace_start_seconds)
        ),
    )


def peak_concurrency(records) -> int:
    """Recompute peak node occupancy from the output, independently of Cluster."""

    deltas: list[tuple[datetime, int]] = []
    for record in records:
        deltas.append((record.start_time, record.nodes_required))
        deltas.append((record.end_time, -record.nodes_required))
    # A release at an instant must be applied before an allocation at the same
    # instant, matching the engine's completions-before-scheduling rule.
    deltas.sort(key=lambda item: (item[0], item[1]))

    busy = 0
    peak = 0
    for _, delta in deltas:
        busy += delta
        peak = max(peak, busy)
    return peak


def synthetic_provider(
    values: tuple[float, ...],
    *,
    start: datetime = BASE,
) -> TimeSeriesCarbonIntensityProvider:
    return TimeSeriesCarbonIntensityProvider(
        tuple(
            CarbonIntensitySample(
                timestamp=start + index * FIFTEEN_MINUTES,
                intensity_gco2e_per_kwh=value,
            )
            for index, value in enumerate(values)
        )
    )


class CountingCarbonScheduler(FCFSScheduler):
    """FCFS that also opts into carbon-intensity boundary events."""

    name = "fcfs-carbon-aware-wakeups"
    wants_carbon_intensity_events = True

    def __init__(self) -> None:
        self.boundaries: list[datetime] = []

    def on_carbon_intensity_change(self, now, simulator) -> None:
        self.boundaries.append(now)


class ExactScheduleTest(unittest.TestCase):
    def test_hand_computed_fcfs_schedule(self) -> None:
        # Two nodes. A occupies both for 100 s, so B and C wait and then run
        # together for 50 s.
        jobs = [
            make_job("A", duration_seconds=100, nodes=2),
            make_job("B", duration_seconds=50, nodes=1),
            make_job("C", duration_seconds=50, nodes=1),
        ]
        result = Simulator(jobs, Cluster(2), FCFSScheduler()).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["A"].start_time, BASE)
        self.assertEqual(by_id["A"].end_time, BASE + timedelta(seconds=100))
        self.assertEqual(by_id["A"].waiting_seconds, 0.0)

        for job_id in ("B", "C"):
            self.assertEqual(by_id[job_id].start_time, BASE + timedelta(seconds=100))
            self.assertEqual(by_id[job_id].end_time, BASE + timedelta(seconds=150))
            self.assertEqual(by_id[job_id].waiting_seconds, 100.0)
            self.assertEqual(by_id[job_id].turnaround_seconds, 150.0)

        self.assertEqual(result.makespan_seconds, 150.0)
        self.assertEqual(result.peak_busy_nodes, 2)
        self.assertAlmostEqual(result.utilisation, 1.0)

    def test_waiting_measured_from_both_reference_points(self) -> None:
        job = make_job("solo", submit_seconds=0, release_seconds=30, duration_seconds=10)
        result = Simulator([job], Cluster(1), FCFSScheduler()).run()
        record = result.records[0]

        self.assertEqual(record.waiting_seconds, 0.0)
        self.assertEqual(record.waiting_seconds_from_submit, 30.0)
        self.assertEqual(record.turnaround_seconds, 10.0)
        self.assertEqual(record.turnaround_seconds_from_submit, 40.0)

    def test_nodes_freed_at_an_instant_are_reusable_at_that_instant(self) -> None:
        jobs = [
            make_job("first", duration_seconds=100, nodes=1),
            make_job("second", release_seconds=100, duration_seconds=10, nodes=1),
        ]
        result = Simulator(jobs, Cluster(1), FCFSScheduler()).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["second"].start_time, BASE + timedelta(seconds=100))
        self.assertEqual(by_id["second"].waiting_seconds, 0.0)


class StrictOrderingTest(unittest.TestCase):
    def test_fcfs_does_not_backfill(self) -> None:
        # `big` blocks the queue; `small` fits in the idle node but must not
        # overtake it.
        jobs = [
            make_job("holder", duration_seconds=100, nodes=3),
            make_job("big", release_seconds=1, duration_seconds=10, nodes=4),
            make_job("small", release_seconds=2, duration_seconds=10, nodes=1),
        ]
        result = Simulator(jobs, Cluster(4), FCFSScheduler()).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["big"].start_time, BASE + timedelta(seconds=100))
        self.assertGreaterEqual(by_id["small"].start_time, by_id["big"].start_time)

    def test_no_job_overtakes_an_earlier_queued_job(self) -> None:
        generator = random.Random(20200506)
        jobs = [
            make_job(
                index,
                release_seconds=generator.randrange(0, 4_000),
                duration_seconds=generator.randrange(10, 900),
                nodes=generator.randrange(1, 33),
            )
            for index in range(400)
        ]
        result = Simulator(jobs, Cluster(64), FCFSScheduler()).run()

        by_id = {record.job_id: record for record in result.records}
        order = sorted(jobs, key=lambda job: (job.release_time, str(job.job_id)))
        for position, job in enumerate(order):
            start = by_id[job.job_id].start_time
            for earlier in order[:position]:
                earlier_start = by_id[earlier.job_id].start_time
                # An earlier-ranked job may start later only because it was not
                # yet eligible when this one started.
                if earlier.release_time <= start:
                    self.assertLessEqual(
                        earlier_start,
                        start,
                        f"job {job.job_id} overtook {earlier.job_id}",
                    )


class CapacityInvariantTest(unittest.TestCase):
    def test_capacity_is_never_exceeded_under_a_random_workload(self) -> None:
        generator = random.Random(7)
        for total_nodes in (1, 8, 64):
            with self.subTest(total_nodes=total_nodes):
                jobs = [
                    make_job(
                        index,
                        release_seconds=generator.randrange(0, 5_000),
                        duration_seconds=generator.randrange(1, 1_200),
                        nodes=generator.randrange(1, total_nodes + 1),
                    )
                    for index in range(300)
                ]
                result = Simulator(jobs, Cluster(total_nodes), FCFSScheduler()).run()

                self.assertLessEqual(result.peak_busy_nodes, total_nodes)
                self.assertEqual(
                    peak_concurrency(result.records),
                    result.peak_busy_nodes,
                    "independent sweep-line disagrees with the cluster counter",
                )

    def test_every_job_runs_exactly_once_for_its_actual_duration(self) -> None:
        generator = random.Random(99)
        jobs = [
            make_job(
                index,
                release_seconds=generator.randrange(0, 2_000),
                duration_seconds=generator.randrange(1, 600),
                nodes=generator.randrange(1, 17),
            )
            for index in range(250)
        ]
        result = Simulator(jobs, Cluster(32), FCFSScheduler()).run()

        self.assertEqual(len(result.records), len(jobs))
        self.assertEqual(
            {record.job_id for record in result.records},
            {job.job_id for job in jobs},
        )
        by_id = {job.job_id: job for job in jobs}
        for record in result.records:
            self.assertAlmostEqual(
                record.runtime_seconds,
                by_id[record.job_id].actual_duration_seconds,
            )
            self.assertGreaterEqual(record.start_time, record.release_time)

    def test_job_larger_than_the_cluster_is_rejected(self) -> None:
        jobs = [make_job("huge", nodes=9)]
        with self.assertRaises(CapacityError):
            Simulator(jobs, Cluster(8), FCFSScheduler())

    def test_cluster_rejects_impossible_transitions(self) -> None:
        cluster = Cluster(4)
        cluster.allocate(4)
        with self.assertRaises(CapacityError):
            cluster.allocate(1)
        cluster.release(4)
        with self.assertRaises(CapacityError):
            cluster.release(1)


class EventSourceTest(unittest.TestCase):
    def test_carbon_boundaries_are_emitted_only_when_requested(self) -> None:
        jobs = [make_job("j", duration_seconds=3_600, nodes=1)]

        blind = Simulator(
            jobs,
            Cluster(1),
            FCFSScheduler(),
            carbon_intensity_granularity=FIFTEEN_MINUTES,
        ).run()
        self.assertEqual(len(blind.records), 1)

        scheduler = CountingCarbonScheduler()
        Simulator(
            jobs,
            Cluster(1),
            scheduler,
            carbon_intensity_granularity=FIFTEEN_MINUTES,
        ).run()

        # A one-hour job starting on a bucket edge crosses three interior
        # boundaries; generation stops once nothing is queued or running.
        self.assertEqual(
            scheduler.boundaries,
            [BASE + index * FIFTEEN_MINUTES for index in (1, 2, 3, 4)],
        )

    def test_wakeup_in_the_past_is_rejected(self) -> None:
        class BadScheduler(FCFSScheduler):
            def select(self, now, queue, cluster, simulator):
                simulator.request_wakeup(now - timedelta(seconds=1))
                return ()

        with self.assertRaises(SimulationError):
            Simulator([make_job("j")], Cluster(1), BadScheduler()).run()

    def test_a_stalled_queue_is_reported(self) -> None:
        class NeverStarts(Scheduler):
            name = "never"

            def select(self, now, queue, cluster, simulator):
                return ()

        with self.assertRaises(SimulationError):
            Simulator([make_job("j")], Cluster(1), NeverStarts()).run()


class JobValidationTest(unittest.TestCase):
    def test_naive_timestamps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Job(
                job_id="j",
                submit_time=datetime(2020, 5, 6, 0, 0),
                release_time=BASE,
                nodes_required=1,
                actual_duration_seconds=10,
                power=JobPowerProfile(duration_seconds=10, average_power_watts=1),
            )

    def test_duration_must_match_the_power_profile(self) -> None:
        with self.assertRaises(ValueError):
            Job(
                job_id="j",
                submit_time=BASE,
                release_time=BASE,
                nodes_required=1,
                actual_duration_seconds=20,
                power=JobPowerProfile(duration_seconds=10, average_power_watts=1),
            )

    def test_release_cannot_precede_submission(self) -> None:
        with self.assertRaises(ValueError):
            Job(
                job_id="j",
                submit_time=BASE,
                release_time=BASE - timedelta(seconds=1),
                nodes_required=1,
                actual_duration_seconds=10,
                power=JobPowerProfile(duration_seconds=10, average_power_watts=1),
            )

    def test_duplicate_job_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Simulator([make_job("j"), make_job("j")], Cluster(4), FCFSScheduler())

    def test_prediction_seam_defaults_to_the_actual_duration(self) -> None:
        job = make_job("j", duration_seconds=100)
        self.assertEqual(job.scheduling_duration_seconds, 100.0)

        predicted = Job(
            job_id="j",
            submit_time=BASE,
            release_time=BASE,
            nodes_required=1,
            actual_duration_seconds=100,
            power=JobPowerProfile(duration_seconds=100, average_power_watts=1),
            predicted_duration_seconds=250,
        )
        self.assertEqual(predicted.scheduling_duration_seconds, 250.0)
        self.assertEqual(predicted.actual_duration_seconds, 100.0)


class CarbonCouplingTest(unittest.TestCase):
    """The energy-aware versus carbon-aware distinction, at schedule level."""

    def _shifted_run(self, offset: timedelta, provider):
        jobs = [
            Job(
                job_id=index,
                submit_time=BASE + offset,
                release_time=BASE + offset,
                nodes_required=1,
                actual_duration_seconds=1_800,
                power=JobPowerProfile(
                    job_id=index,
                    duration_seconds=1_800,
                    average_power_watts=2_000.0,
                ),
            )
            for index in range(4)
        ]
        result = Simulator(jobs, Cluster(2), FCFSScheduler()).run()
        return account_schedule(result, jobs, provider)

    def test_shifting_a_schedule_preserves_energy_but_changes_emissions(self) -> None:
        # A 12 h day/night cycle: clean at 100, dirty at 500 gCO2e/kWh. The whole
        # schedule fits inside one half-cycle, so a 12 h shift moves it wholly
        # from the clean block into the dirty one.
        values: list[float] = []
        for block in range(4):
            values.extend([100.0 if block % 2 == 0 else 500.0] * 48)
        provider = synthetic_provider(tuple(values))

        early = self._shifted_run(timedelta(0), provider)
        late = self._shifted_run(timedelta(hours=12), provider)

        self.assertAlmostEqual(
            total_energy_kwh(early),
            total_energy_kwh(late),
            places=12,
            msg="shifting a schedule must not change the energy it consumes",
        )
        self.assertAlmostEqual(
            total_emissions_gco2e(late) / total_emissions_gco2e(early),
            5.0,
            places=9,
            msg="emissions must follow the intensity the schedule actually meets",
        )

    def test_running_outside_the_cached_series_is_reported_clearly(self) -> None:
        provider = synthetic_provider((200.0,) * 4)  # one hour of coverage
        jobs = [make_job("long", duration_seconds=7_200, average_power_watts=1_000)]
        result = Simulator(jobs, Cluster(1), FCFSScheduler()).run()

        with self.assertRaises(CoverageError):
            account_schedule(result, jobs, provider)

    def test_constant_intensity_reproduces_energy_times_intensity(self) -> None:
        provider = synthetic_provider((400.0,) * 8)
        jobs = [make_job("j", duration_seconds=3_600, average_power_watts=1_000)]
        result = account_schedule(
            Simulator(jobs, Cluster(1), FCFSScheduler()).run(),
            jobs,
            provider,
        )
        record = result.records[0]

        self.assertAlmostEqual(record.energy_kwh, 1.0)
        self.assertAlmostEqual(record.emissions_gco2e, 400.0)


class TraceReplayTest(unittest.TestCase):
    def test_replay_starts_jobs_at_their_recorded_times(self) -> None:
        jobs = [
            make_job(
                "late",
                release_seconds=0,
                trace_start_seconds=500,
                duration_seconds=10,
                nodes=1,
            ),
            make_job(
                "early",
                release_seconds=0,
                trace_start_seconds=100,
                duration_seconds=10,
                nodes=1,
            ),
        ]
        result = Simulator(jobs, Cluster(1), TraceReplayScheduler()).run()
        by_id = {record.job_id: record for record in result.records}

        self.assertEqual(by_id["early"].start_time, BASE + timedelta(seconds=100))
        self.assertEqual(by_id["late"].start_time, BASE + timedelta(seconds=500))
        self.assertEqual(by_id["early"].delay_vs_trace_seconds, 0.0)
        self.assertEqual(by_id["late"].delay_vs_trace_seconds, 0.0)

    def test_replay_requires_a_recorded_start_time(self) -> None:
        with self.assertRaises(SimulationError):
            Simulator([make_job("j")], Cluster(1), TraceReplayScheduler()).run()

    def test_replay_reports_a_capacity_that_is_too_small(self) -> None:
        jobs = [
            make_job("a", trace_start_seconds=0, duration_seconds=100, nodes=1),
            make_job("b", trace_start_seconds=0, duration_seconds=100, nodes=1),
        ]
        with self.assertRaises(SimulationError):
            Simulator(jobs, Cluster(1), TraceReplayScheduler()).run()


@unittest.skipUnless(DEBUG_TRACE.exists(), f"{DEBUG_TRACE.name} is not present")
class PM100TraceTest(unittest.TestCase):
    """End-to-end checks against the committed PM100 debug subset."""

    jobs: tuple[Job, ...]

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from hpc_sim.workload import load_jobs
        except ImportError as error:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"pyarrow is unavailable: {error}") from error
        cls.jobs = load_jobs(DEBUG_TRACE, limit=5_000)

    def test_replay_reproduces_the_historical_schedule_exactly(self) -> None:
        result = Simulator(self.jobs, Cluster(880), TraceReplayScheduler()).run()

        deltas = [record.delay_vs_trace_seconds for record in result.records]
        self.assertTrue(all(delta == 0.0 for delta in deltas))
        self.assertLessEqual(result.peak_busy_nodes, 880)
        self.assertEqual(peak_concurrency(result.records), result.peak_busy_nodes)

        # Waiting times must fall out of the trace itself, not out of the policy.
        by_id = {job.job_id: job for job in self.jobs}
        for record in result.records:
            job = by_id[record.job_id]
            assert job.trace_start_time is not None
            self.assertAlmostEqual(
                record.waiting_seconds,
                (job.trace_start_time - job.release_time).total_seconds(),
            )

    def test_fcfs_respects_capacity_on_the_real_trace(self) -> None:
        result = Simulator(self.jobs, Cluster(880), FCFSScheduler()).run()

        self.assertEqual(len(result.records), len(self.jobs))
        self.assertLessEqual(result.peak_busy_nodes, 880)
        self.assertEqual(peak_concurrency(result.records), result.peak_busy_nodes)
        for record in result.records:
            self.assertGreaterEqual(record.start_time, record.release_time)

    @unittest.skipUnless(CARBON_CACHE.exists(), "carbon-intensity cache is not present")
    def test_energy_is_schedule_invariant_on_the_real_trace(self) -> None:
        provider = TimeSeriesCarbonIntensityProvider.load(CARBON_CACHE)

        replay = account_schedule(
            Simulator(self.jobs, Cluster(880), TraceReplayScheduler()).run(),
            self.jobs,
            provider,
        )
        fcfs = account_schedule(
            Simulator(self.jobs, Cluster(880), FCFSScheduler()).run(),
            self.jobs,
            provider,
        )

        self.assertAlmostEqual(
            total_energy_kwh(replay),
            total_energy_kwh(fcfs),
            places=6,
            msg="the same jobs must consume the same energy under any schedule",
        )
        self.assertNotAlmostEqual(
            total_emissions_gco2e(replay),
            total_emissions_gco2e(fcfs),
            places=3,
            msg="different schedules must meet different carbon intensity",
        )

    @unittest.skipUnless(CARBON_CACHE.exists(), "carbon-intensity cache is not present")
    def test_average_model_tracks_the_measured_profile(self) -> None:
        provider = TimeSeriesCarbonIntensityProvider.load(CARBON_CACHE)
        result = account_schedule(
            Simulator(self.jobs, Cluster(880), TraceReplayScheduler()).run(),
            self.jobs,
            provider,
        )

        measured_kwh = total_energy_kwh(result)
        average_kwh = sum(
            record.energy_kwh_average_model or 0.0 for record in result.records
        )
        # The loader uses the duration-weighted mean, so the two models must
        # consume identical energy by construction; only their emissions differ.
        self.assertAlmostEqual(measured_kwh, average_kwh, places=6)

        measured_g = total_emissions_gco2e(result)
        average_g = sum(
            record.emissions_gco2e_average_model or 0.0 for record in result.records
        )
        relative_gap = abs(measured_g - average_g) / measured_g
        self.assertLess(relative_gap, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
