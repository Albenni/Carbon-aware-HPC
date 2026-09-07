"""Deterministic checks for the scheduling comparison matrix.

The whole point of the matrix is that a difference between two rows is
attributable to one axis, so what is checked here are the invariants that make
that true: one cohort everywhere, a forecast that never postdates the decision
reading it, and an oracle bounded by the same reach as the forecast run it is
the denominator for. Everything runs on a hand-made grid signal and a hand-made
archive, so no data files and no network are involved.

The archive is wrong on purpose: it puts the clean window two hours *before* the
one the grid really had. A run that quietly read the observations instead of the
forecast would land on the true window and be caught.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # scripts/ is not a package

from carbon_accounting import JobPowerProfile
from carbon_intensity import CarbonIntensitySample, TimeSeriesCarbonIntensityProvider
from carbon_intensity.scheduling_impact import archive_reach
from carbon_intensity.series import CarbonIntensityForecast
from carbon_intensity.snapshots import ForecastArchive
from hpc_sim import Job

from compare_baselines import (
    DERIVED_FIELDS,
    IDENTITY_FIELDS,
    align_arms,
    build_parser,
    compare,
    parse_labelled_path,
    write_rows,
)


UTC = timezone.utc
BASE = datetime(2020, 5, 6, 0, 0, tzinfo=UTC)
QUARTER = timedelta(minutes=15)
DIRTY = 400.0
CLEAN = 100.0
#: The clean window the grid really had, and the one the archive predicts. They
#: do not overlap, so a decision can be attributed to the series it read.
ACTUAL_CLEAN = (BASE + timedelta(hours=4), BASE + timedelta(hours=6))
FORECAST_CLEAN = (BASE + timedelta(hours=2), BASE + timedelta(hours=4))
HORIZON_HOURS = 12
CADENCE_MINUTES = 60
#: Every job fits inside a clean window, and the cluster holds them all at once,
#: so which window a policy aims at is the only thing the emissions can reflect.
DURATION_SECONDS = 7_200.0
NODES = 4
#: One runtime is too short to reach either window, so the duration-scaled
#: family is given the fixed family's budget rather than a budget of nothing.
DELAY_HOURS = 6.0
DELAY_FRACTION = 3.0

#: Columns the CSV has to carry whatever the run was, on top of the identity
#: ones. These are the figures the comparison is read for.
REQUIRED_METRICS = (
    "total_emissions_tco2e",
    "emissions_saved_vs_easy",
    "oracle_recovery",
    "total_energy_mwh",
    "peak_power_mw",
    "waiting_mean_s",
    "waiting_p95_s",
    "bounded_slowdown_mean",
    "bounded_slowdown_p95",
    "utilisation",
    "makespan_days",
    "throughput_jobs_per_hour",
)


def make_job(job_id: object, *, release_seconds: float = 0.0) -> Job:
    release = BASE + timedelta(seconds=release_seconds)
    return Job(
        job_id=job_id,
        submit_time=release,
        release_time=release,
        nodes_required=1,
        actual_duration_seconds=DURATION_SECONDS,
        power=JobPowerProfile(
            job_id=job_id,
            duration_seconds=DURATION_SECONDS,
            average_power_watts=1_000.0,
        ),
        time_limit_seconds=2 * DURATION_SECONDS,
    )


def cohort() -> tuple[Job, ...]:
    return tuple(make_job(index, release_seconds=index * 900) for index in range(4))


def predicted(jobs: tuple[Job, ...], factor: float) -> tuple[Job, ...]:
    """The same jobs as a model would see them: estimates only, truth intact."""

    return tuple(
        replace(
            job,
            predicted_duration_seconds=job.actual_duration_seconds * factor,
            predicted_average_power_watts=900.0,
        )
        for job in jobs
    )


def intensity(timestamp: datetime, window: tuple[datetime, datetime]) -> float:
    return CLEAN if window[0] <= timestamp < window[1] else DIRTY


def provider() -> TimeSeriesCarbonIntensityProvider:
    return TimeSeriesCarbonIntensityProvider(
        [
            CarbonIntensitySample(
                BASE + index * QUARTER, intensity(BASE + index * QUARTER, ACTUAL_CLEAN)
            )
            for index in range(192)  # two days of 15-minute buckets
        ]
    )


class SpyArchive(ForecastArchive):
    """An archive that records what each decision asked for and was given."""

    #: ``(as_of, issue_time)`` per call; a class attribute because the base is a
    #: frozen slots dataclass and this is one archive per test.
    calls: list[tuple[datetime, datetime]] = []

    def get_forecast(self, issue_time, horizon=None):
        snapshot = super().get_forecast(issue_time, horizon)
        SpyArchive.calls.append((issue_time, snapshot.issue_time))
        return snapshot


def archive() -> SpyArchive:
    """Hourly snapshots, each predicting a clean window the grid never had."""

    buckets = HORIZON_HOURS * 4
    snapshots = tuple(
        CarbonIntensityForecast(
            issue,
            tuple(
                CarbonIntensitySample(
                    issue + index * QUARTER,
                    intensity(issue + index * QUARTER, FORECAST_CLEAN),
                )
                for index in range(buckets)
            ),
        )
        for issue in (
            BASE + hour * timedelta(minutes=CADENCE_MINUTES) for hour in range(24)
        )
    )
    return SpyArchive(
        {
            "schema_version": 1,
            "horizon_hours": HORIZON_HOURS,
            "cadence_minutes": CADENCE_MINUTES,
        },
        snapshots,
    )


def arguments(*extra: str):
    return build_parser().parse_args(
        [
            "--nodes",
            str(NODES),
            "--no-replay",
            "--runtime-estimate",
            "scheduling",
            "--max-delay-hours",
            str(DELAY_HOURS),
            "--max-delay-fraction",
            str(DELAY_FRACTION),
            *extra,
        ]
    )


def matrix() -> list[dict[str, object]]:
    """The full matrix: two job-information arms, actual and forecast carbon."""

    jobs = cohort()
    arms = [
        ("actual", "actual", jobs),
        ("predicted", "optimistic", predicted(jobs, 0.5)),
    ]
    SpyArchive.calls = []
    return compare(arguments(), arms, provider(), {"wrong": archive()})


def selected(rows, **identity) -> list[dict[str, object]]:
    return [
        row for row in rows if all(row[key] == value for key, value in identity.items())
    ]


class LabelledArtifactTest(unittest.TestCase):
    def test_a_label_names_the_arm(self) -> None:
        self.assertEqual(
            parse_labelled_path("ridge=data/job_predictions/ridge/test.parquet"),
            ("ridge", Path("data/job_predictions/ridge/test.parquet")),
        )

    def test_a_bare_path_names_itself(self) -> None:
        label, path = parse_labelled_path("data/job_predictions/ridge_baseline/test.parquet")
        self.assertEqual(label, "ridge_baseline")
        self.assertEqual(path, Path("data/job_predictions/ridge_baseline/test.parquet"))

    def test_a_half_written_label_is_refused(self) -> None:
        with self.assertRaises(Exception):
            parse_labelled_path("=data/test.parquet")


class CohortTest(unittest.TestCase):
    """Requirement 1: every configuration is scored on exactly one cohort."""

    def test_the_arms_hold_the_same_jobs_in_the_same_order(self) -> None:
        jobs = cohort()
        arms = align_arms(
            {
                "gradient": (jobs, predicted(jobs, 0.5)),
                "ridge": (jobs, predicted(jobs, 2.0)),
            }
        )

        self.assertEqual([information for information, _, _ in arms], ["actual", "predicted", "predicted"])
        self.assertEqual([model for _, model, _ in arms], ["actual", "gradient", "ridge"])
        identifiers = {tuple(job.job_id for job in arm_jobs) for _, _, arm_jobs in arms}
        self.assertEqual(len(identifiers), 1)
        self.assertEqual(identifiers.pop(), tuple(job.job_id for job in jobs))
        # Only the estimates differ; the ground truth is the same object graph.
        for _, _, arm_jobs in arms:
            self.assertEqual(
                [job.actual_duration_seconds for job in arm_jobs],
                [job.actual_duration_seconds for job in jobs],
            )

    def test_a_job_one_artifact_misses_is_dropped_from_every_arm(self) -> None:
        jobs = cohort()
        partial = jobs[:-1]
        arms = align_arms(
            {
                "gradient": (jobs, predicted(jobs, 0.5)),
                "ridge": (partial, predicted(partial, 2.0)),
            }
        )

        for _, _, arm_jobs in arms:
            self.assertEqual(
                [job.job_id for job in arm_jobs], [job.job_id for job in partial]
            )

    def test_the_limit_truncates_every_arm_alike(self) -> None:
        jobs = cohort()
        arms = align_arms({"gradient": (jobs, predicted(jobs, 0.5))}, limit=2)
        for _, _, arm_jobs in arms:
            self.assertEqual([job.job_id for job in arm_jobs], [0, 1])

    def test_disjoint_artifacts_are_refused(self) -> None:
        jobs = cohort()
        other = tuple(make_job(f"other-{index}") for index in range(2))
        with self.assertRaises(ValueError):
            align_arms(
                {
                    "gradient": (jobs, predicted(jobs, 0.5)),
                    "ridge": (other, predicted(other, 0.5)),
                }
            )

    def test_every_row_of_the_matrix_scores_the_same_jobs(self) -> None:
        rows = matrix()
        self.assertEqual({row["jobs"] for row in rows}, {len(cohort())})
        # Energy is a property of the jobs, not of the schedule, so one cohort
        # scored everywhere means one energy figure everywhere.
        self.assertEqual(len({round(row["total_energy_mwh"], 9) for row in rows}), 1)


class ForecastCausalityTest(unittest.TestCase):
    """Requirement 2: no decision reads a forecast issued after it."""

    def test_every_snapshot_predates_the_decision_that_read_it(self) -> None:
        matrix()

        self.assertTrue(SpyArchive.calls, "no decision ever consulted the archive")
        for as_of, issued in SpyArchive.calls:
            self.assertLessEqual(issued, as_of)

    def test_a_forecast_run_follows_the_archive_and_not_the_observations(self) -> None:
        rows = matrix()
        for delay_policy in ("fixed", "duration-scaled"):
            with self.subTest(delay_policy=delay_policy):
                oracle = selected(
                    rows,
                    job_information="actual",
                    delay_policy=delay_policy,
                    carbon_information="actual",
                    scheduler_family="carbon",
                    oracle_recovery=1.0,
                )
                forecast = selected(
                    rows,
                    job_information="actual",
                    delay_policy=delay_policy,
                    carbon_information="forecast",
                )
                self.assertEqual(len(oracle), 1)
                self.assertEqual(len(forecast), 1)
                # The archive points at a window the grid never had, so a run
                # reading it must do worse than the one reading the truth.
                self.assertGreater(
                    forecast[0]["total_emissions_tco2e"],
                    oracle[0]["total_emissions_tco2e"],
                )


class ReachTest(unittest.TestCase):
    """Requirement 3: the oracle a forecast is scored against shares its reach."""

    def test_the_bounded_oracle_and_every_forecast_share_one_reach(self) -> None:
        rows = matrix()
        expected = archive_reach(archive()) / timedelta(hours=1)

        bounded = [row for row in rows if row["oracle_recovery"] != ""]
        self.assertTrue(bounded)
        for row in bounded:
            self.assertEqual(row["forecast_reach_hours"], expected)
        # The unbounded oracle is still reported, and is not mistaken for one.
        unbounded = selected(
            rows, scheduler_family="carbon", carbon_information="actual", forecast_reach_hours=""
        )
        self.assertTrue(unbounded)
        for row in unbounded:
            self.assertEqual(row["oracle_recovery"], "")

    def test_the_oracle_is_its_own_denominator(self) -> None:
        rows = matrix()
        for row in rows:
            if row["carbon_information"] == "actual" and row["forecast_reach_hours"] != "":
                self.assertEqual(row["oracle_recovery"], 1.0)

    def test_recovery_is_the_share_of_the_bounded_oracle_saving(self) -> None:
        rows = matrix()
        for row in selected(rows, carbon_information="forecast"):
            arm = selected(
                rows,
                job_information=row["job_information"],
                job_model=row["job_model"],
                delay_policy=row["delay_policy"],
                carbon_information="actual",
                oracle_recovery=1.0,
            )
            easy = selected(
                rows,
                job_information=row["job_information"],
                job_model=row["job_model"],
                scheduler_family="easy",
            )
            available = easy[0]["total_emissions_tco2e"] - arm[0]["total_emissions_tco2e"]
            if available <= 0.0:
                continue
            self.assertAlmostEqual(
                row["oracle_recovery"],
                (easy[0]["total_emissions_tco2e"] - row["total_emissions_tco2e"]) / available,
            )


class IdentityColumnTest(unittest.TestCase):
    """Requirement 4: every row says which cell of the matrix it is."""

    def test_every_row_carries_every_identity_column(self) -> None:
        for row in matrix():
            for field in IDENTITY_FIELDS + DERIVED_FIELDS:
                self.assertIn(field, row)

    def test_the_axes_hold_the_values_they_claim(self) -> None:
        rows = matrix()
        self.assertEqual(
            {row["scheduler_family"] for row in rows},
            {"fcfs", "easy", "power-cap-easy", "carbon", "power-cap-carbon"},
        )
        self.assertEqual(
            {row["delay_policy"] for row in rows}, {"none", "fixed", "duration-scaled"}
        )
        self.assertEqual({row["job_information"] for row in rows}, {"actual", "predicted"})
        self.assertEqual({row["job_model"] for row in rows}, {"actual", "optimistic"})
        self.assertEqual(
            {row["carbon_information"] for row in rows}, {"none", "actual", "forecast"}
        )
        for row in rows:
            # A forecast model is named exactly when a forecast was read.
            self.assertEqual(
                bool(row["forecast_model"]), row["carbon_information"] == "forecast"
            )
            # A delay budget is recorded exactly when one was granted.
            granted = bool(row["max_delay_hours"] != "") or bool(row["max_delay_fraction"] != "")
            self.assertEqual(granted, row["delay_policy"] != "none")
            if row["delay_policy"] == "fixed":
                self.assertEqual(row["max_delay_hours"], DELAY_HOURS)
            if row["delay_policy"] == "duration-scaled":
                self.assertEqual(row["max_delay_fraction"], DELAY_FRACTION)

    def test_the_identity_columns_identify_a_row_uniquely(self) -> None:
        rows = matrix()
        keys = [tuple(row[field] for field in IDENTITY_FIELDS) for row in rows]
        self.assertEqual(len(set(keys)), len(keys))


class CsvOutputTest(unittest.TestCase):
    """Requirement 5: one CSV row per configuration, nothing dropped."""

    def test_the_csv_holds_one_row_per_configuration(self) -> None:
        rows = matrix()
        # 1 FCFS, plus per arm: EASY, power-capped EASY, and per delay family
        # the unbounded oracle, the bounded one and one run per archive, with
        # the power-capped carbon policy on the fixed family only.
        self.assertEqual(len(rows), 1 + 2 * (2 + (1 + 1 + 1 + 1) + (1 + 1 + 1)))

        with tempfile.TemporaryDirectory() as directory:
            destination = write_rows(rows, Path(directory) / "matrix.csv")
            with destination.open(newline="") as handle:
                reader = csv.DictReader(handle)
                written = list(reader)
                header = reader.fieldnames

        self.assertEqual(len(written), len(rows))
        for field in IDENTITY_FIELDS + REQUIRED_METRICS:
            self.assertIn(field, header)
        # The identity columns lead, so the table reads as keys then measures.
        self.assertEqual(tuple(header[: len(IDENTITY_FIELDS)]), IDENTITY_FIELDS)
        self.assertEqual(
            [row["scheduler_family"] for row in written],
            [row["scheduler_family"] for row in rows],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
