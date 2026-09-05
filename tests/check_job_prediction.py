"""Short checks for temporal job models and simulator prediction injection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_accounting import JobPowerProfile
from hpc_sim import (
    Cluster,
    Job,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
    Simulator,
)
from job_prediction import (
    SchedulingPrediction,
    attach_predictions,
    fit_job_predictor,
    regression_metrics,
    temporal_split,
)
from job_prediction.experiment import rolling_folds
from job_prediction.model import TargetConfig, fit_gradient_predictor
from job_prediction.features import (
    FeatureSpec,
    build_features,
    causal_group_stats,
    causal_inflight,
)
from job_prediction.scheduling_impact import duration_band_table, impact_summary


UTC = timezone.utc
BASE = datetime(2020, 1, 1, tzinfo=UTC)


def model_frame(count: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(count):
        submit_time = BASE + timedelta(hours=index)
        duration = float(10 + index % 7)
        power = float(100 + 10 * (index % 5))
        rows.append(
            {
                "job_id": index,
                "submit_time": submit_time,
                "end_time": submit_time + timedelta(minutes=1),
                "time_limit": 30 + index,
                "num_nodes_req": 1 + index % 3,
                "num_cores_req": 4 + index % 8,
                "num_tasks": None if index % 7 == 0 else 1 + index % 4,
                "cores_per_task": 1 + index % 2,
                "num_gpus_req": index % 2,
                "mem_req": 100 + index,
                "priority": 10 + index,
                "qos": str(index % 2),
                "duration_seconds": duration,
                "average_power_watts": power,
                "energy_kwh": duration * power / 3_600_000.0,
            }
        )
    return pd.DataFrame(rows[::-1])  # deliberately newest first


def history_frame() -> pd.DataFrame:
    """Three users, overlapping runs, and one job that ends far in the future."""

    rows = [
        # job, user, submit offset (h), duration (s)
        (0, "a", 0, 100.0),
        (1, "a", 1, 200.0),
        (2, "b", 2, 400.0),
        (3, "a", 3, 800.0),
        (4, "b", 4, 1_600.0),
        (5, "a", 4, 3_200.0),
    ]
    frame = pd.DataFrame(
        [
            {
                "job_id": job_id,
                "user_id": user,
                "submit_time": BASE + timedelta(hours=offset),
                # Every job finishes ten minutes after submission except job 3,
                # which is still running when jobs 4 and 5 are submitted.
                "end_time": BASE
                + timedelta(hours=offset)
                + (timedelta(days=30) if job_id == 3 else timedelta(minutes=10)),
                "duration_seconds": duration,
                "average_power_watts": 100.0 + job_id,
                "energy_kwh": duration * (100.0 + job_id) / 3_600_000.0,
                "time_limit": 60,
            }
            for job_id, user, offset, duration in rows
        ]
    )
    return frame


class AntiLeakageTest(unittest.TestCase):
    """Guards on the one place a feature is allowed to read a past outcome."""

    def test_history_excludes_self_running_and_future_jobs(self) -> None:
        frame = history_frame()
        stats = causal_group_stats(
            frame,
            keys=("user_id",),
            value_column="duration_seconds",
            prefix="user_duration",
        )
        counts = stats["user_duration_count"].to_numpy()

        # Job 0 is user a's first job: it has no history and must not see itself.
        self.assertEqual(counts[0], np.log1p(0))
        # Job 1 sees only job 0.
        self.assertEqual(counts[1], np.log1p(1))
        self.assertAlmostEqual(stats["user_duration_mean"].iloc[1], np.log1p(100.0))
        # Job 5 belongs to user a and is submitted while job 3 is still running,
        # so it may see jobs 0 and 1 only.
        self.assertEqual(counts[5], np.log1p(2))
        self.assertAlmostEqual(
            stats["user_duration_mean"].iloc[5],
            (np.log1p(100.0) + np.log1p(200.0)) / 2.0,
        )
        # Job 2 is user b's first job and never sees user a's finished jobs.
        self.assertEqual(counts[2], np.log1p(0))

    def test_inflight_counts_are_observable_at_submission(self) -> None:
        counts = causal_inflight(history_frame(), keys=("user_id",), prefix="user_queue")
        # When job 5 is submitted, user a has three earlier jobs and job 3 of
        # them is still running.
        self.assertAlmostEqual(counts["user_queue_submitted"].iloc[5], np.log1p(3))
        self.assertAlmostEqual(counts["user_queue_inflight"].iloc[5], np.log1p(1))

    def test_features_ignore_every_outcome_recorded_after_the_cutoff(self) -> None:
        """The decisive check: rewrite the future, and the past must not move."""

        spec = FeatureSpec(
            groups=("base", "derived", "user_hist", "signature_hist", "queue"),
        )
        frame = model_frame(60)
        frame["user_id"] = frame.job_id % 4
        frame["group_id"] = "g"
        frame["shared"] = "OK"
        frame["req_switch"] = 0
        frame["threads_per_core"] = None
        frame = frame.sort_values(["submit_time", "job_id"], ignore_index=True)

        cutoff = frame.submit_time.iloc[len(frame) // 2]
        corrupted = frame.copy()
        future = pd.to_datetime(corrupted.end_time, utc=True) > pd.Timestamp(cutoff)
        for column in ("duration_seconds", "average_power_watts", "energy_kwh"):
            corrupted.loc[future, column] = corrupted.loc[future, column] * 1_000.0 + 7.0

        observable = pd.to_datetime(frame.submit_time, utc=True) <= pd.Timestamp(cutoff)
        pd.testing.assert_frame_equal(
            build_features(frame, spec).loc[observable],
            build_features(corrupted, spec).loc[observable],
        )
        self.assertTrue(future.any() and observable.any())

    def test_rolling_folds_never_reach_past_their_cutoff(self) -> None:
        frame = model_frame(200).sort_values(["submit_time", "job_id"], ignore_index=True)
        folds = rolling_folds(frame, folds=3, validation_fraction=0.2)

        self.assertGreater(len(folds), 1)
        for fold in folds:
            submitted = pd.to_datetime(fold.train.submit_time, utc=True)
            finished = pd.to_datetime(fold.train.end_time, utc=True)
            # A fold may fit only on jobs both submitted and finished before it.
            self.assertTrue((submitted < fold.cutoff).all())
            self.assertTrue((finished <= fold.cutoff).all())
            self.assertTrue(
                (pd.to_datetime(fold.validation.submit_time, utc=True) >= fold.cutoff).all()
            )


def gradient_frame(count: int = 120) -> pd.DataFrame:
    frame = model_frame(count).sort_values(["submit_time", "job_id"], ignore_index=True)
    frame["user_id"] = frame.job_id % 5
    frame["group_id"] = "g"
    frame["shared"] = "OK"
    frame["req_switch"] = 0
    frame["threads_per_core"] = None
    return frame


class GradientPredictorTest(unittest.TestCase):
    def test_artifact_round_trips_and_stays_physically_consistent(self) -> None:
        import tempfile

        data = gradient_frame()
        split = temporal_split(data, train_fraction=0.6, validation_fraction=0.2)
        spec = FeatureSpec(groups=("base", "derived", "user_hist"))
        configs = {
            target: TargetConfig(overrides=(("max_iter", 10),))
            for target in ("duration_seconds", "average_power_watts", "energy_kwh")
        }
        training = fit_gradient_predictor(data, split, spec=spec, configs=configs)

        predictions = training.predictor.predict(data)
        values = predictions.drop(columns="job_id").to_numpy()
        self.assertTrue(np.all(np.isfinite(values)) and np.all(values > 0.0))
        implied = (
            predictions.predicted_average_power_watts
            * predictions.predicted_duration_seconds
            / 3_600_000.0
        )
        np.testing.assert_allclose(predictions.predicted_energy_kwh, implied)

        with tempfile.TemporaryDirectory() as directory:
            path = training.predictor.save(
                Path(directory) / "job_models.joblib", metadata={"source": "test"}
            )
            self.assertTrue(path.with_suffix(".json").exists())
            reloaded = type(training.predictor).load(path)
        pd.testing.assert_frame_equal(reloaded.predict(data), predictions)

    def test_fitting_never_uses_a_label_unavailable_at_its_cutoff(self) -> None:
        data = gradient_frame()
        # One early job finishes long after the test boundary, so its label was
        # not observable at any fit cutoff and must be dropped from all three.
        data.loc[0, "end_time"] = data.submit_time.max() + timedelta(days=365)
        split = temporal_split(data, train_fraction=0.6, validation_fraction=0.2)
        training = fit_gradient_predictor(
            data,
            split,
            spec=FeatureSpec(groups=("base",)),
            configs={
                target: TargetConfig(overrides=(("max_iter", 5),))
                for target in ("duration_seconds", "average_power_watts", "energy_kwh")
            },
        )
        self.assertEqual(training.fit_counts["train"], len(split.train) - 1)
        self.assertEqual(
            training.fit_counts["development"],
            len(split.train) + len(split.validation) - 1,
        )


class JobPredictionTest(unittest.TestCase):
    def test_scheduling_impact_summarises_job_level_changes(self) -> None:
        frame = pd.DataFrame(
            {
                "actual_duration_s": [2.0, 30.0, 3_600.0],
                "duration_absolute_relative_error": [100.0, 1.0, 0.1],
                "power_absolute_relative_error": [0.3, 0.2, 0.1],
                "target_start_delta_s": [900.0, 0.0, -900.0],
                "absolute_target_start_delta_s": [900.0, 0.0, 900.0],
                "simulated_start_delta_s": [1_800.0, 0.0, -900.0],
                "absolute_simulated_start_delta_s": [1_800.0, 0.0, 900.0],
            }
        )

        summary = impact_summary(frame)
        self.assertEqual(summary["changed_start_jobs"], 2)
        self.assertEqual(summary["absolute_start_delta_median_s"], 900.0)
        self.assertEqual(duration_band_table(frame).jobs.tolist(), [1, 1, 1])

    def test_temporal_split_sorts_and_keeps_timestamp_ties_together(self) -> None:
        timestamps = [2, 1, 4, 1, 3, 0]
        data = pd.DataFrame(
            {
                "job_id": range(len(timestamps)),
                "submit_time": [BASE + timedelta(hours=value) for value in timestamps],
            }
        )
        split = temporal_split(data, train_fraction=0.5, validation_fraction=1 / 6)

        self.assertLess(split.train.submit_time.max(), split.validation.submit_time.min())
        self.assertLess(split.validation.submit_time.max(), split.test.submit_time.min())
        tied = data.loc[data.submit_time == BASE + timedelta(hours=1), "job_id"]
        self.assertTrue(set(tied).issubset(set(split.train.job_id)))

    def test_predictions_are_positive_and_physically_consistent(self) -> None:
        data = model_frame()
        data.loc[data.job_id == 0, "end_time"] = BASE + timedelta(days=10)
        split = temporal_split(data, train_fraction=0.6, validation_fraction=0.2)
        training = fit_job_predictor(split, alphas=(0.01, 0.1))
        self.assertEqual(training.fit_counts["train"], len(split.train) - 1)
        predictor = training.predictor
        predictions = predictor.predict(split.test)

        values = predictions.drop(columns="job_id").to_numpy()
        self.assertTrue(np.all(np.isfinite(values)) and np.all(values > 0.0))
        implied = (
            predictions.predicted_average_power_watts
            * predictions.predicted_duration_seconds
            / 3_600_000.0
        )
        np.testing.assert_allclose(predictions.predicted_energy_kwh, implied)

    def test_predictions_drive_decisions_but_not_simulated_completion(self) -> None:
        jobs = [
            Job(
                job_id=job_id,
                submit_time=BASE,
                release_time=BASE,
                nodes_required=1,
                actual_duration_seconds=60.0,
                power=JobPowerProfile(
                    job_id=job_id,
                    duration_seconds=60.0,
                    average_power_watts=100.0,
                ),
            )
            for job_id in ("a", "b")
        ]
        prediction = SchedulingPrediction(
            10.0,
            800.0,
            800.0 * 10.0 / 3_600_000.0,
        )
        predicted = attach_predictions(jobs, {"a": prediction, "b": prediction})
        scheduler = PowerCappedEASYScheduler(
            1_000.0,
            runtime_estimate=RuntimeEstimateSource.SCHEDULING,
        )
        result = Simulator(predicted, Cluster(2), scheduler).run()
        records = {record.job_id: record for record in result.records}

        self.assertEqual(scheduler.first_reservations["b"], BASE + timedelta(seconds=10))
        self.assertEqual(records["b"].start_time, BASE + timedelta(seconds=60))
        self.assertTrue(all(record.runtime_seconds == 60.0 for record in records.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
