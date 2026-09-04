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
    temporal_split,
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
