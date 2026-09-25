"""Predictive baseline leakage, cohort alignment, shared metrics, and CSV export."""

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_prediction_baselines as comparison
from check_job_prediction import BASE, model_frame
from carbon_intensity.baselines import BaselineCarbonIntensityProvider
from carbon_intensity.evaluate import error_metrics, evaluate_forecast
from carbon_intensity.protocol import TemporalProtocol
from carbon_intensity.series import (
    CarbonIntensitySample, FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider,
)
from carbon_intensity.snapshots import ForecastArchive
from job_prediction import temporal_split
from job_prediction.evaluation import baseline_estimates, job_metric_row, regression_metrics, runtime_band_metrics
from train_job_models import _metric_rows
from job_prediction.data import PredictionComposition


class PredictiveBaselineTest(unittest.TestCase):
    def test_train_only_medians_and_node_fallback(self):
        split = temporal_split(model_frame(10), train_fraction=0.5, validation_fraction=0.2)
        split.train["average_power_watts"] = [10.0, 30.0, 50.0, 70.0, 1e9]
        split.train["duration_seconds"] = [100.0, 200.0, 300.0, 500.0, 1e9]
        split.train["num_nodes_alloc"] = [1, 1, 2, 2, 99]
        split.train.loc[3, "end_time"] = split.train_until  # Available exactly at cutoff.
        split.train.loc[4, "end_time"] = split.validation_until
        split.test["num_nodes_alloc"] = [1, 99, np.nan]
        split.test["time_limit"] = [0.5, 2.0, 10.0]
        estimates = baseline_estimates(split, runtime_by_nodes=True, power_by_nodes=True)
        predictions = {name: values for name, _, _, values in estimates}
        np.testing.assert_array_equal(predictions["time_limit * 60"], [30.0, 120.0, 600.0])
        np.testing.assert_array_equal(predictions["training median runtime"], [250.0] * 3)
        np.testing.assert_array_equal(predictions["training median runtime by nodes"], [150.0, 250.0, 250.0])
        np.testing.assert_array_equal(predictions["training median power"], [40.0] * 3)
        np.testing.assert_array_equal(predictions["training median power by nodes"], [20.0, 40.0, 40.0])
        # Corrupt all unavailable outcomes: no fitted value may change.
        split.train.loc[4, "average_power_watts"] = np.nan
        split.train.loc[4, "duration_seconds"] = np.nan
        split.validation["average_power_watts"] = -1e12
        split.validation["duration_seconds"] = -1e12
        split.test["average_power_watts"] = np.nan
        split.test["duration_seconds"] = np.nan
        for before, after in zip(estimates, baseline_estimates(
            split, runtime_by_nodes=True, power_by_nodes=True,
        ), strict=True):
            np.testing.assert_array_equal(before[3], after[3])
        for invalid in (0.0, -1.0, np.nan, np.inf):
            split.test.loc[0, "time_limit"] = invalid
            with self.assertRaises(ValueError):
                baseline_estimates(split)

    def test_bias_and_actual_runtime_band_boundaries(self):
        frame = pd.DataFrame({"duration_seconds": [10, 30], "average_power_watts": [10, 30]})
        for target, unit in (("duration_seconds", "s"), ("average_power_watts", "W")):
            row = job_metric_row(frame, "test", target, unit, [20, 10])
            self.assertEqual(row["bias"], -5)
            self.assertEqual(row["relative_bias"], -0.25)
            self.assertEqual(row["weighted_absolute_relative_error"], 0.75)
            self.assertEqual(row["mae"], 15)
            self.assertAlmostEqual(row["rmse"], np.sqrt(250))
        actual = np.array([1, 299, 300, 1_799, 1_800, 3_599, 3_600, 14_400, 14_401])
        bands = runtime_band_metrics(actual, actual / 2)
        self.assertEqual(bands["count"].tolist(), [2, 2, 2, 2, 1])
        np.testing.assert_allclose(bands.relative_bias, -0.5)
        np.testing.assert_allclose(bands.weighted_absolute_relative_error, 0.5)
        np.testing.assert_allclose(bands.bias, -bands.actual_runtime_mean_s / 2)
        self.assertEqual(bands.actual_runtime_median_s.iloc[3], 9_000)
        sparse = runtime_band_metrics([10], [20])
        self.assertEqual(sparse["count"].tolist(), [1, 0, 0, 0, 0])
        self.assertTrue(sparse.mae.iloc[1:].isna().all())

    def test_forecast_bias_and_horizon_bands(self):
        protocol = TemporalProtocol(BASE + 16 * STEP, BASE + 112 * STEP, BASE, BASE + 8 * STEP,
                                    observation_delay=STEP)
        actual = TimeSeriesCarbonIntensityProvider([
            CarbonIntensitySample(BASE + i * STEP, 100 + i) for i in range(112)
        ])
        result = evaluate_forecast(actual, protocol, BaselineCarbonIntensityProvider(actual, protocol).get_forecast,
                                   partition="test", horizon=96 * STEP)
        self.assertEqual(result["count"], 96)
        bands = result["by_horizon_band"]
        self.assertEqual([row["count"] for row in bands], [4, 8, 12, 24, 48])
        self.assertEqual(bands[0]["bias_gco2e_per_kwh"], -3.5)
        self.assertEqual(bands[1]["bias_gco2e_per_kwh"], -9.5)
        self.assertAlmostEqual(result["relative_bias"], sum(range(-97, -1)) / sum(range(116, 212)))
        self.assertAlmostEqual(result["weighted_absolute_relative_error"], -result["relative_bias"])
        self.assertIsNone(error_metrics([2, 10, 100, -10, 0])["relative_bias"])
        self.assertIsNone(error_metrics([2, 10, 100, -10, 0])["weighted_absolute_relative_error"])

    def test_comparison_exports_and_rejects_mismatched_inputs(self):
        data = model_frame()
        data["num_nodes_alloc"] = 1
        split = temporal_split(data)
        expected_split = {
            "train_count": len(split.train), "validation_count": len(split.validation),
            "test_count": len(split.test), "validation_start": split.train_until.isoformat(),
            "test_start": split.validation_until.isoformat(),
        }
        predictions = split.test[["job_id", "duration_seconds", "average_power_watts", "energy_kwh"]].rename(
            columns={key: f"predicted_{key}" for key in ("duration_seconds", "average_power_watts", "energy_kwh")}
        ).iloc[::-1]
        offset = 7 * 96
        protocol = TemporalProtocol(BASE + (offset + 16) * STEP, BASE + (offset + 24) * STEP,
                                    BASE, BASE + (offset + 8) * STEP,
                                    observation_delay=STEP)
        actual = TimeSeriesCarbonIntensityProvider([
            CarbonIntensitySample(BASE + index * STEP, 100.0 + index) for index in range(offset + 32)
        ])
        persistence = BaselineCarbonIntensityProvider(actual, protocol)
        issue = protocol.test_start + STEP / 2
        self.assertEqual(
            {sample.intensity_gco2e_per_kwh for sample in persistence.get_forecast(issue, 8 * STEP).samples},
            {114.0 + offset},
        )
        past = TimeSeriesCarbonIntensityProvider(actual.samples[:offset + 15])
        self.assertEqual(BaselineCarbonIntensityProvider(past, protocol).get_forecast(issue, 8 * STEP),
                         persistence.get_forecast(issue, 8 * STEP))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model_path = directory / "job_models.json"
            model_path.write_text(json.dumps({"metadata": {"split": expected_split}}))
            prediction_path = directory / "test_predictions.parquet"
            predictions.to_parquet(prediction_path, index=False)
            actual_path = actual.save(directory / "actual.json")
            actual_hash = sha256(actual_path.read_bytes()).hexdigest()
            protocol_path = directory / "protocol.json"
            protocol_path.write_text(json.dumps({**protocol.metadata(), "actual_sha256": actual_hash}))
            snapshots = directory / "snapshots"
            ForecastArchive.generate(
                "existing_model", persistence.get_forecast, protocol, horizon=4 * STEP, cadence=STEP,
                metadata={**protocol.metadata(), "actual_sha256": actual_hash, "training_cutoff": None},
            ).save(snapshots / "test_existing_model.json")
            output = directory / "output"
            argv = ["compare_prediction_baselines", "--job-run", f"existing={directory}",
                    "--actual", str(actual_path), "--protocol", str(protocol_path),
                    "--snapshots", str(snapshots), "--horizon-hours", "1", "--cadence-minutes", "15",
                    "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(comparison, "load_job_data", return_value=data):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(comparison.main(), 0)
            self.assertEqual(len(list(output.glob("*.csv"))), 5)
            for target in ("runtime", "power"):
                table = pd.read_csv(output / f"test_{target}_comparison.csv")
                self.assertEqual(set(table["count"]), {len(split.test)})
                self.assertEqual(table.loc[table.model == "existing", "mae"].item(), 0.0)
                self.assertEqual(set(table.long_job_count), {0})
            runtime = pd.read_csv(output / "test_runtime_comparison.csv").iloc[0]
            expected = regression_metrics(split.test.duration_seconds, split.test.time_limit * 60)
            self.assertEqual(runtime.mae, expected.mae)
            carbon = pd.read_csv(output / "test_carbon_intensity_comparison.csv")
            self.assertEqual(set(carbon.model), {"persistence", "seasonal_daily", "seasonal_weekly", "existing_model"})
            self.assertEqual(set(carbon.forecasts), {5})
            self.assertEqual(set(carbon.predicted_buckets), {20})
            self.assertEqual(carbon.loc[carbon.model == "persistence", "mae_gco2e_per_kwh"].item(), 3.5)
            bands = pd.read_csv(output / "test_runtime_band_comparison.csv")
            self.assertEqual(bands.groupby("model")["count"].sum().tolist(), [len(split.test)] * 4)
            horizons = pd.read_csv(output / "test_carbon_intensity_horizon_comparison.csv")
            self.assertEqual(set(horizons.horizon_band), {"0–1 h"})
            self.assertEqual(set(horizons["count"]), {20})
            metadata = json.loads((output / "comparison_metadata.json").read_text())
            self.assertEqual(metadata["power_baseline"]["fit_count"], len(split.train))
            components = {target: split.test[target].to_numpy() for target in
                          ("duration_seconds", "average_power_watts", "energy_kwh")}
            rows = _metric_rows(split.test, components, predictions.iloc[::-1],
                                PredictionComposition.DURATION_POWER, baseline_estimates(split))
            self.assertEqual(next(row["mae"] for row in rows if row["estimate"] == "time_limit * 60"), expected.mae)
            predictions.iloc[:-1].to_parquet(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly.*test job IDs"):
                comparison.job_comparison(split, {"existing": directory})
            model_path.write_text(json.dumps({"metadata": {"split": {}}}))
            with self.assertRaisesRegex(ValueError, "different temporal split"):
                comparison.job_comparison(split, {"existing": directory})
            for rules, digest in ((replace(protocol, observation_delay=timedelta(0)), actual_hash),
                                  (protocol, "wrong cache")):
                with self.assertRaises(ValueError):
                    comparison.carbon_comparison(actual, rules, digest, snapshots,
                                                 horizon=4 * STEP, cadence=STEP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
