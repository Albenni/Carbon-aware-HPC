"""Small dataset-free check of robustness selection, cache keys and paired tails."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from carbon_accounting import JobPowerProfile
from carbon_intensity.series import CarbonIntensitySample, TimeSeriesCarbonIntensityProvider
from experiments import improved_scheduler as ex
from experiments import robustness as r


def main():
    start = datetime(2020, 5, 6, tzinfo=timezone.utc)
    jobs = tuple(ex.Job(str(i), start + timedelta(days=i), start + timedelta(days=i), 1,
                        3600., JobPowerProfile(duration_seconds=3600., average_power_watts=1000.))
                 for i in range(8))
    with patch.object(ex, "cohort", return_value=(jobs, jobs)):
        windows = r.windows(jobs)
        ids = [ex.subset_ids(window)[0] for window in windows]
        assert ids == [["0", "1"], ["2", "3"], ["4", "5"], ["6", "7"]]
        assert ex.subset_ids(replace(windows[0], subset=1))[0] == ["0"]
        assert len(r.matrix(windows)) == 222
    try:
        ex.Scenario("test", window_start=start.isoformat())
    except ValueError:
        pass
    else:
        raise AssertionError("one-sided release window accepted")

    # Old keys are stable; windows and decision grids cannot collide with them.
    item = ex.task(ex.Scenario("test"), r.qc.IMPROVED)
    with patch.object(ex, "input_hashes", return_value={}):
        identity = {"scenario": {"stage": "test", "subset": None, "max_nodes": None, "contention": False, "nodes": 880},
                    "job_information": "predicted", "carbon_information": "forecast", "policy": asdict(r.qc.IMPROVED), "inputs": {}}
        expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        assert ex.run_directory(item)[0] == expected
        assert len({ex.run_directory((replace(item[0], grid_minutes=grid), *item[1:]))[0]
                    for grid in (15, 60, 240)}) == 3

    frame = pd.DataFrame({"job_id": [str(i) for i in range(100)], "waiting_s": [0.] * 98 + [100., 1000.],
                          "actual_duration_s": [3599., 3600.] * 50, "nodes": [8, 9] * 50,
                          "bounded_slowdown": [1.] * 98 + [2., 11.], "energy_kwh": 1.,
                          "emissions_gco2e": 90., "start_time": [start] * 100})
    easy = frame.assign(waiting_s=0., emissions_gco2e=100.)
    decisions = pd.DataFrame(columns=["job_id", "decision_time", "target"])
    paired = r.paired_jobs(frame, easy, decisions)
    metrics = r.describe(paired)
    assert (metrics["waiting_p95_s"], metrics["waiting_p99_s"], metrics["waiting_max_s"]) == (0., 100., 1000.)
    assert metrics["emissions_saved_pct"] == 10.
    assert np.isclose(metrics["top_1pct_positive_extra_wait_share"], 1000 / 1100)
    groups = pd.DataFrame(r.classes(paired))
    for _, members in groups.groupby("dimension"):
        assert members.jobs.sum() == 100 and np.isclose(members.emissions_saved_kg.sum(), 1.)
    assert groups[groups.dimension == "duration"].set_index("class").jobs.to_dict() == {"long": 50, "short": 50}
    assert np.isnan(r.describe(r.paired_jobs(easy, easy, decisions))["top_1pct_positive_extra_wait_share"])
    try:
        r.paired_jobs(frame, easy.iloc[:-1], decisions)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete job pairing accepted")

    # A real synthetic simulation checks the grid reaches the scheduler and
    # windows reach the cohort selector, without touching experiment artifacts.
    step = timedelta(minutes=15)
    provider = TimeSeriesCarbonIntensityProvider(tuple(
        CarbonIntensitySample(start + n * step, 50. if 1 <= n < 5 else 400.) for n in range(192)))
    with tempfile.TemporaryDirectory() as directory, patch.object(ex, "OUTPUT", Path(directory)), \
            patch.object(ex, "cohort", return_value=(jobs[:1], jobs[:1])), \
            patch.object(ex, "input_hashes", return_value={}), \
            patch.object(ex, "carbon", return_value=(provider, provider, timedelta(hours=23))):
        starts = []
        for grid in (15, 240):
            scenario = replace(windows[0], grid_minutes=grid, nodes=1)
            row = ex.execute(ex.task(scenario, ex.GREEDY, ex.ORACLE))
            assert row["decision_granularity_minutes"] == grid and row["jobs"] == 1
            starts.append(pd.read_parquet(ex.OUTPUT / "runs" / row["run_key"] / "jobs.parquet").start_time.iloc[0])
        assert starts == [start + step, start]
    assert r.ABLATIONS["without_benefit_rate"].threshold == r.qc.IMPROVED.threshold
    assert r.ABLATIONS["with_congestion"].objective != "benefit-rate"
    print("Robustness checks passed.")


class RobustnessChecks(unittest.TestCase):
    def test_checks_pass(self):
        main()


if __name__ == "__main__":
    main()
