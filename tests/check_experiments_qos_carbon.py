"""Dataset-free regression check: python tests/check_experiments_qos_carbon.py."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from carbon_accounting import JobPowerProfile
from carbon_intensity.series import CarbonIntensitySample, TimeSeriesCarbonIntensityProvider
from experiments import improved_scheduler as ex
from experiments.milp import planning_emissions, replay, solve
from experiments.qos_carbon import marginal_row, pareto_mask, ratio, summary_table, comparisons, frontier_table
from hpc_sim import CarbonSignal, Job, account_schedule


def main():
    # A QoS projection is not the five-dimensional selection frontier. A point
    # dominated in mean waiting may still be efficient for the p95 slowdown.
    points = pd.DataFrame({"carbon": [10, 12, 13, 10], "wait": [8, 2, 9, 8], "p95": [4, 8, 1, 4]})
    assert pareto_mask(points, ("carbon", "wait")) == [True, True, False, True]
    assert pareto_mask(points, ("carbon", "p95")) == [True, False, True, True]
    for denominator in (0., -1., float("nan")):
        value, reason = ratio(2., denominator)
        assert math.isnan(value) and reason
    assert ratio(-1., 2.)[0] == -.5 and ratio(3., 2.)[0] == 1.5
    before = {"total_emissions_gco2e": 5000., "waiting_mean_s": 20., "bounded_slowdown_p95": 2., "jobs": 10}
    after = before | {"total_emissions_gco2e": 6000., "waiting_mean_s": 30.}
    cost = marginal_row(before, after)
    assert cost["extra_carbon_saved_kg"] == -1 and math.isnan(cost["mean_wait_s_per_extra_kg"])
    assert marginal_row(before, after | {"total_emissions_gco2e": 4000.})["mean_wait_s_per_extra_kg"] == 10

    # Deliberately distinct EASY totals: retention and loss shares must keep a
    # fixed reference, whereas operational saving uses the matching runtime.
    rows = []
    for information, carbon in (("actual", "none"), ("predicted", "none"),
                                ("actual", "actual"), ("actual", "forecast"),
                                ("predicted", "actual"), ("predicted", "forecast")):
        total = {("actual", "none"): 100., ("predicted", "none"): 105.,
                 ("actual", "actual"): 80., ("actual", "forecast"): 85.,
                 ("predicted", "actual"): 90., ("predicted", "forecast"): 92.}[information, carbon]
        modes = ("easy",) if carbon == "none" else ("once", "receding")
        for mode in modes:
            row = {"run_key": f"{information}-{carbon}-{mode}", "policy_id": "easy" if mode == "easy" else mode,
                   "policy_mode": mode, "policy_delay_hours": 6., "job_information": information,
                   "carbon_information": carbon, "total_emissions_gco2e": total, "jobs": 10}
            row |= {metric: 1. for metric in ex.FRONTIER if metric != "total_emissions_gco2e"}
            rows.append(row)
    summary = summary_table(pd.DataFrame(rows))
    realistic = summary[(summary.policy == "improved") & (summary.information_scenario == "realistic")].iloc[0]
    assert np.isclose(realistic.carbon_benefit_retained, .4)
    assert np.isclose(realistic.emissions_saved_pct, 100 * 13 / 105)
    assert np.isclose(realistic.emissions_saved_common_pct, 8.)
    _, _, losses = comparisons(summary)
    front = frontier_table(summary)
    assert front.pareto_joint.dtype == bool and front.pareto_mean_wait.any()
    for loss in losses.itertuples():
        assert np.isclose(loss.forecast_loss_share + loss.job_model_loss_share + loss.interaction_loss_share, .6)

    # Runtime-only and power-only controls must retain the same physical job.
    start = datetime(2020, 5, 6, tzinfo=timezone.utc)
    job = Job("j", start, start, 1, 1800., JobPowerProfile(duration_seconds=1800., average_power_watts=1000., job_id="j"))
    predicted = replace(job, predicted_duration_seconds=900., predicted_average_power_watts=2000.)
    with patch.object(ex, "cohort", return_value=((job,), (predicted,))):
        runtime = ex.scenario_jobs(ex.Scenario("test"), "runtime-only")[0][0]
        power = ex.scenario_jobs(ex.Scenario("test"), "power-only")[0][0]
        assert (runtime.scheduling_duration_seconds, runtime.scheduling_average_power_watts) == (900., 1000.)
        assert (power.scheduling_duration_seconds, power.scheduling_average_power_watts) == (1800., 2000.)
        assert runtime.power is job.power and power.actual_duration_seconds == job.actual_duration_seconds

    # Off-grid CI boundary: accounting samples the first 20 seconds at 400,
    # but the solver integrates only five dirty seconds. Never certify a gap
    # with the sampled accounting value instead of the solver's objective.
    step = timedelta(minutes=15)
    provider = TimeSeriesCarbonIntensityProvider(tuple(
        CarbonIntensitySample(start + n * step, 400. if n < 4 else 100.) for n in range(96)))
    signal = CarbonSignal(provider)
    release = start + timedelta(seconds=3595)
    shifted = replace(job, submit_time=release, release_time=release)
    starts, info = solve((shifted,), signal, nodes=1, max_delay=timedelta(0), granularity=step, delay_weight=0)
    exact = planning_emissions((shifted,), starts, signal)
    recorded = account_schedule(replay((shifted,), starts, 1), (shifted,), provider).records[0]
    assert np.isclose(exact, info["objective_gco2e"], rtol=0, atol=1e-9)
    assert abs(exact - recorded.emissions_gco2e_average_model) > 1.
    # A shared reach can remove otherwise cleaner starts from the MILP domain.
    starts, _ = solve((job,), signal, nodes=1, max_delay=timedelta(hours=3), granularity=step,
                      delay_weight=0, reach=timedelta(minutes=30))
    assert starts["j"] == start
    print("QoS/carbon consolidation checks passed.")


class ConsolidationChecks(unittest.TestCase):
    def test_checks_pass(self):
        main()


if __name__ == "__main__":
    main()
