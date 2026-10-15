"""Consolidated QoS/carbon experiments: ``python -m experiments.qos_carbon run|report``.

Reuse the frozen improved-scheduler inputs, policies and content-addressed runs.
Only missing information cells and runtime/power controls are simulated. Reports
read the aggregate CSVs, so regenerating a figure never launches a simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments import improved_scheduler as ex
from experiments.milp import planning_emissions, replay, solve
from hpc_sim import CarbonSignal, account_schedule, carbon_cost_gco2e, schedule_metrics
from hpc_sim.metrics import Distribution


OUTPUT = ex.PROJECT_ROOT / "results/qos_carbon"
CACHE = ex.PROJECT_ROOT / "data/experiments/qos_carbon"
# Already selected on validation; do not choose a new threshold on this test.
IMPROVED = ex.Policy("receding", objective="benefit-rate", threshold=0.014930658589279368)
POLICIES = {"greedy": ex.GREEDY, "improved": IMPROVED}
CELLS = {
    ("actual", "actual"): "perfect_information",
    ("actual", "forecast"): "forecast_only_error",
    ("predicted", "actual"): "job_model_only_error",
    ("predicted", "forecast"): "realistic",
}
FACTORS = {"actual": (False, False), "runtime-only": (True, False),
           "power-only": (False, True), "predicted": (True, True)}
QOS = ("waiting_mean_s", "waiting_p95_s", "waiting_max_s", "turnaround_mean_s",
       "bounded_slowdown_mean", "bounded_slowdown_p95", "utilisation", "peak_power_mw")


def relative(path: Path) -> str:
    return str(path.relative_to(ex.PROJECT_ROOT))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def preflight() -> dict:
    """Refuse stale inputs before consulting the historical simulation cache."""
    frozen = ex.manifest()
    inputs = ex.INPUTS | {"validation_predictions": ex.OUTPUT / "validation_predictions.parquet"}
    hashes = {name: ex.sha256(path) for name, path in inputs.items()}
    if hashes != frozen["input_sha256"]:
        raise ValueError("inputs differ from improved_scheduler/manifest.json; prepare a new experiment")
    configs, selection = ex.selected()
    if IMPROVED not in configs:
        raise ValueError("the improved policy is not in the frozen validation selection")
    sources = sorted(path for path in (ex.PROJECT_ROOT / "src").rglob("*.py"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m experiments.qos_carbon run",
        "input_sha256": hashes,
        "source_sha256": {relative(path): ex.sha256(path) for path in sources},
        "selection_sha256": ex.sha256(ex.OUTPUT / "selection.json"),
        "selection_frozen": selection["frozen"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ex.PROJECT_ROOT, text=True).strip(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "matplotlib", "pyarrow")},
        "cohort": {"stage": "test", "jobs": frozen["cohorts"]["test"]["jobs"], "nodes": ex.NODES},
        "delay_hours": list(ex.DELAY_SWEEP), "grid_minutes": 15,
        "policies": {name: asdict(policy) for name, policy in POLICIES.items()},
        "information_cells": list(CELLS.values()),
        "retention_reference": "same policy/budget perfect-information saving, fixed actual-job EASY",
        "saving_reference": "EASY with the same runtime information; common-EASY saving also exported",
        "qos": "from release; BSLD=max(1,(wait+runtime)/max(runtime,10s)); nearest-rank percentiles",
        "evaluation": "measured power profiles and actual CI; peak power is a job-mean proxy",
        "scope": "retrospective test cohort; no training, retuning or fresh holdout claim",
    }


def schedules(row) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = ex.OUTPUT / "runs" / row["run_key"]
    return pd.read_parquet(directory / "jobs.parquet"), pd.read_parquet(directory / "decisions.parquet")


def augment(row: dict) -> dict:
    """Validate reused schedules and add the tail metrics missing in old CSVs."""
    frame, decisions = schedules(row)
    ids = frame["job_id"]
    if ids.duplicated().any() or len(frame) != row["jobs"]:
        raise ValueError(f"invalid job set in {row['run_key']}")
    runtime = (frame.end_time - frame.start_time).dt.total_seconds()
    wait = (frame.start_time - frame.release_time).dt.total_seconds()
    assert (wait >= 0).all() and np.allclose(runtime, frame.actual_duration_s, rtol=0, atol=1e-6)
    assert np.allclose(wait, frame.waiting_s, rtol=0, atol=1e-6)
    assert np.isclose(frame.emissions_gco2e.sum(), row["total_emissions_gco2e"], rtol=1e-10)
    # A cached result is still checked for physical feasibility, not just totals.
    events = sorted([(t, 1, n) for t, n in zip(frame.start_time, frame.nodes)]
                    + [(t, 0, -n) for t, n in zip(frame.end_time, frame.nodes)])
    occupancy = np.cumsum([event[2] for event in events])
    assert occupancy.min() >= 0 and occupancy.max() <= row["nodes"] and occupancy[-1] == 0
    if len(decisions):
        assert (decisions.target <= decisions.deadline).all()
        issued = decisions.dropna(subset=["forecast_issue_time"])
        assert (issued.forecast_issue_time <= issued.decision_time).all()
    extra = {}
    for name, values in (("turnaround", wait + runtime), ("bounded_slowdown", frame.bounded_slowdown)):
        distribution = Distribution.of(values.tolist())
        extra |= {f"{name}_p95": distribution.p95, f"{name}_p99": distribution.p99,
                  f"{name}_max" + ("_s" if name == "turnaround" else ""): distribution.maximum}
    # Keep seconds in every turnaround column name.
    extra["turnaround_p95_s"] = extra.pop("turnaround_p95")
    extra["turnaround_p99_s"] = extra.pop("turnaround_p99")
    held = ex.voluntary_delay(decisions, frame)
    return row | extra | {
        "voluntary_delay_hours": float(held.sum() / 3600),
        "held_share": float((held > 0).mean()),
        "source_run": relative(ex.OUTPUT / "runs" / row["run_key"]),
    }


def ratio(numerator: float, denominator: float) -> tuple[float, str]:
    if not np.isfinite(denominator) or denominator <= 1e-9:
        return float("nan"), "nonpositive or zero reference benefit"
    return numerator / denominator, ""


def saving_columns(row, easy, common) -> dict:
    saved = easy["total_emissions_gco2e"] - row["total_emissions_gco2e"]
    common_saved = common["total_emissions_gco2e"] - row["total_emissions_gco2e"]
    return {
        "easy_run_key": easy["run_key"], "common_easy_run_key": common["run_key"],
        "emissions_saved_kg": saved / 1000,
        "emissions_saved_pct": 100 * saved / easy["total_emissions_gco2e"],
        "emissions_saved_common_kg": common_saved / 1000,
        "emissions_saved_common_pct": 100 * common_saved / common["total_emissions_gco2e"],
        "added_mean_wait_s": row["waiting_mean_s"] - easy["waiting_mean_s"],
        "added_job_wait_hours": (row["waiting_mean_s"] - easy["waiting_mean_s"]) * row["jobs"] / 3600,
    }


def summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    easy = raw[raw.policy_id == "easy"].set_index("job_information")
    common = easy.loc["actual"]
    rows = []
    for _, row in raw[raw.policy_id != "easy"].iterrows():
        reference = easy.loc["predicted" if FACTORS[row.job_information][0] else "actual"]
        cell = (row.job_information, row.carbon_information)
        policy = "greedy" if row.policy_mode == "once" else "improved"
        rows.append(row.to_dict() | saving_columns(row, reference, common) | {
            "policy": policy, "delay_hours": row.policy_delay_hours,
            "information_scenario": CELLS.get(cell, f"{cell[0]}_{cell[1]}"),
            "runtime_predicted": FACTORS[cell[0]][0], "power_predicted": FACTORS[cell[0]][1],
            "carbon_forecast": cell[1] == "forecast",
        })
    frame = pd.DataFrame(rows)
    for _, indexes in frame.groupby(["policy", "delay_hours"]).groups.items():
        group = frame.loc[indexes]
        perfect = group[group.information_scenario == "perfect_information"].iloc[0]
        for index in indexes:
            retained, reason = ratio(frame.loc[index, "emissions_saved_common_kg"], perfect.emissions_saved_common_kg)
            frame.loc[index, "carbon_benefit_retained"] = retained
            frame.loc[index, "carbon_benefit_retained_reason"] = reason
    for (information, carbon), scenario in CELLS.items():
        reference = easy.loc[information]
        for hours in ex.DELAY_SWEEP:
            rows.append(reference.to_dict() | saving_columns(reference, reference, common) | {
                "policy": "EASY", "delay_hours": hours, "information_scenario": scenario,
                "carbon_benefit_retained": np.nan,
                "carbon_benefit_retained_reason": "carbon-blind reference; no policy benefit denominator",
            })
    baselines = pd.DataFrame([row for row in rows if row["policy"] == "EASY"])
    return pd.concat([frame, baselines], ignore_index=True).sort_values(
        ["information_scenario", "delay_hours", "policy"], kind="stable").reset_index(drop=True)


def pareto_mask(frame: pd.DataFrame, metrics) -> list[bool]:
    points = frame[list(metrics)].to_numpy(float)
    if not np.isfinite(points).all():
        raise ValueError("Pareto metrics must be finite")
    return [not any(ex.dominates(other, point) for other in points) for point in points]


def frontier_table(summary: pd.DataFrame) -> pd.DataFrame:
    points = summary[summary.information_scenario.isin(CELLS.values())].drop_duplicates(
        ["information_scenario", "run_key"]).copy()
    for flag in ("pareto_mean_wait", "pareto_p95_bsld", "pareto_joint"):
        points[flag] = False
    # All points in a cell use the same EASY denominator, so minimising carbon
    # is exactly equivalent to maximising the reported saving.
    for _, group in points.groupby("information_scenario"):
        for name, metrics in {
            "pareto_mean_wait": ("total_emissions_gco2e", "waiting_mean_s"),
            "pareto_p95_bsld": ("total_emissions_gco2e", "bounded_slowdown_p95"),
            "pareto_joint": ("total_emissions_gco2e", "waiting_mean_s", "bounded_slowdown_p95"),
        }.items():
            points.loc[group.index, name] = pareto_mask(group, metrics)
    return points


def marginal_row(before, after) -> dict:
    gain = (before["total_emissions_gco2e"] - after["total_emissions_gco2e"]) / 1000
    wait = after["waiting_mean_s"] - before["waiting_mean_s"]
    bsld = after["bounded_slowdown_p95"] - before["bounded_slowdown_p95"]
    rate, reason = ratio(wait, gain)
    return {
        "extra_carbon_saved_kg": gain, "delta_mean_wait_s": wait, "delta_p95_bsld": bsld,
        "mean_wait_s_per_extra_kg": rate,
        "job_wait_hours_per_extra_kg": ratio(wait * after["jobs"] / 3600, gain)[0],
        "p95_bsld_per_extra_kg": ratio(bsld, gain)[0],
        "marginal_cost_reason": reason,
    }


def comparisons(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = summary[(summary.policy != "EASY") & summary.information_scenario.isin(CELLS.values())]
    paired, marginal, losses = [], [], []
    for (scenario, hours), group in core.groupby(["information_scenario", "delay_hours"]):
        group = group.set_index("policy")
        greedy, improved = group.loc["greedy"], group.loc["improved"]
        paired.append({
            "information_scenario": scenario, "delay_hours": hours,
            **marginal_row(greedy, improved),
            "mean_wait_reduction_pct": 100 * ratio(greedy.waiting_mean_s - improved.waiting_mean_s, greedy.waiting_mean_s)[0],
            "p95_bsld_reduction_pct": 100 * ratio(greedy.bounded_slowdown_p95 - improved.bounded_slowdown_p95, greedy.bounded_slowdown_p95)[0],
            "saving_ratio_improved_over_greedy": ratio(improved.emissions_saved_kg, greedy.emissions_saved_kg)[0],
            "saving_ratio_reason": ratio(improved.emissions_saved_kg, greedy.emissions_saved_kg)[1],
            "improved_dominates_greedy": ex.dominates(ex.vector(improved), ex.vector(greedy)),
        })
    for (scenario, policy), group in core.groupby(["information_scenario", "policy"]):
        ordered = group.sort_values("delay_hours").to_dict("records")
        for before, after in zip(ordered, ordered[1:]):
            marginal.append({"information_scenario": scenario, "policy": policy,
                             "from_delay_hours": before["delay_hours"], "to_delay_hours": after["delay_hours"],
                             **marginal_row(before, after)})
    for (policy, hours), group in core.groupby(["policy", "delay_hours"]):
        cells = group.set_index("information_scenario")
        perfect = cells.loc["perfect_information"]
        f, j, both = (cells.loc[name].total_emissions_gco2e - perfect.total_emissions_gco2e
                      for name in ("forecast_only_error", "job_model_only_error", "realistic"))
        row = {"policy": policy, "delay_hours": hours}
        for name, loss in (("forecast", f), ("job_model", j), ("interaction", both - f - j), ("combined", both)):
            row[f"{name}_loss_kg"] = loss / 1000
            row[f"{name}_loss_share"], row["loss_share_reason"] = ratio(loss / 1000, perfect.emissions_saved_common_kg)
        losses.append(row)
    return pd.DataFrame(paired), pd.DataFrame(marginal), pd.DataFrame(losses)


def factor_effects(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full 2x2x2 at 6 h isolates runtime, power and CI errors and interactions."""
    rows, bands = [], []
    actual, predicted = ex.cohort("test")
    predictions = pd.DataFrame({
        "job_id": [job.job_id for job in actual],
        "duration_s": [job.actual_duration_seconds for job in actual],
        "runtime_ratio": [p.scheduling_duration_seconds / a.actual_duration_seconds
                          for a, p in zip(actual, predicted)],
        "runtime_error_s": [p.scheduling_duration_seconds - a.actual_duration_seconds
                            for a, p in zip(actual, predicted)],
    }).set_index("job_id")
    dimensions = ("runtime_predicted", "power_predicted", "carbon_forecast")
    for policy, group in summary[(summary.delay_hours == 6) & (summary.policy != "EASY")].groupby("policy"):
        cells = {tuple(bool(row[name]) for name in dimensions): row for _, row in group.iterrows()}
        for state, before in cells.items():
            for axis, factor in enumerate(("runtime", "power", "carbon_forecast")):
                if state[axis]:
                    continue
                changed = list(state)
                changed[axis] = True
                after = cells[tuple(changed)]
                a, da = schedules(before)
                b, db = schedules(after)
                joined = a.set_index("job_id").join(b.set_index("job_id"), rsuffix="_after", validate="one_to_one")
                if joined.start_time_after.isna().any() or len(joined) != len(b):
                    raise ValueError("factor controls must contain identical jobs")
                joined["start_delta_s"] = (joined.start_time_after - joined.start_time).dt.total_seconds()
                first_a = da.drop_duplicates("job_id").set_index("job_id").target
                first_b = db.drop_duplicates("job_id").set_index("job_id").target
                joined["target_changed"] = first_a.reindex(joined.index) != first_b.reindex(joined.index)
                context = {"policy": policy, "factor": factor, "from_scenario": before.information_scenario,
                           "to_scenario": after.information_scenario}
                rows.append(context | {
                    "emissions_increase_kg": (after.total_emissions_gco2e - before.total_emissions_gco2e) / 1000,
                    "mean_wait_delta_s": after.waiting_mean_s - before.waiting_mean_s,
                    "p95_wait_delta_s": after.waiting_p95_s - before.waiting_p95_s,
                    "p95_bsld_delta": after.bounded_slowdown_p95 - before.bounded_slowdown_p95,
                    "changed_start_share": float((joined.start_delta_s != 0).mean()),
                    "changed_initial_target_share": float(joined.target_changed.mean()),
                    "absolute_start_shift_mean_s": float(joined.start_delta_s.abs().mean()),
                })
                if factor != "runtime":
                    continue
                joined = joined.join(predictions, validate="one_to_one")
                joined["runtime_error_band"] = pd.cut(joined.runtime_ratio,
                    [0, .5, .8, 1.25, 2, np.inf], right=False,
                    labels=["<0.5x", "0.5-0.8x", "0.8-1.25x", "1.25-2x", ">=2x"])
                joined["duration_band"] = pd.cut(joined.duration_s,
                    [0, 60, 600, 3600, 10800, np.inf], right=False, labels=[band[2] for band in ex.BANDS])
                for dimension in ("runtime_error_band", "duration_band"):
                    for band, members in joined.groupby(dimension, observed=True):
                        bands.append(context | {
                            "dimension": dimension, "band": str(band), "jobs": len(members),
                            "runtime_mae_s": float(members.runtime_error_s.abs().mean()),
                            "median_runtime_ratio": float(members.runtime_ratio.median()),
                            "changed_start_share": float((members.start_delta_s != 0).mean()),
                            "changed_initial_target_share": float(members.target_changed.mean()),
                            "mean_wait_delta_s": float(members.start_delta_s.mean()),
                            "waiting_after_p95_s": Distribution.of(members.waiting_s_after.tolist()).p95,
                            "bsld_after_p95": Distribution.of(members.bounded_slowdown_after.tolist()).p95,
                            "emissions_increase_kg": float((members.emissions_gco2e_after - members.emissions_gco2e).sum() / 1000),
                        })
    return pd.DataFrame(rows), pd.DataFrame(bands)


def forecast_diagnostics(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual, forecast, _ = ex.carbon()
    signal = CarbonSignal(actual)
    jobs = {job.job_id: job for job in ex.cohort("test")[0]}
    rows, used_issues = [], set()
    for _, row in summary[(summary.delay_hours == 6) & (summary.policy != "EASY")].iterrows():
        _, decisions = schedules(row)
        used_issues.update(decisions.forecast_issue_time.dropna())
        first = decisions.drop_duplicates("job_id")
        held = first[first.target > first.decision_time]
        # Counterfactual only: no capacity constraint and no later replanning.
        benefit = np.array([
            carbon_cost_gco2e(jobs[item.job_id], item.decision_time, signal)
            - carbon_cost_gco2e(jobs[item.job_id], item.target, signal)
            for item in held.itertuples()
        ])
        predicted_benefit = (held.cost_now_gco2e - held.cost_target_gco2e).to_numpy(float)
        rows.append({"policy": row.policy, "information_scenario": row.information_scenario,
                     "initial_deferrals": len(held),
                     "locally_harmful_deferral_share": float(np.mean(benefit < -1e-9)) if len(held) else np.nan,
                     "local_benefit_prediction_mae_g": float(np.mean(abs(predicted_benefit - benefit))) if len(held) else np.nan,
                     "initial_local_benefit_g": float(benefit.sum()),
                     "effective_budget_mean_h": row.effective_budget_mean_h,
                     "reach_limited_share": row.reach_limited_share,
                     "target_changes": row.target_changes, "held_share": row.held_share})
    errors = []
    for snapshot in forecast.archive.snapshots:
        if snapshot.issue_time not in used_issues:
            continue
        for sample in snapshot.samples:
            errors.append({
                "lead_hours": (sample.timestamp - snapshot.issue_time).total_seconds() / 3600,
                "error": sample.intensity_gco2e_per_kwh - actual.get_actual(sample.timestamp),
            })
    accuracy = []
    for lead, members in pd.DataFrame(errors).groupby("lead_hours"):
        values = members.error.to_numpy()
        accuracy.append({"lead_hours": lead, "trajectories": len(values),
                         "bias_g_per_kwh": float(values.mean()), "mae_g_per_kwh": float(abs(values).mean()),
                         "rmse_g_per_kwh": float(np.sqrt(np.mean(values ** 2)))})
    return pd.DataFrame(rows), pd.DataFrame(accuracy)


def milp_comparison(workers: int, time_limit: float) -> pd.DataFrame:
    """Small-instance Pareto controls; one shared domain for all prices/policies."""
    cases = [(100, 128, hours, 15, weight) for hours in ex.DELAY_SWEEP for weight in (0., 1., 10.)]
    cases += [(1000, nodes, 6., 60, 0.) for nodes in (880, 256, 128)]
    cases += [(1000, 128, 6., 60, weight) for weight in (1., 10.)]
    tasks = [ex.task(ex.Scenario("test", subset=size, max_nodes=128, nodes=nodes),
                     replace(policy, delay_hours=hours) if name != "EASY" else policy, ex.ORACLE)
             for size, nodes, hours, _, _ in cases for name, policy in {"EASY": ex.EASY, **POLICIES}.items()]
    ex.run_all(tasks, workers)
    actual, _, reach = ex.carbon()
    signal = CarbonSignal(actual)
    rows = []
    for size, nodes, hours, grid, weight in cases:
        scenario = ex.Scenario("test", subset=size, max_nodes=128, nodes=nodes)
        jobs, _, skipped = ex.scenario_jobs(scenario, "actual")
        references, frames, reference_rows = [], {}, {}
        for name, policy in {"EASY": ex.EASY, **POLICIES}.items():
            policy = replace(policy, delay_hours=hours) if name != "EASY" else policy
            row = augment(ex.execute(ex.task(scenario, policy, ex.ORACLE)))
            frame, _ = schedules(row)
            frames[name], reference_rows[name] = frame, row
            references.append(SimpleNamespace(records=[SimpleNamespace(job_id=item.job_id, start_time=item.start_time)
                                                         for item in frame.itertuples()]))
        identity = {"jobs": size, "nodes": nodes, "delay_hours": hours, "grid_minutes": grid,
                    "delay_weight_g_per_job_hour": weight, "time_limit_s": time_limit,
                    "reference_keys": [row["run_key"] for row in reference_rows.values()],
                    "reference_sha256": [ex.sha256(ex.OUTPUT / "runs" / row["run_key"] / "jobs.parquet")
                                         for row in reference_rows.values()],
                    "solver_source_sha256": ex.sha256(Path(__file__).with_name("milp.py"))}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        directory = CACHE / "milp" / key
        if (directory / "solver.json").exists():
            solver = json.loads((directory / "solver.json").read_text())["solver"]
            plan_frame = pd.read_parquet(directory / "starts.parquet")
            starts = dict(zip(plan_frame.job_id, plan_frame.start_time))
        else:
            starts, solver = solve(jobs, signal, nodes=nodes, max_delay=timedelta(hours=hours),
                granularity=timedelta(minutes=grid), references=references, delay_weight=weight,
                time_limit=time_limit, reach=reach)
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(starts.items(), columns=["job_id", "start_time"]).to_parquet(directory / "starts.parquet", index=False)
            write_json(directory / "solver.json", {"identity": identity, "solver": solver})
            print(f"MILP {size} jobs / {nodes} nodes / {hours:g} h / w={weight:g}: "
                  f"{solver['solution_status']} ({solver['solve_seconds']:.1f}s)", flush=True)
        values = {}
        for name, frame in frames.items():
            planning = planning_emissions(jobs, dict(zip(frame.job_id, frame.start_time)), signal)
            row = reference_rows[name]
            values[name] = row | {"planning_emissions_g": planning,
                                 "objective_g": planning + weight * frame.waiting_s.sum() / 3600}
        if starts:
            result = account_schedule(replay(jobs, starts, nodes), jobs, actual)
            metrics = schedule_metrics(result)
            planning = planning_emissions(jobs, starts, signal)
            objective = planning + weight * sum(record.waiting_seconds for record in result.records) / 3600
            if not np.isclose(objective, solver["objective_gco2e"], rtol=1e-9, atol=1e-5):
                raise AssertionError("replayed MILP objective differs from the solver coefficients")
            values["MILP"] = metrics.as_row() | {
                "planning_emissions_g": planning, "objective_g": objective,
                "total_emissions_gco2e": metrics.total_emissions_gco2e,
                "bounded_slowdown_p99": metrics.bounded_slowdown.p99,
                "turnaround_p95_s": metrics.turnaround.p95, "turnaround_p99_s": metrics.turnaround.p99,
                "turnaround_max_s": metrics.turnaround.maximum,
            }
        feasible_bound = min(row["objective_g"] for row in values.values())
        dual = solver.get("dual_bound_gco2e")
        for name, value in values.items():
            row = value | {k: v for k, v in identity.items() if k not in ("reference_keys", "reference_sha256")}
            row |= {"policy": name, "subset_skipped_jobs": skipped, "milp_key": key,
                    "source_solver": relative(directory), **solver,
                    "emissions_saved_pct": 100 * (1 - value["total_emissions_gco2e"] / values["EASY"]["total_emissions_gco2e"]),
                    "optimality_gap_lower_pct": 100 * max(0., value["objective_g"] / feasible_bound - 1),
                    "optimality_gap_upper_pct": 100 * max(0., value["objective_g"] / dual - 1) if dual and dual > 0 else np.nan,
                    "gap_objective": "carbon" if weight == 0 else "carbon_plus_delay",
                    "gap_domain": "candidate grid + all reference starts; shared reach; offline future arrivals",
                    "saving_captured_vs_milp": ratio(values["EASY"]["planning_emissions_g"] - value["planning_emissions_g"],
                        values["EASY"]["planning_emissions_g"] - values["MILP"]["planning_emissions_g"])[0] if starts else np.nan,
            }
            rows.append(row)
        if not starts:
            rows.append({**identity, "policy": "MILP", "milp_key": key, **solver})
    return pd.DataFrame(rows)


def run(arguments) -> None:
    protocol = preflight()
    scenario = ex.Scenario("test")
    tasks = [ex.task(scenario, ex.EASY, cell) for cell in (ex.ORACLE, ex.REALISTIC)]
    tasks += [ex.task(scenario, replace(policy, delay_hours=hours), cell)
              for policy in POLICIES.values() for hours in ex.DELAY_SWEEP for cell in CELLS]
    tasks += [ex.task(scenario, policy, (information, carbon)) for policy in POLICIES.values()
              for information in ("runtime-only", "power-only") for carbon in ("actual", "forecast")]
    unique = list(dict.fromkeys(tasks))
    reused = sum((ex.run_directory(item)[1] / "run.json").exists() for item in unique)
    print(f"Main matrix: {reused}/{len(unique)} runs reusable", flush=True)
    raw = pd.DataFrame([augment(row) for row in ex.run_all(unique, arguments.workers)])
    if not np.allclose(raw.total_energy_mwh, raw.total_energy_mwh.iloc[0], rtol=1e-10):
        raise AssertionError("energy changed across schedules of the same jobs")
    expected_ids = {job.job_id for job in ex.cohort("test")[0]}
    easy_schedules = {row.job_information: schedules(row)[0].set_index("job_id")
                     for _, row in raw[raw.policy_id == "easy"].iterrows()}
    for _, row in raw.iterrows():
        frame = schedules(row)[0].set_index("job_id")
        if set(frame.index) != expected_ids:
            raise ValueError("a cached schedule does not contain exactly the frozen cohort")
        if row.policy_delay_hours == 0:
            reference = easy_schedules["predicted" if FACTORS[row.job_information][0] else "actual"]
            if not frame.start_time.equals(reference.start_time.reindex(frame.index)):
                raise AssertionError("zero budget must reproduce matching EASY starts")
    summary = summary_table(raw)
    paired, marginal, losses = comparisons(summary)
    effects, bands = factor_effects(summary)
    diagnostics, accuracy = forecast_diagnostics(summary)
    outputs = {"delay_summary": summary[summary.information_scenario.isin(CELLS.values())],
               "information_factorial": summary[(summary.delay_hours == 6) & (summary.policy != "EASY")],
               "pareto": frontier_table(summary), "greedy_vs_improved": paired,
               "marginal_cost": marginal, "information_loss": losses,
               "factor_effects": effects, "runtime_error_bands": bands,
               "decision_diagnostics": diagnostics, "forecast_accuracy": accuracy}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    milp = milp_comparison(arguments.workers, arguments.milp_time_limit)
    milp.to_csv(OUTPUT / "milp_comparison.csv", index=False)
    protocol |= {"main_runs_reused": reused, "main_runs_total": len(unique),
                 "milp_time_limit_s": arguments.milp_time_limit,
                 "run_keys": raw.run_key.tolist(),
                 "csv_sha256": {path.name: ex.sha256(path) for path in sorted(OUTPUT.glob("*.csv"))}}
    write_json(OUTPUT / "protocol.json", protocol)
    report(arguments)


def report(arguments) -> None:
    """Regenerate tables and scientific figures without datasets or simulation."""
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    for name, expected in protocol["csv_sha256"].items():
        if ex.sha256(OUTPUT / name) != expected:
            raise ValueError(f"{name} differs from the recorded experiment")
    read = lambda name: pd.read_csv(OUTPUT / f"{name}.csv")
    summary, frontier, paired, marginal = [read(name) for name in
        ("delay_summary", "pareto", "greedy_vs_improved", "marginal_cost")]
    factorial, losses, effects, bands, diagnostics, accuracy, milp = [read(name) for name in
        ("information_factorial", "information_loss", "factor_effects", "runtime_error_bands",
         "decision_diagnostics", "forecast_accuracy", "milp_comparison")]
    plt = ex._plt()
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)

    def save(figure, name):
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
            figure.savefig(figures / f"{name}.{suffix}", dpi=180, bbox_inches="tight", metadata=metadata)
        plt.close(figure)

    colors = {"EASY": ex.MUTED, "greedy": ex.PALETTE[1], "improved": ex.PALETTE[0], "MILP": ex.PALETTE[2]}
    titles = dict(zip(CELLS.values(), ("Perfect information", "Forecast-only error", "Job-model-only error", "Realistic")))
    for metric, flag, name, axis, scale in (
        ("waiting_mean_s", "pareto_mean_wait", "pareto_wait", "Mean waiting (h)", 1 / 3600),
        ("bounded_slowdown_p95", "pareto_p95_bsld", "pareto_bsld_p95", "Bounded slowdown p95", 1),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, scenario in zip(axes.flat, CELLS.values()):
            cell = frontier[frontier.information_scenario == scenario]
            for policy, members in cell.groupby("policy", sort=False):
                ax.scatter(members[metric] * scale, members.emissions_saved_pct, label=policy,
                           color=colors[policy], marker="s" if policy == "EASY" else "o", s=30, zorder=3)
                if policy != "EASY":
                    for point in members[members.delay_hours == 6].itertuples():
                        ax.annotate("6h", (getattr(point, metric) * scale, point.emissions_saved_pct),
                                    xytext=(5, 5), textcoords="offset points", fontsize=8)
            efficient = cell[cell[flag]].sort_values(metric).drop_duplicates([metric, "emissions_saved_pct"])
            ax.plot(efficient[metric] * scale, efficient.emissions_saved_pct, "k--", linewidth=1,
                    alpha=.6, label="Discrete Pareto frontier")
            ax.axhline(0, color=ex.MUTED, linewidth=.7)
            ax.set(title=titles[scenario], xlabel=axis, ylabel="Emissions saved vs matched EASY (%)")
            if metric == "bounded_slowdown_p95":
                ax.set_xscale("log")
                ax.set_xlabel(axis + " (log scale)")
        axes[0, 0].legend(fontsize=8)
        figure.suptitle("Delay budgets: 0, 1, 3, 6, 12, 24 h (6 h labelled)", fontsize=10)
        save(figure, name)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, policy in zip(axes, POLICIES):
        for index, scenario in enumerate(("forecast_only_error", "job_model_only_error", "realistic")):
            group = summary[(summary.policy == policy) & (summary.information_scenario == scenario)].sort_values("delay_hours")
            ax.plot(group.delay_hours, 100 * group.carbon_benefit_retained, "o-", label=titles[scenario], color=ex.PALETTE[index])
        ax.axhline(100, color=ex.MUTED, linestyle="--", linewidth=1)
        ax.axhline(0, color=ex.MUTED, linewidth=.7)
        ax.set(title=policy, xlabel="Nominal delay budget (h)", ylabel="Carbon benefit retained (%)")
    axes[0].legend(fontsize=8, frameon=True, facecolor="white", framealpha=1, edgecolor="none")
    save(figure, "benefit_retained")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for index, policy in enumerate(POLICIES):
        group = marginal[(marginal.policy == policy) & (marginal.information_scenario == "realistic")]
        x = np.arange(len(group)) + (index - .5) * .35
        for ax, metric in zip(axes, ("extra_carbon_saved_kg", "job_wait_hours_per_extra_kg")):
            ax.bar(x, group[metric], width=.32, label=policy, color=colors[policy])
            for position, value in zip(x, group[metric]):
                if pd.isna(value):
                    ax.text(position, 0, "n/a", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(np.arange(len(group)), [f"{a:g}→{b:g}" for a, b in zip(group.from_delay_hours, group.to_delay_hours)])
            ax.set_xlabel("Delay budget increase (h)")
            ax.axhline(0, color=ex.MUTED, linewidth=.7)
    axes[0].set_ylabel("Incremental saving (kgCO₂e)")
    axes[1].set_ylabel("Additional job-wait hours / extra kgCO₂e")
    axes[0].set_title("Realistic: marginal carbon benefit", loc="left")
    axes[1].set_title("QoS cost (n/a when no extra saving)", loc="left")
    axes[0].legend()
    save(figure, "marginal_cost")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(accuracy.lead_hours, accuracy.mae_g_per_kwh, label="MAE", color=ex.PALETTE[0])
    axes[0].plot(accuracy.lead_hours, accuracy.rmse_g_per_kwh, label="RMSE", color=ex.PALETTE[1])
    axes[0].set(xlabel="Forecast lead (h)", ylabel="CI error (gCO₂e/kWh)", title="Snapshots consulted by the 6 h runs")
    axes[0].legend()
    for index, policy in enumerate(POLICIES):
        group = diagnostics[(diagnostics.policy == policy) & diagnostics.information_scenario.isin(CELLS.values())]
        group = group.set_index("information_scenario").reindex(CELLS.values())
        axes[1].bar(np.arange(4) + (index - .5) * .35, 100 * group.locally_harmful_deferral_share,
                    width=.32, color=colors[policy], label=policy)
    axes[1].set_xticks(range(4), ["Perfect", "Forecast\nerror", "Job\nerror", "Realistic"])
    axes[1].set(ylabel="Initial deferrals increasing local carbon (%)", title="Actual runtime + CI counterfactual (6 h)")
    axes[1].legend()
    save(figure, "forecast_error")

    figure, axes = plt.subplots(1, 3, figsize=(12, 6.5))
    for index, policy in enumerate(POLICIES):
        group = effects[effects.policy == policy].sort_values(["factor", "from_scenario"])
        y = np.arange(len(group)) + (index - .5) * .35
        for ax, metric, unit in zip(axes,
            ("emissions_increase_kg", "mean_wait_delta_s", "changed_start_share"),
            ("Emission increase (kgCO₂e)", "Mean waiting change (s)", "Jobs starting at a different time (%)")):
            ax.barh(y, group[metric] * (100 if metric == "changed_start_share" else 1), height=.32,
                    color=colors[policy], label=policy)
            ax.axvline(0, color=ex.MUTED, linewidth=.7)
            ax.set_xlabel(unit)
            ax.set_yticks(np.arange(len(group)))
            ax.invert_yaxis() if index == 0 else None
        contexts = {"perfect_information": "perfect inputs", "job_model_only_error": "runtime + power error",
                    "forecast_only_error": "CI error", "runtime-only_actual": "runtime error",
                    "power-only_actual": "power error", "runtime-only_forecast": "runtime + CI error",
                    "power-only_forecast": "power + CI error"}
        labels = [f"Add {r.factor.replace('carbon_forecast', 'CI')} | {contexts[r.from_scenario]}" for r in group.itertuples()]
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[0].legend(fontsize=8)
    figure.suptitle("Turning on one error source, holding the other two fixed — 6 h budget")
    save(figure, "error_effects")

    small = milp[(milp.jobs == 100) & (milp.nodes == 128)].dropna(
        subset=["total_emissions_gco2e", "waiting_mean_s", "bounded_slowdown_p95"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, scale, title in zip(axes, ("waiting_mean_s", "bounded_slowdown_p95"),
        (1 / 3600, 1), ("Mean waiting (h)", "Bounded slowdown p95")):
        for policy in ("EASY", "greedy", "improved", "MILP"):
            group = small[small.policy == policy].drop_duplicates(["delay_hours", metric, "emissions_saved_pct"])
            ax.scatter(group[metric] * scale, group.emissions_saved_pct, label=policy, color=colors[policy],
                       marker="D" if policy == "MILP" else "o", alpha=.8, s=28)
        efficient = small.loc[pareto_mask(small, ("total_emissions_gco2e", metric))].sort_values(metric)
        ax.plot(efficient[metric] * scale, efficient.emissions_saved_pct, "k--", linewidth=1, alpha=.5)
        ax.set(xlabel=title, ylabel="Emissions saved vs EASY (%)", title="100 jobs / 128 nodes / perfect information")
        if metric == "bounded_slowdown_p95":
            ax.set_xscale("log")
            ax.set_xlabel(title + " (log scale)")
    axes[0].legend(fontsize=8)
    save(figure, "milp_frontier")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    large = milp[(milp.jobs == 1000) & (milp.delay_weight_g_per_job_hour == 0)]
    for policy in POLICIES:
        group = large[large.policy == policy].sort_values("nodes")
        axes[0].errorbar(group.nodes, group.optimality_gap_lower_pct,
            yerr=[np.zeros(len(group)), (group.optimality_gap_upper_pct - group.optimality_gap_lower_pct).clip(lower=0)],
            marker="o", capsize=3, color=colors[policy], label=policy)
    axes[0].set(xlabel="Nodes (1,000-job fixed subset)", ylabel="Carbon objective gap (%)", title="Certified lower–upper gap interval")
    axes[0].set_xticks([128, 256, 880])
    axes[0].legend()
    for policy in ("EASY", "greedy", "improved", "MILP"):
        group = milp[(milp.jobs == 1000) & (milp.nodes == 128) & (milp.policy == policy)]
        group = group.drop_duplicates(["waiting_mean_s", "emissions_saved_pct"])
        axes[1].scatter(group.waiting_mean_s / 3600, group.emissions_saved_pct, label=policy,
                       marker="D" if policy == "MILP" else "o", color=colors[policy])
        if policy == "MILP":
            for point in group.itertuples():
                axes[1].annotate(f"w={point.delay_weight_g_per_job_hour:g}",
                    (point.waiting_mean_s / 3600, point.emissions_saved_pct), xytext=(3, 4), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Mean waiting (h)", ylabel="Emissions saved vs EASY (%)", title="1,000 jobs / 128 nodes / 6 h budget")
    axes[1].legend(fontsize=8)
    save(figure, "milp_gaps")

    table = ex._markdown
    six = summary[(summary.information_scenario == "realistic") & (summary.delay_hours == 6)].set_index("policy")
    g, i = six.loc["greedy"], six.loc["improved"]
    comp = paired[(paired.information_scenario == "realistic") & (paired.delay_hours == 6)].iloc[0]
    effects_by_context = effects.set_index(["policy", "factor", "from_scenario"])
    runtime_g = effects_by_context.loc[("greedy", "runtime", "perfect_information")]
    runtime_i = effects_by_context.loc[("improved", "runtime", "perfect_information")]
    forecast_g = effects_by_context.loc[("greedy", "carbon_forecast", "perfect_information")]
    forecast_i = effects_by_context.loc[("improved", "carbon_forecast", "perfect_information")]
    compensating = effects_by_context.loc[("improved", "carbon_forecast", "job_model_only_error")]
    under = bands[(bands.dimension == "runtime_error_band") & (bands.band == "<0.5x")
                  & (bands.from_scenario == "perfect_information")].set_index("policy")
    increments = marginal[(marginal.policy == "improved") & (marginal.information_scenario == "realistic")].set_index("to_delay_hours")
    small_gap = milp[(milp.jobs == 1000) & (milp.nodes == 128) & (milp.delay_weight_g_per_job_hour == 0)].set_index("policy")
    priced = milp[(milp.jobs == 1000) & (milp.nodes == 128) & (milp.policy == "MILP")].set_index("delay_weight_g_per_job_hour")
    milp_conclusion = "Alcune istanze non hanno una soluzione MILP o un bound duale finito: consultare status e gap nel CSV; non si dichiara ottimalità."
    if np.isfinite(small_gap.optimality_gap_upper_pct).all() and np.isfinite(priced.waiting_mean_s).all():
        milp_conclusion = (
            f"Su questo prefisso da 1.000 job a 128 nodi, la greedy è entro "
            f"{small_gap.loc['greedy', 'optimality_gap_upper_pct']:.5f}% dall'ottimo carbonico discreto. "
            f"Improved lascia un gap del {small_gap.loc['improved', 'optimality_gap_lower_pct']:.3f}%, "
            f"riducendo il waiting medio del "
            f"{100 * (1 - small_gap.loc['improved', 'waiting_mean_s'] / small_gap.loc['greedy', 'waiting_mean_s']):.1f}% "
            f"e il p95 BSLD da {small_gap.loc['greedy', 'bounded_slowdown_p95']:.1f} a "
            f"{small_gap.loc['improved', 'bounded_slowdown_p95']:.1f}. "
            f"Il MILP a w=1 conserva il "
            f"{100 * priced.loc[1., 'emissions_saved_pct'] / priced.loc[0., 'emissions_saved_pct']:.2f}% "
            f"del saving misurato del MILP carbon-only e porta il waiting da "
            f"{priced.loc[0., 'waiting_mean_s']:.0f} a {priced.loc[1., 'waiting_mean_s']:.0f} s: "
            "il minimo delle sole emissioni può spendere molto waiting per benefici marginali piccoli. "
            "Le differenze minime nelle emissioni misurate fra greedy e MILP non certificano ottimalità "
            "su un obiettivo diverso dai coefficienti integrati."
        )
    brief = {"policy": "policy", "budget h": "delay_hours:g", "CO₂ t": "total_emissions_tco2e:.4f",
             "saved %": "emissions_saved_pct:.3f", "wait mean s": "waiting_mean_s:.1f",
             "wait p95 s": "waiting_p95_s:.0f", "wait max s": "waiting_max_s:.0f",
             "turnaround mean s": "turnaround_mean_s:.1f", "BSLD mean": "bounded_slowdown_mean:.2f",
             "BSLD p95": "bounded_slowdown_p95:.2f", "utilisation": "utilisation:.2%", "peak MW": "peak_power_mw:.3f"}
    parts = ["# Trade-off QoS–carbon emissions: esperimenti consolidati", "",
        f"Sulla coorte test di **{protocol['cohort']['jobs']:,} job**, a 880 nodi e budget 6 h, "
        f"lo scheduler improved risparmia **{i.emissions_saved_pct:.3f}%** rispetto a EASY con runtime predetto, "
        f"contro **{g.emissions_saved_pct:.3f}%** della greedy. "
        f"Evita {comp.extra_carbon_saved_kg:.2f} kgCO₂e aggiuntivi, riduce il waiting medio del "
        f"{comp.mean_wait_reduction_pct:.1f}% e il p95 bounded slowdown del {comp.p95_bsld_reduction_pct:.1f}%.", "",
        f"Il carbon benefit retained, con EASY actual comune e il perfect-information della stessa policy, "
        f"è **{100 * i.carbon_benefit_retained:.2f}%** per improved e **{100 * g.carbon_benefit_retained:.2f}%** per greedy. "
        "Questi risultati sono descrittivi di una coorte già analizzata; non costituiscono un nuovo holdout.", "",
        "## Protocollo e significato delle metriche", "",
        "- Stessi job e profili misurati, cluster omogeneo, scheduling non preemptive; energia invariata. "
        "QoS da `release_time`, non dalla submission. Percentili nearest-rank, BSLD = "
        "`max(1, (wait + runtime) / max(runtime, 10 s))`.",
        "- Greedy: scelta carbonica al rilascio (`once`). Improved: benefit-rate receding horizon, "
        "soglia **0.014930658589279368 gCO₂e/job-hour**, già congelata su validation. Nessun nuovo tuning.",
        "- Budget nominali 0, 1, 3, 6, 12, 24 h; griglia online 15 minuti. Budget effettivo "
        "`min(budget, max(0, 23 h - runtime stimato))`; la contesa può portare il waiting oltre il budget volontario.",
        "- `emissions_saved_pct` e kg: EASY con la **stessa informazione sul runtime**. "
        "`emissions_saved_common_*`: EASY con runtime reale per tutte le celle. Sono due confronti espliciti.",
        "- `carbon_benefit_retained = (C(EASY actual) - C(cella)) / "
        "(C(EASY actual) - C(perfect, stessa policy e budget))`. Denominatore non positivo: NA con motivo; "
        "valori negativi o superiori al 100% non vengono troncati. Perfect information di una euristica non è un limite teorico.",
        "- Emissioni ex post sui profili e sulla CI osservata. Il peak power è un proxy ricavato dalle potenze medie dei job. "
        "Utilisation = node-seconds / (nodi × makespan), non una percentuale di energia.", "",
        "## Frontiere di Pareto", "",
        "Le frontiere sono calcolate separatamente per scenario, sulla stessa coorte e capacità. "
        "Le linee uniscono soltanto punti non dominati; non implicano che i punti intermedi siano realizzabili. "
        "`pareto.csv` distingue le due proiezioni e la frontiera congiunta carbonio/wait mean/BSLD p95. "
        "Un semplice sweep con punti collegati non è necessariamente una frontiera.", "",
        "![Carbon saving e mean waiting](figures/pareto_wait.png)", "",
        "![Carbon saving e p95 bounded slowdown](figures/pareto_bsld_p95.png)", "",
        "## Tabella per delay budget", "",
        "Risparmio rispetto a EASY abbinato; turnaround e bounded slowdown hanno anche mediana, p95, p99 e massimo nel CSV. "
        "EASY è ripetuto per budget per agevolare il confronto, senza ripetere la simulazione.", ""]
    for scenario in CELLS.values():
        parts += [f"### {titles[scenario]}", "", table(summary[summary.information_scenario == scenario], brief), ""]
    parts += ["## Greedy vs improved e costo marginale", "",
        "I delta sono improved meno greedy, salvo il carbon saving che è positivo quando improved evita più emissioni. "
        "La dominanza richiede carbonio, waiting mean/p95 e BSLD mean/p95 non peggiori. "
        "Il costo marginale di ampliare un budget è Δ attesa / Δ kg evitati; "
        "i job-wait hours sommano l'attesa di tutti i job. Non è un costo economico.", "",
        table(paired[paired.information_scenario == "realistic"], {
            "budget h": "delay_hours:g", "extra saving kg": "extra_carbon_saved_kg:.3f",
            "Δ wait mean s": "delta_mean_wait_s:.1f", "Δ BSLD p95": "delta_p95_bsld:.2f",
            "wait reduction %": "mean_wait_reduction_pct:.2f", "dominates": "improved_dominates_greedy"}), "",
        table(marginal[marginal.information_scenario == "realistic"], {
            "policy": "policy", "from h": "from_delay_hours:g", "to h": "to_delay_hours:g",
            "extra saving kg": "extra_carbon_saved_kg:.3f", "Δ wait mean s": "delta_mean_wait_s:.1f",
            "job-wait h / kg": "job_wait_hours_per_extra_kg:.2f", "Δ BSLD p95 / kg": "p95_bsld_per_extra_kg:.2f"}), "",
        "Una cella NA indica che l'incremento di budget non compra altro carbon saving. "
        "Non si assume monotonicità: forecast, code e aggiornamenti possono rendere un budget maggiore peggiore.", "",
        f"Per improved, passare da 3 a 6 h costa {increments.loc[6., 'job_wait_hours_per_extra_kg']:.1f} "
        f"job-wait hours per kg aggiuntivo. Da 6 a 12 h il costo sale a "
        f"{increments.loc[12., 'job_wait_hours_per_extra_kg']:.1f}, con soli "
        f"{increments.loc[12., 'extra_carbon_saved_kg']:.2f} kg aggiuntivi. Da 12 a 24 h "
        f"si perdono {-increments.loc[24., 'extra_carbon_saved_kg']:.2f} kg di saving e il waiting aumenta: "
        "il punto a 24 h è dominato da quello a 12 h su entrambe le proiezioni. "
        "È un risultato retrospettivo, non una nuova selezione del budget su un holdout.", "",
        "![Costo marginale](figures/marginal_cost.png)", "",
        "## Informazione e carbon benefit retained", "",
        table(factorial[factorial.information_scenario.isin(CELLS.values())], {
            "policy": "policy", "scenario": "information_scenario", "CO₂ t": "total_emissions_tco2e:.4f",
            "saved matched %": "emissions_saved_pct:.3f", "saved common %": "emissions_saved_common_pct:.3f",
            "retained": "carbon_benefit_retained:.2%", "wait mean s": "waiting_mean_s:.1f",
            "BSLD p95": "bounded_slowdown_p95:.2f"}), "",
        "La perdita rispetto al perfect-information è scomposta in forecast, modelli dei job e interazione: "
        "`C(F)-C(P)`, `C(J)-C(P)`, `C(R)-C(F)-C(J)+C(P)`. Le quote usano il beneficio perfect "
        "con EASY actual comune e sommano a `1 - retained(realistic)`. Un'interazione negativa indica compensazione, "
        "non un errore di segno.", "",
        table(losses[losses.delay_hours == 6], {
            "policy": "policy", "forecast loss": "forecast_loss_share:.2%",
            "job loss": "job_model_loss_share:.2%", "interaction": "interaction_loss_share:.2%",
            "combined loss": "combined_loss_share:.2%"}), "",
        "![Beneficio conservato per budget](figures/benefit_retained.png)", "",
        "## Runtime prediction e carbon-intensity forecast error", "",
        "La cella job-model-only include **runtime e potenza**. Il controllo fattoriale 2×2×2 a 6 h "
        "li separa: ogni riga seguente attiva una sola fonte di errore lasciando fisse le altre due. "
        "Runtime reale e profilo misurato continuano a governare esecuzione e valutazione in ogni controllo.", "",
        table(effects, {"policy": "policy", "errore attivato": "factor", "contesto iniziale": "from_scenario",
            "Δ CO₂ kg": "emissions_increase_kg:.3f", "Δ wait mean s": "mean_wait_delta_s:.1f",
            "Δ BSLD p95": "p95_bsld_delta:.2f", "start cambiati": "changed_start_share:.1%",
            "target iniziali cambiati": "changed_initial_target_share:.1%"}), "",
        "Gli errori di runtime cambiano sia l'integrazione del costo futuro sia le reservation EASY e "
        "il budget effettivo. La potenza moltiplica tutti i candidati della greedy: da sola non ne cambia "
        "il minimo; può invece cambiare il superamento della soglia benefit-rate di improved. "
        "La matrice misura questi effetti anche in presenza del forecast.", "",
        f"Con potenza e CI esatte, introdurre il solo errore di runtime costa "
        f"**{runtime_g.emissions_increase_kg:.2f} kg** alla greedy e **{runtime_i.emissions_increase_kg:.2f} kg** "
        f"a improved; cambia lo start rispettivamente del {runtime_g.changed_start_share:.1%} e "
        f"{runtime_i.changed_start_share:.1%} dei job. Con tutti gli input dei job esatti, il solo forecast costa "
        f"{forecast_g.emissions_increase_kg:.2f} e {forecast_i.emissions_increase_kg:.2f} kg. "
        f"Con i job già predetti, invece, il forecast fa diminuire le emissioni di improved di "
        f"{-compensating.emissions_increase_kg:.2f} kg e aumenta l'attesa media di "
        f"{compensating.mean_wait_delta_s:.0f} s. Questo cambio di segno mostra la compensazione fra errori "
        "e replanning: una previsione meno accurata non implica sempre uno schedule peggiore su questa coorte.", "",
        "![Effetti isolati delle tre fonti di errore](figures/error_effects.png)", "",
        "Le fasce di errore usano runtime predetto/reale (1 = esatto). Le emissioni sono differenze "
        "per job fra simulazioni abbinate; comprendono gli effetti indiretti della contesa. "
        "Le fasce descrivono l'associazione e non identificano un effetto causale indipendente per job. "
        "`runtime_error_bands.csv` ripete il confronto anche per durata reale e per gli altri contesti.", "",
        f"I **{int(under.loc['greedy', 'jobs']):,} job sottostimati di oltre un fattore 2** "
        f"({under.loc['greedy', 'jobs'] / protocol['cohort']['jobs']:.1%} della coorte) contribuiscono "
        f"{under.loc['greedy', 'emissions_increase_kg']:.2f} kg all'aumento della greedy e "
        f"{under.loc['improved', 'emissions_increase_kg']:.2f} kg a quello di improved. "
        "La concentrazione nei runtime fortemente sottostimati è più informativa del solo errore medio globale.", "",
        table(bands[(bands.dimension == "runtime_error_band") & (bands.from_scenario == "perfect_information")], {
            "policy": "policy", "runtime ratio": "band", "jobs": "jobs:d", "runtime MAE s": "runtime_mae_s:.1f",
            "start cambiati": "changed_start_share:.1%", "Δ wait mean s": "mean_wait_delta_s:.1f",
            "BSLD p95 dopo": "bsld_after_p95:.2f", "Δ CO₂ kg": "emissions_increase_kg:.3f"}), "",
        "Il MAE/RMSE del forecast è misurato sugli snapshot effettivamente consultati dai run a 6 h, "
        "per lead; le traiettorie si sovrappongono. Il grafico a destra valuta il target della **prima decisione** "
        "con runtime e CI reali: una deferral dannosa aumenta il costo locale integrato. "
        "Questo controfattuale ignora capacità e replanning e non sostituisce le emissioni dello schedule.", "",
        f"Sugli {int(accuracy.trajectories.iloc[0])} snapshot consultati il MAE passa da "
        f"{accuracy.mae_g_per_kwh.iloc[0]:.2f} gCO₂e/kWh al lead iniziale a "
        f"{accuracy.mae_g_per_kwh.iloc[-1]:.2f} al lead {accuracy.lead_hours.iloc[-1]:g} h. "
        "L'errore cresce con l'orizzonte; la valutazione delle decisioni mostra perché contano anche "
        "il segno del vantaggio e la durata del job, non soltanto il MAE della serie.", "",
        "![Forecast error e decisioni](figures/forecast_error.png)", "",
        table(diagnostics[diagnostics.information_scenario.isin(CELLS.values())], {
            "policy": "policy", "scenario": "information_scenario", "deferrals iniziali": "initial_deferrals:d",
            "localmente dannose": "locally_harmful_deferral_share:.2%", "budget effettivo h": "effective_budget_mean_h:.2f",
            "reach limited": "reach_limited_share:.2%", "target changes": "target_changes:d"}), "",
        "## MILP su piccole istanze", "",
        "Primi 100 o 1.000 job cronologici del medesimo test, filtrati a richieste ≤128 nodi, "
        "poi mantenuti identici fra capacità. Il CSV conta gli esclusi. Nessuna generalizzazione "
        "dal prefisso alla coorte completa: la finestra carbonica e la contesa sono diverse.", "",
        "La griglia MILP è 15 min sui 100 job e 60 min sui 1.000, aumentata con **tutti gli start di EASY, "
        "greedy e improved**. È lo stesso dominio per ciascun prezzo e contiene le tre soluzioni online. "
        "Si applica lo stesso reach dei benchmark. Gli start di riferimento oltre il budget, dovuti alla contesa, "
        "restano ammissibili e sono contati in `starts_beyond_budget`: l'ottimo riguarda questo dominio esteso.", "",
        "Obiettivo `planning_emissions_g + w × job-wait-hours`, w = 0, 1, 10 gCO₂e/job-hour. "
        "Il costo è ricalcolato con **gli stessi coefficienti integrati** del solver. L'accounting "
        "campionato a potenza media, usato dai vecchi CSV, può differire anche a potenza costante e "
        "non è usato per certificare il gap. Le emissioni misurate restano la metrica della frontiera.", "",
        "`optimality_gap_lower_pct = 100 × (obiettivo policy / miglior soluzione ammissibile - 1)`; "
        "il bound duale fornisce il limite superiore. Sono gap al **dominio discreto**. A w>0 il gap "
        "è sull'obiettivo carbonio+attesa, non sulle sole emissioni. `optimal` significa entro la "
        f"tolleranza relativa 10⁻⁶; timeout dopo {protocol['milp_time_limit_s']:g} s conservati come tali. "
        "Il MILP conosce anche i job futuri: è un riferimento offline, non una policy online distribuibile.", "",
        table(milp[(milp.jobs == 1000) & (milp.delay_weight_g_per_job_hour == 0)], {
            "nodes": "nodes:d", "policy": "policy", "CO₂ t": "total_emissions_tco2e:.5f",
            "saved %": "emissions_saved_pct:.3f", "wait mean s": "waiting_mean_s:.1f",
            "BSLD p95": "bounded_slowdown_p95:.2f", "gap low %": "optimality_gap_lower_pct:.5f",
            "gap high %": "optimality_gap_upper_pct:.5f", "MILP status": "solution_status"}), "",
        milp_conclusion, "",
        "![MILP e politiche online sulle istanze da 100 job](figures/milp_frontier.png)", "",
        "![Gap e costo QoS sulle istanze da 1000 job](figures/milp_gaps.png)", "",
        "### Tabella per budget: 100 job, 128 nodi, informazione perfetta", "",
        table(small.drop_duplicates(["policy", "delay_hours", "total_emissions_gco2e", "waiting_mean_s"]),
              {**brief, "w g/job-h": "delay_weight_g_per_job_hour:g"}), "",
        "## Riproduzione e artifact", "",
        "```bash\n# Riusa le cache e calcola soltanto i run mancanti\n"
        "make qos-carbon ARGS='--workers 2'\n"
        "# Solo figure e tabelle dai CSV, senza simulare né leggere workload/forecast\n"
        "make qos-carbon-report\n"
        "# Check sintetico, senza dataset\n"
        ".venv/bin/python tests/check_experiments_qos_carbon.py\n```", "",
        "Se si parte dai soli input originali: `make improved-scheduler PHASE=prepare` e "
        "`make improved-scheduler PHASE=validation` ricreano il manifest, le predizioni di validation "
        "e la selezione richiesta. La consolidazione non riaddestra i modelli.", "",
        "- `delay_summary.csv`: matrice completa per budget e tutte le metriche richieste; "
        "`pareto.csv`: flag di non dominanza per proiezione e congiunti.",
        "- `greedy_vs_improved.csv`, `marginal_cost.csv`, `information_loss.csv`: differenze abbinate, "
        "costi incrementali, quote della perdita e motivi dei valori non definiti.",
        "- `information_factorial.csv`, `factor_effects.csv`, `runtime_error_bands.csv`, "
        "`decision_diagnostics.csv`, `forecast_accuracy.csv`: errori e conseguenze sullo schedule.",
        "- `milp_comparison.csv`: QoS, emissioni, obiettivo, incumbent, dual bound, gap, status e dominio per ogni istanza.",
        "- `protocol.json`: versioni, hash SHA-256 di input/sorgenti/CSV, selezione e run key. "
        "`source_run` punta agli schedule e decisioni riusati in `data/experiments/improved_scheduler/runs/`; "
        "`source_solver` ai piani e metadati in `data/experiments/qos_carbon/milp/`.",
        "- `figures/`: PNG e PDF da matplotlib. Questo report è generato dagli stessi CSV, non trascritto a mano.", "",
        "## Limiti", "",
        "Operational carbon soltanto; potenza e durata non cambiano con lo start time; nessun idle/sleep, "
        "DVFS, migrazione o preemption. La coorte principale include COMPLETED con profili validi e parte "
        "da cluster vuoto: non è l'intera domanda storica. Qui il protocollo principale resta a 880 nodi/15 min "
        "su una coorte, con subset MILP separati. Il [report di robustezza](robustness/README.md) "
        "completa quattro finestre di release, 880/512/256 nodi, griglie 15/60/240 min, domanda terminale, "
        "ablation e classi di job. Le finestre restano vicine e retrospettive; non sono un holdout indipendente. "
        "Nessun intervallo di confidenza IID su job o traiettorie correlate. "
        "Gli schedule sono deterministici; tempi e incumbent MILP fermati dal timeout dipendono dalla macchina.", ""]
    (OUTPUT / "README.md").write_text("\n".join(parts))
    print(f"Report, CSV and figures: {OUTPUT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "report"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--milp-time-limit", type=float, default=60.)
    arguments = parser.parse_args()
    if arguments.workers < 1 or not np.isfinite(arguments.milp_time_limit) or arguments.milp_time_limit <= 0:
        parser.error("workers and MILP time limit must be positive")
    (run if arguments.phase == "run" else report)(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
