"""Thesis robustness experiments using frozen policies: run | report.

No scheduler or predictor is fitted here. Reuse content-addressed schedules;
reports need only the aggregate CSVs and their SHA-256 manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments import improved_scheduler as ex
from experiments import qos_carbon as qc
from hpc_sim import CarbonSignal, carbon_cost_gco2e
from hpc_sim.metrics import Distribution


OUTPUT = qc.OUTPUT / "robustness"
POLICIES = {"EASY": ex.EASY, **qc.POLICIES}
ABLATIONS = {
    "improved": qc.IMPROVED,
    "without_replanning": replace(qc.IMPROVED, mode="once"),
    "without_benefit_rate": replace(qc.IMPROVED, objective="carbon"),
    "without_threshold": replace(qc.IMPROVED, threshold=0.),
    "with_delay_scaling": replace(qc.IMPROVED, alpha=1.),
    "with_congestion": replace(qc.IMPROVED, mode="congestion", objective="carbon", congestion_price=1.),
}
# A congestion penalty in grams cannot be added to a benefit-rate ratio. Its
# control uses the existing carbon objective, with the same frozen threshold.
CONTRASTS = {
    "replanning": ("without_replanning", "improved"),
    "benefit_rate": ("without_benefit_rate", "improved"),
    "threshold": ("without_threshold", "improved"),
    "delay_scaling": ("improved", "with_delay_scaling"),
    "congestion_awareness": ("without_benefit_rate", "with_congestion"),
}


def windows(jobs) -> list[ex.Scenario]:
    """Four disjoint release windows; the final window ends with the cohort."""
    start = min(job.release_time for job in jobs)
    end = max(job.release_time for job in jobs) + timedelta(microseconds=1)
    bounds = [start + timedelta(days=2 * index) for index in range(4)] + [end]
    if bounds[-2] >= end:
        raise ValueError("four release windows require more than six days of workload")
    return [ex.Scenario("test", window_start=a.isoformat(), window_end=b.isoformat())
            for a, b in zip(bounds, bounds[1:])]


def matrix(window_scenarios) -> list[tuple]:
    entries = []

    def sweep(study, case, scenario, cells=(ex.REALISTIC,)):
        for cell in cells:
            for name, policy in POLICIES.items():
                for hours in ((0.,) if name == "EASY" else ex.DELAY_SWEEP):
                    config = policy if name == "EASY" else replace(policy, delay_hours=hours)
                    entries.append((study, case, name, ex.task(scenario, config, cell)))

    for index, scenario in enumerate(window_scenarios, 1):
        sweep("temporal", f"W{index}", scenario, (ex.ORACLE, ex.REALISTIC))
    for nodes in (880, 512, 256):
        sweep("capacity", str(nodes), ex.Scenario("test", nodes=nodes))
    for grid in (15, 60, 240):
        sweep("granularity", str(grid), ex.Scenario("test", grid_minutes=grid))
    for contention in (False, True):
        sweep("contention", "terminal" if contention else "clean", ex.Scenario("test", contention=contention))
    for nodes in (880, 256):
        for name, policy in {"EASY": ex.EASY, **ABLATIONS}.items():
            entries.append(("ablation", str(nodes), name, ex.task(ex.Scenario("test", nodes=nodes), policy)))
    return entries


def paired_jobs(frame, baseline, decisions) -> pd.DataFrame:
    joined = frame.merge(baseline[["job_id", "waiting_s", "bounded_slowdown", "emissions_gco2e"]],
                        on="job_id", how="outer", suffixes=("", "_easy"),
                        validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("paired fairness requires identical job ids")
    joined["added_wait_s"] = joined.waiting_s - joined.waiting_s_easy
    joined["saved_gco2e"] = joined.emissions_gco2e_easy - joined.emissions_gco2e
    joined["voluntary_delay_s"] = joined.job_id.map(ex.voluntary_delay(decisions, frame))
    return joined


def describe(frame: pd.DataFrame) -> dict:
    """Job-weighted summaries, using the project's nearest-rank convention."""
    result = {"jobs": len(frame), "total_emissions_gco2e": float(frame.emissions_gco2e.sum()),
              "energy_kwh": float(frame.energy_kwh.sum())}
    quantities = {"waiting": frame.waiting_s,
                  "turnaround": frame.waiting_s + frame.actual_duration_s,
                  "bounded_slowdown": frame.bounded_slowdown,
                  "added_wait": frame.added_wait_s,
                  "voluntary_delay": frame.voluntary_delay_s}
    for name, values in quantities.items():
        distribution = Distribution.of(values.tolist())
        for stat, field in (("mean", "mean"), ("p95", "p95"), ("p99", "p99"), ("max", "maximum")):
            result[f"{name}_{stat}" + ("" if name == "bounded_slowdown" else "_s")] = getattr(distribution, field)
    extra = frame.added_wait_s.clip(lower=0).sort_values(ascending=False)
    tail_count = max(1, int(np.ceil(.01 * len(extra))))
    tail_share, reason = qc.ratio(float(extra.iloc[:tail_count].sum()), float(extra.sum()))
    return result | {
        "emissions_saved_kg": float(frame.saved_gco2e.sum() / 1000),
        "emissions_saved_pct": 100 * qc.ratio(float(frame.saved_gco2e.sum()), float(frame.emissions_gco2e_easy.sum()))[0],
        "worse_wait_share": float((frame.added_wait_s > 1e-6).mean()),
        "extra_wait_gt_1h_share": float((frame.added_wait_s > 3600).mean()),
        "extra_wait_gt_6h_share": float((frame.added_wait_s > 21600).mean()),
        "held_share": float((frame.voluntary_delay_s > 0).mean()),
        "top_1pct_positive_extra_wait_share": tail_share,
        "top_1pct_positive_extra_wait_reason": reason,
    }


def classes(frame):
    # Fixed physical cutoffs, never estimated from policy outcomes.
    duration = np.where(frame.actual_duration_s < 3600, "short", "long")
    size = np.where(frame.nodes <= 8, "small", "large")
    for dimension, labels in (("duration", duration), ("size", size),
                              ("duration_size", np.char.add(np.char.add(duration, "/"), size))):
        for label in sorted(set(labels)):
            members = frame[labels == label]
            yield {"dimension": dimension, "class": label, **describe(members)}


def ablation_effects(summary):
    rows = []
    for case, group in summary[summary.study == "ablation"].groupby("case"):
        lookup = group.set_index("policy")
        for mechanism, (before, after) in CONTRASTS.items():
            a, b = lookup.loc[before], lookup.loc[after]
            rows.append({"nodes": int(case), "mechanism": mechanism, "before": before, "after": after,
                         "before_run_key": a.run_key, "after_run_key": b.run_key,
                         "extra_saved_kg": (a.total_emissions_gco2e - b.total_emissions_gco2e) / 1000,
                         "saving_delta_pp": b.emissions_saved_pct - a.emissions_saved_pct,
                         **{f"delta_{metric}": b[metric] - a[metric] for metric in (
                             "waiting_mean_s", "waiting_p95_s", "waiting_p99_s", "waiting_max_s",
                             "bounded_slowdown_p95", "bounded_slowdown_p99", "bounded_slowdown_max")}})
    return pd.DataFrame(rows)


def run(arguments):
    started = perf_counter()
    protocol = qc.preflight()
    actual_jobs, _ = ex.cohort("test")
    window_scenarios = windows(actual_jobs)
    entries = matrix(window_scenarios)
    tasks = list(dict.fromkeys(entry[3] for entry in entries))
    protocol |= {
        "command": "python -m experiments.robustness run",
        "window_scenarios": [asdict(scenario) for scenario in window_scenarios],
        "capacities": [880, 512, 256], "decision_grids_minutes": [15, 60, 240],
        "ablations": {name: asdict(policy) for name, policy in ABLATIONS.items()},
        "ablation_contrasts": CONTRASTS,
        "class_definitions": {"short": "actual runtime < 3600 s", "long": ">= 3600 s",
                              "small": "requested nodes <= 8", "large": "> 8"},
        "windows": "half-open release windows, 2 days; last truncated to cohort end; cold start and full drain",
        "continuous_control": "the same release cohorts scored within full-trace schedules; no reset",
        "variability": "min/median/max across four windows; descriptive, not IID confidence intervals",
        "saving_reference": "EASY with identical cohort, capacity, grid and runtime information; main_delay.csv also retains the original common-EASY columns",
        "retention_reference": "only in main_delay.csv, as defined by the main QoS/carbon protocol",
        "reused_runs": sum((ex.run_directory(item)[1] / "run.json").exists() for item in tasks),
        "unique_runs": len(tasks), "workers": arguments.workers,
    }
    # Written before simulation: choices and contrasts are reviewable, not fitted to results.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design_path = OUTPUT / "design.json"
    if design_path.exists():
        frozen = json.loads(design_path.read_text())
        for key in ("input_sha256", "selection_sha256", "policies", "window_scenarios", "capacities",
                    "decision_grids_minutes", "ablations", "ablation_contrasts", "class_definitions"):
            if json.dumps(frozen[key], sort_keys=True) != json.dumps(protocol[key], sort_keys=True):
                raise ValueError(f"{key} differs from the frozen robustness design")
    else:
        qc.write_json(design_path, protocol)
    protocol["design_sha256"] = ex.sha256(design_path)
    raw = {row["run_key"]: qc.augment(row) for row in ex.run_all(tasks, arguments.workers)}
    frame_cache = {}

    def frames(row):
        key = row["run_key"]
        if key not in frame_cache:
            frame_cache[key] = qc.schedules(row)
        return frame_cache[key]

    def reference(item):
        scenario, information, _, _ = item
        key = ex.run_directory(ex.task(scenario, ex.EASY, (information, "actual")))[0]
        return raw[key]

    summary, fairness, grouped = [], [], []
    for study, case, name, item in entries:
        row = raw[ex.run_directory(item)[0]]
        easy = reference(item)
        frame, decisions = frames(row)
        baseline = frames(easy)[0]
        paired = paired_jobs(frame, baseline, decisions)
        if not np.isclose(frame.energy_kwh.sum(), baseline.energy_kwh.sum(), rtol=1e-10):
            raise AssertionError("paired schedules changed physical energy")
        if name != "EASY" and item[3].delay_hours == 0:
            if not (paired.added_wait_s == 0).all():
                raise AssertionError("zero delay does not reproduce matched EASY")
        context = {"study": study, "case": case, "policy": name, "run_key": row["run_key"],
                   "information_scenario": "perfect_information" if item[1] == "actual" else "realistic",
                   "delay_hours": 0. if name == "EASY" else item[3].delay_hours,
                   "nodes": item[0].nodes, "grid_minutes": item[0].grid_minutes}
        savings = {key: value for key, value in qc.saving_columns(row, easy, easy).items()
                   if "_common_" not in key and key != "common_easy_run_key"}
        summary.append(row | savings | context)
        fairness.append(context | describe(paired))
        if context["delay_hours"] in (0., 6., 24.) or study == "ablation":
            grouped.extend(context | group for group in classes(paired))
    summary = pd.DataFrame(summary)
    # The existing full information matrix is copied, never re-transcribed.
    main = pd.read_csv(qc.OUTPUT / "delay_summary.csv")
    main_protocol = json.loads((qc.OUTPUT / "protocol.json").read_text())
    if ex.sha256(qc.OUTPUT / "delay_summary.csv") != main_protocol["csv_sha256"]["delay_summary.csv"]:
        raise ValueError("the main delay summary differs from its frozen protocol")
    window_rows, signal_rows, continuous = [], [], []
    actual, _, reach = ex.carbon()
    for index, scenario in enumerate(window_scenarios, 1):
        jobs, _, _ = ex.scenario_jobs(scenario, "actual")
        start, end = pd.Timestamp(scenario.window_start), pd.Timestamp(scenario.window_end)
        samples = actual.get_actual_range(start.floor("15min").to_pydatetime(), end.ceil("15min").to_pydatetime())
        intensities = np.array([sample.intensity_gco2e_per_kwh for sample in samples])
        key = ex.run_directory(ex.task(scenario, ex.EASY))[0]
        easy = raw[key]
        window_rows.append({"window": f"W{index}", "release_start": start, "release_end_exclusive": end,
                            "jobs": len(jobs), "release_span_hours": (end - start).total_seconds() / 3600,
                            "ci_mean_g_per_kwh": intensities.mean(), "ci_std_g_per_kwh": intensities.std(),
                            "ci_min_g_per_kwh": intensities.min(), "ci_max_g_per_kwh": intensities.max(),
                            "ci_cv": intensities.std() / intensities.mean(), "ci_buckets": len(samples),
                            "node_hours": sum(j.nodes_required * j.actual_duration_seconds / 3600 for j in jobs),
                            "easy_utilisation": easy["utilisation"]})
        signal_rows.extend({"window": f"W{index}", "timestamp": sample.timestamp,
                            "intensity_g_per_kwh": sample.intensity_gco2e_per_kwh} for sample in samples)
        for _, row in main[main.information_scenario.isin(("realistic", "perfect_information"))].drop_duplicates(
                ["information_scenario", "run_key"]).iterrows():
            frame, decisions = frames(row)
            baseline = frames({"run_key": row.easy_run_key})[0]
            paired = paired_jobs(frame, baseline, decisions)
            members = paired[(paired.release_time >= start) & (paired.release_time < end)]
            continuous.append({"window": f"W{index}", "policy": row.policy,
                               "information_scenario": row.information_scenario, "delay_hours": row.delay_hours,
                               "run_key": row.run_key, **describe(members)})
    # One deterministic job illustrates CI integration, without selecting a best case.
    example = next(job for job in actual_jobs if 3600 <= job.actual_duration_seconds <= 7200)
    signal = CarbonSignal(actual)
    costs = pd.DataFrame([{"job_id": example.job_id, "start_time": example.release_time + timedelta(minutes=15 * n),
                          "delay_hours": n / 4, "runtime_s": example.actual_duration_seconds,
                          "planning_emissions_g": carbon_cost_gco2e(example, example.release_time + timedelta(minutes=15 * n), signal)}
                         for n in range(97)])
    # Fixed editorial examples, not a newly selected winner on the test results.
    chosen = main[(main.information_scenario == "realistic") & (
        ((main.policy == "EASY") & (main.delay_hours == 0)) |
        ((main.policy == "greedy") & (main.delay_hours == 6)) |
        ((main.policy == "improved") & main.delay_hours.isin((1, 6, 24))))].copy()
    outputs = {"summary": summary, "fairness": pd.DataFrame(fairness), "job_classes": pd.DataFrame(grouped),
               "ablation_effects": ablation_effects(summary), "windows": pd.DataFrame(window_rows),
               "carbon_signal": pd.DataFrame(signal_rows), "job_start_cost": costs,
               "temporal_continuous": pd.DataFrame(continuous), "main_delay": main,
               "representative_configurations": chosen}
    if sum(row["jobs"] for row in window_rows) != len(actual_jobs):
        raise AssertionError("release windows do not partition the complete test cohort")
    temporal = summary[(summary.study == "temporal") & (summary.policy != "EASY")]
    outputs["temporal_variability"] = temporal.groupby(["information_scenario", "policy", "delay_hours"]).agg(
        windows=("case", "nunique"), saving_min_pct=("emissions_saved_pct", "min"),
        saving_median_pct=("emissions_saved_pct", "median"), saving_max_pct=("emissions_saved_pct", "max"),
        wait_mean_min_s=("waiting_mean_s", "min"), wait_mean_max_s=("waiting_mean_s", "max"),
        wait_p99_min_s=("waiting_p99_s", "min"), wait_p99_max_s=("waiting_p99_s", "max"),
        bsld_p99_min=("bounded_slowdown_p99", "min"), bsld_p99_max=("bounded_slowdown_p99", "max")).reset_index()
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    all_keys = set(raw) | set(main.run_key)
    protocol |= {"elapsed_seconds": perf_counter() - started, "forecast_reach_hours": reach.total_seconds() / 3600,
                 "main_protocol_sha256": ex.sha256(qc.OUTPUT / "protocol.json"),
                 "run_artifact_sha256": {qc.relative(ex.OUTPUT / "runs" / key / filename): ex.sha256(ex.OUTPUT / "runs" / key / filename)
                                          for key in sorted(all_keys) for filename in ("run.json", "jobs.parquet", "decisions.parquet")},
                 "csv_sha256": {f"{name}.csv": ex.sha256(OUTPUT / f"{name}.csv") for name in outputs}}
    qc.write_json(OUTPUT / "protocol.json", protocol)
    report(arguments)


def report(arguments):
    """Build final figures and thesis tables from CSVs alone."""
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    for name, digest in protocol["csv_sha256"].items():
        if ex.sha256(OUTPUT / name) != digest:
            raise ValueError(f"{name} differs from the recorded experiment")
    read = lambda name: pd.read_csv(OUTPUT / f"{name}.csv")
    summary, main, fair, groups, win, effects, selected = [read(name) for name in (
        "summary", "main_delay", "fairness", "job_classes", "windows", "ablation_effects", "representative_configurations")]
    plt = ex._plt()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    directory = OUTPUT / "figures"
    directory.mkdir(exist_ok=True)
    colors = {"EASY": ex.MUTED, "greedy": ex.PALETTE[1], "improved": ex.PALETTE[0]}

    def save(figure, name):
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(directory / f"{name}.{suffix}", dpi=220, bbox_inches="tight",
                           metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else {})
        plt.close(figure)

    def curves(ax, frame, metric, ylabel, scale=1):
        for name in POLICIES:
            points = frame[frame.policy == name].sort_values("delay_hours")
            if name == "EASY":
                ax.axhline(points[metric].iloc[0] * scale, color=colors[name], linestyle="--", label=name)
            else:
                ax.plot(points.delay_hours, points[metric] * scale, "s--" if name == "greedy" else "o-",
                        color=colors[name], label=name)
        ax.set(xlabel="Nominal delay budget (h)", ylabel=ylabel, xticks=ex.DELAY_SWEEP)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    for ax, (scenario, title) in zip(axes.flat, (
        ("perfect_information", "Perfect information"), ("forecast_only_error", "Forecast error only"),
        ("job_model_only_error", "Job-model error only"), ("realistic", "Realistic"))):
        curves(ax, main[main.information_scenario == scenario], "emissions_saved_pct", "Emissions saved vs EASY (%)")
        ax.set_title(title)
    axes[0, 0].legend()
    save(figure, "delay_budget_vs_emissions_saved")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    realistic = main[main.information_scenario == "realistic"]
    for ax, (metric, label, scale) in zip(axes.flat, (
        ("waiting_mean_s", "Mean waiting (h)", 1 / 3600), ("waiting_p95_s", "Waiting p95 (h)", 1 / 3600),
        ("waiting_p99_s", "Waiting p99 (h)", 1 / 3600), ("waiting_max_s", "Maximum waiting (h)", 1 / 3600),
        ("bounded_slowdown_p99", "Bounded slowdown p99", 1), ("bounded_slowdown_max", "Maximum bounded slowdown", 1))):
        curves(ax, realistic, metric, label, scale)
        if metric.startswith("bounded"):
            ax.set_yscale("log")
            ax.set_ylabel(label + " (log)")
    axes[0, 0].legend()
    figure.suptitle("Realistic information · 23,560 jobs · 880 nodes · 15 min grid")
    save(figure, "delay_budget_vs_qos")

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    temporal = summary[(summary.study == "temporal") & (summary.information_scenario == "realistic")]
    for ax, (window, group) in zip(axes.flat, temporal.groupby("case")):
        curves(ax, group, "emissions_saved_pct", "Emissions saved vs local EASY (%)")
        info = win.set_index("window").loc[window]
        ax.set_title(f"{window}: {info.jobs:,} jobs · CI {info.ci_mean_g_per_kwh:.0f} ± {info.ci_std_g_per_kwh:.0f} g/kWh")
    axes[0, 0].legend()
    save(figure, "temporal_tradeoff")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, (study, label) in enumerate((("capacity", "Nodes"), ("granularity", "Grid (min)"), ("contention", "Demand"))):
        data = summary[(summary.study == study) & (summary.policy == "improved")]
        for case, members in data.groupby("case", sort=False):
            members = members.sort_values("delay_hours")
            axes[0, column].plot(members.delay_hours, members.emissions_saved_pct, "o-", label=f"{label}: {case}")
            axes[1, column].plot(members.delay_hours, members.waiting_p99_s / 3600, "o-", label=f"{label}: {case}")
        axes[0, column].set(ylabel="Saved vs matching EASY (%)", title=study.capitalize())
        axes[1, column].set(ylabel="Waiting p99 (h)")
        for ax in axes[:, column]:
            ax.set(xlabel="Nominal delay budget (h)", xticks=ex.DELAY_SWEEP)
        axes[0, column].legend(fontsize=8)
    save(figure, "sensitivity")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, (nodes, members) in enumerate(effects.groupby("nodes", sort=False)):
        x = np.arange(len(members)) + (index - .5) * .35
        axes[0].bar(x, members.saving_delta_pp, .35, label=f"{nodes} nodes")
        axes[1].bar(x, members.delta_waiting_p99_s / 3600, .35, label=f"{nodes} nodes")
    for ax in axes:
        ax.set_xticks(np.arange(len(CONTRASTS)), [name.replace("_", "\n") for name in CONTRASTS], fontsize=8)
        ax.axhline(0, color=ex.MUTED, linewidth=.8)
    axes[0].set(ylabel="Saving change (percentage points)", title="Enable one mechanism")
    axes[1].set(ylabel="Waiting p99 change (h)", title="QoS effect (negative is better)")
    axes[0].legend()
    save(figure, "ablation")

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    class_points = groups[(groups.study == "capacity") & (groups.case.astype(str) == "880") &
                          ((groups.policy == "EASY") | (groups.delay_hours == 6))]
    for column, (dimension, labels) in enumerate((("duration", ("short", "long")), ("size", ("small", "large")))):
        for index, policy in enumerate(POLICIES):
            members = class_points[(class_points.dimension == dimension) & (class_points.policy == policy)].set_index("class").loc[list(labels)]
            x = np.arange(2) + (index - 1) * .24
            axes[0, column].bar(x, members.emissions_saved_pct, .24, label=policy, color=colors[policy])
            axes[1, column].bar(x, members.bounded_slowdown_p99, .24, label=policy, color=colors[policy])
        for ax in axes[:, column]:
            ax.set_xticks(range(2), labels)
        axes[0, column].set(ylabel="Saved vs class-matched EASY (%)", title=dimension.capitalize())
        axes[0, column].axhline(0, color=ex.MUTED, linewidth=.8)
        axes[1, column].set(ylabel="Bounded slowdown p99 (log)", yscale="log")
    axes[0, 0].legend()
    figure.suptitle("Job classes · 6 h nominal budget · realistic information · 880 nodes")
    save(figure, "job_classes")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    signal, costs = read("carbon_signal"), read("job_start_cost")
    for window, points in signal.groupby("window"):
        axes[0].plot(pd.to_datetime(points.timestamp, utc=True), points.intensity_g_per_kwh, linewidth=1, label=window)
    axes[0].set(ylabel="Actual CI (gCO₂e/kWh)", xlabel="UTC date")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(ncol=2)
    axes[1].plot(costs.delay_hours, costs.planning_emissions_g, color=ex.PALETTE[0])
    axes[1].set(xlabel="Shift from release (h)", ylabel="Planning emissions (gCO₂e)",
                title=f"Job {costs.job_id.iloc[0]} · runtime {costs.runtime_s.iloc[0] / 3600:.2f} h")
    save(figure, "carbon_intensity_and_start_cost")

    table = ex._markdown
    brief = {"policy": "policy", "budget h": "delay_hours:g", "saved %": "emissions_saved_pct:.3f",
             "wait mean s": "waiting_mean_s:.1f", "wait p95 s": "waiting_p95_s:.0f",
             "wait p99 s": "waiting_p99_s:.0f", "wait max s": "waiting_max_s:.0f",
             "BSLD p99": "bounded_slowdown_p99:.2f", "BSLD max": "bounded_slowdown_max:.2f"}
    selected_table = table(selected, {"CO₂ t": "total_emissions_tco2e:.5f", **brief})
    (OUTPUT / "representative_configurations.md").write_text(selected_table + "\n")
    latex = [r"\begin{tabular}{lrrrrrrrr}", r"\hline",
             r"Policy & Budget (h) & CO$_2$e (t) & Saved (\%) & Mean wait (s) & p95 (s) & p99 (s) & Max (s) & BSLD p99 \\", r"\hline"]
    for row in selected.itertuples():
        values = (row.delay_hours, row.total_emissions_tco2e, row.emissions_saved_pct, row.waiting_mean_s,
                  row.waiting_p95_s, row.waiting_p99_s, row.waiting_max_s, row.bounded_slowdown_p99)
        latex.append(row.policy + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")
    (OUTPUT / "representative_configurations.tex").write_text("\n".join(latex + [r"\hline", r"\end{tabular}", ""]))
    six = temporal[(temporal.delay_hours == 6) & (temporal.policy != "EASY")]
    tail = fair[(fair.study == "capacity") & (fair.case.astype(str) == "880") &
                ((fair.policy == "EASY") | (fair.delay_hours == 6))]
    class_six = groups[(groups.study == "capacity") & (groups.case.astype(str) == "880") &
                       ((groups.policy == "EASY") | (groups.delay_hours == 6)) & (groups.dimension != "duration_size")]
    by_class = class_six.set_index(["policy", "class"])
    by_policy = tail.set_index("policy")
    large = by_class.loc[("improved", "large")]
    full_six = main[(main.information_scenario == "realistic") & (main.delay_hours == 6)].set_index("policy")
    improved, greedy = full_six.loc["improved"], full_six.loc["greedy"]
    variability = read("temporal_variability")
    interval = variability[(variability.information_scenario == "realistic") &
                           (variability.policy == "improved") & (variability.delay_hours == 6)].iloc[0]
    stress = summary[(summary.study == "capacity") & (summary.policy == "improved") &
                     (summary.delay_hours == 6)].set_index("nodes").loc[256]
    intermediate = summary[(summary.study == "capacity") & (summary.policy == "improved") &
                           (summary.delay_hours == 6)].set_index("nodes").loc[512]
    grids = summary[(summary.study == "granularity") & (summary.policy == "improved") &
                    (summary.delay_hours == 6)].sort_values("grid_minutes")
    continuous = read("temporal_continuous")
    same_windows = continuous[(continuous.policy == "improved") & (continuous.delay_hours == 6) &
                             (continuous.information_scenario == "realistic")].set_index("window")
    resets = six[six.policy == "improved"].set_index("case").emissions_saved_pct - same_windows.emissions_saved_pct
    parts = ["# Robustezza e output finali per la tesi", "",
        "Validazione retrospettiva delle policy esistenti, senza nuove feature dello scheduler, refit o selezione sul test. "
        "Le configurazioni e i contrasti sono registrati prima delle simulazioni in `design.json`; "
        "i CSV, gli schedule per job e gli input sono identificati da hash SHA-256 in `protocol.json`.", "",
        f"Sul test completo, a 6 h improved risparmia **{improved.emissions_saved_pct:.3f}%** rispetto a EASY "
        f"contro {greedy.emissions_saved_pct:.3f}% della greedy, con attesa media inferiore del "
        f"{100 * (1 - improved.waiting_mean_s / greedy.waiting_mean_s):.1f}%. "
        f"Tuttavia il p99 del waiting è {improved.waiting_p99_s:.0f} s contro {greedy.waiting_p99_s:.0f} s "
        f"e il massimo {improved.waiting_max_s:.0f} s contro {greedy.waiting_max_s:.0f} s: "
        "il vantaggio medio non è un miglioramento uniforme delle code.", "",
        f"A 6 h il saving di improved nelle quattro finestre varia da **{interval.saving_min_pct:.3f}% "
        f"a {interval.saving_max_pct:.3f}%**, con mediana {interval.saving_median_pct:.3f}%. "
        f"Sulla coorte completa il saving è {intermediate.emissions_saved_pct:.3f}% a 512 nodi "
        f"e {stress.emissions_saved_pct:.3f}% a 256 nodi; in quest'ultimo caso il p99 del waiting "
        f"raggiunge {stress.waiting_p99_s / 3600:.2f} h. Capacità, periodo e code QoS vanno quindi "
        "considerati insieme al saving del caso nominale.", "",
        "## Configurazioni rappresentative", "", selected_table, "",
        "Esempi editoriali fissati: EASY, greedy a 6 h, improved a 1/6/24 h. Non sono una nuova graduatoria ottimizzata sul test. "
        "Il riferimento è sempre EASY con le stesse informazioni sui runtime, coorte e capacità. "
        "Un saving negativo indica più emissioni di EASY. La tabella è disponibile anche in CSV e LaTeX.", "",
        "![Delay budget vs emissions saved](figures/delay_budget_vs_emissions_saved.png)", "",
        "![Delay budget vs QoS](figures/delay_budget_vs_qos.png)", "",
        "## Finestre temporali e carbon intensity", "",
        "Quattro finestre contigue e disgiunte per release coprono tutti i job del test; le prime tre durano 48 h, "
        "l'ultima finisce con la coorte. Ogni simulazione riparte vuota e viene drenata fino all'ultimo completamento: "
        "nessun job o ritardo viene censurato alla fine della finestra. I bucket CI includono i bordi arrotondati a 15 min.", "",
        table(win, {"window": "window", "start UTC": "release_start", "end UTC (exclusive)": "release_end_exclusive",
                    "jobs": "jobs:d", "hours": "release_span_hours:.2f", "CI mean": "ci_mean_g_per_kwh:.2f",
                    "CI std": "ci_std_g_per_kwh:.2f", "EASY utilisation": "easy_utilisation:.1%"}), "",
        table(six, {"window": "case", **brief}), "",
        "![Finestre](figures/temporal_tradeoff.png)", "",
        "`temporal_variability.csv` riporta min/mediana/max tra finestre per ciascun budget e policy, "
        "sia con informazione perfetta sia realistica. Sono misure descrittive: quattro finestre vicine, "
        "con possibili code di esecuzione sovrapposte, non sono repliche IID. Cambiano sia CI sia mix/carico dei job; "
        "l'associazione non identifica un effetto causale della sola CI. `temporal_continuous.csv` valuta gli stessi "
        "gruppi di release nello schedule completo, conservando la contesa tra finestre e misurando l'effetto del reset. "
        f"Per improved a 6 h la differenza cold-start meno continuo nel saving locale varia da "
        f"{resets.min():+.3f} a {resets.max():+.3f} punti percentuali. Le percentuali usano ciascuna il proprio EASY; "
        "il CSV conserva anche le emissioni e i saving assoluti per confrontare i due contesti.", "",
        "![CI e costo del job](figures/carbon_intensity_and_start_cost.png)", "",
        "Il job illustrativo è il primo cronologico con durata reale tra 1 e 2 h; il costo integra CI e potenza media "
        "senza vincoli di capacità o reach del forecast. È un esempio del modello di costo, non uno schedule "
        "né una nuova misura di saving.", "",
        "## Capacity, contention e decision granularity", "",
        "La coorte completa resta identica a 880/512/256 nodi: nessun job valutato è escluso. "
        "Le griglie 15/60/240 min cambiano i candidati e i risvegli della policy; la risoluzione CI resta 15 min "
        "e il simulatore conserva timestamp continui. Arrivi e completamenti possono comunque innescare decisioni "
        "fuori griglia. EASY è abbinato a ogni scenario: variando capacità cambiano anche i suoi start e le sue emissioni, "
        "quindi il denominatore del saving relativo. "
        "La domanda terminale non-COMPLETED è aggiunta a 880 nodi e conta soltanto per le risorse; "
        "le emissioni restano quelle dei COMPLETED, non del sistema completo. L'utilizzo include tutti i job simulati.", "",
        "A 6 h, passando da 15 a 60 a 240 min, improved risparmia rispettivamente "
        + ", ".join(f"{value:.3f}%" for value in grids.emissions_saved_pct)
        + "; l'attesa media è " + ", ".join(f"{value / 60:.1f}" for value in grids.waiting_mean_s)
        + " min. La griglia è quindi parte del trade-off: una riduzione del waiting può accompagnarsi "
        "a un minor beneficio carbonico. Nessun parametro viene riselezionato usando questo sweep.", "",
        table(summary[(summary.study.isin(("capacity", "contention", "granularity"))) &
                      ((summary.policy == "EASY") | (summary.delay_hours == 6))], {"study": "study", "case": "case", **brief,
                                                   "utilisation": "utilisation:.1%", "resource jobs": "resource_only_jobs:d"}), "",
        "![Sensitivity](figures/sensitivity.png)", "",
        "## Ablation dei cinque meccanismi", "",
        "A 6 h su 880 e 256 nodi, i contrasti cambiano una scelta alla volta, con la stessa coorte realistica. "
        "Replanning significa aggiornamento sugli snapshot nuovi (once → receding), non `replan_on_events`, "
        "che resta disattivo. Benefit-rate confronta il massimo ΔCO₂/ora con il minimo costo carbonico, "
        "mantenendo la soglia. Threshold confronta 0 con θ=0.014930658589279368 gCO₂e/job-hour. "
        "Delay scaling confronta il budget fisso con min(runtime previsto, budget), α=1. "
        "Congestion awareness confronta il minimo costo con e senza prezzo storico 1, scope=start. "
        "Quest'ultimo contrasto usa l'obiettivo carbonico: il codice rifiuta una penalità in grammi aggiunta a un rapporto. "
        "L'improved selezionato non contiene congestion awareness o delay scaling: questi due sono controlli aggiuntivi, "
        "non rimozioni da una inesistente configurazione che combina tutto. Gli effetti non sono additivi.", "",
        table(effects, {"nodes": "nodes:d", "mechanism": "mechanism", "extra saved kg": "extra_saved_kg:.3f",
                        "Δ saving pp": "saving_delta_pp:+.3f", "Δ wait mean s": "delta_waiting_mean_s:+.1f",
                        "Δ wait p99 s": "delta_waiting_p99_s:+.0f", "Δ max wait s": "delta_waiting_max_s:+.0f"}), "",
        "![Ablation](figures/ablation.png)", "",
        "## Fairness, tail QoS e classi di job", "",
        "Waiting e turnaround partono da eligible/release; BSLD=max(1,(wait+runtime)/max(runtime,10 s)). "
        "P95 e p99 usano nearest rank, coerente con il simulatore. `fairness.csv` include media, p95, p99 e massimo "
        "di waiting, turnaround, BSLD, attesa aggiunta rispetto allo stesso job in EASY e ritardo volontario. "
        "La quota top-1% misura quanta attesa aggiuntiva positiva ricade sul peggior 1% dei job (arrotondato per eccesso); "
        "resta non definita con denominatore nullo. Non è una metrica di equità tra utenti e non dimostra assenza di starvation.", "",
        table(tail, {**brief, "worse wait": "worse_wait_share:.1%", "extra >6h": "extra_wait_gt_6h_share:.1%",
                     "top 1% burden": "top_1pct_positive_extra_wait_share:.1%"}), "",
        f"Improved riduce la quota di job con attesa maggiore di EASY dal "
        f"{by_policy.loc['greedy', 'worse_wait_share']:.1%} della greedy al "
        f"{by_policy.loc['improved', 'worse_wait_share']:.1%}. La quota di attesa aggiuntiva positiva "
        f"concentrata nel peggior 1% passa però da {by_policy.loc['greedy', 'top_1pct_positive_extra_wait_share']:.1%} "
        f"a {by_policy.loc['improved', 'top_1pct_positive_extra_wait_share']:.1%}: è una concentrazione relativa "
        "su una platea penalizzata più piccola, non da sola una prova di maggior danno assoluto.", "",
        "Classi fissate prima dell'analisi: short <1 h e long ≥1 h di runtime reale; small ≤8 nodi e large >8. "
        "Il runtime reale definisce solo gruppi ex post, non informazioni per lo scheduler. "
        "Ogni percentuale di saving usa l'EASY degli stessi job della classe. `job_classes.csv` include anche "
        "le quattro intersezioni non vuote, i budget 0/6/24 h, tutte le sensitivity e le ablation; "
        "ogni dimensione partiziona la coorte. Classi piccole producono p99 prossimi al massimo; "
        "i conteggi accompagnano sempre i quantili.", "",
        table(class_six, {"class": "class", "jobs": "jobs:d", **brief}), "",
        f"I {int(large.jobs):,} job large ({large.jobs / protocol['cohort']['jobs']:.1%} della coorte) "
        f"producono {large.emissions_saved_kg:.2f} kg, ossia "
        f"{large.emissions_saved_kg / improved.emissions_saved_kg:.1%} del saving di improved, "
        f"ma {large.worse_wait_share:.1%} di essi aspetta più di EASY. "
        f"Fra gli short il p99 BSLD scende da {by_class.loc[('greedy', 'short'), 'bounded_slowdown_p99']:.1f} "
        f"a {by_class.loc[('improved', 'short'), 'bounded_slowdown_p99']:.1f}, rimanendo sopra "
        f"{by_class.loc[('EASY', 'short'), 'bounded_slowdown_p99']:.1f} di EASY. "
        f"Per i long il saving cambia segno: {by_class.loc[('greedy', 'long'), 'emissions_saved_kg']:+.2f} kg "
        f"con greedy, {by_class.loc[('improved', 'long'), 'emissions_saved_kg']:+.2f} kg con improved. "
        "Durata e dimensione descrivono oneri differenti; le medie globali non li rappresentano da sole.", "",
        "![Classi di job](figures/job_classes.png)", "",
        "## Assumptions, limitations e threats to validity", "",
        "- Operational carbon dei profili dei nodi; niente embodied carbon, overhead esterno/PUE o idle dei nodi liberi. "
        "La baseline presente nei profili è già attribuita ai job. L'energia dei medesimi job è invariata allo start.",
        "- Cluster omogeneo, risorsa espressa in nodi, job non preemptive; nessuna migrazione, DVFS o dipendenza "
        "di runtime/potenza dallo start. 880 è la capacità nominale ricostruita dagli ID, non una misura di disponibilità "
        "istantanea; 512/256 sono stress test. 774/787 non sono nuove misure sperimentali di questa matrice.",
        "- Decisioni realistiche: runtime/potenza previsti e snapshot emessi non dopo la decisione. "
        "Runtime reale per restituire risorse; profili misurati e CI reale solo per la valutazione. "
        "L'oracle usa durata e potenza media reali e CI futura; non ottimizza il profilo misurato e non è un bound.",
        "- Il budget limita il rinvio volontario, non l'attesa da contesa. Il reach condiviso è 23 h meno il runtime "
        "di planning: 24 h nominali non significano 24 h disponibili. Il fallback senza forecast offre il job a EASY "
        "ed è contato in `forecast_unavailable_decisions`, particolarmente rilevante a fine archivio.",
        "- Filtering COMPLETED/profili validi, avvio a cluster vuoto, nessun backlog anteriore alla coorte; "
        "domanda terminale parziale con larghezza osservata e durata di planning da time limit. Il confronto contention "
        "non ricostruisce tutti i job esclusi o l'energia dei terminali.",
        "- Un sito, una zona elettrica e un periodo breve del 2020; test già consultato nei round precedenti. "
        "Finestre e sensitivity sono descrittive, non un nuovo holdout né evidenza di robustezza stagionale. "
        "Il massimo descrive solo i job osservati; non dà garanzie QoS future.",
        "- La CI storica rappresenta il segnale della cache, non emissioni marginali causalmente evitate; "
        "l'accounting campiona CI all'inizio dei segmenti di potenza nominalmente da 20 s. "
        "Le predizioni restano congelate; nessun intervallo IID su job correlati, nessuna significatività inferenziale dichiarata.", "",
        "## Riproduzione", "",
        "```bash\nmake robustness ARGS='--workers 2'\nmake robustness-report\n"
        ".venv/bin/python tests/check_experiments_robustness.py\n```", "",
        f"{protocol['unique_runs']} simulazioni distinte richieste, {protocol['reused_runs']} già in cache all'avvio. "
        "Schedule e decisioni restano in `data/experiments/improved_scheduler/runs/`. "
        "`make robustness-report` legge solo CSV e protocollo: non lancia simulazioni e non richiede i dataset originali. "
        "Input mancanti: vedere il protocollo di [consolidamento principale](../README.md). "
        "Le figure sono esportate in PNG e PDF vettoriale; le tabelle provengono dagli stessi CSV.", ""]
    (OUTPUT / "README.md").write_text("\n".join(parts))
    qc.write_json(OUTPUT / "report_artifacts.json", {
        "report_source_sha256": ex.sha256(Path(__file__)),
        "artifacts": {qc.relative(path): ex.sha256(path) for path in [
            *sorted(directory.glob("*.pdf")), *sorted(directory.glob("*.png")), OUTPUT / "README.md",
            OUTPUT / "representative_configurations.md", OUTPUT / "representative_configurations.tex"]}})
    print(f"Robustness report and thesis figures: {OUTPUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "report"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    (run if args.phase == "run" else report)(args)


if __name__ == "__main__":
    main()
