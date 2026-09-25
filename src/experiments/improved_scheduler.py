"""Delay-aware carbon scheduling, from frozen inputs to figures and tables.

The carbon-aware policies minimise each job's predicted emissions and let delay
fall where it may. This experiment asks whether pricing the delay keeps the
carbon while giving queue time back: a gCO2e-per-hour delay price
(``carbon-delay``), a minimum saving per hour waited (``benefit-rate`` and the
threshold filter), a budget proportional to the predicted duration, and a
coordination of the queue that prices contention over the whole estimated run
and replans on every event.

Phases, each reading only what the previous ones saved:

    prepare     the saved split, the validation cohort (48 h embargo before the
                carbon cutoff), train-only validation predictions, input hashes
    smoke       the synthetic checks and 1,000 chronological validation jobs
    validation  corrected baselines, pilot grids, sweeps, frozen selection
    test        the frozen selection in the four information cells, the delay
                sweep, resource-only contention and the capacity subsets
    milp        offline optimum on the capacity subsets, one domain per instance
    report      figures and tables from the saved artifacts alone

Every run is keyed by its configuration, the input hashes and the simulator
source; a key that already has a result is loaded, never recomputed or
overwritten. Scoring is ex post: measured profiles and actual intensity.

The September/October period already belongs to the carbon model's test and the
scheduler test cohort was analysed in earlier rounds, so this is a
retrospective comparison whose new parameters are chosen on validation, not a
fresh joint test.

Run it with ``python -m experiments.improved_scheduler <phase>``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from carbon_accounting import PowerModel, account_emissions
from carbon_intensity.scheduling_impact import archive_reach
from carbon_intensity.series import TimeSeriesCarbonIntensityProvider
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive
from hpc_sim import (
    CarbonAwareScheduler,
    CarbonSignal,
    Cluster,
    CongestionAwareRecedingScheduler,
    EASYBackfillScheduler,
    FCFSScheduler,
    Job,
    RecedingHorizonCarbonAwareScheduler,
    RuntimeEstimateSource,
    Simulator,
    account_schedule,
    schedule_metrics,
)
from hpc_sim.metrics import bounded_slowdown


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "data/experiments/improved_scheduler"
INPUTS = {
    "workload": PROJECT_ROOT / "data/processed/pm100_clean.parquet",
    "test_predictions": PROJECT_ROOT / "data/job_predictions/test_predictions.parquet",
    "actual": PROJECT_ROOT / "data/carbon_intensity/actual/actual.json",
    "archive": PROJECT_ROOT / "data/carbon_intensity/snapshots/test_boosted_ridge.json",
    "contention": PROJECT_ROOT / "data/job_table.parquet",
}
#: Written by the long-job duration round with the same frozen recipe; only
#: compared against, never read as an input.
ARCHIVED_VALIDATION = (
    PROJECT_ROOT / "data/job_predictions/model_ensamble_test/validation_predictions.parquet"
)
#: The simulator source a run depends on; a change there invalidates the runs.
SIMULATOR_SOURCE = sorted((PROJECT_ROOT / "src/hpc_sim").glob("*.py"))

UTC = timezone.utc
SPLIT_COUNTS = (109_676, 23_826, 23_560)
TRAIN_UNTIL = datetime(2020, 9, 25, 4, 51, 34, tzinfo=UTC)
VALIDATION_UNTIL = datetime(2020, 10, 5, 14, 0, 51, tzinfo=UTC)
#: Validation decisions must be scored on carbon observations before this.
CARBON_CUTOFF = datetime(2020, 10, 5, 14, tzinfo=UTC)

NODES = 880
GRANULARITY = timedelta(minutes=15)
BUDGET_HOURS = 6.0
DELAY_SWEEP = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0)
ALPHAS = (0.25, 0.5, 1.0, 2.0)
PLANNING_JOBS = 32
CAPACITIES = (880, 512, 256, 128)
SUBSET_MAX_NODES = 128
FALLBACK_PRICES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
MILP_LIMIT_SECONDS = 600.0
MILP_GRID = timedelta(hours=1)

REALISTIC = ("predicted", "forecast")
ORACLE = ("actual", "actual")
CELLS = (ORACLE, ("predicted", "actual"), ("actual", "forecast"), REALISTIC)
#: Minimised together; non-dominance is judged on this vector only.
FRONTIER = (
    "total_emissions_gco2e",
    "waiting_mean_s",
    "waiting_p95_s",
    "bounded_slowdown_mean",
    "bounded_slowdown_p95",
)
RTOL, ATOL = 1e-8, 1e-9
BANDS = (
    (0.0, 60.0, "<60"),
    (60.0, 600.0, "[60,600)"),
    (600.0, 3_600.0, "[600,3600)"),
    (3_600.0, 10_800.0, "[3600,10800)"),
    (10_800.0, float("inf"), ">=10800"),
)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """One scheduler configuration. ``delay_hours`` caps a scaled budget."""

    mode: str  # fcfs, easy, once, receding, congestion
    objective: str = "carbon"
    delay_price: float = 0.0  # lambda, gCO2e per job-hour
    threshold: float = 0.0  # theta, gCO2e per job-hour
    delay_hours: float = BUDGET_HOURS
    alpha: float | None = None
    congestion_price: float = 0.0
    scope: str = "start"
    replan: bool = False
    planning_jobs: int | None = None
    overrun_to_horizon: bool = False

    def __post_init__(self) -> None:
        # The greedy is carbon-delay at zero price and zero threshold: one name
        # for one policy, so the same run is never computed twice.
        if self.objective in ("carbon", "carbon-delay"):
            priced = bool(self.delay_price or self.threshold)
            object.__setattr__(self, "objective", "carbon-delay" if priced else "carbon")

    @property
    def id(self) -> str:
        if self.mode in ("fcfs", "easy"):
            return self.mode
        parts = [self.mode, self.objective]
        if self.objective == "carbon-delay":
            parts.append(f"lambda={self.delay_price:.6g}")
        if self.objective == "benefit-rate" or self.threshold:
            parts.append(f"theta={self.threshold:.6g}")
        parts.append(
            f"fixed={self.delay_hours:g}h"
            if self.alpha is None
            else f"alpha={self.alpha:g},cap={self.delay_hours:g}h"
        )
        if self.mode == "congestion":
            parts.append(f"price={self.congestion_price:g}")
            if self.scope != "start":
                parts.append(self.scope)
            if self.replan:
                parts.append(f"replan{self.planning_jobs or ''}")
            if self.overrun_to_horizon:
                parts.append("overrun-horizon")
        return "|".join(parts)


FCFS = Policy("fcfs")
EASY = Policy("easy")
GREEDY = Policy("once")
RECEDING = Policy("receding")
#: The operational scheduler this round has to beat, with its historical parameters.
INCUMBENT = Policy("congestion", congestion_price=1.0)


def coordinated(policy: Policy, *, interval: bool, replan: bool, overrun: bool = True) -> Policy:
    return replace(
        policy,
        scope="interval" if interval else "start",
        replan=replan,
        planning_jobs=PLANNING_JOBS if replan else None,
        overrun_to_horizon=replan and overrun,
    )


@dataclass(frozen=True)
class Scenario:
    """The jobs and the machine: a cohort, an optional chronological subset, contention."""

    stage: str  # validation or test
    subset: int | None = None
    max_nodes: int | None = None
    contention: bool = False
    nodes: int = NODES

    @property
    def id(self) -> str:
        name = self.stage
        if self.subset:
            name += f"-first{self.subset}" + (f"-le{self.max_nodes}" if self.max_nodes else "")
        if self.contention:
            name += "-resource-only"
        return f"{name}@{self.nodes}"


Task = tuple[Scenario, str, str, Policy]  # scenario, job information, carbon information, policy


def task(scenario: Scenario, policy: Policy, cell: tuple[str, str] = REALISTIC) -> Task:
    information, carbon = cell
    if policy.mode in ("fcfs", "easy"):
        carbon = "none"
        if policy.mode == "fcfs":
            information = "actual"  # reads no estimate, so one run serves every arm
    return scenario, information, carbon, policy


# ---------------------------------------------------------------------------
# Inputs, cohorts and provenance
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict:
    path = OUTPUT / "manifest.json"
    if not path.exists():
        raise SystemExit("run the prepare phase first")
    return json.loads(path.read_text())


_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


class SealedArchive(ArchiveCarbonIntensityProvider):
    """The forecast a realistic decision may read, and nothing it may not."""

    def get_actual(self, timestamp):
        raise AssertionError("a realistic decision read the actual series")

    def get_actual_range(self, start, end):
        raise AssertionError("a realistic decision read the actual series")


def carbon():
    """Actual provider, the sealed forecast provider and the archive's reach."""

    def build():
        actual = TimeSeriesCarbonIntensityProvider.load(INPUTS["actual"])
        archive = ForecastArchive.load(INPUTS["archive"])
        return actual, SealedArchive(actual, archive), archive_reach(archive)

    return _cached("carbon", build)


def cohort(stage: str) -> tuple[tuple[Job, ...], tuple[Job, ...]]:
    """The stage's jobs with actual inputs and with predictions, in (release, id) order."""

    def build():
        from job_prediction.integration import attach_predictions, load_prediction_file
        from hpc_sim.workload import load_jobs

        expected = manifest()["cohorts"][stage]["job_ids"]
        path = INPUTS["test_predictions"] if stage == "test" else OUTPUT / "validation_predictions.parquet"
        actual = load_jobs(INPUTS["workload"], job_ids=set(expected))
        actual = tuple(sorted(actual, key=lambda job: (job.release_time, str(job.job_id))))
        if [job.job_id for job in actual] != expected:
            raise ValueError(f"the {stage} cohort no longer matches the manifest")
        return actual, attach_predictions(actual, load_prediction_file(path))

    return _cached(("cohort", stage), build)


def subset_ids(scenario: Scenario) -> tuple[list[object], int]:
    """The first ``subset`` jobs within the node limit, and how many were skipped.

    The filter is applied once, on the chronological cohort, and the same ids
    serve every policy and every capacity.
    """

    kept, skipped = [], 0
    for job in cohort(scenario.stage)[0]:
        if scenario.max_nodes is not None and job.nodes_required > scenario.max_nodes:
            skipped += 1
            continue
        kept.append(job.job_id)
        if len(kept) == scenario.subset:
            break
    return kept, skipped


def require_predictions(jobs: tuple[Job, ...]) -> None:
    """A realistic arm must never fall back to the actual duration or power."""

    missing = [
        job.job_id
        for job in jobs
        if job.predicted_duration_seconds is None
        or (job.power is not None and job.predicted_average_power_watts is None)
    ]
    if missing:
        raise ValueError(f"{len(missing)} predicted-arm jobs lack predictions, e.g. {missing[:3]}")


def resource_only(stage: str, information: str, jobs: tuple[Job, ...]) -> tuple[Job, ...]:
    """Terminal non-COMPLETED demand around the cohort, as every policy sees it.

    The allocated width is observed, which is privileged information about
    demand. On the predicted arm the planned duration is the requested time
    limit, or the largest positive train limit when none was requested; the
    observed runtime still governs when the nodes come back.
    """

    def build():
        from hpc_sim.workload import load_contention_jobs

        return load_contention_jobs(
            INPUTS["contention"],
            window_start=min(job.release_time for job in jobs),
            window_end=max(job.release_time for job in jobs) + timedelta(microseconds=1),
        )

    extra = _cached(("contention", stage, len(jobs)), build)
    if information == "predicted":
        fallback = manifest()["max_train_time_limit_s"]
        extra = tuple(
            replace(job, predicted_duration_seconds=job.time_limit_seconds or fallback)
            for job in extra
        )
    return extra


def scenario_jobs(scenario: Scenario, information: str) -> tuple[tuple[Job, ...], tuple[Job, ...], int]:
    """Evaluated jobs, resource-only jobs, and jobs the subset filter skipped."""

    actual, predicted = cohort(scenario.stage)
    jobs = predicted if information == "predicted" else actual
    skipped = 0
    if scenario.subset:
        ids, skipped = subset_ids(scenario)
        keep = set(ids)
        jobs = tuple(job for job in jobs if job.job_id in keep)
    extra = resource_only(scenario.stage, information, jobs) if scenario.contention else ()
    return jobs, extra, skipped


def build_scheduler(policy: Policy, carbon_information: str):
    actual, forecast, reach = carbon()
    estimate = RuntimeEstimateSource.SCHEDULING
    if policy.mode == "fcfs":
        return FCFSScheduler()
    if policy.mode == "easy":
        return EASYBackfillScheduler(runtime_estimate=estimate)
    realistic = carbon_information == "forecast"
    options = dict(
        runtime_estimate=estimate,
        decision_granularity=GRANULARITY,
        reach=reach,
        forecast=realistic,
        objective=policy.objective,
        delay_price_gco2e_per_hour=policy.delay_price,
        min_carbon_saving_per_delay=policy.threshold,
    )
    if policy.alpha is None:
        options["max_delay"] = timedelta(hours=policy.delay_hours)
    else:
        options["max_delay_fraction"] = policy.alpha
        options["max_delay_cap"] = timedelta(hours=policy.delay_hours)
    provider = forecast if realistic else actual
    if policy.mode == "once":
        return CarbonAwareScheduler(provider, **options)
    if policy.mode == "receding":
        return RecedingHorizonCarbonAwareScheduler(provider, replan_on_events=policy.replan, **options)
    return CongestionAwareRecedingScheduler(
        provider,
        congestion_price=policy.congestion_price,
        congestion_scope=policy.scope,
        replan_on_events=policy.replan,
        planning_jobs=policy.planning_jobs,
        overrun_to_horizon=policy.overrun_to_horizon,
        **options,
    )


def input_hashes() -> dict[str, str]:
    """Hashes a run depends on: the frozen inputs and the simulator source."""

    def build():
        hashes = dict(manifest()["input_sha256"])
        simulator = hashlib.sha256()
        for path in SIMULATOR_SOURCE:
            simulator.update(path.read_bytes())
        hashes["simulator_source"] = simulator.hexdigest()
        return hashes

    return _cached("hashes", build)


def run_directory(item: Task) -> tuple[str, Path]:
    scenario, information, carbon_information, policy = item
    identity = {
        "scenario": asdict(scenario),
        "job_information": information,
        "carbon_information": carbon_information,
        "policy": asdict(policy),
        "inputs": input_hashes(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return key, OUTPUT / "runs" / key


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def execute(item: Task) -> dict:
    """Simulate one configuration, account it, and save schedule and decisions."""

    key, directory = run_directory(item)
    saved = directory / "run.json"
    if saved.exists():
        return json.loads(saved.read_text())["row"]

    scenario, information, carbon_information, policy = item
    jobs, extra, skipped = scenario_jobs(scenario, information)
    if information == "predicted":
        require_predictions(jobs + extra)
    actual, _, _ = carbon()
    scheduler = build_scheduler(policy, carbon_information)

    started = perf_counter()
    result = Simulator(jobs + extra, Cluster(scenario.nodes), scheduler).run()
    wall = perf_counter() - started

    evaluated = {job.job_id for job in jobs}
    evaluation = result.replace_records(
        tuple(record for record in result.records if record.job_id in evaluated)
    )
    accounted = account_schedule(evaluation, jobs, actual)
    metrics = schedule_metrics(accounted)
    by_id = {job.job_id: job for job in jobs}
    records = accounted.records
    schedule = pd.DataFrame({
        "job_id": [record.job_id for record in records],
        "nodes": [record.nodes_required for record in records],
        "release_time": [record.release_time for record in records],
        "start_time": [record.start_time for record in records],
        "end_time": [record.end_time for record in records],
        "actual_duration_s": [by_id[record.job_id].actual_duration_seconds for record in records],
        "waiting_s": [record.waiting_seconds for record in records],
        "bounded_slowdown": [bounded_slowdown(record) for record in records],
        "energy_kwh": [record.energy_kwh for record in records],
        "emissions_gco2e": [record.emissions_gco2e for record in records],
        "emissions_gco2e_average_model": [record.emissions_gco2e_average_model for record in records],
    })
    decisions = pd.DataFrame(
        scheduler.decision_records() if hasattr(scheduler, "decision_records") else [],
        columns=[
            "job_id", "decision_time", "forecast_issue_time", "previous_target", "target",
            "deadline", "budget_nominal_s", "budget_effective_s", "cost_now_gco2e",
            "cost_target_gco2e", "congestion_penalty_gco2e", "best_rate", "reason", "policy",
        ],
    )
    first = decisions.drop_duplicates("job_id") if len(decisions) else decisions
    scored = first[first["reason"] != "resource-only"] if len(first) else first

    row = {
        "run_key": key,
        "stage": scenario.stage,
        "scenario": scenario.id,
        "subset": scenario.subset or "",
        "subset_skipped_jobs": skipped,
        "contention": "resource-only" if scenario.contention else "clean",
        "nodes": scenario.nodes,
        "job_information": information,
        "carbon_information": carbon_information,
        "policy_id": policy.id,
        **{f"policy_{name}": value for name, value in asdict(policy).items()},
        "workload": INPUTS["workload"].name,
        "job_predictions": (
            "" if information == "actual"
            else INPUTS["test_predictions"].name if scenario.stage == "test"
            else "validation_predictions.parquet"
        ),
        "forecast_archive": INPUTS["archive"].stem if carbon_information == "forecast" else "",
        "decision_granularity_minutes": GRANULARITY / timedelta(minutes=1),
        "evaluated_jobs": len(jobs),
        "resource_only_jobs": len(extra),
        **metrics.as_row(),
        "total_emissions_gco2e": metrics.total_emissions_gco2e,
        "peak_power_is_proxy_from_job_means": True,
        "schedule_end": max(record.end_time for record in records).isoformat(),
        "wall_seconds": wall,
        "decisions": getattr(scheduler, "decision_count", 0),
        "decision_seconds": getattr(scheduler, "decision_seconds", 0.0),
        "target_changes": max(0, len(decisions) - len(first)),
        "revisits": getattr(scheduler, "revisit_count", 0),
        "reach_limited_share": (
            float((scored["budget_effective_s"] < scored["budget_nominal_s"]).mean()) if len(scored) else 0.0
        ),
        "nominal_budget_mean_h": float(scored["budget_nominal_s"].mean() / 3600) if len(scored) else 0.0,
        "effective_budget_mean_h": float(scored["budget_effective_s"].mean() / 3600) if len(scored) else 0.0,
        "forecast_unavailable_decisions": int((decisions["reason"] == "forecast-unavailable").sum()),
    }
    directory.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(directory / "jobs.parquet", index=False)
    decisions.to_parquet(directory / "decisions.parquet", index=False)
    saved.write_text(json.dumps({"key": key, "row": row}, indent=1, default=str))
    return row


def run_all(items: list[Task], workers: int) -> list[dict]:
    """Every task's row, computing only the ones without a saved result."""

    items = list(dict.fromkeys(items))
    pending = [item for item in items if not (run_directory(item)[1] / "run.json").exists()]
    if pending:
        print(f"  {len(pending)} of {len(items)} runs to compute", flush=True)
    if workers > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, item): item for item in pending}
            for future in as_completed(futures):
                _progress(futures[future], future.result())
    else:
        for item in pending:
            _progress(item, execute(item))
    return [execute(item) for item in items]


def _progress(item: Task, row: dict) -> None:
    print(
        f"  {item[0].id:<28} {item[1]:>9}/{item[2]:<8} {item[3].id:<70} "
        f"{row['total_emissions_tco2e']:.5f} t  wait {row['waiting_mean_s']:>8.0f} s  "
        f"{row['wall_seconds']:>6.0f} s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------


def voluntary_delay(decisions: pd.DataFrame, schedule: pd.DataFrame) -> pd.Series:
    """Seconds each job was actually held back by the policy.

    The union of ``[t_k, min(target_k, t_{k+1}, start)]`` over the job's
    decisions, positive intervals only; a revisit that kept the standing
    target is not recorded and does not change the union. Neither
    ``start - release`` nor the targets' distance from release is used: both
    count contention as if the policy had chosen it.
    """

    zero = pd.Series(0.0, index=schedule["job_id"].to_numpy())
    if decisions.empty:
        return zero
    held = decisions[["job_id", "decision_time", "target"]].merge(
        schedule[["job_id", "start_time"]], on="job_id", validate="many_to_one"
    )
    held = held.reset_index(drop=True)
    held = held.sort_values(["job_id", "decision_time"], kind="stable")
    following = held.groupby("job_id")["decision_time"].shift(-1).fillna(held["start_time"])
    end = pd.concat([held["target"], following, held["start_time"]], axis=1).min(axis=1)
    seconds = (end - held["decision_time"]).dt.total_seconds().clip(lower=0.0)
    return zero.add(seconds.groupby(held["job_id"]).sum(), fill_value=0.0)


def ratio(numerator: float, denominator: float) -> tuple[float, str]:
    """A saving per unit of cost, or NaN with the reason when the cost is not positive."""

    if denominator > 0.0:
        return numerator / denominator, ""
    return float("nan"), "denominator not positive"


def isolated_emissions(scenario: Scenario) -> pd.Series:
    """Each job's measured emissions had it started at release on an empty machine.

    A counterfactual that ignores capacity, so not necessarily feasible; it is
    reported next to the paired saving against EASY, never instead of it.
    """

    path = OUTPUT / f"isolated_{scenario.stage}.parquet"
    if not path.exists():
        actual, _, _ = carbon()
        jobs = cohort(scenario.stage)[0]
        pd.DataFrame({
            "job_id": [job.job_id for job in jobs],
            "emissions_gco2e": [
                account_emissions(
                    job.power, job.release_time, actual.get_actual, power_model=PowerModel.MEASURED
                ).emissions_gco2
                for job in jobs
            ],
        }).to_parquet(path, index=False)
    frame = pd.read_parquet(path)
    return frame.set_index("job_id")["emissions_gco2e"]


def paired(row: dict, reference: dict) -> tuple[dict, list[dict]]:
    """Per-job comparison of one run with EASY on the same jobs and job information."""

    schedule = pd.read_parquet(OUTPUT / "runs" / row["run_key"] / "jobs.parquet")
    decisions = pd.read_parquet(OUTPUT / "runs" / row["run_key"] / "decisions.parquet")
    easy = pd.read_parquet(OUTPUT / "runs" / reference["run_key"] / "jobs.parquet")
    joined = schedule.merge(
        easy[["job_id", "emissions_gco2e", "waiting_s", "bounded_slowdown"]],
        on="job_id", how="outer", suffixes=("", "_easy"), indicator=True, validate="one_to_one",
    )
    if (joined["_merge"] != "both").any():
        raise ValueError(f"run {row['run_key']} and EASY do not hold the same jobs")
    held = voluntary_delay(decisions, schedule)
    joined["voluntary_delay_s"] = joined["job_id"].map(held).to_numpy()
    joined["saved_gco2e"] = joined["emissions_gco2e_easy"] - joined["emissions_gco2e"]
    joined["added_wait_s"] = joined["waiting_s"] - joined["waiting_s_easy"]
    joined["added_bsld"] = joined["bounded_slowdown"] - joined["bounded_slowdown_easy"]
    isolated = isolated_emissions(Scenario(row["stage"]))
    joined["saved_vs_isolated_gco2e"] = joined["job_id"].map(isolated).to_numpy() - joined["emissions_gco2e"]

    def summary(frame: pd.DataFrame) -> dict:
        saved = float(frame["saved_gco2e"].sum())
        hours = float(frame["voluntary_delay_s"].sum()) / 3600.0
        was_held = frame["voluntary_delay_s"] > 0.0
        kept = frame[was_held]
        per_wait, per_wait_reason = ratio(saved, float(frame["added_wait_s"].sum()) / 3600.0)
        per_hold, per_hold_reason = ratio(saved, hours)
        per_bsld, per_bsld_reason = ratio(saved, float(frame["added_bsld"].sum()))
        return {
            "jobs": len(frame),
            "saved_vs_easy_gco2e": saved,
            "saved_vs_isolated_gco2e": float(frame["saved_vs_isolated_gco2e"].sum()),
            "added_wait_hours": float(frame["added_wait_s"].sum()) / 3600.0,
            "added_bsld_sum": float(frame["added_bsld"].sum()),
            "voluntary_delay_hours": hours,
            "held_jobs": int(was_held.sum()),
            "held_share": float(was_held.mean()) if len(frame) else 0.0,
            "held_delay_p50_h": float(kept["voluntary_delay_s"].median() / 3600) if len(kept) else 0.0,
            "held_delay_p95_h": float(kept["voluntary_delay_s"].quantile(0.95) / 3600) if len(kept) else 0.0,
            "gco2e_per_added_wait_hour": per_wait,
            "gco2e_per_added_wait_hour_reason": per_wait_reason,
            "gco2e_per_voluntary_hour": per_hold,
            "gco2e_per_voluntary_hour_reason": per_hold_reason,
            "gco2e_per_added_bsld": per_bsld,
            "gco2e_per_added_bsld_reason": per_bsld_reason,
            "held_saving_negative": int((kept["saved_gco2e"] < 0).sum()),
            "held_saving_zero": int((kept["saved_gco2e"] == 0).sum()),
            "held_saving_below_1g": int(((kept["saved_gco2e"] > 0) & (kept["saved_gco2e"] < 1.0)).sum()),
            "held_saving_below_0p1pct": int(
                ((kept["saved_gco2e"] > 0) & (kept["saved_gco2e"] < 1e-3 * kept["emissions_gco2e_easy"])).sum()
            ),
        }

    duration = joined["actual_duration_s"]
    bands = [
        {"run_key": row["run_key"], "band": label, **summary(joined[(duration >= low) & (duration < high)])}
        for low, high, label in BANDS
    ]
    return summary(joined), bands


def analyse(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every row with its paired comparison, and the per-duration-band table."""

    easy = {
        (row["scenario"], row["job_information"]): row for row in rows if row["policy_id"] == "easy"
    }
    energy: dict[str, float] = {}
    out, bands = [], []
    for row in rows:
        # The energy of the evaluated jobs cannot depend on the schedule.
        reference_energy = energy.setdefault(row["scenario"], row["total_energy_mwh"])
        if abs(row["total_energy_mwh"] - reference_energy) > 1e-6 * reference_energy:
            raise AssertionError(f"energy moved in run {row['run_key']}")
        reference = easy.get((row["scenario"], row["job_information"]))
        if reference is None:
            out.append(dict(row))
            continue
        summary, band_rows = paired(row, reference)
        summary["saved_vs_easy"] = summary["saved_vs_easy_gco2e"] / reference["total_emissions_gco2e"]
        summary["easy_waiting_mean_s"] = reference["waiting_mean_s"]
        out.append(row | summary)
        bands.extend(band | {"policy_id": row["policy_id"], "scenario": row["scenario"],
                             "cell": f"{row['job_information']}/{row['carbon_information']}"}
                     for band in band_rows)
    return pd.DataFrame(out), pd.DataFrame(bands)


# ---------------------------------------------------------------------------
# Frontier and selection
# ---------------------------------------------------------------------------


def dominates(a, b) -> bool:
    """``a`` is no worse than ``b`` on every frontier metric and better on one."""

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    close = np.isclose(a, b, rtol=RTOL, atol=ATOL)
    return bool(np.all((a <= b) | close) and np.any((a < b) & ~close))


def vector(row) -> tuple[float, ...]:
    return tuple(float(row[metric]) for metric in FRONTIER)


def non_dominated(frame: pd.DataFrame) -> pd.Series:
    points = [vector(row) for _, row in frame.iterrows()]
    return pd.Series(
        [not any(dominates(other, point) for other in points if other is not point) for point in points],
        index=frame.index,
    )


def representatives(frame: pd.DataFrame) -> list[str]:
    """Up to three non-dominated configurations: least carbon, least mean wait, least p95 bsld.

    Points saving carbon against EASY are preferred; exact ties fall to the
    metric vector, then to the configuration id.
    """

    front = frame[non_dominated(frame)]
    saving = front[front["saved_vs_easy_gco2e"] > 0.0]
    pool = saving if len(saving) else front
    chosen: list[str] = []
    for metric in ("total_emissions_gco2e", "waiting_mean_s", "bounded_slowdown_p95"):
        best = min(
            pool.to_dict("records"),
            key=lambda point: (point[metric], vector(point), point["policy_id"]),
        )
        if best["policy_id"] not in chosen:
            chosen.append(best["policy_id"])
    return chosen


def pilot_grids(decisions: pd.DataFrame) -> dict:
    """Thresholds and delay prices sized from each job's best positive benefit rate at release."""

    first = decisions.drop_duplicates("job_id")
    rates = first["best_rate"].dropna()
    rates = rates[rates > 0.0].to_numpy()
    if not len(rates):
        return {"positive_jobs": 0, "quantiles": {}, "thresholds": [0.0],
                "prices": list(FALLBACK_PRICES), "note": "no positive opportunity in the pilot"}
    quantiles = {str(q): float(np.quantile(rates, q / 100)) for q in (10, 25, 50, 75, 90)}
    thresholds = [0.0] + sorted(set(quantiles.values()))
    prices = [0.0] + [float(value) for value in np.geomspace(quantiles["10"] / 10, quantiles["90"] * 10, 7)]
    return {"positive_jobs": int(len(rates)), "jobs": int(len(first)), "quantiles": quantiles,
            "thresholds": thresholds, "prices": prices}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def prepare(arguments: argparse.Namespace) -> None:
    """Freeze the split, the validation cohort and its train-only predictions."""

    from job_prediction.data import (
        AVERAGE_POWER_WATTS, COMPLETION_TIME, DURATION_SECONDS, JOB_ID, SUBMIT_TIME,
        load_job_data, temporal_split,
    )
    from job_prediction.features import build_features
    from job_prediction.gradient import _fit_target
    from hpc_sim.workload import load_jobs

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import train_job_models as frozen  # the recipe the incumbent model was trained with

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_job_data(INPUTS["workload"])
    split = temporal_split(data)
    if split.counts != SPLIT_COUNTS or (split.train_until, split.validation_until) != (TRAIN_UNTIL, VALIDATION_UNTIL):
        raise ValueError(f"the saved split did not reproduce: {split.counts}")

    submitted = pd.to_datetime(data[SUBMIT_TIME], utc=True)
    completed = pd.to_datetime(data[COMPLETION_TIME], utc=True)
    train = (submitted < TRAIN_UNTIL).to_numpy()
    fit_rows = np.flatnonzero(train & (completed <= TRAIN_UNTIL).to_numpy())
    validation = ((submitted >= TRAIN_UNTIL) & (submitted < VALIDATION_UNTIL)).to_numpy()
    # Only outcomes observable before the test starts: the rest belong to it.
    evaluate_rows = np.flatnonzero(validation & (completed <= VALIDATION_UNTIL).to_numpy())

    matrix = build_features(data, frozen.SELECTED_FEATURE_SPEC).to_numpy(dtype=float)
    predictions = pd.DataFrame({JOB_ID: data[JOB_ID].to_numpy()[evaluate_rows]})
    for target, column in (
        (DURATION_SECONDS, "predicted_duration_seconds"),
        (AVERAGE_POWER_WATTS, "predicted_average_power_watts"),
    ):
        model = _fit_target(
            matrix[fit_rows],
            pd.to_numeric(data[target]).to_numpy(float)[fit_rows],
            target_name=target,
            config=frozen.SELECTED_CONFIGS[target],
            seed=0,
        )
        predictions[column] = model.predict(matrix[evaluate_rows])
    predictions["predicted_energy_kwh"] = (
        predictions["predicted_average_power_watts"] * predictions["predicted_duration_seconds"] / 3.6e6
    )
    predictions.to_parquet(OUTPUT / "validation_predictions.parquet", index=False)

    agreement = {}
    if ARCHIVED_VALIDATION.exists():
        archived = pd.read_parquet(ARCHIVED_VALIDATION).set_index("job_id")
        mine = predictions.set_index(JOB_ID)
        for column, theirs in (("predicted_duration_seconds", "incumbent"),
                               ("predicted_average_power_watts", "predicted_average_power_watts")):
            relative = (mine[column] / archived.loc[mine.index, theirs] - 1.0).abs()
            agreement[column] = float(relative.max())

    cutoff = CARBON_CUTOFF - timedelta(hours=arguments.embargo_hours)
    candidates = load_jobs(INPUTS["workload"], job_ids=set(predictions[JOB_ID]))
    validation_jobs = sorted(
        (job for job in candidates if job.release_time < cutoff),
        key=lambda job: (job.release_time, str(job.job_id)),
    )
    test_ids = pd.read_parquet(INPUTS["test_predictions"], columns=["job_id"])["job_id"]
    test_jobs = sorted(
        load_jobs(INPUTS["workload"], job_ids=set(test_ids)),
        key=lambda job: (job.release_time, str(job.job_id)),
    )
    limits = pd.to_numeric(data["time_limit"]).to_numpy(float)[train]
    record = {
        "created": datetime.now(UTC).isoformat(),
        "split_counts": list(split.counts),
        "train_until": TRAIN_UNTIL.isoformat(),
        "validation_until": VALIDATION_UNTIL.isoformat(),
        "carbon_cutoff": CARBON_CUTOFF.isoformat(),
        "embargo_hours": arguments.embargo_hours,
        "validation_fit_rows": int(len(fit_rows)),
        "validation_outcome_rows": int(len(evaluate_rows)),
        "validation_recipe": {
            "features": frozen.SELECTED_FEATURE_SPEC.name,
            "configs": {name: config.as_row() for name, config in frozen.SELECTED_CONFIGS.items()
                        if name in (DURATION_SECONDS, AVERAGE_POWER_WATTS)},
            "seed": 0,
        },
        "agreement_with_archived_validation_max_relative": agreement,
        "max_train_time_limit_s": float(limits[limits > 0].max() * 60.0),
        "cohorts": {
            "validation": {"jobs": len(validation_jobs), "job_ids": [job.job_id for job in validation_jobs],
                           "released_before": cutoff.isoformat()},
            "test": {"jobs": len(test_jobs), "job_ids": [job.job_id for job in test_jobs]},
        },
        "input_sha256": {
            **{name: sha256(path) for name, path in INPUTS.items()},
            "validation_predictions": sha256(OUTPUT / "validation_predictions.parquet"),
        },
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                   capture_output=True, text=True).stdout.strip(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"validation cohort        {len(validation_jobs):,} jobs released before {cutoff.isoformat()}")
    print(f"test cohort              {len(test_jobs):,} jobs")
    print(f"validation fit rows      {len(fit_rows):,}; agreement with archive {agreement}")


def check_cutoff(rows: list[dict]) -> None:
    late = [row for row in rows if row["stage"] == "validation"
            and datetime.fromisoformat(row["schedule_end"]) >= CARBON_CUTOFF]
    if late:
        raise SystemExit(
            f"{len(late)} validation runs end after the carbon cutoff; rerun prepare with "
            "--embargo-hours increased by 24 and start the validation again"
        )


def smoke(arguments: argparse.Namespace) -> None:
    """Synthetic checks, then 1,000 chronological validation jobs through every mechanism."""

    for check in ("check_hpc_sim_delay_objectives.py", "check_experiments_improved_scheduler.py"):
        subprocess.run([sys.executable, str(PROJECT_ROOT / "tests" / check)], check=True)
    scenario = Scenario("validation", subset=1_000)
    probe = Policy("congestion", objective="carbon-delay", delay_price=1.0, threshold=1.0,
                   alpha=1.0, congestion_price=1.0)
    policies = [
        GREEDY, RECEDING, INCUMBENT,
        Policy("congestion", objective="carbon-delay", delay_price=1.0, congestion_price=1.0),
        Policy("receding", objective="benefit-rate", threshold=1.0),
        probe,
        coordinated(probe, interval=True, replan=True),
        replace(coordinated(probe, interval=True, replan=True), delay_hours=0.0),
        replace(INCUMBENT, delay_hours=0.0),
    ]
    items = [task(scenario, FCFS), task(scenario, EASY, ORACLE), task(scenario, EASY)]
    items += [task(scenario, policy, cell) for policy in policies for cell in (ORACLE, REALISTIC)]
    rows = run_all(items, arguments.workers)
    frame, _ = analyse(rows)
    for _, row in frame.iterrows():
        if row["policy_delay_hours"] == 0.0 and row["policy_mode"] not in ("fcfs", "easy"):
            assert row["saved_vs_easy_gco2e"] == 0.0 and row["added_wait_hours"] == 0.0, row["policy_id"]
    for row in rows:
        decisions = pd.read_parquet(OUTPUT / "runs" / row["run_key"] / "decisions.parquet")
        if len(decisions):
            assert (decisions["target"] <= decisions["deadline"]).all()
            issued = decisions.dropna(subset=["forecast_issue_time"])
            assert (issued["forecast_issue_time"] <= issued["decision_time"]).all()
    frame.to_csv(OUTPUT / "smoke_runs.csv", index=False)
    print(frame[["policy_id", "job_information", "carbon_information", "saved_vs_easy",
                 "waiting_mean_s", "voluntary_delay_hours", "wall_seconds"]].to_string())


def family_rows(frame: pd.DataFrame, policies: list[Policy]) -> pd.DataFrame:
    ids = [policy.id for policy in policies]
    return frame[frame["policy_id"].isin(ids) & (frame["job_information"] == "predicted")
                 & (frame["carbon_information"] == "forecast")]


def validation(arguments: argparse.Namespace) -> None:
    """Baselines, pilot, sweeps and selection, all on the validation cohort."""

    scenario = Scenario("validation")
    workers = arguments.workers
    all_rows: list[dict] = []

    def stage(label: str, items: list[Task]) -> pd.DataFrame:
        print(f"{label}", flush=True)
        all_rows.extend(run_all(items, workers))
        check_cutoff(all_rows)
        frame, _ = analyse(list({row["run_key"]: row for row in all_rows}.values()))
        return frame

    baselines = [
        task(scenario, FCFS), task(scenario, EASY, ORACLE), task(scenario, EASY),
        task(scenario, GREEDY, ORACLE), task(scenario, INCUMBENT, ORACLE),
        task(scenario, GREEDY), task(scenario, RECEDING), task(scenario, INCUMBENT),
    ]
    stage("corrected baselines", baselines)

    grids_path = OUTPUT / "grids.json"
    incumbent_key = run_directory(task(scenario, INCUMBENT))[0]
    grids = pilot_grids(pd.read_parquet(OUTPUT / "runs" / incumbent_key / "decisions.parquet"))
    if grids_path.exists() and json.loads(grids_path.read_text())["prices"] != grids["prices"]:
        raise SystemExit("the frozen grids no longer match the pilot")
    grids_path.write_text(json.dumps(grids, indent=1))
    print(f"grids                    thresholds {grids['thresholds']}\n"
          f"                         prices {grids['prices']}")

    families: dict[str, list[Policy]] = {
        "scalarized": [
            Policy("congestion", objective="carbon-delay", delay_price=price, congestion_price=1.0)
            for price in grids["prices"]
        ],
        "benefit-rate": [
            Policy("receding", objective="benefit-rate", threshold=threshold)
            for threshold in grids["thresholds"]
        ],
    }
    frame = stage("price and threshold sweeps", [task(scenario, p) for f in families.values() for p in f])
    by_id = {policy.id: policy for f in families.values() for policy in f}
    reps = {name: representatives(family_rows(frame, policies)) for name, policies in families.items()}

    for name in ("scalarized", "benefit-rate"):
        families[f"{name}-adaptive"] = [
            replace(by_id[rep], alpha=alpha) for rep in reps[name] for alpha in ALPHAS
        ]
    sensitivity = [
        replace(by_id[reps[name][0]], alpha=alpha, delay_hours=12.0)
        for name in ("scalarized", "benefit-rate") for alpha in ALPHAS
    ]
    items = [task(scenario, p) for name in ("scalarized-adaptive", "benefit-rate-adaptive")
             for p in families[name]] + [task(scenario, p) for p in sensitivity]
    frame = stage("fixed against adaptive", items)
    by_id |= {policy.id: policy for name in ("scalarized-adaptive", "benefit-rate-adaptive")
              for policy in families[name]}
    by_id |= {policy.id: policy for policy in sensitivity}
    for name in ("scalarized-adaptive", "benefit-rate-adaptive"):
        reps[name] = representatives(family_rows(frame, families[name]))

    quantiles = grids["quantiles"]
    combined_thresholds = [0.0] + [quantiles[q] for q in ("50", "90") if q in quantiles] if quantiles else [0.0]
    families["combined"] = list(dict.fromkeys(
        replace(by_id[rep], threshold=threshold)
        for rep in reps["scalarized-adaptive"] for threshold in combined_thresholds
    ))
    controls = list(dict.fromkeys(
        control
        for policy in families["combined"]
        for control in (replace(policy, delay_price=0.0), replace(policy, threshold=0.0),
                        replace(policy, alpha=None, delay_hours=BUDGET_HOURS))
    ))
    frame = stage("combination and one-at-a-time controls",
                  [task(scenario, p) for p in families["combined"] + controls])
    by_id |= {policy.id: policy for policy in families["combined"] + controls}
    reps["combined"] = representatives(family_rows(frame, families["combined"]))

    bases = [by_id[rep] for rep in reps["combined"]] + [INCUMBENT]
    families["coordinated"] = list(dict.fromkeys(
        coordinated(base, interval=interval, replan=replan)
        for base in bases for interval, replan in ((True, False), (False, True), (True, True))
    ))
    overrun_ablation = [coordinated(base, interval=True, replan=True, overrun=False) for base in bases]
    frame = stage("congestion and trigger ablation",
                  [task(scenario, p) for p in families["coordinated"] + overrun_ablation])
    by_id |= {policy.id: policy for policy in families["coordinated"] + overrun_ablation}
    reps["coordinated"] = representatives(family_rows(frame, families["coordinated"]))

    rows, bands = analyse(list({row["run_key"]: row for row in all_rows}.values()))
    membership = {policy.id: name for name, policies in families.items() for policy in policies}
    rows["family"] = rows["policy_id"].map(membership).fillna("")
    realistic = (rows["job_information"] == "predicted") & (rows["carbon_information"] == "forecast")
    rows["family_non_dominated"] = False
    for name in families:
        members = rows[realistic & (rows["family"] == name)]
        rows.loc[members.index, "family_non_dominated"] = non_dominated(members)
    rows["representative"] = rows["policy_id"].isin({rep for chosen in reps.values() for rep in chosen})
    rows.to_csv(OUTPUT / "validation_runs.csv", index=False)
    bands.to_csv(OUTPUT / "validation_bands.csv", index=False)

    test_configs = list(dict.fromkeys(rep for name in families for rep in reps.get(name, [])))
    selection = {
        "frozen": datetime.now(UTC).isoformat(),
        "representatives": reps,
        "test_configs": [asdict(by_id[rep]) for rep in test_configs],
        "test_config_ids": test_configs,
        "incumbent": asdict(INCUMBENT),
        "grids": grids,
        "decision_rule": (
            "keep the new code iff at least one test configuration Pareto-dominates the "
            "incumbent in the realistic test cell on "
            + ", ".join(FRONTIER) + f" (rtol {RTOL}, atol {ATOL}); otherwise restore"
        ),
    }
    path = OUTPUT / "selection.json"
    if path.exists() and json.loads(path.read_text())["test_config_ids"] != test_configs:
        raise SystemExit("a different selection is already frozen; refusing to overwrite it")
    path.write_text(json.dumps(selection, indent=1))
    print("representatives")
    for name, chosen in reps.items():
        print(f"  {name:<22} {chosen}")


def selected() -> tuple[list[Policy], dict]:
    path = OUTPUT / "selection.json"
    if not path.exists():
        raise SystemExit("the test phase needs the frozen validation selection")
    selection = json.loads(path.read_text())
    return [Policy(**config) for config in selection["test_configs"]], selection


def test(arguments: argparse.Namespace) -> None:
    """The frozen selection on the test cohort; no parameter is chosen here."""

    configs, selection = selected()
    main = list(dict.fromkeys([GREEDY, RECEDING, INCUMBENT] + configs))
    scenario = Scenario("test")
    workers = arguments.workers
    all_rows: list[dict] = []

    def stage(label: str, items: list[Task]) -> None:
        print(label, flush=True)
        all_rows.extend(run_all(items, workers))

    stage("four information cells", [task(scenario, FCFS), task(scenario, EASY, ORACLE), task(scenario, EASY)]
          + [task(scenario, policy, cell) for cell in CELLS for policy in main])
    rows, _ = analyse(list({row["run_key"]: row for row in all_rows}.values()))
    decide(rows, selection)

    stage("delay sweep", [task(scenario, replace(policy, delay_hours=hours))
                          for policy in main for hours in DELAY_SWEEP])
    contended = Scenario("test", contention=True)
    stage("resource-only contention", [task(contended, EASY)] + [task(contended, p) for p in main])
    for size in (1_000, 5_000):
        for nodes in CAPACITIES:
            subset = Scenario("test", subset=size, max_nodes=SUBSET_MAX_NODES, nodes=nodes)
            stage(f"{size} jobs on {nodes} nodes",
                  [task(subset, EASY, ORACLE), task(subset, EASY)]
                  + [task(subset, policy, cell) for policy in main for cell in (ORACLE, REALISTIC)])

    rows, bands = analyse(list({row["run_key"]: row for row in all_rows}.values()))
    rows["preregistered"] = rows["policy_id"].isin(selection["test_config_ids"])
    rows.to_csv(OUTPUT / "test_runs.csv", index=False)
    bands.to_csv(OUTPUT / "test_bands.csv", index=False)


def decide(rows: pd.DataFrame, selection: dict) -> dict:
    """Apply the rule frozen with the selection to the realistic test cell."""

    cell = rows[(rows["scenario"] == Scenario("test").id) & (rows["job_information"] == "predicted")
                & (rows["carbon_information"] == "forecast")].set_index("policy_id")
    incumbent = cell.loc[INCUMBENT.id]
    verdicts = {
        config: {
            "dominates_incumbent": dominates(vector(cell.loc[config]), vector(incumbent)),
            "dominated_by_incumbent": dominates(vector(incumbent), vector(cell.loc[config])),
            **{metric: float(cell.loc[config][metric]) for metric in FRONTIER},
            "saved_vs_easy": float(cell.loc[config]["saved_vs_easy"]),
        }
        for config in selection["test_config_ids"]
    }
    decision = {
        "rule": selection["decision_rule"],
        "incumbent": {metric: float(incumbent[metric]) for metric in FRONTIER}
        | {"saved_vs_easy": float(incumbent["saved_vs_easy"])},
        "configs": verdicts,
        "keep": any(verdict["dominates_incumbent"] for verdict in verdicts.values()),
    }
    (OUTPUT / "decision.json").write_text(json.dumps(decision, indent=1))
    print(f"decision                 {'KEEP' if decision['keep'] else 'RESTORE'}")
    return decision


def milp_instance(instance: dict) -> list[dict]:
    """One offline optimum on a capacity subset, against every policy run there."""

    from experiments.milp import replay, solve

    scenario = Scenario("test", subset=instance["jobs"], max_nodes=SUBSET_MAX_NODES, nodes=instance["nodes"])
    jobs, _, _ = scenario_jobs(scenario, "actual")
    actual, _, _ = carbon()
    references = []
    for policy in instance["policies"]:
        key, directory = run_directory(task(scenario, policy, ORACLE))
        schedule = pd.read_parquet(directory / "jobs.parquet")
        references.append((policy.id, schedule))

    def starts_of(frame: pd.DataFrame) -> SimpleNamespace:  # all solve() reads of a schedule
        return SimpleNamespace(records=[
            SimpleNamespace(job_id=job_id, start_time=start)
            for job_id, start in zip(frame["job_id"], frame["start_time"].dt.to_pydatetime())
        ])

    weight = instance["delay_price"]
    starts, solver = solve(
        jobs, CarbonSignal(actual), nodes=instance["nodes"], max_delay=timedelta(hours=BUDGET_HOURS),
        granularity=instance["granularity"], references=[starts_of(frame) for _, frame in references],
        delay_weight=weight, time_limit=MILP_LIMIT_SECONDS,
    )
    measured: dict[str, dict] = {}
    for name, frame in references:
        measured[name] = {
            "measured_gco2e": float(frame["emissions_gco2e"].sum()),
            "model_gco2e": float(frame["emissions_gco2e_average_model"].sum()),
            "delay_hours": float(frame["waiting_s"].sum() / 3600),
            "waiting_mean_s": float(frame["waiting_s"].mean()),
        }
    if starts:
        plan = account_schedule(replay(jobs, starts, instance["nodes"]), jobs, actual)
        measured["milp"] = {
            "measured_gco2e": sum(record.emissions_gco2e for record in plan.records),
            "model_gco2e": sum(record.emissions_gco2e_average_model for record in plan.records),
            "delay_hours": sum(record.waiting_seconds for record in plan.records) / 3600,
            "waiting_mean_s": float(np.mean([record.waiting_seconds for record in plan.records])),
        }
    for values in measured.values():
        values["weighted_gco2e"] = values["model_gco2e"] + weight * values["delay_hours"]

    rows = []
    easy = measured["easy"]
    for name, values in measured.items():
        row = {
            "jobs": instance["jobs"], "nodes": instance["nodes"],
            "grid_minutes": instance["granularity"] / timedelta(minutes=1),
            "delay_price": weight, "configuration": name, **values,
        }
        for metric in ("measured", "model", "weighted"):
            if "milp" in measured:
                value, reason = ratio(
                    easy[f"{metric}_gco2e"] - values[f"{metric}_gco2e"],
                    easy[f"{metric}_gco2e"] - measured["milp"][f"{metric}_gco2e"],
                )
            else:
                value, reason = float("nan"), "no MILP solution"
            row[f"captured_{metric}"] = value
            row[f"captured_{metric}_reason"] = reason
        rows.append(row | (solver if name == "milp" else {}))
    if not starts:
        rows.append({"jobs": instance["jobs"], "nodes": instance["nodes"], "delay_price": weight,
                     "grid_minutes": instance["granularity"] / timedelta(minutes=1),
                     "configuration": "milp", **solver})
    return rows


def milp(arguments: argparse.Namespace) -> None:
    """Offline optimum on the capacity subsets, one domain per instance and price."""

    configs, _ = selected()
    main = list(dict.fromkeys([EASY, GREEDY, RECEDING, INCUMBENT] + configs))
    prices = sorted({policy.delay_price for policy in configs if policy.objective == "carbon-delay"} - {0.0})
    smoke_scenario = Scenario("test", subset=100, max_nodes=SUBSET_MAX_NODES)
    run_all([task(smoke_scenario, policy, ORACLE) for policy in main], arguments.workers)

    def instances(size: int, capacities, grids, weights) -> list[dict]:
        return [{"jobs": size, "nodes": nodes, "granularity": grid, "delay_price": weight, "policies": main}
                for nodes in capacities for grid in grids for weight in weights]

    plan = instances(100, (NODES,), (MILP_GRID, GRANULARITY), (0.0,)) + instances(
        1_000, CAPACITIES, (MILP_GRID,), [0.0] + prices)
    rows = _solve_all(plan, arguments.workers)
    if all(row.get("solution_status") == "optimal" for row in rows
           if row["configuration"] == "milp" and row["jobs"] == 1_000):
        rows += _solve_all(instances(5_000, (NODES,), (MILP_GRID,), [0.0] + prices), arguments.workers)
    else:
        print("5,000-job MILP skipped: a 1,000-job instance did not finish within the limit")
    pd.DataFrame(rows).to_csv(OUTPUT / "milp_results.csv", index=False)


def _solve_all(plan: list[dict], workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(milp_instance, instance) for instance in plan]):
            result = future.result()
            solver = next(row for row in result if row["configuration"] == "milp")
            print(f"  MILP {solver['jobs']} jobs {solver['nodes']} nodes grid {solver['grid_minutes']:g} min "
                  f"lambda {solver['delay_price']:g}: {solver.get('solution_status')} "
                  f"in {solver.get('solve_seconds', float('nan')):.0f} s", flush=True)
            rows.extend(result)
    return rows


#: Categorical slots in fixed order (the dataviz reference palette, light surface).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
NAMES = {GREEDY.id: "greedy", RECEDING.id: "receding", INCUMBENT.id: "incumbent", "easy": "EASY", "fcfs": "FCFS"}
POLICY_FIELDS = tuple(f"policy_{name}" for name in Policy.__dataclass_fields__)


def label(row) -> str:
    """A short name for a configuration, for legends and tables."""

    if row["policy_id"] in NAMES:
        return NAMES[row["policy_id"]]
    parts = []
    if row["policy_objective"] == "benefit-rate":
        parts.append(f"rate>{row['policy_threshold']:.3g}")
    else:
        if row["policy_delay_price"]:
            parts.append(f"λ={row['policy_delay_price']:.3g}")
        if row["policy_threshold"]:
            parts.append(f"θ={row['policy_threshold']:.3g}")
    if not pd.isna(row["policy_alpha"]):
        parts.append(f"α={row['policy_alpha']:g}")
    if row["policy_scope"] == "interval":
        parts.append("interval")
    if row["policy_replan"]:
        parts.append("replan")
    if row["policy_mode"] == "receding" and row["policy_objective"] != "benefit-rate":
        parts.append("receding")
    return " ".join(parts) or row["policy_id"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })
    return plt


def _save(plt, figure, name: str) -> None:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _markdown(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    """A Markdown table of the chosen columns, formatted by the given specs."""

    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for name, spec in columns.items():
            key, _, fmt = spec.partition(":")
            value = label(row) if key == "label" else row[key]
            cells.append("-" if isinstance(value, float) and np.isnan(value)
                         else format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines)


SUMMARY = {
    "policy": "label", "saving vs EASY": "saved_vs_easy:.3%", "tCO2e": "total_emissions_tco2e:.4f",
    "wait mean s": "waiting_mean_s:,.0f", "wait p95 s": "waiting_p95_s:,.0f",
    "bsld mean": "bounded_slowdown_mean:.2f", "bsld p95": "bounded_slowdown_p95:.2f",
    "held": "held_share:.1%", "voluntary h": "voluntary_delay_hours:,.0f",
    "g per voluntary h": "gco2e_per_voluntary_hour:,.1f", "g per added wait h": "gco2e_per_added_wait_hour:,.2f",
}


def report(arguments: argparse.Namespace) -> None:
    """Figures and tables from the saved CSVs and parquets only; nothing is simulated."""

    plt = _plt()
    validation_rows = pd.read_csv(OUTPUT / "validation_runs.csv")
    rows = pd.read_csv(OUTPUT / "test_runs.csv")
    bands = pd.read_csv(OUTPUT / "test_bands.csv")
    selection = json.loads((OUTPUT / "selection.json").read_text())
    decision = json.loads((OUTPUT / "decision.json").read_text())
    milp_rows = pd.read_csv(OUTPUT / "milp_results.csv") if (OUTPUT / "milp_results.csv").exists() else None
    def cell(frame, information, carbon_information, scenario=Scenario("test").id):
        return frame[(frame["scenario"] == scenario) & (frame["job_information"] == information)
                     & (frame["carbon_information"] == carbon_information)]

    realistic = cell(rows, *REALISTIC)
    main6 = realistic[realistic["policy_delay_hours"] == BUDGET_HOURS]
    # Eight categorical slots: the three references, then the preregistered
    # configurations saving most in the realistic cell. Every row is in the tables.
    ranked = (main6[main6["policy_id"].isin(selection["test_config_ids"])]
              .sort_values("saved_vs_easy", ascending=False)["policy_id"].tolist())
    shown = list(dict.fromkeys([GREEDY.id, RECEDING.id, INCUMBENT.id] + ranked))[: len(PALETTE)]
    color = {pid: PALETTE[index] for index, pid in enumerate(shown)}
    names = {pid: label(main6[main6["policy_id"] == pid].iloc[0]) for pid in shown}
    posthoc_path = OUTPUT / "test_posthoc_grid_runs.csv"
    posthoc = pd.read_csv(posthoc_path) if posthoc_path.exists() else None

    def sweep(pid: str) -> pd.DataFrame:
        """The delay-sweep points of one 6 h configuration, budget ascending."""

        reference = main6[main6["policy_id"] == pid].iloc[0]
        mask = np.ones(len(realistic), dtype=bool)
        for field in POLICY_FIELDS:
            if field != "policy_delay_hours":
                left, right = realistic[field], reference[field]
                mask &= (left.isna() & pd.isna(right)) | (left == right)
        return realistic[mask].sort_values("policy_delay_hours")

    # Frontier: saving against mean wait and against bounded slowdown p95, along the delay sweep.
    for name, metric, axis in (("frontier_wait", "waiting_mean_s", "mean waiting time (h)"),
                               ("frontier_bsld_p95", "bounded_slowdown_p95", "bounded slowdown p95")):
        figure, ax = plt.subplots(figsize=(7.2, 4.6))
        for pid in shown:
            if pid not in set(main6["policy_id"]):
                continue
            points = sweep(pid)
            scale = 3600.0 if metric == "waiting_mean_s" else 1.0
            ax.plot(points[metric] / scale, 100 * points["saved_vs_easy"], "-o", color=color[pid],
                    label=names[pid])
            six = points[points["policy_delay_hours"] == BUDGET_HOURS]
            ax.plot(six[metric] / scale, 100 * six["saved_vs_easy"], "o", color=color[pid],
                    markersize=9, markeredgecolor="#fcfcfb", markeredgewidth=2)
        if posthoc is not None:
            grid = posthoc[posthoc["policy_mode"] != "easy"]
            ax.plot(grid[metric] / scale, 100 * grid["saved_vs_easy"], "D", color=MUTED, markersize=5,
                    markerfacecolor="none", label="validation grids on test (post hoc, descriptive)")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel(axis + " (delay sweep 0-24 h; large marker = 6 h)")
        ax.set_ylabel("emissions saved vs EASY (%)")
        ax.set_title("Test cohort, predicted jobs + forecast carbon", loc="left")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _save(plt, figure, name)

    # Delay budget against saving and QoS, with the budget the reach leaves.
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for pid in shown:
        if pid not in set(main6["policy_id"]):
            continue
        points = sweep(pid)
        for ax, metric, scale in zip(axes, ("saved_vs_easy", "waiting_mean_s", "bounded_slowdown_p95"),
                                     (100.0, 1 / 3600.0, 1.0)):
            ax.plot(points["policy_delay_hours"], points[metric] * scale, "-o", color=color[pid],
                    label=names[pid])
    for ax, title in zip(axes, ("emissions saved vs EASY (%)", "mean waiting time (h)", "bounded slowdown p95")):
        ax.set_title(title, loc="left")
        ax.set_xlabel("nominal delay budget / cap (h)")
    figure.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=4,
                  bbox_to_anchor=(0.5, -0.02))
    figure.subplots_adjust(bottom=0.28)
    _save(plt, figure, "delay_budget")

    # Fixed against adaptive, on validation.
    vrealistic = cell(validation_rows, *REALISTIC, scenario=Scenario("validation").id)
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    for index, family in enumerate(("scalarized", "scalarized-adaptive", "benefit-rate", "benefit-rate-adaptive")):
        members = vrealistic[vrealistic["family"] == family]
        ax.plot(members["waiting_mean_s"] / 3600, 100 * members["saved_vs_easy"], "o",
                color=PALETTE[index], label=family)
    incumbent = vrealistic[vrealistic["policy_id"] == INCUMBENT.id]
    ax.plot(incumbent["waiting_mean_s"] / 3600, 100 * incumbent["saved_vs_easy"], "*", color=INK,
            markersize=12, label="incumbent")
    ax.set_xlabel("mean waiting time (h)")
    ax.set_ylabel("emissions saved vs EASY (%)")
    ax.set_title("Validation cohort: fixed 6 h against duration-scaled budgets", loc="left")
    ax.legend(fontsize=7)
    _save(plt, figure, "fixed_vs_adaptive")

    # Distributions of voluntary delay and of the saving it bought, per held job.
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    easy_key = cell(rows, "predicted", "none").iloc[0]
    for pid in shown:
        match = main6[main6["policy_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        schedule = pd.read_parquet(OUTPUT / "runs" / row["run_key"] / "jobs.parquet")
        decisions = pd.read_parquet(OUTPUT / "runs" / row["run_key"] / "decisions.parquet")
        easy = pd.read_parquet(OUTPUT / "runs" / easy_key["run_key"] / "jobs.parquet")
        held = voluntary_delay(decisions, schedule)
        joined = schedule.set_index("job_id").join(easy.set_index("job_id")[["emissions_gco2e"]], rsuffix="_easy")
        joined["held_h"] = held / 3600
        kept = joined[joined["held_h"] > 0]
        rate = (kept["emissions_gco2e_easy"] - kept["emissions_gco2e"]) / kept["held_h"]
        for ax, values in zip(axes, (kept["held_h"], rate)):
            ordered = np.sort(values.to_numpy())
            ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=color[pid],
                    linewidth=1.6, label=names[pid])
    axes[0].set_xlabel("voluntary delay per held job (h)")
    axes[1].set_xlabel("gCO2e saved vs EASY per voluntary hour, per held job")
    axes[1].set_xscale("symlog", linthresh=1.0)
    for ax in axes:
        ax.set_ylabel("share of held jobs")
    axes[0].legend(fontsize=6.5)
    _save(plt, figure, "voluntary_delay_distribution")

    # Duration bands.
    table = bands.merge(rows[["run_key", *POLICY_FIELDS]], on="run_key")
    table = table[(table["scenario"] == Scenario("test").id) & (table["cell"] == "predicted/forecast")
                  & (table["policy_delay_hours"] == BUDGET_HOURS) & table["policy_id"].isin(shown)]
    figure, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / max(1, len(shown))
    for index, pid in enumerate(shown):
        members = table[table["policy_id"] == pid].set_index("band").reindex([b[2] for b in BANDS])
        ax.bar(np.arange(len(BANDS)) + index * width, members["saved_vs_easy_gco2e"] / 1e3, width * 0.9,
               color=color[pid], label=names[pid])
    ax.set_xticks(np.arange(len(BANDS)) + 0.4 - width / 2, [b[2] for b in BANDS])
    ax.set_xlabel("actual duration band (s)")
    ax.set_ylabel("kgCO2e saved vs EASY")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(fontsize=6.5)
    _save(plt, figure, "duration_bands")

    # Four information cells.
    figure, ax = plt.subplots(figsize=(8, 3.8))
    labels = ["jobs actual\ncarbon actual", "jobs predicted\ncarbon actual", "jobs actual\ncarbon forecast",
              "jobs predicted\ncarbon forecast"]
    for index, pid in enumerate(shown):
        values = []
        for information, carbon_information in CELLS:
            match = cell(rows, information, carbon_information)
            match = match[(match["policy_id"] == pid)]
            values.append(100 * match["saved_vs_easy"].iloc[0] if len(match) else np.nan)
        ax.bar(np.arange(4) + index * width, values, width * 0.9, color=color[pid],
               label=names[pid])
    ax.set_xticks(np.arange(4) + 0.4 - width / 2, labels)
    ax.set_ylabel("emissions saved vs EASY (%)")
    ax.legend(fontsize=6.5)
    _save(plt, figure, "information_cells")

    # Contention: capacity subsets, and the MILP where it finished.
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for pid in shown:
        points = []
        for nodes in CAPACITIES:
            scenario = Scenario("test", subset=1_000, max_nodes=SUBSET_MAX_NODES, nodes=nodes).id
            match = cell(rows, *REALISTIC, scenario=scenario)
            match = match[match["policy_id"] == pid]
            if len(match):
                points.append((match["utilisation"].iloc[0], 100 * match["saved_vs_easy"].iloc[0]))
        if points:
            x, y = zip(*sorted(points))
            axes[0].plot(x, y, "-o", color=color[pid], label=names[pid])
    axes[0].set_xlabel("utilisation (1,000-job subset at 880/512/256/128 nodes)")
    axes[0].set_ylabel("emissions saved vs EASY (%), realistic")
    axes[0].legend(fontsize=6.5)
    if milp_rows is not None:
        carbon_only = milp_rows[(milp_rows["jobs"] == 1_000) & (milp_rows["delay_price"] == 0.0)
                                & (milp_rows["grid_minutes"] == 60)]
        for pid in shown:
            group = carbon_only[carbon_only["configuration"] == pid].sort_values("nodes")
            axes[1].plot(group["nodes"], 100 * group["captured_model"], "-o", color=color[pid], label=names[pid])
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(CAPACITIES, [str(nodes) for nodes in CAPACITIES])
        axes[1].set_xlabel("nodes (1,000-job subset, jobs and carbon actual, 60-min MILP grid)")
        axes[1].set_ylabel("share of the MILP saving captured (%), planning model")
        axes[1].legend(fontsize=6)
    _save(plt, figure, "contention_milp")

    # Tables.
    parts = ["# Delay-aware scheduler: generated tables", "",
             f"Decision rule: {decision['rule']}", f"Decision: {'KEEP' if decision['keep'] else 'RESTORE'}", ""]
    vbase = cell(validation_rows, *REALISTIC, scenario=Scenario("validation").id)
    parts += ["## Validation, realistic cell", "", _markdown(
        vbase.sort_values(["family", "total_emissions_gco2e"]),
        {"family": "family", **SUMMARY, "non-dominated": "family_non_dominated", "rep": "representative"}), ""]
    for information, carbon_information in CELLS:
        frame = cell(rows, information, carbon_information)
        frame = frame[frame["policy_delay_hours"].isin([BUDGET_HOURS]) | frame["policy_mode"].isin(["fcfs", "easy"])]
        parts += [f"## Test, jobs {information} / carbon {carbon_information}", "",
                  _markdown(frame.drop_duplicates("policy_id"), SUMMARY), ""]
    parts += ["## Test delay sweep, realistic", "", _markdown(
        realistic.sort_values(["policy_id", "policy_delay_hours"]),
        {"delay h": "policy_delay_hours:g", **SUMMARY, "effective budget h": "effective_budget_mean_h:.2f",
         "reach-limited": "reach_limited_share:.1%"}), ""]
    for contention in (True,):
        frame = cell(rows, *REALISTIC, scenario=Scenario("test", contention=True).id)
        parts += ["## Test with resource-only contention, realistic", "", _markdown(frame, {
            **SUMMARY, "evaluated": "evaluated_jobs:,", "resource-only": "resource_only_jobs:,",
            "util": "utilisation:.1%"}), ""]
    for size in (1_000, 5_000):
        for nodes in CAPACITIES:
            scenario = Scenario("test", subset=size, max_nodes=SUBSET_MAX_NODES, nodes=nodes).id
            for information, carbon_information in (ORACLE, REALISTIC):
                frame = cell(rows, information, carbon_information, scenario=scenario)
                if len(frame):
                    parts += [f"## {scenario}, jobs {information} / carbon {carbon_information}", "",
                              _markdown(frame, {**SUMMARY, "util": "utilisation:.1%",
                                                "skipped >128": "subset_skipped_jobs"}), ""]
    band_table = table.assign(policy_id=table["policy_id"])
    parts += ["## Duration bands, test realistic 6 h", "", _markdown(band_table, {
        "policy": "label", "band": "band", "jobs": "jobs:,", "held": "held_jobs:,",
        "saved g": "saved_vs_easy_gco2e:,.0f", "voluntary h": "voluntary_delay_hours:,.0f",
        "g per voluntary h": "gco2e_per_voluntary_hour:,.1f", "held negative": "held_saving_negative",
        "held <1 g": "held_saving_below_1g", "held <0.1%": "held_saving_below_0p1pct"}), ""]
    if milp_rows is not None:
        milp_rows = milp_rows.assign(policy_id=milp_rows["configuration"])
        parts += ["## MILP (actual jobs, actual carbon)", "", _markdown(milp_rows, {
            "jobs": "jobs", "nodes": "nodes", "grid min": "grid_minutes:g", "lambda": "delay_price:.4g",
            "configuration": "configuration", "measured g": "measured_gco2e:,.0f", "model g": "model_gco2e:,.0f",
            "weighted g": "weighted_gco2e:,.0f", "captured measured": "captured_measured:.1%",
            "captured model": "captured_model:.1%", "captured weighted": "captured_weighted:.1%",
            "status": "solution_status", "gap": "mip_gap", "seconds": "solve_seconds"})]
    (OUTPUT / "report_tables.md").write_text("\n".join(parts) + "\n")
    print(f"figures and tables written to {OUTPUT}")


PHASES = {
    "prepare": prepare,
    "smoke": smoke,
    "validation": validation,
    "test": test,
    "milp": milp,
    "report": report,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=tuple(PHASES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embargo-hours", type=float, default=48.0,
                        help="prepare: validation jobs must be released this long before the carbon cutoff")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    started = perf_counter()
    PHASES[arguments.phase](arguments)
    log = OUTPUT / "commands.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"{datetime.now(UTC).isoformat()}\t{' '.join(sys.argv)}\t"
                     f"{perf_counter() - started:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
