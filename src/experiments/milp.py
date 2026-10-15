"""Offline optimum of carbon-aware scheduling, as a mixed-integer linear program.

The greedy policy in :mod:`hpc_sim.carbon_aware` picks each job's cleanest start
on its own and lets contention sort out the rest. With perfect information it is
a benchmark, not a bound: every job aims at the same clean hour and nothing asks
which of them should get it. This module asks, on small instances, by choosing
every start at once.

Job ``j`` may start at any instant of its candidate set ``S_j``; ``x[j, s] = 1``
when it does:

    minimise    sum_j sum_s x[j, s] * (c[j, s] + w * delay[j, s])
    subject to  sum_s x[j, s] = 1                                  each job starts once
                sum_{j, s : s <= t < s + d_j} n_j * x[j, s] <= N    for every t in T
                x[j, s] in {0, 1}

``c[j, s]`` is :func:`hpc_sim.carbon_cost_gco2e`, the constant-average-power cost
the greedy scores its own candidates with, so both minimise the same quantity.
``T`` is the union of all candidate starts: occupancy only rises when a job
starts, so capacity checked at every possible start holds at every instant.
``w`` is gCO2e per hour of delay; the default only breaks ties toward the
earliest start, as the greedy does, and a larger one turns the objective into a
CO2/QoS trade-off.

``S_j`` is the greedy's own candidate set (release, then the decision grid up to
``release + max_delay``) plus the instant each reference schedule actually
started ``j``. Every reference schedule is therefore a feasible solution, so the
optimum provably matches or beats each of them, and the gap to the greedy is a
proven lower bound on its distance from optimal. The same union keeps the
program feasible when contention pushed a reference job past its budget, which
the greedy allows because its budget bounds voluntary delay only.

The plan is replayed through the simulator and accounted like any other run, so
its metrics are comparable and a capacity violation raises instead of being
reported.

Run it with ``python -m experiments.milp``.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timedelta
from math import fsum
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from carbon_intensity.series import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    PM100_PARTITION_1_NODES,
    CarbonAwareScheduler,
    CarbonSignal,
    Cluster,
    EASYBackfillScheduler,
    Job,
    RuntimeEstimateSource,
    SimulationResult,
    Simulator,
    TraceReplayScheduler,
    account_schedule,
    carbon_cost_gco2e,
    schedule_metrics,
)
from hpc_sim.carbon_aware import candidate_start_times
from hpc_sim.models import seconds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = PROJECT_ROOT / "data/processed/pm100_debug_5000.parquet"
DEFAULT_ACTUAL = PROJECT_ROOT / "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/experiments"

HOUR = timedelta(hours=1)
#: gCO2e per hour of delay. Breaks ties toward the earliest start, and cannot
#: move the optimum by more than this times the total delay in hours.
TIE_BREAK = 1e-3
#: Relative gap at which HiGHS may declare optimality.
MIP_GAP = 1e-6


def planning_emissions(jobs, starts, signal: CarbonSignal) -> float:
    """Score the exact MILP coefficients, including CI bucket boundaries.

    Ex-post average-power accounting samples CI every power-profile interval;
    it is close to this integral but is not the objective a solver bounds.
    """
    return fsum(carbon_cost_gco2e(job, starts[job.job_id], signal) for job in jobs)


def solve(
    jobs: tuple[Job, ...],
    signal: CarbonSignal,
    *,
    nodes: int,
    max_delay: timedelta,
    granularity: timedelta,
    references: Iterable[SimulationResult] = (),
    delay_weight: float = TIE_BREAK,
    time_limit: float = 600.0,
    reach: timedelta | None = None,
) -> tuple[dict[object, datetime], dict[str, object]]:
    """Every job's start in the cheapest schedule, and how the solver got there.

    ``solution_status`` separates a plan proven optimal within the gap, a
    feasible plan found before the time limit (not a proof of optimality), and
    no plan at all, in which case the returned mapping is empty.

    ``reach`` applies the online policy's effective budget, capped at
    ``max(0, reach - scheduling_duration)``. Reference starts remain in the
    domain even when capacity pushed them past that voluntary budget.
    """

    if max_delay < timedelta(0) or granularity <= timedelta(0):
        raise ValueError("max_delay must be nonnegative and granularity positive")
    budgets = [max_delay if reach is None else min(
        max_delay, max(timedelta(0), reach - seconds(job.scheduling_duration_seconds))
    ) for job in jobs]
    extra: dict[object, set[datetime]] = defaultdict(set)
    for result in references:
        for record in result.records:
            extra[record.job_id].add(record.start_time)
    candidates = [
        sorted(
            set(candidate_start_times(job, max_delay=budget, granularity=granularity))
            | extra[job.job_id]
        )
        for job, budget in zip(jobs, budgets)
    ]
    times = sorted({start for starts in candidates for start in starts})
    # The domain holds every reference start, including ones contention pushed
    # past the voluntary budget; count them so the domain is stated, not implied.
    beyond_budget = sum(
        start > job.release_time + budget
        for job, starts, budget in zip(jobs, candidates, budgets)
        for start in starts
    )

    cost: list[float] = []
    owner: list[int] = []
    rows: list[int] = []
    columns: list[int] = []
    load: list[int] = []
    for index, (job, starts) in enumerate(zip(jobs, candidates)):
        runtime = seconds(job.scheduling_duration_seconds)
        for start in starts:
            column = len(cost)
            cost.append(
                carbon_cost_gco2e(job, start, signal)
                + delay_weight * ((start - job.release_time) / HOUR)
            )
            owner.append(index)
            # Rows are the instants this start keeps the job running: [s, s + d).
            first, last = bisect_left(times, start), bisect_left(times, start + runtime)
            rows.extend(range(first, last))
            columns.extend([column] * (last - first))
            load.extend([job.nodes_required] * (last - first))

    variables = len(cost)
    started = perf_counter()
    result = milp(
        np.array(cost),
        integrality=np.ones(variables),
        bounds=Bounds(0, 1),
        constraints=[
            LinearConstraint(
                coo_array((np.ones(variables), (owner, np.arange(variables))),
                          shape=(len(jobs), variables)),
                1, 1,
            ),
            LinearConstraint(
                coo_array((load, (rows, columns)), shape=(len(times), variables)),
                -np.inf, nodes,
            ),
        ],
        options={"time_limit": time_limit, "mip_rel_gap": MIP_GAP},
    )
    elapsed = perf_counter() - started

    chosen: dict[object, datetime] = {}
    if result.x is not None:
        offset = 0
        for job, starts in zip(jobs, candidates):
            chosen[job.job_id] = starts[int(np.argmax(result.x[offset:offset + len(starts)]))]
            offset += len(starts)
    return chosen, {
        "solution_status": (
            "optimal" if result.status == 0
            else "no-solution" if result.x is None
            else "feasible-timeout" if result.status == 1
            else "feasible-other"
        ),
        "starts_beyond_budget": beyond_budget,
        "solver_status": result.message,
        "proven_optimal": result.status == 0,
        "mip_gap": result.mip_gap,
        "objective_gco2e": result.fun,
        "dual_bound_gco2e": result.mip_dual_bound,
        "branch_nodes": result.mip_node_count,
        "solve_seconds": elapsed,
        "variables": variables,
        "capacity_rows": len(times),
    }


def replay(jobs: tuple[Job, ...], starts: dict[object, datetime], nodes: int) -> SimulationResult:
    """Run a fixed plan through the simulator, which raises if it overfills the machine."""

    scheduler = TraceReplayScheduler()
    scheduler.name = "milp-optimum"
    planned = tuple(replace(job, trace_start_time=starts[job.job_id]) for job in jobs)
    result = Simulator(planned, Cluster(nodes), scheduler).run()
    # The plan borrowed the trace-start field; restore the recorded one so the
    # delay-vs-trace columns mean the same thing in every row.
    recorded = {job.job_id: job.trace_start_time for job in jobs}
    return result.replace_records(tuple(
        replace(record, trace_start_time=recorded[record.job_id]) for record in result.records
    ))


def compare(
    jobs: tuple[Job, ...],
    provider: TimeSeriesCarbonIntensityProvider,
    *,
    nodes: int = PM100_PARTITION_1_NODES,
    max_delay: timedelta = timedelta(hours=6),
    granularity: timedelta = HOUR,
    delay_weight: float = TIE_BREAK,
    time_limit: float = 600.0,
) -> list[dict[str, object]]:
    """EASY, the greedy oracle and the MILP optimum on one instance, scored alike."""

    def run(scheduler) -> SimulationResult:
        return account_schedule(Simulator(jobs, Cluster(nodes), scheduler).run(), jobs, provider)

    estimate = RuntimeEstimateSource.SCHEDULING
    easy = run(EASYBackfillScheduler(runtime_estimate=estimate))
    greedy = run(CarbonAwareScheduler(
        provider, max_delay=max_delay, decision_granularity=granularity, runtime_estimate=estimate
    ))
    starts, solver = solve(
        jobs, CarbonSignal(provider), nodes=nodes, max_delay=max_delay, granularity=granularity,
        references=(easy, greedy), delay_weight=delay_weight, time_limit=time_limit,
    )
    if not starts:
        raise ValueError(f"no schedule found on {nodes} nodes: {solver['solver_status']}")
    runs = {
        "easy": easy,
        "carbon_greedy": greedy,
        "milp_optimum": account_schedule(replay(jobs, starts, nodes), jobs, provider),
    }

    metrics = {name: schedule_metrics(result) for name, result in runs.items()}
    measured = {name: scored.total_emissions_gco2e for name, scored in metrics.items()}
    # Optimality is proven on the average-power model the program minimises, so
    # the optimality figures are taken there; measured emissions stay alongside.
    signal = CarbonSignal(provider)
    model = {name: planning_emissions(
        jobs, {record.job_id: record.start_time for record in result.records}, signal
    ) for name, result in runs.items()}
    available = model["easy"] - model["milp_optimum"]
    budget = max_delay.total_seconds()

    return [
        {
            "configuration": name,
            "nodes": nodes,
            "max_delay_hours": max_delay / HOUR,
            "decision_granularity_minutes": granularity / timedelta(minutes=1),
            "delay_weight_gco2e_per_hour": delay_weight,
            "emissions_saved_vs_easy": 1.0 - measured[name] / measured["easy"],
            "model_emissions_tco2e": model[name] / 1e6,
            "optimal_saving_captured": (
                (model["easy"] - model[name]) / available if available else float("nan")
            ),
            "gap_to_optimum": model[name] / model["milp_optimum"] - 1.0,
            "jobs_past_budget": sum(record.waiting_seconds > budget for record in result.records),
            **metrics[name].as_row(),
            **(solver if name == "milp_optimum" else {}),
        }
        for name, result in runs.items()
    ]


def _report(rows: list[dict[str, object]]) -> None:
    header = f"{'configuration':<15}{'nodes':>6}{'tCO2e':>10}{'saved':>8}{'captured':>10}"
    print(f"{header}{'gap':>9}{'wait mean':>11}{'wait p95':>10}{'bsld mean':>11}{'past budget':>13}")
    for row in rows:
        print(
            f"{row['configuration']:<15}{row['nodes']:>6}{row['total_emissions_tco2e']:>10.4f}"
            f"{row['emissions_saved_vs_easy']:>8.2%}{row['optimal_saving_captured']:>10.2%}"
            f"{row['gap_to_optimum']:>9.3%}{row['waiting_mean_s']:>11,.0f}"
            f"{row['waiting_p95_s']:>10,.0f}{row['bounded_slowdown_mean']:>11.2f}"
            f"{row['jobs_past_budget']:>13}"
        )
        if "solver_status" in row:
            print(
                f"  solver: {row['solver_status']} in {row['solve_seconds']:.1f} s, "
                f"{row['variables']:,} variables, {row['capacity_rows']:,} capacity rows, "
                f"gap {row['mip_gap']:.1e}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--carbon-cache", type=Path, default=DEFAULT_ACTUAL)
    parser.add_argument("--limit", type=int, default=500,
                        help="contiguous prefix of the chronological workload")
    parser.add_argument("--nodes", type=int, nargs="+", default=[PM100_PARTITION_1_NODES])
    parser.add_argument("--max-delay-hours", type=float, default=6.0)
    parser.add_argument("--decision-granularity-minutes", type=float, default=60.0)
    parser.add_argument("--delay-weight", type=float, default=TIE_BREAK,
                        help="gCO2e per hour of delay; the default only breaks ties")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="solver seconds per instance")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-output", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        from hpc_sim.workload import load_jobs

        jobs = load_jobs(arguments.workload, limit=arguments.limit)
        provider = TimeSeriesCarbonIntensityProvider.load(arguments.carbon_cache)
        rows = [
            row
            for nodes in arguments.nodes
            for row in compare(
                jobs,
                provider,
                nodes=nodes,
                max_delay=timedelta(hours=arguments.max_delay_hours),
                granularity=timedelta(minutes=arguments.decision_granularity_minutes),
                delay_weight=arguments.delay_weight,
                time_limit=arguments.time_limit,
            )
        ]
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"MILP comparison failed: {error}", file=sys.stderr)
        return 1

    print(f"jobs                     {len(jobs):,}")
    print(f"maximum carbon delay     {arguments.max_delay_hours:g} h")
    print(f"decision granularity     {arguments.decision_granularity_minutes:g} min")
    print()
    _report(rows)

    if arguments.no_output:
        return 0
    import pandas as pd

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / (
        f"milp_{len(jobs)}jobs_{arguments.max_delay_hours:g}h_"
        f"{arguments.decision_granularity_minutes:g}min_w{arguments.delay_weight:g}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nmetrics written          {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
