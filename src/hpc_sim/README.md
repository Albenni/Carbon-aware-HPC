# Discrete-event HPC simulator

This package simulates when HPC jobs run under a finite node budget and measures the resulting time, energy, and carbon costs.

Scheduling is separate from accounting: all energy and emission calculations come from `carbon_accounting`, and all grid signals come from `carbon_intensity`.

## Event model

The simulator uses continuous timestamps rather than fixed ticks.

| Event                     | Meaning                                   |
| ------------------------- | ----------------------------------------- |
| `RELEASE`                 | a job becomes eligible at `release_time`  |
| `COMPLETION`              | a running job ends and releases its nodes |
| `CARBON_INTENSITY_CHANGE` | a grid-signal boundary                    |
| `TIMER`                   | a scheduler-requested wakeup              |

All events at the same timestamp are processed before scheduling. Completions are applied before releases, so nodes released at `t` can be reused immediately at `t`.

Ties are resolved by a monotonic sequence counter, making runs deterministic.

Carbon-aware policies normally decide once at release and request a `TIMER` for the selected start rather than waking at every carbon bucket. A deferred job therefore adds one event, not one event per bucket.

## Resources

`Cluster` models equivalent nodes by count rather than identity. Trace node IDs are used for validation only.

Default capacity is **880 nodes** (`PM100_PARTITION_1_NODES`), corresponding to the distinct node IDs observed in PM100 partition 1.

Observed peak occupancy is lower:

- raw completed trace: 787 nodes;
- cleaned power-profile dataset: 774 nodes.

These are lower bounds on capacity and useful sensitivity-analysis values.

All allocations enforce:

```text
0 <= busy <= total
```

Violations raise `CapacityError`. Jobs larger than the cluster are rejected when the `Simulator` is constructed.

## Schedulers

```python
class Scheduler(ABC):
    name: str

    def select(self, now, queue, cluster, simulator) -> tuple[Job, ...]: ...
    def on_release(self, job, now, simulator) -> None: ...
```

### FCFS

`FCFSScheduler` considers jobs in `(release_time, job_id)` order and stops at the first job that does not fit. Smaller jobs never overtake a blocked head job.

### EASY backfilling

`EASYBackfillScheduler` reserves capacity for the first queued job that cannot start, the **pivot**.

Later jobs may backfill only if they cannot delay the pivot's reservation. A backfill is safe when it either:

- completes before the reservation; or
- uses only capacity the pivot will not require at the reservation.

Only the pivot is reserved, preventing starvation of the queue head.

### Power-capped EASY

`PowerCappedEASYScheduler` applies EASY while enforcing an aggregate power limit.

### Carbon-aware EASY

`CarbonAwareScheduler` holds each job until the lowest-carbon candidate within its delay budget, then returns it to normal EASY placement.

Its main options are:

- `max_delay`: fixed voluntary delay budget;
- `max_delay_fraction`: budget proportional to predicted runtime;
- `forecast=True`: use the latest issued forecast rather than future actuals;
- `reach`: limit the budget to the horizon a forecast archive can safely cover.

Scheduler names encode the selected combination, including:

- `carbon-aware`
- `carbon-scaled-delay`
- `carbon-aware-forecast`
- `carbon-forecast-scaled-delay`

with `-bounded` when applicable.

### Carbon + power cap

`PowerCappedCarbonAwareScheduler` first applies the carbon target, then admits jobs only when both node and aggregate-power constraints allow them to start.

### Trace replay

`TraceReplayScheduler` starts every job at its recorded `start_time` and is used to validate the event engine.

Schedulers may call:

```python
simulator.request_wakeup(when)
```

Past wakeups raise an error. EASY does not require explicit wakeups because completions already trigger reservation recomputation.

## Scheduling information

Backfilling requires runtime estimates. `RuntimeEstimateSource` makes their source explicit:

| Source       | Estimate                          | Use                                                                 |
| ------------ | --------------------------------- | ------------------------------------------------------------------- |
| `TIME_LIMIT` | requested walltime                | classic EASY                                                        |
| `SCHEDULING` | `Job.scheduling_duration_seconds` | predicted runtime, or actual runtime when no prediction is attached |

PM100 jobs are typically much shorter than their requested walltime: median runtime is **2.5%** of the requested limit. Walltime-based EASY therefore creates conservative reservations.

Power decisions similarly use:

```python
Job.scheduling_average_power_watts
```

Policies never read measured execution profiles directly.

Models in `src/job_prediction/` populate both scheduling fields from prediction-only Parquet artifacts. Actual completion times still control resource release, while energy and emissions are scored from measured profiles.

## Power-aware baseline

Changing a start time does not change a job's duration or power profile, so total energy is schedule-invariant in this model.

The power-aware baseline therefore constrains **when** power is consumed rather than trying to reduce total energy:

```text
sum(running job average power) <= power_cap_watts
```

The cap applies to both job starts and EASY reservations.

Jobs whose own scheduling-power estimate exceeds the cap are rejected rather than left permanently queued.

## Carbon-aware scheduling

`CarbonAwareScheduler` evaluates candidate starts once when a job becomes eligible.

| Component    | Definition                                                        |
| ------------ | ----------------------------------------------------------------- |
| delay budget | measured from `release_time`                                      |
| candidates   | release time plus carbon-signal boundaries up to the budget       |
| cost         | `energy × mean carbon intensity` over `[start, start + duration]` |
| power model  | scheduling average power                                          |
| tie-break    | earliest start                                                    |

Important invariants:

- `max_delay=0` reproduces EASY exactly;
- a held job is removed from placement and cannot become the EASY pivot;
- the delay budget bounds voluntary carbon delay, not later queueing delay;
- once the target time arrives, the job competes normally for resources.

The cheapest candidate is fixed at release, so the target is not reconsidered later.

`CarbonSignal` materializes the source series once and stores a cumulative integral, making interval scoring constant-time. Requests outside available signal coverage raise an error.

Jobs without a power value use ordinary EASY placement and receive no carbon target. This allows terminal PM100 executions to contribute node contention without assigning artificial energy data.

The policy is greedy per job. Jobs may independently converge on the same clean interval, creating queueing and peak-power effects that are not part of the carbon objective.

## Duration-scaled delay

With:

```python
max_delay_fraction=f
```

the voluntary delay budget is:

```text
scheduling_duration_seconds * f
```

A factor of `1.0` permits a delay up to one predicted runtime. A factor of zero reproduces EASY.

Candidate starts remain aligned to the carbon grid, so a budget shorter than the next grid boundary produces no shift.

## Carbon scheduling under a power cap

`PowerCappedCarbonAwareScheduler` combines the carbon target with the same aggregate-power constraint as `PowerCappedEASYScheduler`.

With `max_delay=0`, both schedulers produce identical starts when given the same runtime estimates.

The cap uses `Job.scheduling_average_power_watts`. With predicted power, the planned schedule may satisfy the cap while measured ex-post power exceeds it.

Resource-only jobs without a power estimate are rejected by power-capped schedulers.

### Debug workload

5,000 jobs, exact scheduling inputs, six-hour fixed delay, duration-scaled factor `1.0`, and a cap equal to 80% of FCFS peak:

| Scheduler          | Emissions (tCO2e) | Peak (MW) | Mean wait (s) | Wait p95 (s) | Mean slowdown |
| ------------------ | ----------------: | --------: | ------------: | -----------: | ------------: |
| EASY               |            4.6411 |     0.606 |          31.2 |          344 |          1.08 |
| Power-capped EASY  |            4.6412 |     0.485 |          36.4 |          376 |          1.10 |
| Carbon-aware       |            4.4245 |     0.586 |       6,391.4 |       20,353 |        225.70 |
| Carbon + power cap |            4.4250 |     0.485 |       6,420.3 |       20,370 |        225.76 |
| Duration-scaled    |            4.3962 |     0.593 |         257.0 |          444 |          1.14 |

The combined carbon/power policy cuts the carbon-aware peak by **17.2%** while retaining **99.7744%** of its saving against EASY.

The duration-scaled policy saves **5.28%** against EASY, compared with **4.67%** for the fixed six-hour delay, while reducing mean wait from 6,391 s to 257 s.

Its maximum wait is 73,972 s because long jobs receive larger budgets; mean bounded slowdown is 1.14 and maximum bounded slowdown is 8.53.

## Per-job records and accounting

`JobRecord` stores simulated start and end times plus waiting and turnaround relative to both:

- `release_time`;
- `submit_time`.

`account_schedule(result, jobs, provider)` adds:

| Field                           | Meaning                                          |
| ------------------------------- | ------------------------------------------------ |
| `energy_kwh`                    | measured PM100 power profile                     |
| `emissions_gco2e`               | measured profile against actual carbon intensity |
| `energy_kwh_average_model`      | average-power approximation                      |
| `emissions_gco2e_average_model` | average-power emissions approximation            |

The measured values are evaluation ground truth. On the debug subset, the average-power emissions approximation differs by about **0.2%** in aggregate.

Accounting runs after simulation, keeping the event engine independent of the carbon provider.

`check_coverage` validates that the final schedule remains inside the available carbon series before accounting begins.

For mixed clean/terminal workloads, `SimulationResult.replace_records` retains only clean evaluation jobs for energy, carbon, and QoS summaries while preserving full-run node occupancy, peak, and makespan.

Average job power defaults to the **duration-weighted mean** of the measured profile. This guarantees that measured and average-power representations consume the same energy.

Use:

```python
average_power_source="stored"
```

to use the trace's stored arithmetic mean instead.

## Metrics

`schedule_metrics(result)` operates only on an accounted `SimulationResult`.

| Group  | Metrics                                                     |
| ------ | ----------------------------------------------------------- |
| Carbon | total and mean emissions, measured and average-power models |
| Energy | total energy, peak aggregate power                          |
| QoS    | waiting, turnaround, bounded slowdown                       |
| System | node utilisation, peak nodes, makespan, throughput          |

Each QoS quantity is reported as a `Distribution` containing mean, median, p95, p99, and maximum.

Bounded slowdown is:

```text
max(1, (wait + runtime) / max(runtime, 10 s))
```

The 10-second floor prevents very short PM100 jobs from dominating the metric.

`reference="release"` is the default QoS origin. `reference="submit"` reports user-perceived delay.

Peak power is reconstructed from each job's duration-weighted mean power. It does not capture within-job 20-second fluctuations, but it matches the quantity controlled by power-capped schedulers.

## Use

```python
import sys
sys.path.insert(0, "src")

from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import (
    Cluster,
    FCFSScheduler,
    Simulator,
    account_schedule,
    format_metrics,
    schedule_metrics,
)
from hpc_sim.workload import load_jobs

jobs = load_jobs("data/processed/pm100_debug_5000.parquet")

result = Simulator(
    jobs,
    Cluster(880),
    FCFSScheduler(),
).run()

provider = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"
)

result = account_schedule(result, jobs, provider)

print(format_metrics(schedule_metrics(result)))
```

Policy examples:

```python
from datetime import timedelta

from hpc_sim import (
    CarbonAwareScheduler,
    EASYBackfillScheduler,
    PowerCappedCarbonAwareScheduler,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
)

# Classic EASY: requested walltime
EASYBackfillScheduler()

# EASY with scheduling-time runtime estimate
EASYBackfillScheduler(
    runtime_estimate=RuntimeEstimateSource.SCHEDULING
)

# Power-aware baseline
PowerCappedEASYScheduler(
    power_cap_watts=680_000.0
)

# Carbon-aware oracle
CarbonAwareScheduler(
    provider,
    max_delay=timedelta(hours=6),
)

# Runtime-scaled delay
CarbonAwareScheduler(
    provider,
    max_delay_fraction=1.0,
)

# Issued forecasts
CarbonAwareScheduler(
    archive_provider,
    forecast=True,
    max_delay=timedelta(hours=6),
    reach=reach,
)

CarbonAwareScheduler(
    archive_provider,
    forecast=True,
    max_delay_fraction=1.0,
    reach=reach,
)

# Carbon + power cap
PowerCappedCarbonAwareScheduler(
    provider,
    680_000.0,
    max_delay=timedelta(hours=6),
)
```

`hpc_sim` itself uses only the standard library. `hpc_sim.workload` additionally requires `pyarrow`.

### CLI examples

```bash
# Replay
.venv/bin/python scripts/run_simulation.py \
  --scheduler replay --limit 5000

# EASY
.venv/bin/python scripts/run_simulation.py \
  --scheduler easy \
  --workload data/processed/pm100_clean.parquet

# Power cap
.venv/bin/python scripts/run_simulation.py \
  --scheduler power-cap \
  --power-cap-mw 0.68

# Fixed-delay carbon oracle
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon \
  --max-delay-hours 6 \
  --runtime-estimate scheduling

# Runtime-scaled carbon oracle
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon-scaled-delay \
  --max-delay-fraction 1 \
  --runtime-estimate scheduling

# Carbon + power cap
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon-power-cap \
  --power-cap-mw 0.485 \
  --max-delay-hours 6 \
  --runtime-estimate scheduling

# Forecast-based carbon scheduling
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon-forecast \
  --carbon-cache data/carbon_intensity/actual/actual.json \
  --forecast-archive data/carbon_intensity/snapshots/test_boosted_ridge.json \
  --max-delay-hours 6 \
  --runtime-estimate scheduling

# Forecast + runtime-scaled delay
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon-forecast-scaled-delay \
  --carbon-cache data/carbon_intensity/actual/actual.json \
  --forecast-archive data/carbon_intensity/snapshots/test_boosted_ridge.json \
  --max-delay-fraction 1 \
  --runtime-estimate scheduling
```

Job-model evaluation:

```bash
.venv/bin/python scripts/train_job_models.py

.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet
```

Baseline comparison:

```bash
.venv/bin/python scripts/compare_baselines.py \
  --limit 5000
```

Carbon and power-aware variants:

```bash
.venv/bin/python scripts/compare_baselines.py \
  --limit 5000 \
  --no-replay \
  --runtime-estimate scheduling \
  --max-delay-hours 6 \
  --max-delay-fraction 1 \
  --power-cap-fraction 0.8
```

Forecast comparison:

```bash
.venv/bin/python scripts/compare_baselines.py \
  --limit 5000 \
  --no-replay \
  --runtime-estimate scheduling \
  --max-delay-hours 6 \
  --max-delay-fraction 1 \
  --carbon-cache data/carbon_intensity/actual/actual.json \
  --forecast-archive data/carbon_intensity/snapshots/test_boosted_ridge.json \
  --forecast-archive data/carbon_intensity/snapshots/test_seasonal_daily.json
```

Full job-information × carbon-signal × delay-policy matrix:

```bash
.venv/bin/python scripts/compare_baselines.py \
  --no-replay \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions gradient=data/job_predictions/test_predictions.parquet \
  --job-predictions ridge=data/job_predictions/ridge_baseline/test_predictions.parquet \
  --carbon-cache data/carbon_intensity/actual/actual.json \
  --forecast-archive data/carbon_intensity/snapshots/test_persistence.json \
  --forecast-archive data/carbon_intensity/snapshots/test_seasonal_daily.json \
  --forecast-archive data/carbon_intensity/snapshots/test_seasonal_weekly.json \
  --forecast-archive data/carbon_intensity/snapshots/test_ridge_direct.json \
  --forecast-archive data/carbon_intensity/snapshots/test_ridge_refit_once.json \
  --forecast-archive data/carbon_intensity/snapshots/test_ridge_refit_14d.json \
  --forecast-archive data/carbon_intensity/snapshots/test_boosted_ridge.json \
  --runtime-estimate scheduling \
  --max-delay-hours 6 \
  --max-delay-fraction 1
```

Each prediction artifact adds a labelled scheduling-input arm. Each forecast archive adds a carbon-signal arm. Oracle and forecast runs share the same reachable horizon, and all runs are scored ex-post from measured power and actual carbon intensity.

Carbon/QoS frontier:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --limit 5000

.venv/bin/python scripts/carbon_tradeoff.py \
  --limit 5000 \
  --max-delay-hours 6 24 \
  --decision-granularity-minutes 15 60 240
```

Predicted-input frontier:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet \
  --max-delay-hours 0 1 3 6 12 24 \
  --decision-granularity-minutes 15
```

Terminal-contention scenario:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --limit 5000 \
  --contention-workload data/job_table.parquet \
  --max-delay-hours 0 6
```

Tests:

```bash
.venv/bin/python -m unittest discover \
  -s tests \
  -p "check_*.py"

.venv/bin/python tests/check_hpc_sim_simulator.py
.venv/bin/python tests/check_hpc_sim_baselines.py
.venv/bin/python tests/check_hpc_sim_carbon_aware.py
.venv/bin/python tests/check_job_prediction.py
```

Editable install:

```bash
.venv/bin/pip install -e .
```

The full 157,062-job trace takes about fourteen minutes for a four-policy comparison. Most runtime is spent integrating 20-second power samples during emission accounting; scheduling itself takes under a second per policy.

## Validation

`TraceReplayScheduler` reproduces every recorded start time exactly on both the 5,000-job debug subset and the full 157,062-job clean trace.

Its peak occupancy is **774 nodes**, matching an independent sweep-line calculation. FCFS reaches the configured capacity of 880 nodes without exceeding it.

The terminal-contention loader admits **50,165** valid non-completed partition-1 executions:

| Status        |   Jobs |
| ------------- | -----: |
| Failed        | 29,561 |
| Cancelled     | 10,876 |
| Timeout       |  8,564 |
| Out of memory |    997 |
| Node failure  |    167 |

These executions consume nodes but have no power profiles.

One additional timed-out execution has eligibility before submission, and 18 records have no positive execution interval; these are rejected rather than repaired.

Across the full clean trace, energy is identical for all schedulers:

```text
553.34 MWh
```

to six decimals.

Emissions differ because schedules overlap the carbon signal differently:

- replay: `156.333 tCO2e`;
- FCFS: `156.536 tCO2e`.

EASY reservations are checked directly through `first_reservations`: no pivot begins after its promised start on synthetic, random, or PM100 workloads.

Tests also cover:

- a short safe backfill;
- a long backfill that would violate the reservation;
- a long backfill using capacity not required by the pivot.

Carbon-aware tests verify:

- candidate cost matches `carbon_accounting` to nine decimals;
- targets remain within delay budgets;
- jobs never start before their selected carbon target;
- zero delay reproduces EASY exactly;
- carbon-aware shifts change emissions without changing energy;
- duration-scaled budgets use predicted duration;
- zero duration factor reproduces EASY;
- zero-delay carbon + power-cap reproduces power-capped EASY.

## Baseline results

Full 157,062-job clean trace, 880 nodes, requested-walltime EASY estimates, power cap at 80% of FCFS peak:

| Metric                |  Replay |    FCFS |    EASY | Power cap |
| --------------------- | ------: | ------: | ------: | --------: |
| Emissions (tCO2e)     | 156.333 | 156.536 | 156.536 |   156.529 |
| Energy (MWh)          |  553.34 |  553.34 |  553.34 |    553.34 |
| Peak power (MW)       |   0.700 |   0.846 |   0.852 |     0.676 |
| Mean wait (s)         | 2,434.3 |   277.8 |   252.7 |     222.9 |
| Wait p99 (s)          |  62,099 |  10,341 |  10,004 |     8,993 |
| Wait max (s)          | 410,317 |  56,649 |  58,043 |    58,043 |
| Mean bounded slowdown |   71.37 |    2.41 |    2.12 |      1.68 |
| Max bounded slowdown  |  35,852 |   4,344 |   4,320 |     2,690 |

The carbon-blind baselines have nearly identical emissions. Their scheduling objectives do not use carbon intensity.

The power cap reduces peak power from `0.846` to `0.676 MW` without changing total energy.

Its lower mean waiting time comes from redistribution rather than a uniformly better schedule. Among the 157,062 jobs:

- 120,113 one-node jobs wait 58 s instead of 87 s;
- 65–256-node jobs wait 1,099 s instead of 520 s.

The maximum wait remains 58,043 s.

## Carbon-aware results

Full trace, 880 nodes, 15-minute carbon grid, `SCHEDULING` runtime estimates:

| Delay budget | Emissions (tCO2e) | Saved | Mean wait (s) | Wait p95 (s) | Mean slowdown | Slowdown p95 | Peak MW |
| ------------ | ----------------: | ----: | ------------: | -----------: | ------------: | -----------: | ------: |
| 0 (= EASY)   |           156.535 | 0.00% |           251 |          111 |          2.09 |         1.51 |   0.852 |
| 1 h          |           155.894 | 0.41% |         1,245 |        3,430 |         32.11 |        240.3 |   0.866 |
| 3 h          |           154.577 | 1.25% |         4,290 |       10,541 |        142.42 |        887.1 |   0.877 |
| 6 h          |           153.008 | 2.25% |         8,402 |       21,236 |        287.60 |      1,776.5 |   0.878 |
| 12 h         |           149.631 | 4.41% |        17,630 |       42,681 |        566.75 |      3,268.4 |   0.857 |
| 24 h         |           145.427 | 7.10% |        49,067 |       85,688 |      1,679.09 |      8,078.6 |   0.903 |

Energy remains `553.34 MWh` in every row.

Carbon savings increase monotonically with delay budget, but QoS costs increase much faster. From a 1-hour to a 24-hour budget, carbon saving grows by about 17× while mean waiting grows by about 39×.

The 2020 IT-NO series also limits the available benefit. Its median daily range is `71 gCO2e/kWh`, or 24.5% of daily mean intensity, while the cleanest quarter-hour is about 11% below the day's mean. A 24-hour scheduler therefore has limited room to reduce emissions even with perfect information.

Peak power increases to `0.903 MW` at a 24-hour budget because many jobs independently target the same low-carbon periods.

Tail QoS is substantially worse than the mean. At a 6-hour budget:

- mean bounded slowdown: `287.60`;
- p95 bounded slowdown: `1,776.5`.

## Decision granularity

5,000-job debug workload:

| Decision grid | 6 h saved | 24 h saved |
| ------------- | --------: | ---------: |
| 15 min        |     4.67% |     11.09% |
| 60 min        |     4.45% |     10.94% |
| 240 min       |     3.67% |      9.94% |

Moving from 15-minute to hourly decisions loses little. A four-hour grid loses roughly one fifth of the saving.

The debug subset reaches larger savings than the full trace because it covers a different period of the carbon series. Frontier values are therefore comparable only within the same workload window.

## Predicted scheduling inputs

The held-out job-model cohort contains **23,560 jobs**. The same carbon experiment is run twice:

1. with actual scheduling durations and powers;
2. with gradient-model predictions.

Completion times, energy, and emissions always use measured outcomes.

`saved` is measured against EASY with actual scheduling inputs (`17.4712 tCO2e`). `retained` is the fraction of perfect-information carbon saving preserved by the predicted-input run.

| Delay | Actual tCO2e | Saved | Predicted tCO2e |  Saved | Retained |
| ----- | -----------: | ----: | --------------: | -----: | -------: |
| 0     |      17.4712 | 0.00% |         17.4754 | -0.02% |        — |
| 1 h   |      17.4347 | 0.21% |         17.4549 |  0.09% |   44.69% |
| 3 h   |      17.3210 | 0.86% |         17.3951 |  0.44% |   50.66% |
| 6 h   |      17.2402 | 1.32% |         17.3727 |  0.56% |   42.65% |
| 12 h  |      16.9118 | 3.20% |         17.1681 |  1.73% |   54.18% |
| 24 h  |      16.7036 | 4.39% |         17.0137 |  2.62% |   59.61% |

The 6-hour row matches `job_prediction.scheduling_impact`.

Predicted scheduling inputs preserve roughly 40–60% of the perfect-information carbon saving. Retention generally improves with wider delay budgets because duration errors have more alternative low-carbon starts available.

The 24-hour perfect-information saving is `4.39%`, lower than the `7.10%` full-trace result because this cohort covers only the last ten days of the trace.

### Prediction effects on EASY

Even at zero carbon delay, predictions change EASY placement.

| Input                       | Mean wait | Mean bounded slowdown |
| --------------------------- | --------: | --------------------: |
| Actual scheduling durations |   422.9 s |                  2.11 |
| Predicted durations         |   458.3 s |                  3.00 |

With actual durations, EASY makes 1,247 placements that differ from FCFS. With predicted durations, it makes none: the resulting schedule is FCFS.

The cause is runtime underprediction. The gradient model has:

- overall duration bias: `-568 s`;
- bias for jobs ≥3 h: `-14,140 s`.

When projected completions fall at or before the current time while jobs are still running, EASY cannot form a reliable future reservation and falls back to strict FCFS for that pass.

This happens on all 1,696 passes containing a pivot. The ridge baseline shows the same behavior. Artificially doubling predicted durations restores 506 of the 1,247 backfills, confirming that the effect is caused by underprediction rather than a model-specific code path.

### Predicted-input QoS

| Delay | Wait actual | Wait predicted | BSLD actual | BSLD predicted | Peak actual | Peak predicted |
| ----- | ----------: | -------------: | ----------: | -------------: | ----------: | -------------: |
| 0     |       422.9 |          458.3 |        2.11 |           3.00 |       0.850 |          0.844 |
| 1 h   |     1,500.7 |        1,524.9 |       72.04 |          72.72 |       0.855 |          0.852 |
| 3 h   |     5,013.8 |        5,036.9 |      310.42 |         312.26 |       0.850 |          0.858 |
| 6 h   |     7,870.7 |        7,922.7 |      459.98 |         463.07 |       0.856 |          0.855 |
| 12 h  |    12,140.3 |       12,400.4 |      656.74 |         674.84 |       0.856 |          0.874 |
| 24 h  |    42,542.2 |       43,584.9 |    2,979.30 |       2,991.06 |       0.870 |          0.904 |

Once carbon delay dominates the schedule, the additional QoS cost of prediction errors is relatively small.

Peak concentration is more sensitive: at a 24-hour budget predicted inputs increase peak power from `0.870` to `0.904 MW`.

Energy remains **67.52 MWh** in every row.

## Terminal-job contention

The main experiments use the clean completed-job cohort.

An additional contention scenario schedules those jobs together with valid terminal executions:

- `FAILED`
- `CANCELLED`
- `TIMEOUT`
- `OUT_OF_MEMORY`
- `NODE_FAIL`

Terminal jobs retain their observed execution interval and node allocation but have no power profile.

On the 5,000-job debug cohort, adding 2,227 terminal jobs:

- increases node utilisation from 9.7% to 20.6%;
- increases mean clean-job EASY waiting from 31.2 s to 87.8 s.

These runs model additional resource contention, not the cause of cancellation or failure. Rescheduled terminal jobs retain their observed duration and allocation.

Selection rules, status counts, and detailed results are documented in [PM100 terminal-job contention](../../docs/PM100_features.md#terminal-job-contention-scenario).
