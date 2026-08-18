# Discrete-event HPC simulator

This package decides _when_ jobs run under a finite node budget, and records
what that costs in time, energy, and carbon. It is the component that turns
`CO2(job, start_time)` from a formula into an experiment.

It reuses `carbon_accounting` for every energy and emission figure and
`carbon_intensity` for every grid signal; it adds no physics of its own.

## Event model

Continuous timestamps, no fixed tick. The clock jumps between event instants:

| Event                     | Meaning                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `RELEASE`                 | a job becomes eligible (`release_time`, i.e. `eligible_time`) |
| `COMPLETION`              | a running job ends and returns its nodes                      |
| `CARBON_INTENSITY_CHANGE` | a grid signal bucket boundary                                 |
| `TIMER`                   | a wakeup a scheduler asked for                                |

Every event sharing an instant is drained before the scheduler is consulted, and
completions are applied before releases. Nodes freed at `t` are therefore
available to a job starting at `t`, with no ordering subtlety between the two
events. Ties are broken by a monotonic sequence counter, so a run is fully
deterministic.

`CARBON_INTENSITY_CHANGE` is emitted only for a scheduler that sets
`wants_carbon_intensity_events`. FCFS is carbon blind and never sees one; the
source exists for the carbon-aware policies of later schedulers, which need to reconsider
a deferred job when the signal moves. Boundary generation stops once nothing is
queued or running, so it cannot keep a simulation alive forever.

## Resources

The model formalization treats the nodes of a partition as equivalent, so
`Cluster` tracks a **count**, not identities. The `nodes` id-list column of the
trace stays validation only.

Capacity defaults to **880 nodes** (`PM100_PARTITION_1_NODES`): the distinct node
ids observed in partition 1 across the raw `COMPLETED` PM100 jobs (ids 20–979).
Observed peak concurrency is lower — 787 in the raw trace, 774 after the dataset
power profile cleaning — so those are lower bounds on the machine rather than its
capacity. All three are worth sweeping in a sensitivity analysis.

Every allocation and release is checked against `0 <= busy <= total` and raises
`CapacityError` on violation instead of clamping: silently over committing would
invalidate every downstream emission figure. A job larger than the whole cluster
is rejected when the `Simulator` is constructed, because it would stall a strict
FCFS queue forever.

## Schedulers

```python
class Scheduler(ABC):
    name: str
    wants_carbon_intensity_events: bool = False

    def select(self, now, queue, cluster, simulator) -> tuple[Job, ...]: ...
    def on_release(self, job, now, simulator) -> None: ...
    def on_carbon_intensity_change(self, now, simulator) -> None: ...
```

- **`FCFSScheduler`** considers jobs in `(release_time, job_id)` order and stops
  at the first that does not fit. That halt _is_ strict FCFS: a small job never
  overtakes a blocked large one, even when idle nodes could hold it. Relaxing
  exactly this rule, under a reservation for the blocked job, is EASY
  backfilling (Phase 4).
- **`TraceReplayScheduler`** starts each job at its recorded `start_time`. It is
  the validation policy rather than a policy under study — see below.

A scheduler may call `simulator.request_wakeup(when)` to be consulted again
later. A wakeup in the past raises, because that is a policy bug rather than a
rounding artefact.

## Per-job records

`JobRecord` carries simulated `start_time`/`end_time`, plus waiting and
turnaround measured from **both** `release_time` and `submit_time`. The project
has not fixed that convention: eligibility is the natural origin for a delay
budget, submission is what a user perceives, so both are recorded and the choice
stays open.

`account_schedule(result, jobs, provider)` then fills in, per job:

| Field                                                       | Meaning                                                                            |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `energy_kwh`, `emissions_gco2e`                             | the measured PM100 profile against the actual signal — the evaluation ground truth |
| `energy_kwh_average_model`, `emissions_gco2e_average_model` | what a scheduler using one mean power would have modelled                          |

Carrying both makes the model error of the simple representation a byproduct of
every run rather than a separate experiment. On the debug subset the gap is
about **0.2%** of total emissions.

Accounting deliberately happens _after_ the event loop. The engine stays
carbon-agnostic, so a forecast provider can be substituted later
without touching it. `check_coverage` fails early and legibly when a schedule
runs past the end of the cached series, instead of dying inside an accounting
loop with a bare missing-bucket error.

Jobs are loaded with the **duration-weighted** mean of the measured profile, not
the stored `node_power_mean_W` arithmetic mean. Only the weighted mean makes the
average and measured models consume identical energy, which is what isolates the
timing effect from a power representation artefact. Pass
`average_power_source="stored"` to compare against the stored value.

## Use

```python
import sys; sys.path.insert(0, "src")

from carbon_intensity import TimeSeriesCarbonIntensityProvider
from hpc_sim import Cluster, FCFSScheduler, Simulator, account_schedule
from hpc_sim.workload import load_jobs

jobs = load_jobs("data/processed/pm100_debug_5000.parquet")
result = Simulator(jobs, Cluster(880), FCFSScheduler()).run()

provider = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"
)
result = account_schedule(result, jobs, provider)
```

`hpc_sim` itself is standard-library only; `hpc_sim.workload` is the single
module that needs pyarrow.

```bash
.venv/bin/python scripts/run_simulation.py --scheduler replay --limit 5000
.venv/bin/python scripts/run_simulation.py --scheduler fcfs \
  --workload data/processed/pm100_clean.parquet
.venv/bin/python tests/check_simulator.py
```

## Validation

`TraceReplayScheduler` re-runs the schedule the real system produced. On both
the 5,000-job debug subset and the full 157,062-job clean trace it reproduces
**every** recorded start time exactly (max delay 0 s), and its peak occupancy of
774 nodes matches an independent sweep-line computation over the source table.
That is the engine's ground-truth anchor. FCFS on the full trace reaches exactly
880 busy nodes and never exceeds them.

Energy is identical across policies (553.34 MWh on the full trace, both
schedulers) while emissions differ (156.333 vs 156.536 tCO2e). That is the
energy-aware versus carbon-aware distinction, reproduced at schedule level.

## Interpreting the comparison — important caveat

**FCFS shows a _lower_ mean waiting time than the historical replay** (278 s vs
2,434 s on the full trace). This is not evidence that FCFS beats the production
scheduler. The simulated workload is a strict subset of what the machine really
ran: the dataset inspection removed all non-`COMPLETED` jobs (50,928 of them) and every job
failing power profile validation, and other partitions are excluded entirely.
Node utilisation is therefore only ~20%, while the recorded waiting times were
produced under the _full_ contention of jobs that are absent here.

Replay's waiting times are consequently **not** a comparable performance
baseline; only its start times are meaningful, as a fidelity check. Scheduling
policies must be compared against each other on the same simulated workload —
which is what Phase 4's FCFS versus EASY comparison does. This is the concrete
form of the filtering bias concern and it should be
quantified before the final experiments.
