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
  overtakes a blocked large one, even when idle nodes could hold it.
- **`EASYBackfillScheduler`** relaxes exactly that rule, under a reservation.
  The first job that does not fit becomes the **pivot** and is promised the
  earliest instant at which enough nodes are projected to free up. Jobs further
  back may then overtake it, but only if they provably cannot push that
  reservation later — each either ends before the reservation, or takes only
  nodes the pivot will not need at it. The single reservation is what keeps the
  head of the queue from starving.
- **`PowerCappedEASYScheduler`** is the energy aware baseline: EASY plus an
  aggregate power budget. See below for why the budget is on power and not on
  energy.
- **`CarbonAwareScheduler`** is EASY plus one carbon decision per job: hold it
  until the cleanest start time within its delay budget. See below.
- **`TraceReplayScheduler`** starts each job at its recorded `start_time`. It is
  the validation policy rather than a policy under study — see below.

A scheduler may call `simulator.request_wakeup(when)` to be consulted again
later. A wakeup in the past raises, because that is a policy bug rather than a
rounding artefact. EASY needs no wakeup: it recomputes the reservation on every
pass, and a completion — the only event that can bring one forward — already
triggers a pass. The carbon-aware policy asks for exactly one, at the instant it
chose for the job.

### What a policy is allowed to know

Backfilling has to guess when running jobs end, and which guess it may use is a
statement about information, not an implementation detail. `RuntimeEstimateSource`
makes it explicit:

| Source       | Estimate                          | Use                                                               |
| ------------ | --------------------------------- | ----------------------------------------------------------------- |
| `TIME_LIMIT` | the walltime the user requested   | classic EASY; the only estimate genuinely available at submission |
| `SCHEDULING` | `Job.scheduling_duration_seconds` | a prediction once one exists, the actual duration until then      |

PM100 users overshoot heavily — the median job runs for **2.5%** of its
requested walltime — so classic reservations sit far in the future and suppress
backfills that perfect information would allow. `SCHEDULING` is the setting that
puts the baselines on the same footing as the carbon-aware policies,
which is what "the same information for every scheduler compared" requires.

Power is read through the matching seam, `Job.scheduling_average_power_watts`,
so no policy ever touches a measured profile.

The submission-time models in `src/job_prediction/` populate both seams from a
prediction only parquet artifact. The event engine still releases resources at
the actual completion time, and accounting still integrates the measured power
profile, so estimates influence decisions without replacing ground truth in the
evaluation.

### Why the energy-aware baseline caps power

Shifting a job changes neither its duration nor its power, so **total energy is
invariant to the schedule** (§2.3 of the model formalization). A policy that
tried to minimise energy would therefore have nothing to optimise, and the runs
confirm it: every baseline reports the same MWh to six decimals.

What a power-aware policy can change is _when_ power is drawn. The baseline is
the one a capped machine actually runs: never let the summed average power of
the running jobs exceed `power_cap_watts`. It moves jobs in time for a power
reason while staying blind to the grid signal — the precise contrast with a
carbon-aware policy, which moves them for the opposite reason. The cap
constrains starting decisions only; the pivot's reservation is still computed
on nodes alone, so it stays a lower bound on when the pivot can really start. A
job whose own average power exceeds the cap could never start, so it is
rejected at release rather than left to stall the queue.

### Carbon-aware scheduling with perfect information

`CarbonAwareScheduler` is the first policy that moves a job _because of the
grid_. It is EASY backfilling plus one decision, taken once when a job becomes
eligible: score every candidate start time inside the delay budget, then hold
the job until the cheapest one.

| Ingredient       | Choice                                                                      |
| ---------------- | --------------------------------------------------------------------------- |
| delay budget     | `max_delay`, measured from `release_time` — the origin §2.6 proposes        |
| candidate starts | the release instant, then the grid signal's own boundaries up to the budget |
| cost of a start  | `energy × mean intensity` over `[t, t + duration]`, average-power model     |
| tie-break        | the earliest candidate, so a job is never held without a strict gain        |

Three properties make the policy safe to compare against the baselines, and they
are what `tests/check_carbon_aware.py` checks rather than any particular
placement:

- **`max_delay=0` reproduces EASY exactly**, start time for start time. The
  sweep therefore begins at the baseline instead of at a different policy.
- **A held job yields its place.** It is withheld from the placement pass rather
  than refused a slot, so it never becomes the pivot and never blocks anyone: the
  jobs behind it move up and the machine keeps working. When its target arrives
  it re-enters at its original queue position, which bounds the harm — a job that
  gives up its turn is at worst as delayed as the queue it rejoins.
- **The budget bounds the _voluntary_ delay only.** Once its target arrives a job
  competes for nodes like any other, and contention can still push it later. The
  parameter bounds the delay the carbon decision is responsible for, which is the
  quantity the trade-off is about; the total delay is what the QoS metrics report.

Because the schedule cannot change the signal and the signal cannot change the
job, the cheapest start is fixed at release and never revisited. The policy asks
for a wakeup at exactly that instant instead of subscribing to every bucket
boundary, so a deferral costs one event per held job rather than one per bucket.

`CarbonSignal` is the supporting piece: it materialises the provider's series
once and keeps a cumulative integral, so scoring a candidate is two lookups no
matter how long the job runs. A window reaching outside the available series
raises instead of being clamped — a policy planning on invented data would
report a saving the evaluation could never confirm.

The decision reads `Job.scheduling_duration_seconds` and
`Job.scheduling_average_power_watts`, the same seam every other policy uses, so
substituting predictions turns the oracle into the realistic scenario without
touching the policy. Evaluation still integrates the measured profile, and
`average_model_gap` reports the residual difference on every run.

The rule is greedy and per job: every
job independently aims at the same clean interval, and the queueing that
collectively creates is a cost the rule does not model. Perfect information
bounds what _this_ policy can do, not what the offline problem admits — that
belongs to the later optimisation formulation (§2.7).

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

## Metrics

`schedule_metrics(result)` scores a finished, accounted run. It reads only a
`SimulationResult`, so it cannot tell which policy produced the schedule — which
is the point: baselines and carbon-aware policies are judged by the same
function on the same workload.

| Group  | Reported                                                                        |
| ------ | ------------------------------------------------------------------------------- |
| Carbon | total and mean-per-job emissions, and the same totals under the average model   |
| Energy | total energy, and the schedule's peak aggregate power                           |
| QoS    | waiting, turnaround, and bounded slowdown — each as mean, median, p95, p99, max |
| System | node utilisation, peak nodes, makespan, throughput                              |

Every QoS quantity is a `Distribution`, not a mean. A policy that defers jobs to
chase clean electricity can improve an average while punishing a minority
badly, so the tail is reported next to the mean by construction rather than on
request.

Bounded slowdown is `max(1, (wait + runtime) / max(runtime, 10 s))`. The 10 second
floor is the usual convention and it matters here: PM100 has many very short
jobs, and without it a job that runs for one second and waits for a hundred
would report a slowdown of 101 and dominate the mean.

`reference="release"` (default) measures waiting from eligibility, the origin
the model formalization proposes for a delay budget; `reference="submit"` gives
the user-perceived figure. Both are available because the project has not fixed
the convention.

Peak power is computed from each job's duration-weighted mean power, recovered
from its accounted energy. It is therefore a floor on the true peak — the
20-second profile fluctuates inside every job — but it is exactly the quantity a
power-capped policy controls.

## Use

```python
import sys; sys.path.insert(0, "src")

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
result = Simulator(jobs, Cluster(880), FCFSScheduler()).run()

provider = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"
)
result = account_schedule(result, jobs, provider)
print(format_metrics(schedule_metrics(result)))
```

Swapping the policy is the only change needed to compare:

```python
from datetime import timedelta

from hpc_sim import (
    CarbonAwareScheduler,
    EASYBackfillScheduler,
    PowerCappedEASYScheduler,
    RuntimeEstimateSource,
)

EASYBackfillScheduler()                                       # classic, plans on requested walltime
EASYBackfillScheduler(runtime_estimate=RuntimeEstimateSource.SCHEDULING)  # perfect information
PowerCappedEASYScheduler(power_cap_watts=680_000.0)           # energy-aware baseline
CarbonAwareScheduler(provider, max_delay=timedelta(hours=6))  # carbon-aware oracle
```

`hpc_sim` itself is standard-library only; `hpc_sim.workload` is the single
module that needs pyarrow.

```bash
.venv/bin/python scripts/run_simulation.py --scheduler replay --limit 5000
.venv/bin/python scripts/run_simulation.py --scheduler easy \
  --workload data/processed/pm100_clean.parquet
.venv/bin/python scripts/run_simulation.py --scheduler power-cap --power-cap-mw 0.68
.venv/bin/python scripts/run_simulation.py --scheduler carbon --max-delay-hours 6 \
  --runtime-estimate scheduling

# held-out jobs, predicted inputs for decisions, actual outcomes for scoring
.venv/bin/python scripts/train_job_models.py
.venv/bin/python scripts/run_simulation.py --scheduler carbon \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet

# every baseline over one workload, side by side, into a CSV
.venv/bin/python scripts/compare_baselines.py --limit 5000

# the carbon / QoS frontier: one run per delay budget, EASY as the reference
.venv/bin/python scripts/carbon_tradeoff.py --limit 5000
.venv/bin/python scripts/carbon_tradeoff.py --limit 5000 \
  --max-delay-hours 6 24 --decision-granularity-minutes 15 60 240

.venv/bin/python tests/check_simulator.py
.venv/bin/python tests/check_baselines.py
.venv/bin/python tests/check_carbon_aware.py
.venv/bin/python tests/check_job_prediction.py
```

The full 157,062-job trace takes roughly fourteen minutes for the four-policy
comparison. Almost all of it is emission accounting, which integrates every
20-second power sample against the grid signal; the scheduling itself is under
a second per policy.

## Validation

`TraceReplayScheduler` re-runs the schedule the real system produced. On both
the 5,000-job debug subset and the full 157,062-job clean trace it reproduces
**every** recorded start time exactly (max delay 0 s), and its peak occupancy of
774 nodes matches an independent sweep-line computation over the source table.
That is the engine's ground-truth anchor. FCFS on the full trace reaches exactly
880 busy nodes and never exceeds them.

Energy is identical across policies (553.34 MWh on the full trace, every
scheduler, to six decimals) while emissions differ (156.333 tCO2e for replay
against 156.536 for FCFS). That is the energy-aware versus carbon-aware
distinction, reproduced at schedule level.

EASY's reservation promise is checked directly rather than through its outcomes:
`EASYBackfillScheduler.first_reservations` records what each pivot was promised,
and no job — on random workloads or on the PM100 subset — ever starts after it.
The hand computed cases in `tests/check_baselines.py` pin the three decisions
that matter: a short job overtakes the blocked head, a long one that would push
the reservation back does not, and a long one that fits in nodes the pivot will
not claim does.

The carbon-aware policy is checked the same way, on its own contract rather than
its placements (`tests/check_carbon_aware.py`): the predicted cost of a start
time agrees with `carbon_accounting` to nine decimals, no job on the PM100
subset is ever held past its budget or started before its target, a zero budget
reproduces EASY start time for start time, and holding jobs changes emissions
while leaving energy identical.

## Baseline results

The four policies over the full 157,062-job trace, 880 nodes, classic
walltime estimates, cap at 80% of the FCFS peak:

| Metric                | replay  | FCFS    | EASY    | power-cap |
| --------------------- | ------- | ------- | ------- | --------- |
| emissions (tCO2e)     | 156.333 | 156.536 | 156.536 | 156.529   |
| energy (MWh)          | 553.34  | 553.34  | 553.34  | 553.34    |
| peak power (MW)       | 0.700   | 0.846   | 0.852   | 0.676     |
| waiting mean (s)      | 2,434.3 | 277.8   | 252.7   | 222.9     |
| waiting p99 (s)       | 62,099  | 10,341  | 10,004  | 8,993     |
| waiting max (s)       | 410,317 | 56,649  | 58,043  | 58,043    |
| bounded slowdown mean | 71.37   | 2.41    | 2.12    | 1.68      |
| bounded slowdown max  | 35,852  | 4,344   | 4,320   | 2,690     |

Three things in that table are worth stating explicitly.

**Emissions barely move between the baselines.** They are not trying to move
them: all three are carbon blind, and the differences here are incidental
consequences of a slightly different overlap with the grid signal. That flatness
is what makes them a fair reference point for the carbon-aware policy, whose
results are below.

**The power cap trades peak power for time, not for energy.** It cuts the peak
from 0.846 to 0.676 MW — the budget it was given, to three decimals — while
consuming exactly the same MWh. That is the energy-aware baseline doing the only
thing it can do in this model.

**The cap's mean waiting time _improves_, and the mean is lying.** Capping power
blocks the head of the queue more often, and every block is a backfill opening
for a short job. Broken down by job width, one-node jobs (120,113 of 157,062)
wait 58s instead of 87s, while 65–256-node jobs wait 1,099s instead of 520s.
The cap does not make the system faster; it moves delay off the many narrow jobs
onto the few wide ones, and the unchanged 58,043s maximum shows the worst-served
job is no better off. This is exactly the failure mode the QoS distributions
exist to catch, and the same one a carbon-aware policy will be tempted to
produce.

## Carbon-aware results

`CarbonAwareScheduler` over the same 157,062-job trace and 880 nodes, on the
provider's 15-minute grid, with perfect information (`scheduling` estimates).
The zero-budget row _is_ EASY, so every column can be read as a cost relative to
it; the carbon-blind baselines agree on emissions to five significant figures
(FCFS 156.536, EASY 156.535 tCO2e), so the saving does not depend on which one
it is measured against.

| delay budget | emissions (tCO2e) | saved | waiting mean (s) | waiting p95 (s) | bounded slowdown mean | bounded slowdown p95 | peak power (MW) |
| ------------ | ----------------- | ----- | ---------------- | --------------- | --------------------- | -------------------- | --------------- |
| 0 (= EASY)   | 156.535           | 0.00% | 251              | 111             | 2.09                  | 1.51                 | 0.852           |
| 1 h          | 155.894           | 0.41% | 1,245            | 3,430           | 32.11                 | 240.3                | 0.866           |
| 3 h          | 154.577           | 1.25% | 4,290            | 10,541          | 142.42                | 887.1                | 0.877           |
| 6 h          | 153.008           | 2.25% | 8,402            | 21,236          | 287.60                | 1,776.5              | 0.878           |
| 12 h         | 149.631           | 4.41% | 17,630           | 42,681          | 566.75                | 3,268.4              | 0.857           |
| 24 h         | 145.427           | 7.10% | 49,067           | 85,688          | 1,679.09              | 8,078.6              | 0.903           |

Energy is 553.34 MWh in every row, as it must be.

**The saving is real, and the exchange rate is poor.** Carbon falls
monotonically with the budget, which is the answer to "does carbon-aware
scheduling do anything at all": it does, and 7.1% is four orders of magnitude
above the difference between the carbon-blind baselines themselves. But the QoS side grows faster
than the carbon side throughout: going from a one hour budget to a
twentyfour hour one multiplies the saving by 17 and the mean waiting time by 39.
The interesting region of this frontier is its left end, not its right.

**The grid signal, not the policy, sets the ceiling.** Over the cached IT-NO
2020 series the median intra-day range is 71 gCO2e/kWh, 24.5% of that day's
mean, and the cleanest quarter-hour of a day sits only about 11% below the daily
mean. A policy that shifts work inside a day therefore cannot save much more
than a tenth of the total no matter how cleverly it aims, and the 7.1% reached
at a 24-hour budget is already close to that. Reporting a saving without the
amplitude of the signal it came from would make this number look like a property
of the scheduler when it is mostly a property of the zone.

**Peak power goes up, not down** — 0.852 to 0.903 MW at the widest budget. The
greedy rule optimises each job in isolation and every job aims at the same clean
interval, so the policy manufactures exactly the concentration the power-capped
baseline exists to prevent. Combining the two is the obvious next experiment,
and the reason the cap was built as a composable variant of EASY.

**And the means understate the damage**, in the same way they did for the power
cap. At a six-hour budget the mean bounded slowdown is 288 while the p95 is
1,777: the average is carried by jobs that were held a little, and the tail by
jobs that were held the whole budget and then met a busy machine. This is the
tail that a delay budget bounds only in its voluntary part.

### Decision granularity

How finely the policy is allowed to aim, on the 5,000-job debug subset:

| decision grid | 6 h budget saved | 24 h budget saved |
| ------------- | ---------------- | ----------------- |
| 15 min        | 4.67%            | 11.09%            |
| 60 min        | 4.45%            | 10.94%            |
| 240 min       | 3.67%            | 9.94%             |

Coarsening the grid from 15 to 60 minutes costs almost nothing, and going to
four hours gives up about a fifth of the saving. The signal simply does not
carry much structure below the hour, which is worth knowing before a forecast is
introduced: a forecast that is accurate hour by hour loses little against a
perfect quarter-hourly one.

The subset saves more than the full trace at the same budget (11.1% against
7.1% at 24 hours) because it covers a different and much shorter stretch of the
series. Frontier numbers are only comparable within one workload window, which
is the sensitivity analysis the final experiments still owe.

The six-point sweep over the full trace takes about half an hour, again almost
entirely emission accounting.

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
policies must be compared against each other on the same simulated workload,
which is what the baseline results above do. This is the concrete form of the
filtering bias concern and it should be quantified before the final
experiments.
