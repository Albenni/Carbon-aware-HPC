# experiments

Experiments that need more than one of the other packages at once. The
simulator, the job models and the carbon-intensity forecasts each have their own
package and their own evaluation; what lives here is what only makes sense when
they are combined.

## Information ablation

`ablation.py` answers one question: of the carbon saving an oracle could take,
how much survives when the scheduler has to predict both the jobs and the grid,
and which of the two predictions costs more?

Six runs on one cohort, one cluster, one delay budget, one decision
granularity and one runtime-estimate source. Two of them are carbon-blind EASY,
for provenance; the four that matter vary exactly one information source at a
time:

| Configuration                           | Job information | Carbon information | Reads             |
| --------------------------------------- | --------------- | ------------------ | ----------------- |
| `carbon_actual_jobs_actual_carbon`      | measured        | observed           | the oracle        |
| `carbon_actual_jobs_forecast_carbon`    | measured        | forecast archive   | grid error only   |
| `carbon_predicted_jobs_actual_carbon`   | model output    | observed           | job error only    |
| `carbon_predicted_jobs_forecast_carbon` | model output    | forecast archive   | the realistic run |

Every run is charged against the observed series after the fact, so the four
totals are comparable and their differences belong to the input that moved.

### What it reports

`emissions_saved_vs_easy` and `oracle_recovery` per configuration, the full
`ScheduleMetrics` from `release_time` plus the `submit_`-prefixed timing view
from `submit_time`, and on the realistic row the split of what the oracle's
saving gave up:

```
carbon_forecast_loss_share  +  job_model_loss_share  +  interaction_loss_share
    =  combined_loss_share  =  1 - oracle_recovery(realistic)
```

Shares are taken over `C(easy) - C(oracle)`. That denominator cancels out of
every difference, so the split does not depend on which carbon-blind run is
used as the reference.

### Latest results

Archive `boosted_ridge`, the selected forecast model.

| configuration                         | job_information | carbon_information | runtime_estimate | nodes | max_delay_hours | forecast_reach_hours | decision_granularity_minutes | emissions_saved_vs_easy | oracle_recovery     | scheduler                     | jobs  | total_emissions_tco2e | mean_emissions_gco2e | total_energy_mwh  | peak_power_mw      | average_model_gap   | waiting_mean_s     | waiting_median_s | waiting_p95_s | waiting_p99_s | waiting_max_s | turnaround_mean_s  | turnaround_median_s | bounded_slowdown_mean | bounded_slowdown_median | bounded_slowdown_p95 | bounded_slowdown_max | total_nodes | peak_busy_nodes | utilisation        | makespan_days     | throughput_jobs_per_hour | delay_vs_trace_mean_s | delay_vs_trace_max_s | submit_waiting_mean_s | submit_waiting_median_s | submit_waiting_p95_s | submit_waiting_p99_s | submit_waiting_max_s | submit_turnaround_mean_s | submit_turnaround_median_s | submit_bounded_slowdown_mean | submit_bounded_slowdown_median | submit_bounded_slowdown_p95 | submit_bounded_slowdown_max | carbon_forecast_loss_share | job_model_loss_share | interaction_loss_share | combined_loss_share |
| ------------------------------------- | --------------- | ------------------ | ---------------- | ----- | --------------- | -------------------- | ---------------------------- | ----------------------- | ------------------- | ----------------------------- | ----- | --------------------- | -------------------- | ----------------- | ------------------ | ------------------- | ------------------ | ---------------- | ------------- | ------------- | ------------- | ------------------ | ------------------- | --------------------- | ----------------------- | -------------------- | -------------------- | ----------- | --------------- | ------------------ | ----------------- | ------------------------ | --------------------- | -------------------- | --------------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ------------------------ | -------------------------- | ---------------------------- | ------------------------------ | --------------------------- | --------------------------- | -------------------------- | -------------------- | ---------------------- | ------------------- |
| easy_actual_jobs_none_carbon          | actual          | none               | scheduling       | 880   | 0.0             | 0.0                  | 0.0                          | 0.0                     | 0.0                 | easy                          | 23560 | 17.471199695944442    | 741.5619565341444    | 67.52420508333333 | 0.8499497519169722 | -0.0002420468463871 | 422.90394736842103 | 0.0              | 0.0           | 15010.0       | 20021.0       | 3176.553310696095  | 2.0                 | 2.1106431117946207    | 1.0                     | 1.0                  | 990.5                | 880         | 880             | 0.4751380821797033 | 7.614016203703704 | 128.9288911926865        | -2842.245713073005    | 19372.0              | 2423.7931239388795    | 0.0                     | 3304.0               | 78038.0              | 509487.0             | 5177.442487266553        | 2.0                        | 3.2885256861701557           | 1.0                            | 3.287878787878788           | 2149.8883248730963          |                            |                      |                        |                     |
| easy_predicted_jobs_none_carbon       | predicted       | none               | scheduling       | 880   | 0.0             | 0.0                  | 0.0                          | -0.0002402457252344     | -0.0219043714286856 | easy                          | 23560 | 17.475397076986113    | 741.7401136241983    | 67.52420508333333 | 0.8443061244198385 | -0.0002538604030989 | 458.2618421052632  | 0.0              | 1341.0        | 14546.0       | 18720.0       | 3211.911205432937  | 2.0                 | 3.0002097530506937    | 1.0                     | 1.852334875893984    | 1445.6               | 880         | 880             | 0.4751380821797033 | 7.614016203703704 | 128.9288911926865        | -2806.887818336163    | 18526.0              | 2459.1510186757214    | 0.0                     | 4340.0               | 78038.0              | 509487.0             | 5212.8003820033955       | 2.0                        | 4.178096571908402            | 1.0                            | 4.7862857142857145          | 2149.8883248730963          |                            |                      |                        |                     |
| carbon_actual_jobs_actual_carbon      | actual          | actual             | scheduling       | 880   | 6.0             | 23.0                 | 15.0                         | 0.0109679351455775      | 1.0                 | carbon-aware-bounded          | 23560 | 17.27957671076389     | 733.4285530884504    | 67.52420508333333 | 0.8459317802504518 | -0.0002832514568118 | 7740.895373514431  | 6059.5           | 21178.0       | 35344.0       | 39975.0       | 10494.544736842105 | 7404.0              | 459.7094449461425     | 49.7772054104055        | 1708.9               | 2158.6               | 880         | 880             | 0.4643366339648876 | 7.791134259259259 | 125.99791429598577       | 4475.745713073005     | 39428.0              | 9741.784550084889     | 6265.5                  | 21515.0              | 85633.0              | 521165.0             | 12495.433913412564       | 7462.0                     | 460.88938609437207           | 51.547995219306145             | 1710.4                      | 2180.096446700508           |                            |                      |                        |                     |
| carbon_actual_jobs_forecast_carbon    | actual          | forecast           | scheduling       | 880   | 6.0             | 23.0                 | 15.0                         | 0.0045286628875436      | 0.412900224831263   | carbon-aware-forecast-bounded | 23560 | 17.392078522280556    | 738.2036724227739    | 67.52420508333333 | 0.8979896537032758 | -0.0001240665284174 | 7635.2475382003395 | 5161.5           | 20894.0       | 29184.0       | 32852.0       | 10388.896901528013 | 6948.0              | 508.3453769954825     | 56.53997212543554       | 1776.5               | 2156.6               | 880         | 880             | 0.4610660740284012 | 7.846400462962963 | 125.11044666919398       | 4370.0978777589135    | 32251.0              | 9636.1367147708       | 5592.5                  | 21346.0              | 85220.0              | 521165.0             | 12389.786078098472       | 7085.0                     | 509.52542850024855           | 58.5                           | 1778.2                      | 2234.9187817258885          |                            |                      |                        |                     |
| carbon_predicted_jobs_actual_carbon   | predicted       | actual             | scheduling       | 880   | 6.0             | 23.0                 | 15.0                         | 0.0051969066636733      | 0.4738272605275946  | carbon-aware-bounded          | 23560 | 17.380403501822222    | 737.7081282607055    | 67.52420508333333 | 0.8563465235195132 | 0.0001674810650541  | 7848.782682512733  | 6204.5           | 21263.0       | 33231.0       | 37670.0       | 10602.432045840407 | 7515.0              | 462.8122430673278     | 48.7                    | 1714.5               | 3342.3               | 880         | 880             | 0.4635275399693621 | 7.804733796296296 | 125.77836634639768       | 4583.633022071307     | 37078.0              | 9849.671859083192     | 6377.5                  | 21560.0              | 84869.0              | 509487.0             | 12603.321222410868       | 7564.0                     | 463.9921884600396            | 49.99499840915102              | 1716.3                      | 3342.3                      |                            |                      |                        |                     |
| carbon_predicted_jobs_forecast_carbon | predicted       | forecast           | scheduling       | 880   | 6.0             | 23.0                 | 15.0                         | 0.0005736052352166      | 0.0522983795585247  | carbon-aware-forecast-bounded | 23560 | 17.461178124333333    | 741.136592713639     | 67.52420508333333 | 0.8977111424482367 | -0.0001205982886482 | 7743.46642614601   | 5555.5           | 20996.0       | 28541.0       | 32272.0       | 10497.115789473684 | 7048.5              | 510.09381677217607    | 58.3                    | 1778.3               | 2794.1               | 880         | 880             | 0.4622306353987823 | 7.826631944444444 | 125.42645082007212       | 4478.316765704584     | 31658.0              | 9744.355602716469     | 5783.0                  | 21456.0              | 86044.0              | 524765.0             | 12498.004966044142       | 7106.0                     | 511.2739064772816            | 60.365289256198345             | 1778.3                      | 2794.1                      | 0.587099775168737          | 0.5261727394724053   | -0.1655708941996669    | 0.9477016204414752  |

### Reproducing it

Everything it reads is an artifact another package already writes: the cohort
and its predictions from `job_prediction`, the observations and the forecast
archive from `carbon_intensity`. Regenerate those first if they are missing.

```bash
.venv/bin/python scripts/train_job_models.py --model gradient   # test_predictions.parquet
.venv/bin/python -m carbon_intensity.snapshots                  # snapshots/test_*.json
.venv/bin/python -m experiments.ablation                        # --archive boosted_ridge
```

The result is one row per configuration in
`data/experiments/<partition>_ablation_<archive>.csv`, carrying the parameters
of the run alongside its metrics so a row is readable without the command that
produced it.

Useful flags: `--archive NAME` picks which forecast archive to replay
(`boosted_ridge` by default, the model selected in
`results/forecast_results.md`: best on validation and best on the held-out
test period); `--nodes`,
`--max-delay-hours` and `--decision-granularity-minutes` move the shared
protocol, and apply to all six runs at once by construction.

### Two guards worth knowing about

The oracle is held to the same reach as the forecast archive
(`archive_reach`, horizon minus cadence). Without that the cells would differ in
delay budget as well as in signal, and the split would be measuring the bound.

Every forecast-driven decision is checked to have read a snapshot issued at or
before the job's own release. A failure raises rather than being reported, so a
leaked observation cannot reach a published number.

### Checks

```bash
.venv/bin/python tests/check_experiments_ablation.py
```

A two-job cohort on a synthetic square-wave grid, with an archive equal to the
observations: the oracle recovers everything, the carbon-forecast share is
exactly zero, and the whole remaining gap is attributed to the job models.

## MILP optimum

`milp.py` answers the question the greedy oracle leaves open: with perfect
information, how far is the per-job greedy from the best schedule the same
candidates allow? The greedy scores every job alone, so every job aims at the
same clean hour and contention decides who gets it. That makes it a benchmark,
not a bound. The MILP picks every start at once and proves the result optimal.

### Formulation

One binary `x[j, s]` per job `j` and candidate start `s`:

```
minimise    sum_j sum_s x[j,s] * (c[j,s] + w * delay_hours[j,s])
subject to  sum_s x[j,s] = 1                                  each job starts once
            sum_{j,s : s <= t < s + d_j} n_j * x[j,s] <= N     every t in T
```

| Piece            | Definition                                                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `c[j, s]`        | `carbon_cost_gco2e`, the constant-average-power cost the greedy scores its own candidates with                             |
| candidates `S_j` | the greedy's own (release, then the decision grid up to `release + max_delay`) plus the start EASY and the greedy gave `j` |
| `T`              | union of all candidates: occupancy only rises when a job starts, so checking there is checking everywhere                  |
| `w`              | gCO2e per hour of delay; the default `0.001` only breaks ties toward the earliest start, as the greedy does                |

The reference starts make EASY and the greedy feasible solutions, so the optimum
provably matches or beats both and the greedy's measured gap is a proven lower
bound on its distance from optimal. They also keep the program feasible when
contention pushed a reference job past its budget; `jobs_past_budget` counts
such jobs, and it is zero in every run below.

The plan is replayed with `TraceReplayScheduler`, which raises if it overfills
the machine, and accounted ex-post like every other run. The solver is
`scipy.optimize.milp` (HiGHS) with a relative gap of 1e-6.

### What it reports

Per capacity, three rows: `easy`, `carbon_greedy` (the oracle
`CarbonAwareScheduler`) and `milp_optimum`, each with the full `ScheduleMetrics`
plus:

- `emissions_saved_vs_easy`, on measured emissions;
- `model_emissions_tco2e`, the average-power model the program minimises;
- `optimal_saving_captured = (M_easy - M_x) / (M_easy - M_opt)` and
  `gap_to_optimum = M_x / M_opt - 1`, on that model, where optimality is proven;
- on the MILP row: `proven_optimal`, `mip_gap`, `objective_gco2e`,
  `dual_bound_gco2e`, `branch_nodes`, `solve_seconds`, `variables`,
  `capacity_rows`.

With a larger `--delay-weight`, `milp_optimum` is the optimum of the weighted
objective, so the greedy can land above 100% on those two columns.

### Latest results

Debug subset, 6 h budget, perfect information.

Real capacity, 5,000 jobs on 880 nodes (EASY utilisation 9.7%):

| Grid   | Configuration |  tCO2e | Saved vs EASY | Optimal saving captured | Mean wait (s) | Mean bsld | Solver                             |
| ------ | ------------- | -----: | ------------: | ----------------------: | ------------: | --------: | ---------------------------------- |
| —      | EASY          | 4.6411 |         0.00% |                   0.00% |            31 |      1.08 |                                    |
| 60 min | greedy        | 4.4347 |         4.45% |                  99.65% |         6,264 |    224.22 |                                    |
| 60 min | MILP          | 4.4340 |         4.46% |                    100% |         5,824 |    196.80 | optimal, 16.8 s, 36,795 variables  |
| 15 min | greedy        | 4.4245 |         4.67% |                  99.69% |         6,391 |    225.70 |                                    |
| 15 min | MILP          | 4.4239 |         4.68% |                    100% |         6,108 |    211.61 | optimal, 88.7 s, 126,773 variables |

Contention, first 1,000 jobs, 60-minute grid (tCO2e):

| Nodes | EASY utilisation |   EASY | greedy |   MILP | Greedy captures | Greedy gap | Mean wait greedy / MILP (s) | Solver                       |
| ----: | ---------------: | -----: | -----: | -----: | --------------: | ---------: | --------------------------: | ---------------------------- |
|   880 |             7.0% | 0.4763 | 0.4638 | 0.4638 |         100.00% |     0.000% |               3,069 / 3,048 | optimal, 0.3 s               |
|   512 |            12.0% | 0.4763 | 0.4638 | 0.4638 |         100.00% |     0.000% |               3,213 / 3,156 | optimal, 0.3 s               |
|   256 |            23.9% | 0.4760 | 0.4670 | 0.4659 |          90.12% |     0.225% |               3,850 / 2,891 | optimal, 6.8 s               |
|   128 |            46.8% | 0.4774 | 0.4757 | 0.4714 |          31.92% |     0.921% |               6,120 / 3,598 | time limit 600 s, gap 8.5e-5 |

Delay price, 5,000 jobs on 880 nodes, 60-minute grid:

| w (gCO2e per job-hour) |  tCO2e | Saved vs EASY | Share of greedy saving | Mean wait (s) | Wait p95 (s) | Mean bsld |
| ---------------------- | -----: | ------------: | ---------------------: | ------------: | -----------: | --------: |
| greedy                 | 4.4347 |         4.45% |                100.00% |         6,264 |       19,951 |    224.22 |
| 0.001                  | 4.4340 |         4.46% |                100.33% |         5,824 |       19,951 |    196.80 |
| 1                      | 4.4345 |         4.45% |                100.07% |         2,544 |       16,929 |      7.15 |
| 10                     | 4.4423 |         4.28% |                 96.31% |         1,245 |        7,343 |      3.17 |

### Reading it

- The gap is contention and nothing else. While nodes do not bind, jobs do not
  compete and the per-job choice already is the optimum. At 23.9% utilisation
  the greedy captures 90% of the optimal saving, at 46.8% only 32%. The latter
  is the utilisation of the test cohort the ablation runs on (47.5% at 880
  nodes), so the greedy oracle cannot be read as an upper bound there.
- Under contention the optimum also waits less: mean wait is 25% lower at 256
  nodes and 41% lower at 128. The greedy's loss is a coordination failure, not a
  trade-off.
- Most of the greedy's QoS cost buys no carbon. At 1 g per job-hour the plan
  keeps 99.7% of the optimal saving, cuts median wait from 5,321 s to 0 and mean
  bounded slowdown from 224 to 7.15: the greedy holds short jobs for hours to
  save a few grams.
- Peak power is not the program's concern: at 880 nodes the optimum draws
  0.616 MW against the greedy's 0.585 MW. Capacity is counted in nodes; the power
  cap stays a separate lever.

### Scope and limits

- Optimal over the candidate set, not in continuous time. A continuous-time
  optimum could only be lower, so the greedy's gap is a lower bound.
- The tie-break moves the optimum by at most `w x total delay hours`: 30 g on
  5,000 jobs, 0.0007% of the total. That bounds its effect on carbon, not on the
  queue: among equally clean starts it picks the earliest, which can move waiting
  time a lot. The delay-aware round therefore uses `w = 0` for the carbon-only
  benchmark and names positive weights as prices, never as a neutral tie-break.
- `solve` labels each plan `optimal` (within the gap), `feasible-timeout` (a plan
  found before the limit, not a proof) or `no-solution` (empty plan), and counts
  the reference starts that lie past the voluntary budget.
- The 128-node run stops at the time limit. Its plan is feasible and replayed,
  so the greedy's 0.921% gap still stands as a lower bound, and the dual bound
  caps how much better the optimum could be at about 0.01%. Its exact figures
  depend on the machine's speed.
- No rolling horizon. Splitting the trace into windows gives up the proof of
  optimality, which is the only reason this program exists, and the cost grows
  fast with contention: 5,000 jobs at 9.7% utilisation close in 17 s, 1,000 at
  46.8% do not close in 600 s. Future work, together with an online greedy that
  prices delay.

### Reproducing it

```bash
.venv/bin/python -m experiments.milp --limit 5000
.venv/bin/python -m experiments.milp --limit 5000 --decision-granularity-minutes 15
.venv/bin/python -m experiments.milp --limit 1000 --nodes 880 512 256 128   # about 10 minutes
.venv/bin/python -m experiments.milp --limit 5000 --delay-weight 1
.venv/bin/python -m experiments.milp --limit 5000 --delay-weight 10
```

One CSV per command in
`data/experiments/milp_<jobs>jobs_<budget>h_<grid>min_w<weight>.csv`, one row per
configuration and capacity.

### Checks

```bash
.venv/bin/python tests/check_experiments_milp.py
```

Two one-hour jobs, one node and one clean hour. The greedy gives the hour to the
first job (4,100 g), the optimum to the one drawing ten times the power
(1,400 g), EASY emits 4,400 g, and the greedy captures exactly 10% of the
optimal saving.

## Delay-aware scheduler

`improved_scheduler.py` asks how much of the carbon-aware policies' queue time is
needed for their saving. It adds three levers to the shared scoring path of
`hpc_sim.carbon_aware` (a gCO2e-per-job-hour delay price, a minimum saving per
hour waited, a duration-scaled budget with a cap) and three to the congestion
policy (penalty over the whole estimated run, replanning on every event over the
first 32 deferrable jobs, overruns held to the end of the plan), then runs one
reproducible protocol from frozen inputs to figures.

| Phase        | What it does                                                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`    | reproduces the saved split, refits the frozen gradient recipe on train outcomes observable at the validation cutoff, fixes the cohorts, hashes the inputs |
| `smoke`      | the two synthetic check files, then 1,000 chronological validation jobs through every mechanism, with invariants asserted                                |
| `validation` | corrected baselines, pilot grids from each job's best benefit rate, price/threshold sweeps, fixed vs adaptive, combination, congestion ablation, selection |
| `test`       | the frozen selection in the four information cells, the decision, the 0-24 h delay sweep, resource-only contention, 1,000/5,000-job capacity subsets      |
| `milp`       | offline optimum on the capacity subsets, one domain (grid plus every compared policy's starts) per instance and price                                     |
| `report`     | figures (PNG and PDF) and generated tables from the saved artifacts only                                                                                  |

The validation cohort is the job-prediction validation split restricted to
outcomes observable before the test starts and to releases at least 48 h before
the carbon cutoff (17,656 jobs; every validation schedule is checked to end
before the cutoff). Its predictions come from the frozen recipe refitted on train
outcomes only, and reproduce the archived ones exactly. Selection keeps, per
family, up to three non-dominated configurations on
`(C, W_mean, W_p95, BSLD_mean, BSLD_p95)` (least carbon, least mean wait, least
p95 slowdown, savers first) with tolerances `rtol=1e-8`, `atol=1e-9` only. The
decision rule is frozen with the selection before the test runs.

Every run lives in `data/experiments/improved_scheduler/runs/<key>/` (`run.json`,
`jobs.parquet`, `decisions.parquet`), keyed by configuration, input hashes and
the simulator source; an existing key is loaded, never recomputed. The voluntary
delay is the union of `[t_k, min(target_k, t_{k+1}, start)]` over a job's
decisions, never `start - release`. Paired savings are per job against EASY with
the same job information; the isolated start-at-release counterfactual is
reported beside them, never instead. Results and discussion:
`results/improved_scheduler_results.md`.

### Reproducing it

```bash
make improved-scheduler PHASE=prepare
make improved-scheduler PHASE=smoke
make improved-scheduler PHASE=validation   # about 45 minutes on 4 workers
make improved-scheduler PHASE=test         # needs selection.json from validation
make improved-scheduler PHASE=milp
make improved-scheduler PHASE=report
```

`ARGS="--workers N"` sets the process count; `--embargo-hours` widens the
validation embargo if a validation schedule crosses the cutoff.

### Checks

```bash
.venv/bin/python tests/check_hpc_sim_delay_objectives.py
.venv/bin/python tests/check_experiments_improved_scheduler.py
```

The first covers the scoring rule (objectives disagreeing where they should,
thresholds, ties, bad parameters), flat/useful/negligible savings, predicted
power flipping an absolute threshold, adaptive budgets and caps, zero budget
equal to EASY record for record, zero price equal to the receding policy under
the same triggers, a forecast newer than its decision refused, the interval
penalty seeing congestion the start penalty misses, determinism under reordered
input, no overcommit and no leftover wakeup. The second covers voluntary delay
with targets pulled earlier, pushed later and re-held, non-positive denominators,
non-dominance and representatives, incomplete joins, the MILP with a positive
price, a non-constant profile and a time-limited solve, and the contention CLI.
