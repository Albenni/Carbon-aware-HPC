# Job prediction models

This package turns PM100 fields known at submission into the duration and mean
whole-job power used by the schedulers. Actual runtime and the measured power
profile remain attached to each simulated job only so the engine can reproduce
what happened and score the result afterwards.

The separation is deliberate:

| Used for scheduling decisions | Used only after the decision |
| ----------------------------- | ---------------------------- |
| predicted duration            | actual runtime               |
| predicted average power       | measured 20-second profile   |
| requested resources           | actual completion time       |
| submission timestamp / QoS    | actual energy and emissions  |
| causal history of past jobs   |                              |

No model predicts CO2. The scheduler combines the job estimates with the
separate carbon-intensity signal.

## Result summary

Both models are evaluated once on the same frozen temporal test partition.

| Test estimate          | Ridge baseline | Gradient boosting |       Goal |
| ---------------------- | -------------: | ----------------: | ---------: |
| duration WAPE          |        101.75% |        **42.90%** | 35% -- 40% |
| power WAPE             |         21.39% |        **19.91%** | 15% -- 20% |
| energy WAPE (composed) |         97.43% |        **55.23%** | 35% -- 40% |
| duration >=1 h         |         99.24% |        **38.53%** |          - |
| oracle recovery        |          5.05% |        **42.65%** |      > 50% |

## Inputs and targets

Two feature families are available, and which ones a model may see is part of
the configuration that validation selects.

**Request and calendar.** Requested walltime, nodes, cores, tasks, cores per
task, GPUs, memory, recorded submission priority, QoS, and hour-of-day and
weekday cycles from `submit_time`. `num_tasks` is median imputed and gets a
missing-value indicator; skewed quantities are `log1p` transformed. Derived
ratios add cores per node, memory per node and per core, GPUs per node, tasks
per node, requested node-seconds and core-seconds, a task/core consistency
indicator, the `shared` and `req_switch` flags, group indicators, and month and
day of month.

**Causal history.** For a job submitted at `t`, statistics over the jobs that
share a key _and had already finished by_ `t`: count, geometric mean, standard
deviation, last value, mean of the last eight, maximum, and time since the most
recent one. They are computed for duration, average power, energy, and for the
ratio of runtime to requested walltime, which converts a limit that is known
now into an expected duration. Two keys are used: `user_id`, and the request
signature `(user_id, time_limit, num_nodes_req)`. Optional wider scopes exist
for the exact resource fingerprint, the global workload regime, and the queue
occupancy at submission; validation did not select them.

`user_id` is the reason this works at all: the trace has 406 users and 99.3% of
test jobs come from a user already seen in training.

The models never use job identifiers, allocated resources, eligibility or start
times, runtime, or measured power as features. `partition` is constant and
`threads_per_core` is 99.6% missing, so neither carries signal.

Targets follow the existing accounting contract:

- duration is `run_time` in seconds;
- energy is the integral of `node_power_consumption`, trimming or extending the
  final 20-second sample exactly as `carbon_accounting` does;
- average power is the duration-weighted whole-job mean
  `energy_kWh * 3,600,000 / duration_s`.

The power trace is already aggregated across the allocated nodes. It is never
multiplied by the node count.

## Anti-leakage contract

Three separate rules, each with an executable check in
`tests/check_job_prediction.py`.

1. **Order.** `temporal_split` stable-sorts by `(submit_time, job_id)` before
   splitting. This is required: the clean parquet contains 77,135 adjacent
   submit-time inversions. The 70/15/15 boundaries are half open, so every
   equal timestamp stays in one partition.

2. **Label availability.** A job may be submitted before a boundary and finish
   after it. Fitting therefore uses only the labels observable at each cutoff:
   109,389 training jobs completed by the validation boundary, 23,534
   validation labels completed by the test boundary, and 133,190 pre-test
   outcomes for the final refit. `end_time` is retained for this check only and
   is never encoded as a feature.

3. **Causal history.** `causal_group_stats` admits a past job only once its
   `end_time` has passed the current job's `submit_time`, so a job never sees
   itself, never sees a job still running, and never sees the future. The
   decisive test rewrites every outcome recorded after a cutoff and asserts
   that not one feature value before that cutoff moves.

On the committed trace the split is:

| Partition  |    Jobs | Starts at                 |
| ---------- | ------: | ------------------------- |
| train      | 109,676 | beginning of the workload |
| validation |  23,826 | 2020-09-25 04:51:34 UTC   |
| test       |  23,560 | 2020-10-05 14:00:51 UTC   |

## Selection protocol

`scripts/experiment_job_models.py` discards the test frame on load and scores
candidates on expanding window folds inside the development period. The last
fold reproduces the frozen train/validation boundary and is the selector; the
earlier folds are a stability report, not a second selector.

The folds are not interchangeable, and this matters for reading every number
below:

| Fold  | Cutoff           |   Train | Validation | Jobs < 10 s |
| ----- | ---------------- | ------: | ---------: | ----------: |
| fold3 | 2020-07-17 10:20 |  53,172 |     19,734 |       10.5% |
| fold2 | 2020-08-18 19:43 |  73,118 |     19,944 |        2.4% |
| fold1 | 2020-09-02 19:34 |  93,364 |     19,744 |        2.4% |
| fold0 | 2020-09-28 14:02 | 113,194 |     19,729 |       66.4% |

The workload changes regime near the end of the trace. Test is 69.1% jobs under
ten seconds, so fold0 is the only fold that resembles it, which is exactly why
it is the selector.

WAPE is total absolute error over total actual quantity, so it is a _long-job_
metric here: jobs under ten seconds are 69% of the test count but 0.04% of its
runtime seconds, while jobs of at least an hour are 92%. A duration WAPE above
100% therefore means the long jobs are wrong, not the short ones.

## What the search found

The full leaderboard is `data/job_predictions/validation_leaderboard.csv`, 44
rows written by one run of `scripts/experiment_job_models.py`. Every number
below is from that run: scored on fold0, with the four-fold mean beside it.

**Feature ablation.** Each rung adds exactly one group to the one above, so any
change is attributable to that group. The model is held fixed at
HistGradientBoosting, `log1p` target, absolute error.

| Features               | Cols | WAPE fold0 | 4-fold mean | >=1 h WAPE |
| ---------------------- | ---: | ---------: | ----------: | ---------: |
| A base                 |   19 |    193.34% |     125.75% |     73.76% |
| B + derived            |   38 |    572.05% |     251.82% |     71.55% |
| C + user_id            |   39 |    215.43% |     125.17% |     62.49% |
| D + user_hist          |   69 |     50.31% |      64.12% |     47.54% |
| **E + signature_hist** |  100 | **39.66%** |  **57.45%** | **36.36%** |
| F + fingerprint_hist   |  131 |     44.83% |      58.94% |     40.89% |
| G + global_hist        |  162 |     43.00% |      59.74% |     38.49% |
| H + queue              |  166 |     49.08% |      59.76% |     45.81% |

**Model families**, on feature set E:

| Model             | Transform | Loss     | WAPE fold0 | 4-fold mean | >=1 h WAPE | Fit (s) |
| ----------------- | --------- | -------- | ---------: | ----------: | ---------: | ------: |
| median (constant) | raw       | -        |    104.11% |      99.39% |     98.81% |     0.0 |
| ridge             | log1p     | squared  |     51.48% |      66.53% |     47.72% |     0.0 |
| decision tree     | log1p     | squared  |     61.04% |      75.75% |     50.02% |     2.4 |
| decision tree     | log1p     | absolute |     44.15% |      73.47% |     37.66% |    19.1 |
| random forest     | log1p     | squared  |     62.06% |      67.77% |     60.40% |    37.7 |
| extra trees       | log1p     | squared  |     56.22% |      65.57% |     55.30% |    15.7 |
| hist_gbr          | log1p     | squared  |     57.20% |      67.31% |     53.95% |    10.4 |
| hist_gbr          | log1p     | poisson  |     58.36% |      67.75% |     57.81% |    10.6 |
| hist_gbr          | raw       | squared  |    537.69% |     206.86% |     26.07% |     9.4 |
| hist_gbr          | raw       | absolute |     70.73% |      67.49% |     31.33% |    13.0 |
| segmented (soft)  | raw       | absolute |     43.89% |      62.69% |     26.47% |    27.1 |
| segmented (hard)  | raw       | absolute |     37.27% |      63.03% |     29.44% |    27.5 |
| **hist_gbr**      | **log1p** | **abs.** | **39.66%** |  **57.45%** | **36.36%** |    12.6 |

**Target weighting and capacity**, on the winner above:

| Variant                               | WAPE fold0 | 4-fold mean | >=1 h WAPE |
| ------------------------------------- | ---------: | ----------: | ---------: |
| no weight                             |     39.66% |      57.45% |     36.36% |
| **weight `log1p(duration)`**          | **32.73%** |  **55.67%** | **25.13%** |
| weight `log1p(duration)^2`            |     41.14% |      60.55% |     29.15% |
| weight `duration^0.25`                |     39.74% |      60.41% |     28.80% |
| + 800 iterations                      |     42.59% |      58.38% |     35.36% |
| + 800 iterations, learning rate 0.03  |     36.99% |      57.11% |     30.47% |
| + 127 leaves                          |     37.32% |      59.13% |     30.54% |
| + 31 leaves, 800 iterations           |     34.78% |      56.20% |     26.24% |
| + minimum 100 samples per leaf        |     35.48% |      55.96% |     27.94% |
| + L2 regularisation 10                |     35.95% |      57.23% |     28.22% |
| segmented (hard) with the same weight |     41.17% |      67.79% |     27.48% |

Five findings, in the order they mattered:

1. **Causal history is the whole story.** Adding `user_hist` moves duration
   WAPE from 572% to 50%, an eleven-fold cut that nothing else approaches. The
   signature key takes it to 39.66%.
2. **Identity is not behaviour.** The raw `user_id` on its own does help
   (572% to 215%), but it is still four times worse than the same user's
   history. Knowing _who_ submitted is worth little; knowing _what they usually
   run_ is worth almost everything.
3. **Derived ratios hurt without history.** Rung B is worse than rung A
   (572% against 193%): more columns and no new information let the booster
   overfit the training regime. They stop hurting once history anchors the
   prediction, and rung E beats rung D, so they are kept.
4. **The loss decides the long jobs.** Squared error in log space fits the
   conditional geometric mean and under-predicts the right tail (57.20%).
   Absolute error in log space is much better (39.66%), and weighting each job
   by `log1p(duration)` is better still (32.73%): it makes the optimiser care
   about the jobs that carry the runtime seconds without giving up log-space
   calibration on the short ones. Training in raw seconds is the opposite
   trade, best on long jobs alone (26-31%) and hopeless overall.
5. **More capacity buys nothing.** Every capacity variant is worse than the
   400-iteration, 63-leaf default. So is a quantile loss at 0.55 to 0.8, which
   was tried specifically to correct the long-job under-prediction and lost on
   every fold (best 37.11% against 32.73%).

Segmenting into a short and a long branch, in soft and hard forms, wins on the
long jobs alone (26.5% and 29.4%) but never on total WAPE, and costs a
classifier plus two regressors. It was not selected.

The wider history scopes deserve a note. `fingerprint_hist`, `global_hist` and
`queue` all lose on fold0. On the four-fold mean the picture is closer, and one
configuration - E plus fingerprint and global history - actually wins it
(54.67% against 55.67%) while losing fold0 (36.65% against 32.73%).

**Power** selects the same features with plain absolute error and no weighting:

| Model         | Transform | Loss     | WAPE fold0 | 4-fold mean | >=1 h WAPE |
| ------------- | --------- | -------- | ---------: | ----------: | ---------: |
| **hist_gbr**  | **log1p** | **abs.** | **12.95%** |  **15.17%** | **12.01%** |
| random forest | log1p     | squared  |     13.12% |      15.48% |     10.69% |
| extra trees   | log1p     | squared  |     13.88% |      15.68% |     10.73% |
| ridge         | log1p     | squared  |     13.89% |      17.37% |     12.14% |
| hist_gbr      | log1p     | squared  |     15.09% |      15.88% |     10.84% |

Power is an easy target: even ridge has good results, and the spread
across families is three points. This is the opposite of duration, and it is
why the ridge baseline was already acceptable here and catastrophic there.

**Energy** predicted directly is poor in every configuration, the best being
94.97% on fold0. That is the finding the composition step then acts on.

## Physical consistency

Independent predictions need not satisfy physics, so two compositions are
compared on validation energy MAE:

1. predict duration and power, then derive energy;
2. predict duration and energy, then derive power.

The first wins decisively on PM100: 0.7959 against 1.6492 kWh validation MAE
for the tuning fit, and 1.1714 against 1.3762 kWh for the final one. Deriving
power from energy is far worse still, because dividing by a predicted duration
amplifies its error. Persisted scheduling inputs therefore always satisfy

```text
predicted_energy_kwh
    = predicted_average_power_watts * predicted_duration_seconds / 3,600,000
```

and `SchedulingPrediction` refuses to load a file where they do not.

## Test-period accuracy

One evaluation, after the configuration was frozen and refitted on the
development set. `MAE` and `RMSE` retain the target unit; bias is the mean
signed residual.

| Estimate            |        MAE |       RMSE |   WAPE | Median rel. |      Bias |    p95 abs | >=1 h WAPE |
| ------------------- | ---------: | ---------: | -----: | ----------: | --------: | ---------: | ---------: |
| duration model      | 1,181.30 s | 6,100.66 s | 42.90% |      31.19% | -568.06 s | 3,875.98 s |     38.53% |
| average-power model |   431.09 W | 3,116.11 W | 19.91% |      13.78% | -184.53 W |   634.83 W |     11.57% |
| energy model        |   1.86 kWh |  24.21 kWh | 64.84% |      43.74% | -1.05 kWh |   1.73 kWh |     62.81% |
| duration x power    |   1.58 kWh |  19.87 kWh | 55.23% |      35.13% | -1.03 kWh |   1.67 kWh |     52.25% |

Against the ridge baseline on the same rows: duration 101.75%, power 21.39%,
composed energy 97.43%, duration >=1 h 99.24%.

Duration by actual band, which is where WAPE is decided:

| Band      |   Jobs |         MAE |     WAPE | Median rel. |         Bias |     p95 abs |
| --------- | -----: | ----------: | -------: | ----------: | -----------: | ----------: |
| < 10 s    | 16,273 |      7.25 s |  443.10% |      26.60% |      +6.59 s |      2.96 s |
| 10-60 s   |    389 |    590.73 s | 2094.00% |     454.10% |    +587.79 s |  1,627.52 s |
| 1-10 min  |  1,749 |    782.22 s |  211.50% |      25.50% |    +620.58 s |  2,511.33 s |
| 10-60 min |  2,737 |  1,157.21 s |   68.40% |      47.60% |    +534.86 s |  2,546.43 s |
| 1-3 h     |  1,397 |  2,773.12 s |   49.30% |      36.70% |  -1,371.50 s |  6,418.68 s |
| >= 3 h    |  1,015 | 18,792.45 s |   36.90% |      40.50% | -14,140.54 s | 61,087.91 s |

The `>= 3 h` band is 4.3% of the jobs and **69% of the total absolute error**;
with `1-3 h` it is 83%. The bias column names the failure directly: the model
still under-predicts the longest jobs by about four hours. The ridge baseline
under-predicted them by 50,739 s, that is, by essentially their whole duration.

## Where the limit is

`data/job_predictions/validation_learning_curve.csv`, measured on fold0.

| Training data         | base features WAPE | chosen features WAPE |
| --------------------- | -----------------: | -------------------: |
| 20% (random)          |            709.60% |               36.73% |
| 40%                   |          1,020.30% |               39.40% |
| 60%                   |            707.83% |               40.06% |
| 80%                   |            992.84% |               39.04% |
| 100%                  |            818.22% |               32.73% |
| 100%, refit in-sample |                  - |               16.48% |

Subsets taken as the most _recent_ rows instead of at random are consistently
worse for the chosen features (50.75% at 20%, 41.81% at 40%, 40.28% at 80%):
the per-user statistics need breadth of history, not just recency, so throwing
away old jobs costs more than throwing away random ones.

This separates the four hypotheses cleanly:

- **Feature-limited: yes, dominantly.** Without causal history the model is at
  818% and adding data does nothing. The entire improvement in this work comes
  from the history features.
- **Data-limited: no.** A fifth of the training rows already reaches 36.7%
  against 32.7% for all of them. Collecting more of the same trace will not
  close the remaining gap.
- **Model-limited: partly, and cheaply fixed.** On the same features, ridge
  scores 51.48% and the tuned booster 32.73%, so non-linearity plus the right
  loss is worth about 19 points. Further capacity is worth nothing: every
  larger configuration tried is worse.
- **Distribution shift: yes, and it is now the binding constraint.** Refitting
  on the validation window's own labels halves the error, 32.73% to 16.48%.
  Half of what remains is the model being out of date, not the model being
  wrong. The same effect explains the validation-to-test gap: the selected
  configuration scores 32.73% on fold0 and 42.90% on test, a period whose job
  mix shifted again.

Energy is duration and power multiplied, so its error compounds
theirs: with actual duration the energy error would be roughly the power error,
and the observed 55.23% is close to what a 42.90% duration error and a 19.91%
power error produce together. Bringing energy into band requires duration below
roughly 25%, which these features do not reach out of time.

The credible next step is not a bigger model but a fresher one: periodic refit,
or an online update of the history features, which the learning curve prices at
up to 16 points of WAPE.

## Scheduling impact on the held-out cohort

Same 23,560 test jobs, 880 nodes, a six-hour maximum carbon delay, the
15-minute grid, actual runtimes for completion, and measured power profiles for
ex-post accounting.

| Configuration                | Emissions (tCO2e) | Saved vs EASY | Oracle recovery | Waiting mean (s) | Bounded slowdown |
| ---------------------------- | ----------------: | ------------: | --------------: | ---------------: | ---------------: |
| EASY, actual jobs            |           17.4712 |         0.00% |               - |            422.9 |             2.11 |
| Carbon, actual jobs          |           17.2402 |         1.32% |         100.00% |          7,870.7 |           459.98 |
| Carbon, ridge predictions    |           17.4595 |         0.07% |           5.05% |          7,663.0 |           463.76 |
| Carbon, gradient predictions |           17.3727 |         0.56% |      **42.65%** |          7,922.7 |           463.07 |

Agreement with the actual-value carbon schedule, which is what the recovery
figure is made of:

| Cohort | Model    | Duration WAPE | Target agreement | Start agreement | Start delta mean |  median |      p95 |      p99 |      max |
| ------ | -------- | ------------: | ---------------: | --------------: | ---------------: | ------: | -------: | -------: | -------: |
| all    | ridge    |       101.75% |           92.87% |          59.53% |          889.6 s |   0.0 s |  5,614 s | 19,024 s | 27,213 s |
| all    | gradient |        42.90% |       **95.10%** |      **68.59%** |      **526.7 s** |   0.0 s |  2,841 s | 16,307 s | 30,453 s |
| >= 1 h | ridge    |        99.24% |           40.34% |          32.79% |        5,396.2 s | 1,800 s | 20,820 s | 21,499 s | 27,213 s |
| >= 1 h | gradient |        38.53% |       **65.96%** |      **60.57%** |    **2,518.8 s** | **0 s** | 18,000 s | 21,211 s | 30,453 s |

The long-job cohort is the one that matters: the carbon target of a job depends
only on its predicted duration, because average power multiplies every
candidate start for that job by the same positive constant and is not a
resource constraint. The command verifies this with a counterfactual run using
predicted duration and actual power; every target and simulated start is
identical. A power model will matter once power enters a cap or a
multi-objective policy.

Target agreement on jobs of at least an hour rises from 40.34% to 65.96% and
start agreement from 32.79% to 60.57%, which is where the recovery comes from.
QoS is unchanged in aggregate: mean waiting moves from 7,663 s to 7,923 s
against 7,871 s for the actual-value schedule, and mean bounded slowdown from
463.76 to 463.07 against 459.98. The remaining 57% of the carbon benefit is
lost to the long jobs whose target still moves, which the band table traces to
the `>= 3 h` under-prediction.

That 42.65% is one point on a frontier, and it is the worst point on it. Swept
over the delay budget, the model retains 44.69% of the available saving at one
hour, 50.66% at three, 42.65% at six, 54.18% at twelve and 59.61% at
twenty-four: a wider budget offers more nearly-as-clean slots for a mistimed
job to land in. The same sweep also shows that predicted durations disable EASY
backfilling outright on this cohort — the reservation guard cannot trust a
projection built from under-predicted runtimes — which costs QoS at every
budget and 0.02% of emissions at a budget of zero. The tables are in
[the simulator README](../hpc_sim/README.md#with-predicted-scheduling-inputs).

## Use

Reproduce the validation search, which never reads test:

```bash
.venv/bin/python scripts/experiment_job_models.py
```

Train, evaluate once on test, and write the artifacts:

```bash
.venv/bin/python scripts/train_job_models.py --model gradient
.venv/bin/python scripts/train_job_models.py --model ridge \
  --output-dir data/job_predictions/ridge_baseline
```

`--model gradient` writes `job_models.joblib` with a readable
`job_models.json` sidecar carrying the features, transforms, split boundaries
and hyperparameters; the ridge path still writes a fully readable
`job_models.json`. Both write `test_predictions.parquet` (job id plus three
consistent predicted values, no actual targets), `test_metrics.csv` and
`test_duration_bands.csv`.

Run the held-out cohort with predicted scheduling inputs:

```bash
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet
```

Measure the scheduling impact of each model, then put them side by side:

```bash
PYTHONPATH=src .venv/bin/python -m job_prediction.scheduling_impact
PYTHONPATH=src .venv/bin/python -m job_prediction.scheduling_impact \
  --predictions data/job_predictions/ridge_baseline/test_predictions.parquet \
  --output-dir data/job_predictions/ridge_baseline
.venv/bin/python scripts/compare_job_prediction_impact.py
```

Sweep the delay budget with predicted inputs against actual ones:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet \
  --max-delay-hours 0 1 3 6 12 24 --decision-granularity-minutes 15
```

Run the checks, which include the anti-leakage guards:

```bash
.venv/bin/python tests/check_job_prediction.py
```

## Caveats

- The boosted artifact is a pickle. Its audit trail is the JSON sidecar, not
  the model file; the ridge artifact remains fully readable and is kept for
  that reason.
- The history features need the whole chronological trace, because a test
  job's history is made of earlier jobs. `GradientJobPredictor.predict` takes
  the complete frame and returns every row for exactly this reason; handing it
  a slice would silently change the features.
- Selection used one fold as the decision rule and three more as a stability
  report. A configuration that wins on the four-fold mean but loses on fold0
  exists (`+fingerprint_hist +global_hist`, 54.67% against 55.67%), so this
  choice is a documented judgement, not a forced one.
