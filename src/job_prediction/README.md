# Job prediction models

This package predicts the runtime and mean whole-job power available to the schedulers at submission time.

Actual runtime and measured power remain attached to simulated jobs only for execution replay and ex-post evaluation.

| Scheduling inputs                | Evaluation only                  |
| -------------------------------- | -------------------------------- |
| predicted duration               | actual runtime                   |
| predicted average power          | measured 20-second power profile |
| requested resources              | actual completion time           |
| submission timestamp / QoS       | actual energy and emissions      |
| causal history of completed jobs |                                  |

Carbon intensity is predicted separately. Job models do not predict CO2.

## Results

Both models are evaluated once on the same frozen temporal test partition.

| Test estimate            | Ridge baseline | Gradient boosting |
| ------------------------ | -------------: | ----------------: |
| Duration WAPE            |        101.75% |        **42.90%** |
| Power WAPE               |         21.39% |        **19.91%** |
| Composed energy WAPE     |         97.43% |        **55.23%** |
| Duration WAPE, jobs ≥1 h |         99.24% |        **38.53%** |
| Oracle recovery          |          5.05% |        **42.65%** |

## Inputs and targets

Two feature families are available.

### Request and calendar

Submission-time features include:

- requested walltime;
- nodes, cores, tasks, cores per task;
- GPUs and memory;
- submission priority and QoS;
- hour-of-day and weekday cycles;
- month and day of month;
- cores per node;
- memory per node and per core;
- GPUs and tasks per node;
- requested node-seconds and core-seconds;
- task/core consistency;
- `shared`, `req_switch`, and group indicators.

`num_tasks` is median-imputed with a missing-value indicator. Skewed numeric fields use `log1p`.

### Causal history

For a job submitted at `t`, history features use only matching jobs that completed before `t`.

Statistics include:

- count;
- geometric mean;
- standard deviation;
- last value;
- mean of the last eight;
- maximum;
- time since the most recent observation.

They are computed for:

- duration;
- average power;
- energy;
- runtime / requested-walltime ratio.

The main grouping keys are:

```text
user_id
(user_id, time_limit, num_nodes_req)
```

Optional wider scopes exist for exact resource fingerprints, the global workload regime, and queue occupancy. Validation did not select them.

The trace contains 406 users, and 99.3% of test jobs belong to users already present in training.

Models never use job identifiers, allocated resources, eligibility/start timestamps, actual runtime, or measured power as features. `partition` is constant and `threads_per_core` is 99.6% missing, so both are excluded.

### Targets

Duration is:

```text
run_time
```

in seconds.

Energy is the integral of `node_power_consumption`, using the same final-sample handling as `carbon_accounting`.

Average whole-job power is:

```text
energy_kWh * 3,600,000 / duration_s
```

The power trace already represents aggregate job power across allocated nodes and is never multiplied by node count.

## Anti-leakage contract

Leakage protection is tested in `tests/check_job_prediction.py`.

### Temporal order

`temporal_split` stable-sorts by:

```text
(submit_time, job_id)
```

before splitting.

The clean Parquet contains 77,135 adjacent submit-time inversions, so source row order cannot be used directly.

The 70/15/15 split uses half-open boundaries and keeps identical submission timestamps in the same partition.

### Label availability

Only labels observable at a cutoff may enter fitting.

Counts on the committed trace:

- 109,389 training jobs completed by the validation boundary;
- 23,534 validation jobs completed by the test boundary;
- 133,190 pre-test outcomes available for the final refit.

`end_time` is used only for availability checks and is never encoded as a feature.

### Causal history

`causal_group_stats` includes a past job only when:

```text
past_job.end_time <= current_job.submit_time
```

A job therefore cannot observe itself, currently running jobs, or future outcomes.

The leakage test modifies every outcome occurring after a cutoff and verifies that earlier feature values remain unchanged.

### Split

| Partition  |    Jobs | Starts at               |
| ---------- | ------: | ----------------------- |
| Train      | 109,676 | beginning of workload   |
| Validation |  23,826 | 2020-09-25 04:51:34 UTC |
| Test       |  23,560 | 2020-10-05 14:00:51 UTC |

## Selection protocol

`scripts/experiment_job_models.py` removes the test frame before model selection.

Candidates are evaluated with expanding-window folds inside the development period. `fold0`, which reproduces the frozen train/validation boundary, selects the configuration. Earlier folds provide stability information.

| Fold  | Cutoff           |   Train | Validation | Jobs <10 s |
| ----- | ---------------- | ------: | ---------: | ---------: |
| fold3 | 2020-07-17 10:20 |  53,172 |     19,734 |      10.5% |
| fold2 | 2020-08-18 19:43 |  73,118 |     19,944 |       2.4% |
| fold1 | 2020-09-02 19:34 |  93,364 |     19,744 |       2.4% |
| fold0 | 2020-09-28 14:02 | 113,194 |     19,729 |      66.4% |

The workload changes substantially near the end of the trace. Test contains 69.1% jobs shorter than 10 seconds, making fold0 the closest development fold.

WAPE is:

```text
sum(abs(prediction - actual)) / sum(actual)
```

Short jobs dominate job count but not runtime. On test, jobs under 10 seconds are 69% of jobs but only 0.04% of runtime seconds, while jobs of at least one hour account for 92%.

## Validation search

The complete leaderboard is:

```text
data/job_predictions/validation_leaderboard.csv
```

with 44 configurations.

### Feature ablation

HistGradientBoosting with `log1p` target and absolute-error loss:

| Features                 |    Cols | WAPE fold0 | 4-fold mean |  ≥1 h WAPE |
| ------------------------ | ------: | ---------: | ----------: | ---------: |
| A base                   |      19 |    193.34% |     125.75% |     73.76% |
| B + derived              |      38 |    572.05% |     251.82% |     71.55% |
| C + `user_id`            |      39 |    215.43% |     125.17% |     62.49% |
| D + `user_hist`          |      69 |     50.31% |      64.12% |     47.54% |
| **E + `signature_hist`** | **100** | **39.66%** |  **57.45%** | **36.36%** |
| F + `fingerprint_hist`   |     131 |     44.83% |      58.94% |     40.89% |
| G + `global_hist`        |     162 |     43.00% |      59.74% |     38.49% |
| H + queue                |     166 |     49.08% |      59.76% |     45.81% |

Most of the gain comes from causal user history. Adding request-signature history improves it further. Wider history and queue features do not improve fold0.

### Model families

Feature set E:

| Model           | Transform | Loss         | WAPE fold0 | 4-fold mean |  ≥1 h WAPE | Fit (s) |
| --------------- | --------- | ------------ | ---------: | ----------: | ---------: | ------: |
| median          | raw       | —            |    104.11% |      99.39% |     98.81% |     0.0 |
| ridge           | log1p     | squared      |     51.48% |      66.53% |     47.72% |     0.0 |
| decision tree   | log1p     | squared      |     61.04% |      75.75% |     50.02% |     2.4 |
| decision tree   | log1p     | absolute     |     44.15% |      73.47% |     37.66% |    19.1 |
| random forest   | log1p     | squared      |     62.06% |      67.77% |     60.40% |    37.7 |
| extra trees     | log1p     | squared      |     56.22% |      65.57% |     55.30% |    15.7 |
| hist GBR        | log1p     | squared      |     57.20% |      67.31% |     53.95% |    10.4 |
| hist GBR        | log1p     | poisson      |     58.36% |      67.75% |     57.81% |    10.6 |
| hist GBR        | raw       | squared      |    537.69% |     206.86% |     26.07% |     9.4 |
| hist GBR        | raw       | absolute     |     70.73% |      67.49% |     31.33% |    13.0 |
| segmented, soft | raw       | absolute     |     43.89% |      62.69% |     26.47% |    27.1 |
| segmented, hard | raw       | absolute     |     37.27% |      63.03% |     29.44% |    27.5 |
| **hist GBR**    | **log1p** | **absolute** | **39.66%** |  **57.45%** | **36.36%** |    12.6 |

Raw-target models perform well on the longest jobs but poorly overall. Segmented models improve long-job error without improving total WAPE enough to justify the additional classifier and regressors.

### Weighting and capacity

Starting from the selected HistGradientBoosting configuration:

| Variant                      | WAPE fold0 | 4-fold mean |  ≥1 h WAPE |
| ---------------------------- | ---------: | ----------: | ---------: |
| no weight                    |     39.66% |      57.45% |     36.36% |
| **weight `log1p(duration)`** | **32.73%** |  **55.67%** | **25.13%** |
| weight `log1p(duration)^2`   |     41.14% |      60.55% |     29.15% |
| weight `duration^0.25`       |     39.74% |      60.41% |     28.80% |
| 800 iterations               |     42.59% |      58.38% |     35.36% |
| 800 iterations, lr 0.03      |     36.99% |      57.11% |     30.47% |
| 127 leaves                   |     37.32% |      59.13% |     30.54% |
| 31 leaves, 800 iterations    |     34.78% |      56.20% |     26.24% |
| minimum 100 samples/leaf     |     35.48% |      55.96% |     27.94% |
| L2 = 10                      |     35.95% |      57.23% |     28.22% |
| segmented hard + same weight |     41.17% |      67.79% |     27.48% |

`log1p(duration)` weighting is selected.

Increasing tree capacity does not improve validation results. Quantile losses from 0.55 to 0.8 were also tested; the best reached 37.11% fold0 WAPE.

One wider-history configuration, E plus fingerprint and global history, has a slightly better four-fold mean (`54.67%` vs `55.67%`) but worse fold0 performance (`36.65%` vs `32.73%`).

### Power model

Power selects the same feature set without duration weighting.

| Model         | Transform | Loss         | WAPE fold0 | 4-fold mean |  ≥1 h WAPE |
| ------------- | --------- | ------------ | ---------: | ----------: | ---------: |
| **hist GBR**  | **log1p** | **absolute** | **12.95%** |  **15.17%** | **12.01%** |
| random forest | log1p     | squared      |     13.12% |      15.48% |     10.69% |
| extra trees   | log1p     | squared      |     13.88% |      15.68% |     10.73% |
| ridge         | log1p     | squared      |     13.89% |      17.37% |     12.14% |
| hist GBR      | log1p     | squared      |     15.09% |      15.88% |     10.84% |

Power is substantially easier to predict than duration, and differences between model families are small.

Direct energy prediction performs poorly; the best validation WAPE is 94.97%.

## Physical consistency

Two consistent output parameterizations were compared:

1. predict duration and power, derive energy;
2. predict duration and energy, derive power.

Duration × power performs better:

| Fit    | Duration × power energy MAE | Duration × energy energy MAE |
| ------ | --------------------------: | ---------------------------: |
| tuning |                  0.7959 kWh |                   1.6492 kWh |
| final  |                  1.1714 kWh |                   1.3762 kWh |

Persisted predictions therefore enforce:

```text
predicted_energy_kwh =
    predicted_average_power_watts
    * predicted_duration_seconds
    / 3,600,000
```

`SchedulingPrediction` rejects inconsistent prediction files.

## Test accuracy

The selected configuration is refitted on the complete development period and evaluated once on test.

| Estimate         |        MAE |       RMSE |   WAPE | Median rel. |      Bias |    p95 abs | ≥1 h WAPE |
| ---------------- | ---------: | ---------: | -----: | ----------: | --------: | ---------: | --------: |
| Duration         | 1,181.30 s | 6,100.66 s | 42.90% |      31.19% | -568.06 s | 3,875.98 s |    38.53% |
| Average power    |   431.09 W | 3,116.11 W | 19.91% |      13.78% | -184.53 W |   634.83 W |    11.57% |
| Direct energy    |   1.86 kWh |  24.21 kWh | 64.84% |      43.74% | -1.05 kWh |   1.73 kWh |    62.81% |
| Duration × power |   1.58 kWh |  19.87 kWh | 55.23% |      35.13% | -1.03 kWh |   1.67 kWh |    52.25% |

Ridge baseline on the same rows:

- duration WAPE: `101.75%`;
- power WAPE: `21.39%`;
- composed energy WAPE: `97.43%`;
- duration WAPE ≥1 h: `99.24%`.

### Duration by runtime

| Actual duration |   Jobs |         MAE |     WAPE | Median rel. |         Bias |     p95 abs |
| --------------- | -----: | ----------: | -------: | ----------: | -----------: | ----------: |
| <10 s           | 16,273 |      7.25 s |  443.10% |      26.60% |      +6.59 s |      2.96 s |
| 10–60 s         |    389 |    590.73 s | 2094.00% |     454.10% |    +587.79 s |  1,627.52 s |
| 1–10 min        |  1,749 |    782.22 s |  211.50% |      25.50% |    +620.58 s |  2,511.33 s |
| 10–60 min       |  2,737 |  1,157.21 s |   68.40% |      47.60% |    +534.86 s |  2,546.43 s |
| 1–3 h           |  1,397 |  2,773.12 s |   49.30% |      36.70% |  -1,371.50 s |  6,418.68 s |
| ≥3 h            |  1,015 | 18,792.45 s |   36.90% |      40.50% | -14,140.54 s | 61,087.91 s |

Jobs lasting at least three hours are 4.3% of test jobs but account for **69% of total absolute duration error**. Together with the 1–3 hour band, they account for 83%.

The main remaining duration error is underprediction of long jobs. For jobs ≥3 h the average bias is `-14,140.54 s`.

The ridge baseline bias on this group is `-50,739 s`.

## Learning curve and distribution shift

`data/job_predictions/validation_learning_curve.csv` contains the fold0 learning curve.

| Training data         | Base features WAPE | Selected features WAPE |
| --------------------- | -----------------: | ---------------------: |
| 20% random            |            709.60% |                 36.73% |
| 40%                   |          1,020.30% |                 39.40% |
| 60%                   |            707.83% |                 40.06% |
| 80%                   |            992.84% |                 39.04% |
| 100%                  |            818.22% |                 32.73% |
| 100%, in-sample refit |                  — |                 16.48% |

Using only the most recent rows performs worse than random subsets:

- 20% recent: `50.75%`;
- 40% recent: `41.81%`;
- 80% recent: `40.28%`.

The causal user statistics benefit from longer historical coverage.

The experiments show:

- causal history is the dominant feature source;
- additional rows from the same distribution provide limited gains;
- nonlinear modeling and the selected loss improve substantially over ridge;
- larger gradient-boosting configurations do not improve validation;
- temporal distribution shift remains significant.

Refitting directly on the validation regime reduces fold0 error from `32.73%` to `16.48%`.

The selected model then moves from `32.73%` on fold0 to `42.90%` on test, whose workload mix changes again.

Because composed energy multiplies predicted duration and power, duration remains the dominant source of energy error.

## Scheduling impact

The scheduling comparison uses the 23,560 held-out jobs, 880 nodes, a six-hour carbon delay budget, 15-minute carbon grid, actual completion times, and measured power for accounting.

| Configuration                | Emissions (tCO2e) | Saved vs EASY | Oracle recovery | Mean wait (s) | Mean bounded slowdown |
| ---------------------------- | ----------------: | ------------: | --------------: | ------------: | --------------------: |
| EASY, actual inputs          |           17.4712 |         0.00% |               — |         422.9 |                  2.11 |
| Carbon, actual inputs        |           17.2402 |         1.32% |         100.00% |       7,870.7 |                459.98 |
| Carbon, ridge predictions    |           17.4595 |         0.07% |           5.05% |       7,663.0 |                463.76 |
| Carbon, gradient predictions |           17.3727 |         0.56% |      **42.65%** |       7,922.7 |                463.07 |

### Scheduling agreement

| Cohort | Model    | Duration WAPE | Target agreement | Start agreement | Mean start delta |  Median |      p95 |      p99 |      Max |
| ------ | -------- | ------------: | ---------------: | --------------: | ---------------: | ------: | -------: | -------: | -------: |
| All    | Ridge    |       101.75% |           92.87% |          59.53% |          889.6 s |     0 s |  5,614 s | 19,024 s | 27,213 s |
| All    | Gradient |        42.90% |       **95.10%** |      **68.59%** |      **526.7 s** |     0 s |  2,841 s | 16,307 s | 30,453 s |
| ≥1 h   | Ridge    |        99.24% |           40.34% |          32.79% |        5,396.2 s | 1,800 s | 20,820 s | 21,499 s | 27,213 s |
| ≥1 h   | Gradient |        38.53% |       **65.96%** |      **60.57%** |    **2,518.8 s** | **0 s** | 18,000 s | 21,211 s | 30,453 s |

For this carbon policy, the selected target depends on predicted duration but not on absolute predicted power: average power multiplies every candidate cost for a job by the same positive constant.

A counterfactual using predicted duration and actual power produces identical carbon targets and starts.

Predicted power becomes operationally relevant when power is used as a scheduling constraint or part of a multi-objective policy.

For jobs of at least one hour, gradient boosting raises target agreement from `40.34%` to `65.96%` and start agreement from `32.79%` to `60.57%`.

The remaining loss in oracle recovery is concentrated in long jobs whose predicted duration still shifts the selected carbon window.

### Recovery by delay budget

Gradient predictions retain:

| Delay budget | Oracle recovery |
| ------------ | --------------: |
| 1 h          |          44.69% |
| 3 h          |          50.66% |
| 6 h          |          42.65% |
| 12 h         |          54.18% |
| 24 h         |          59.61% |

Wider budgets generally provide more alternative low-carbon start times, reducing the scheduling impact of a duration error.

Predicted runtimes also affect EASY reservations independently of carbon scheduling. On this test cohort, underprediction prevents reliable EASY reservations and causes the scheduler to fall back to FCFS whenever a pivot is present.

Details of the full delay sweep are in [the simulator README](../hpc_sim/README.md#predicted-scheduling-inputs).

## Use

Run validation search:

```bash
.venv/bin/python scripts/experiment_job_models.py
```

Train the selected gradient model and ridge baseline:

```bash
.venv/bin/python scripts/train_job_models.py --model gradient

.venv/bin/python scripts/train_job_models.py \
  --model ridge \
  --output-dir data/job_predictions/ridge_baseline
```

The gradient model writes:

```text
job_models.joblib
job_models.json
test_predictions.parquet
test_metrics.csv
test_duration_bands.csv
```

`job_models.json` records feature definitions, transforms, split boundaries, and hyperparameters.

The ridge artifact remains fully represented in JSON.

`test_predictions.parquet` contains only job IDs and the three internally consistent predicted quantities. Actual targets are not included.

Run scheduling with predicted inputs:

```bash
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet
```

Compare scheduling impact:

```bash
.venv/bin/python -m job_prediction.scheduling_impact

.venv/bin/python -m job_prediction.scheduling_impact \
  --predictions data/job_predictions/ridge_baseline/test_predictions.parquet \
  --output-dir data/job_predictions/ridge_baseline

.venv/bin/python scripts/compare_job_prediction_impact.py
```

Sweep carbon-delay budgets:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet \
  --max-delay-hours 0 1 3 6 12 24 \
  --decision-granularity-minutes 15
```

Run leakage and consistency checks:

```bash
.venv/bin/python tests/check_job_prediction.py
```

## Caveats

The gradient-boosting artifact is serialized with pickle. Its readable audit trail is `job_models.json`; the ridge artifact remains fully readable.

History features require the complete chronological frame because test-job features depend on earlier completed jobs. `GradientJobPredictor.predict` therefore accepts the complete frame and returns predictions for all rows. Predicting from an isolated slice would change the causal history.

Model selection uses fold0 as the decision fold and the other three folds as stability checks. One configuration with additional fingerprint and global history performs slightly better on the four-fold mean (`54.67%` vs `55.67%`) but worse on fold0 (`36.65%` vs `32.73%`).
