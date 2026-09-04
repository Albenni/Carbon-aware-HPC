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

No model predicts CO2. The scheduler combines the job estimates with the
separate carbon-intensity signal.

## Inputs and targets

The initial feature set is intentionally small:

- requested walltime, nodes, cores, tasks, cores per task, GPUs, and memory;
- recorded submission-time priority and QoS;
- hour-of-day and weekday cycles derived from `submit_time`.

`num_tasks` is median imputed and gets a missing value indicator. Numeric
requests are transformed with `log1p`; QoS is one-hot encoded. Model selection
fits imputation, vocabulary, centering, and scaling on train alone. The final
artifact refits them on train plus validation; test rows are never consulted.

The model does not use job/user/group identifiers, allocated resources,
eligibility/start/end times, runtime, or measured power as features. Partition
is constant in the retained trace; `threads_per_core` and `req_nodes` are mostly
missing, so they are omitted from this first compact model.

Targets follow the existing accounting contract:

- duration is `run_time` in seconds;
- energy is the integral of `node_power_consumption`, trimming or extending the
  final 20-second sample exactly as `carbon_accounting` does;
- average power is the duration-weighted whole-job mean
  `energy_kWh * 3,600,000 / duration_s`.

The power trace is already aggregated across the allocated nodes. It is never
multiplied by the node count.

## Temporal evaluation

`temporal_split` first stable-sorts by `(submit_time, job_id)`. This is required:
the full clean parquet contains 77,135 adjacent submit-time inversions. The
70% / 15% / 15% boundaries are half open and keep every equal timestamp in one
partition.

On the committed trace the split is:

| Partition  |    Jobs | Starts at                 |
| ---------- | ------: | ------------------------- |
| train      | 109,676 | beginning of the workload |
| validation |  23,826 | 2020-09-25 04:51:34 UTC   |
| test       |  23,560 | 2020-10-05 14:00:51 UTC   |

Submission order alone is not enough to prevent target leakage: a job may be
submitted before a boundary and finish after it. The tuning fit therefore uses
only the 109,389 training jobs completed by the validation boundary, and only
23,534 validation labels completed by the test boundary. The final refit uses
the 133,190 pre-test jobs whose outcomes are known by that instant. `end_time`
is retained for this availability check only and is never encoded as a feature.

Three log-ridge regressors predict duration, average power, and energy.
Ridge strength is selected on the future validation period, then the chosen
models are refitted on train plus validation before one test evaluation. This
keeps the implementation dependencies light and makes the baseline easy to audit.
Predictions are bounded by the target range observed in that development data,
which prevents uncontrolled log-space extrapolation without consulting test
outcomes.

Independent predictions need not satisfy physics, so two compositions are
compared on validation energy MAE:

1. predict duration and power, then derive energy;
2. predict duration and energy, then derive power.

The first wins on PM100 (2.1892 versus 2.3050 kWh validation MAE). Persisted
scheduling inputs therefore always satisfy

```text
predicted_energy_kwh
    = predicted_average_power_watts * predicted_duration_seconds / 3,600,000
```

## Test-period accuracy

`MAE` and `RMSE` retain the target unit. `WAPE` is total absolute error divided
by the total actual quantity; mean and median absolute relative errors expose
per-job behavior.

| Estimate            |         MAE |         RMSE |    WAPE | Mean relative | Median relative |
| ------------------- | ----------: | -----------: | ------: | ------------: | --------------: |
| duration model      | 2,801.922 s | 11,805.622 s | 101.75% |     8,095.67% |       7,389.04% |
| average-power model |   463.123 W |  2,847.098 W |  21.39% |        21.85% |          17.28% |
| energy model        |   2.876 kWh |   27.925 kWh | 100.35% |    11,535.41% |      10,726.34% |
| duration × power    |   2.792 kWh |   27.814 kWh |  97.43% |     8,699.71% |       8,428.98% |

The workload shifts sharply: test contains a large mass of one-to-three-second jobs, so a
small absolute miss can be thousands of percent.

## Use

Train, evaluate, and write the model plus held-out predictions:

```bash
.venv/bin/python scripts/train_job_models.py
```

This writes under `data/job_predictions/`:

- `job_models.json`: preprocessing, coefficients, split metadata, and selected
  composition;
- `test_predictions.parquet`: job id plus three consistent predicted values,
  with no actual targets;
- `test_metrics.csv`: the test report above.

Run the held-out cohort with predicted scheduling inputs and actual ex-post
accounting:

```bash
.venv/bin/python scripts/run_simulation.py \
  --scheduler carbon \
  --workload data/processed/pm100_clean.parquet \
  --job-predictions data/job_predictions/test_predictions.parquet
```

The prediction file selects its own job ids. Unless explicitly overridden,
backfilling then uses `scheduling` runtime estimates; without a prediction file,
the command retains classic `time_limit` behavior. The runner first verifies
that every artifact id exists in the workload, then applies optional release
filters or a chronological `--limit`; a wrong or partial workload cannot be
silently treated as the intended test cohort.

## Scheduling impact on the held-out cohort

The scheduling-impact command compares the same 23,560 test jobs in three
configurations: EASY with actual job values, carbon-aware scheduling with
actual duration and power, and carbon-aware scheduling with the persisted
predictions. All runs use 880 nodes, a six-hour maximum carbon delay, the
15-minute grid, actual runtimes for completion, and measured power profiles for
ex-post accounting:

```bash
PYTHONPATH=src .venv/bin/python -m job_prediction.scheduling_impact
```

| Configuration          | Emissions (tCO2e) | Saved vs EASY | Waiting mean (s) | Bounded slowdown mean |
| ---------------------- | -----------------: | ------------: | ---------------: | --------------------: |
| EASY, actual jobs      |            17.4712 |         0.00% |            422.9 |                  2.11 |
| Carbon, actual jobs    |            17.2402 |         1.32% |          7,870.7 |                459.98 |
| Carbon, predicted jobs |            17.4595 |         0.07% |          7,663.0 |                463.76 |

Predictions therefore retain 5.05% of the carbon saving available with actual
job values. Relative to that actual-value carbon schedule, emissions increase
by 1.27%, mean waiting decreases by 2.64%, and mean bounded slowdown increases
by 0.82%. The loss is material because almost all of the available carbon
benefit disappears while the voluntary delay remains.

The predicted target changes for 1,679 jobs (7.13%), and contention propagates
those differences to 9,534 simulated starts (40.47%). The absolute start-time
difference is 889.6 seconds on average, zero at the median, 5,614 seconds at
p95, and 27,213 seconds at the maximum. Large relative duration errors are
concentrated in very short jobs, but long jobs drive the direct carbon choice:
59.66% of jobs lasting at least one hour receive a different target, compared
with 0.32% of jobs shorter than ten seconds.

Average power does not affect the choice in the current policy: it multiplies
every candidate cost for a job by the same positive value and is not a resource
constraint. The command verifies this with a counterfactual run using predicted
duration and actual power; every target and simulated start is identical to the
fully predicted run. A power model will matter once power is part of a cap or a
multi-objective policy.

The command writes three reproducible artifacts beside the model outputs:

- `scheduling_impact_metrics.csv`: aggregate scheduler and QoS metrics;
- `scheduling_impact_jobs.parquet`: prediction errors, targets, simulated
  starts, waiting, slowdown, and emissions for every job;
- `scheduling_impact_duration_bands.csv`: duration-error and start-change
  summaries by actual runtime.

These results justify improving the duration model next; they do not justify
adding complexity to the power model for this scheduler.

Run the short deterministic checks with:

```bash
.venv/bin/python tests/check_job_prediction.py
```
