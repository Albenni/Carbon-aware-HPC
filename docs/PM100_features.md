# Job features description

| Column                 | Description                                                          | Type      |
| ---------------------- | -------------------------------------------------------------------- | --------- |
| cores_alloc_layout     | Map: list of cores allocated per node                                | string    |
| cores_allocated        | Map: number of cores allocated per node                              | string    |
| cores_per_task         | Number of cores required for each task                               | int       |
| derived_ec             | Highest exit code of all job steps                                   | string    |
| eligible_time          | Time job is eligible for running                                     | timestamp |
| end_time               | Time of termination                                                  | timestamp |
| group_id               | Group job submitted as                                               | int       |
| job_id                 | Job ID (anonymized)                                                  | int       |
| job_state              | State of the job, see enum job_states for possible values            | string    |
| nodes                  | List of nodes allocated to job                                       | List[int] |
| num_cores_req          | Number of cores requested by the user                                | int       |
| num_cores_alloc        | Number of cores allocated to the job                                 | int       |
| num_gpus_req           | Number of GPUs requested by the user                                 | int       |
| num_gpus_alloc         | Number of GPUs allocated to the job                                  | int       |
| num_nodes_req          | Number of nodes requested by the user                                | int       |
| num_nodes_alloc        | Number of nodes allocated to the job                                 | int       |
| mem_req                | Amount of memory (RAM) requested by the user                         | int       |
| mem_alloc              | Amount of memory (RAM) allocated to the job                          | int       |
| num_tasks              | Number of tasks requested by a job or job step                       | float     |
| partition              | Name of assigned partition (anonymized)                              | string    |
| priority               | Relative priority of the job, 0=held, 1=required nodes DOWN/DRAINED  | int       |
| qos                    | Quality of Service (anonymized, categorical)                         | string    |
| req_nodes              | Comma-separated list of required nodes                               | string    |
| run_time               | Job run time (seconds)                                               | int       |
| shared                 | 1 if job can share nodes with other jobs, 0 otherwise                | string    |
| start_time             | Time execution begins (actual or expected)                           | timestamp |
| state_reason           | Reason job still pending or failed,see slurm.h:enum job_state_reason | string    |
| submit_time            | Time of job submission                                               | timestamp |
| threads_per_core       | Threads per core required by job                                     | float     |
| time_limit             | Maximum run time in minutes or INFINITE                              | int       |
| user_id                | User ID for a job or job step (anonymized)                           | int       |
| node_power_consumption | Power consumption of the job, recorded at Node level                 | List[int] |
| cpu_power_consumption  | Power consumption of the job, recorded at CPU level                  | List[int] |
| mem_power_consumption  | Power consumption of the job, recorded at Memory level               | List[int] |

## Terminal-job contention scenario

The optional scheduling scenario combines the clean `COMPLETED` evaluation
cohort with terminal non-completed executions from the raw PM100 trace. These
jobs enter the same release queue, are placed by the same scheduler, allocate
nodes, run for their observed duration, and release the nodes normally.

Partition `1` contains 50,165 usable terminal contenders:

| Status          | Jobs   |
| --------------- | -----: |
| `FAILED`        | 29,561 |
| `CANCELLED`     | 10,876 |
| `TIMEOUT`       |  8,564 |
| `OUT_OF_MEMORY` |    997 |
| `NODE_FAIL`     |    167 |

A contender is loaded only when `submit_time <= eligible_time <= start_time <
end_time` and `num_nodes_alloc` is positive and matches a non-empty list of
unique node ids. Missing or inconsistent data is not replaced. This excludes
18 zero-duration rows and one `TIMEOUT` whose eligibility precedes submission.
The 149 valid request/allocation mismatches use `num_nodes_alloc`, because the
scenario models the resources actually held. Together the valid contenders
represent 575,866.80 allocated node-hours.

The full mixed workload therefore contains 207,227 jobs: 157,062 clean jobs and
50,165 contention-only jobs. A temporal subset includes terminal jobs that
arrived before the end of the subset and had not historically completed before
its start. They are still rescheduled from their original eligibility time; no
historical start is imposed on FCFS, EASY or the carbon-aware scheduler.

Resource and evaluation semantics remain separate:

- all mixed-workload jobs contribute to busy-node time, peak nodes, utilisation,
  queue order, FCFS blocking and EASY reservations;
- only the clean `COMPLETED` ids contribute to waiting/turnaround summaries,
  energy and emissions;
- terminal contenders carry `power=None`; no zero-valued or synthetic profile
  is created;
- `CarbonAwareScheduler` offers contenders to EASY immediately and never
  computes a carbon target for them;
- the power-cap policy is not included because aggregate power cannot be
  enforced honestly for jobs without a scheduling power estimate.

Run the original clean experiment by omitting `--contention-workload`, or the
additional scenario with:

```bash
.venv/bin/python scripts/carbon_tradeoff.py \
  --limit 5000 \
  --contention-workload data/job_table.parquet \
  --max-delay-hours 0 6
```

The CSV records both `scheduled_jobs` and `contention_only_jobs`. System metrics
describe the whole simulated workload; per-job QoS and carbon fields describe
the unchanged clean cohort.

On the committed 5,000-job debug cohort, the mixed workload contains 2,227
terminal contenders:

| Scenario            | Scheduled jobs | FCFS wait mean | EASY wait mean | EASY utilisation |
| ------------------- | -------------: | -------------: | -------------: | ---------------: |
| clean cohort        |          5,000 |         31.4 s |         31.2 s |             9.7% |
| terminal contention |          7,227 |         91.1 s |         87.8 s |            20.6% |

At a six-hour carbon delay budget, evaluated-job emissions are 4.4268 tCO2e
(4.62% below mixed-workload EASY) and mean evaluated waiting is 6,541.9 s. The
same clean cohort without terminal contention reports 4.4245 tCO2e, 4.67%, and
6,391.4 s. The scheduling comparison therefore changes while the evaluated ids
and accounting boundary do not. The terminal result is still a counterfactual:
each job keeps its observed duration after being rescheduled, and its failure or
cancellation mechanism is not simulated. The 20,983 completed partition-1
executions excluded from the clean cohort also remain outside this scenario.
