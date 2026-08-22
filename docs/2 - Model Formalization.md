## 2. Formalizzazione del modello di carbon footprint

Definire un modello sufficientemente semplice ma utile per lo scheduling e la simulazione.

### Possibili modelli iniziali

- Modello a **potenza media costante**
- Modello a **energia totale + carbon intensity media**
- Eventuali semplificazioni progressive da raffinarsi in seguito

### Questioni aperte

- Come gestire i **job multi-nodo**?
  - Somma sui nodi?
  - Somma su partizioni?
- Consideriamo differenze di **efficienza hardware** tra nodi/partizioni?
- È davvero necessario prevedere direttamente la CO₂ come target ML?
  - Probabilmente **no**
  - Più plausibile combinare predizioni di:
    - energia,
    - potenza,
    - tempo/durata

### Domande guida

- La **carbon intensity** va presa:
  - come dato storico reale,
  - come forecast,
  - come serie sintetica?
- Quale **granularità temporale** usare?
  - 5 minuti
  - 15 minuti
  - 1 ora
- Il simulatore opererà in **tempo continuo** o **tempo discreto**?
- Consideriamo solo **operational carbon** oppure anche **embodied carbon**?
  - Suggerimento: considerare **solo operational carbon**

---

## 2. Formalization of the Carbon Footprint Model

### 2.1 Objective and Model Boundaries

The objective is to estimate the emissions associated with the execution of a job and use
this estimate to compare possible start times. The model must therefore
separate two elements:

- the job's energy consumption, described by its duration and power;
- the carbon intensity of electricity, which depends on time and on the
  electricity grid zone of the datacenter.

The first version assigns to the job the **operational emissions associated with the energy
supplied to the allocated nodes during execution**. The measurement also includes any
baseline power present in the node profiles, but excludes embodied carbon and
datacenter consumption outside those profiles, such as cooling and networking. The
result is therefore not the complete carbon footprint of the site. A PUE factor
may be added once reliable data becomes available.

Under the provider contract, $CI(t)$ is expressed in `gCO2e/kWh` and $C_j$ in
`gCO2e`. The accounting result retains the earlier abbreviated field name
`emissions_gco2`, but represents grams of CO₂-equivalent when this provider is
used.

At present, carbon accounting, the two power models, an actual-only
carbon-intensity provider backed by a real Electricity Maps `IT-NO` series, the
discrete-event simulator, and the reference baselines — FCFS, EASY
backfilling, and a power-capped energy-aware policy — are implemented, together
with the evaluation metrics used to compare them. Forecast retrieval, the
carbon-aware policies, and the ML models described in the following sections
have not yet been implemented.

### 2.2 Notation

| Symbol      | Meaning                                              | Unit      |
| ----------- | ---------------------------------------------------- | --------- |
| $j$         | job under consideration                              | —         |
| $s_j$       | selected start time for the job                      | timestamp |
| $d_j$       | job duration                                         | s         |
| $P_j(\tau)$ | total job power $\tau$ seconds after start           | W         |
| $CI(t)$     | electricity-grid carbon intensity at time $t$        | gCO₂e/kWh |
| $E_j$       | energy consumed by the job                           | kWh       |
| $C_j(s_j)$  | operational emissions of the job if started at $s_j$ | gCO₂e     |

$P_j$ always represents the power of the entire job, not the power of a single node.
All timestamps must be timezone-aware, and elapsed times are
calculated in UTC.

### 2.3 General Model

With $\tau$ expressed in seconds, the energy consumed by the job is:

$$
E_j = \frac{1}{3{,}600{,}000}
      \int_0^{d_j} P_j(\tau)\,d\tau.
$$

The factor $3{,}600{,}000$ converts watt-seconds to kWh. The emissions produced
by starting the job at $s_j$ are:

$$
C_j(s_j) = \frac{1}{3{,}600{,}000}
           \int_0^{d_j} P_j(\tau)\,CI(s_j+\tau)\,d\tau.
$$

This equation highlights the essential difference between energy-aware
and carbon-aware scheduling. If duration, resources, and the power profile do not change,
shifting a job in time does not change $E_j$, but it may change $C_j(s_j)$ because it changes
the portion of the $CI(t)$ time series that overlaps with the execution. With a constant
carbon intensity $CI_0$, we instead obtain:

$$
C_j = E_j\,CI_0,
$$

which is useful as a sanity check for the calculation, but does not allow the benefit
of temporal shifting to be evaluated. It is also the limitation of the constant
national-factor model applied to PM100 by
[Shim](https://doi.org/10.1186/s42162-025-00586-6), which the dynamic signal introduced
here makes it possible to overcome.

In discrete form, execution is divided into segments $m$ with duration
$\delta_{j,m}$, offset $\tau_{j,m}$, and power $P_{j,m}$, where
$\sum_m\delta_{j,m}=d_j$:

$$
e_{j,m} = \frac{P_{j,m}\,\delta_{j,m}}{3{,}600{,}000},
\qquad
E_j = \sum_m e_{j,m},
\qquad
C_j(s_j) = \sum_m e_{j,m}\,CI(s_j+\tau_{j,m}).
$$

Carbon intensity is sampled at the beginning of each segment and kept constant
within that segment. For exact integration of two piecewise-constant signals, the
segments must also be split at every change in
$CI(t)$. The current implementation uses the former approach, with power segments
nominally 20 seconds long.

The expression "total energy × average carbon intensity" remains correct only if
the average is weighted by the energy of the segments. A simple time average is
equivalent only when power is constant.

### 2.4 The Two Models Used in the Project

#### Initial Model: Constant Average Power

For the scheduler, a constant average power
$\bar P_j$ is initially adopted. If the carbon intensity is $CI_k$ during time
interval $k$, and $\ell_{j,k}(s_j)$ is the number of seconds of overlap between that
interval and the execution of the job, then:

$$
E_j^{\mathrm{avg}} =
\frac{\bar P_j d_j}{3{,}600{,}000},
\qquad
C_j^{\mathrm{avg}}(s_j) =
\frac{\bar P_j}{3{,}600{,}000}
\sum_k CI_k\,\ell_{j,k}(s_j).
$$

This is the reference model for the future scheduler because it requires
only duration, average power, and the carbon-intensity time series, while keeping
the start time as a variable. The overlap-based formula is exact.

#### Evaluation Model: Measured Power Profile

To construct the perfect information benchmark and evaluate the error of the
simple model, the PM100 profile measured every 20 seconds is used. Each sample
becomes a $P_{j,m}$ value in the previous discrete formula. The final interval
is shortened to match the exact duration of the job; if the missing tail is no longer than
one interval, the last observed value is retained, while larger discrepancies are
rejected.

To compare the two models at equal energy consumption, the average power consistent with
the profile is the duration-weighted average:

$$
\bar P_j = \frac{\sum_m P_{j,m}\,\delta_{j,m}}{d_j}.
$$

This is the average that must be used for an equal-energy comparison. The
preprocessed `node_power_mean_W` field is currently an arithmetic mean of the samples
and may differ slightly when the final segment is partial; for this
comparison, the weighted average must therefore be recomputed.

The measured profile is information that becomes available only after execution:
it is therefore used for benchmarking and evaluation, not as input to a
realistic scheduler.

### 2.5 Decisions on Open Issues

| Issue                | Initial Choice                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-node jobs      | In general, the power of the nodes actually allocated to the job is summed. In [PM100](https://dl.acm.org/doi/10.1145/3624062.3624263), this sum, including both power sockets, is already contained in `node_power_consumption`: the profile must therefore be used directly, without multiplying it by `num_nodes_alloc` and without adding CPU or memory profiles. Partitions are not summed.                                                           |
| Hardware differences | The first analysis uses only anonymized partition `1` and assumes a single execution class. The measured profiles incorporate the differences actually observed, but no node- or partition-specific coefficient is introduced yet. Explicit heterogeneity remains an extension.                                                                                                                                                                            |
| ML target            | CO₂ is not predicted directly. Instead, $\hat d_j$ and $\hat{\bar P}_j$ are predicted, or $\hat E_j$ together with duration, using only features known at submission time. These quantities are then combined with the external carbon-intensity signal.                                                                                                                                                                                                   |
| Carbon intensity     | Actual historical and synthetic series are supported. Actual values and forecasts have separate access paths, and forecast retrieval remains part of later work. The `IT-NO` actual series is cached and covers 2020-04-30 to 2020-11-01, spanning the whole workload with 19 days of headroom for delayed schedules; it is used for accounting, ex-post evaluation, and the perfect-information benchmark, while synthetic values serve controlled tests. |
| Cluster capacity     | 880 nodes, the distinct node ids observed in partition `1` across the raw `COMPLETED` trace (ids 20-979). Observed peak concurrency is lower (787 raw, 774 after cleaning) and is a lower bound on the machine rather than its capacity. The value is configurable and is a candidate for sensitivity analysis.                                                                                                                                            |
| Granularity          | Initial choice: piecewise-constant carbon intensity at 15-minute resolution, explicitly requested from Electricity Maps. PM100 profiles remain at 20-second resolution; 5- and 60-minute resolutions will be used for sensitivity analysis.                                                                                                                                                                                                                |
| Temporal model       | A discrete-event simulator with continuous timestamps is planned. Submission/eligibility, completion, and changes in carbon intensity are events; a fixed tick is not required.                                                                                                                                                                                                                                                                            |
| Emissions boundary   | Only operational carbon assigned to the input power profile of the nodes is considered. Embodied carbon and overhead external to the profile are excluded.                                                                                                                                                                                                                                                                                                 |

For direct historical replay, the time series must refer to the electricity grid zone and
timestamps of the workload; any remapping to another period must be explicitly stated.
The source, version, resolution, signal semantics (for example average or
marginal, CO₂ or CO₂e), and missing-data policy must be recorded.
For a forecast, its issuance time must also be preserved,
so that the scheduler cannot access actual future values.
[Radovanović et al.](https://arxiv.org/abs/2106.11750) and
[Wiesner et al.](https://arxiv.org/abs/2110.13234) motivate this separation between
the actual signal, forecast, and decision.

Separating the job model from carbon intensity makes it possible to evaluate multiple
start times and isolate the two sources of error. If only total energy
is predicted, duration and an assumption about its temporal distribution are also required;
in the first version, uniform power is assumed.

The implemented provider uses timezone-aware UTC buckets and half-open ranges.
It raises an explicit error for a missing bucket instead of interpolating or
extrapolating. The Electricity Maps client is configured for North Italy
(`IT-NO`), the zone containing the CINECA site at Casalecchio di Reno, and keeps
the API provenance and historical estimation flags in the cache format when a
download succeeds.

### 2.6 Minimal Connection to Scheduling

The decision variable is $s_j$, and the schedule cost is
$\sum_j C_j(s_j)$. In the first simulator, jobs are non-preemptive and the nodes in the
partition are treated as equivalent, so capacity is tracked as a node count and
the `nodes` identifier list remains validation-only. A start time is feasible only if
the job is eligible, enough nodes are available, and any maximum delay constraint is
satisfied.

The implementation lives in `src/hpc_sim/`. It uses continuous timestamps with
no fixed tick; releases, completions, and carbon-intensity bucket boundaries are
events, and the last of these is delivered only to a policy that registers
interest, so a carbon-blind baseline never observes one. Actual duration always
governs when nodes are released, while a policy reads
`Job.scheduling_duration_seconds`, which returns a prediction when one exists —
this is the single seam that keeps §2.7's separation of decision information
from simulated execution honest once a latest predictions arrive. Emissions are
computed after the event loop rather than inside it, so the engine is
signal-agnostic and a forecast provider substitutes for an actual one without
any change to the simulator. Initially, this delay is proposed to be measured from `eligible_time`;
this convention will need to be compared with submission time and the baseline.

The reference baselines are FCFS, EASY backfilling, and a power-capped variant
of EASY. EASY grants the blocked head of the queue a reservation — the earliest
instant at which the jobs holding its resources are projected to release them —
and lets later jobs overtake it only when they provably cannot push that
reservation back. This is the property that prevents starvation, and it is
what the validation checks rather than any particular backfilling outcome.

Which runtime estimate the reservation is built on is an information choice, not
an implementation detail, so it is explicit. Classic EASY uses the walltime the
user requested; the alternative reads the same prediction seam the carbon-aware
policies will use, which is what makes "the same information for every
scheduler compared" verifiable rather than assumed. In PM100 the two differ
sharply: the median job runs for 2.5% of its requested walltime, so classic
reservations sit far in the future and suppress backfills that perfect
information would allow.

Because shifting a job changes neither its duration nor its power, total energy
is invariant to the schedule, and every baseline consumes identical energy on
the same workload. Within this model an energy-minimising baseline therefore has
nothing to optimise, and the meaningful power-side policy is instead a cap on the
aggregate power drawn at any instant. It moves jobs in time for a power reason
while remaining blind to the grid signal, which is the exact counterpart of a
carbon-aware policy and makes the distinction measurable rather than rhetorical.
The reservation is computed on both budgets under the cap, so a job blocked by
the power limit is protected from being overtaken by jobs that would consume the
power it is waiting for.

The carbon-aware policy implemented on top of these baselines is greedy and
deliberately simple. When a job becomes eligible it scores every candidate start
time within its delay budget — the release instant, then the carbon-intensity
grid up to `release + max_delay` — using the constant-average-power model of
§2.4, and holds the job until the cheapest one. A held job is withheld from the
placement pass rather than refused a slot, so it never occupies the head of the
queue: the jobs behind it move up, and it re-enters at its original position
when its target arrives. The budget therefore bounds the delay the carbon
decision is responsible for, while contention can still add its own; a zero
budget reproduces EASY exactly, which is what anchors the sweep.

Because the schedule cannot change the signal and the signal cannot change the
job, the choice is fixed at release. With actual future intensity this is the
perfect-information benchmark of §2.7, and it is a benchmark rather than a
bound: the policy optimises each job independently and ignores the contention
its own decisions create, so every job aims at the same clean interval and the
resulting queueing is a cost the greedy rule does not model. Establishing the
true optimum is the task of the later optimisation formulation.

Policies are scored by one function on the same workload: total and per-job
emissions, total energy, peak power, node utilisation, makespan, throughput,
and — as full distributions rather than means — waiting time, turnaround, and
bounded slowdown with the conventional ten-second floor.

By varying the maximum delay, the frontier between emissions and QoS can be estimated
without immediately introducing arbitrary weights between objectives. Tail statistics
and maxima of waiting time or bounded slowdown should also be reported: an improved
average may hide severely penalized jobs ([DIWS](preEditoriale.pdf), pp. 9–11). The full
MILP/CP formulation and core- or GPU-specific constraints belong to the subsequent
optimization phase.

### 2.7 Oracle, Predictions, and Evaluation

The following four configurations separate the sources of error. PM100 values
are historical ground truth reused at the new simulated start time.

| Configuration                 | Information Available at Decision Time                                | Purpose                                                  |
| ----------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| Perfect information benchmark | actual duration, actual PM100 profile, actual future carbon intensity | Measures the potential without prediction errors         |
| Forecast only error           | actual job data, carbon-intensity forecast                            | Isolates the effect of electricity-signal forecast error |
| Job model only error          | predicted duration and power, actual future carbon intensity          | Isolates the effect of job predictions                   |
| Realistic scenario            | predicted duration and power, forecast available at decision time     | Evaluates the system that can actually be used online    |

In the simulator, actual duration and completion determine when resources
are released; predicted duration and power affect decisions only. Final
emissions are then calculated using the PM100 profile and actual carbon intensity
at the simulated execution times. The perfect-information benchmark is not automatically
a theoretical optimum: it becomes one only if the offline problem is solved with
proven optimality. A greedy policy using perfect data remains a benchmark,
not an upper bound on the achievable savings.

The comparison with FCFS/EASY and with the realistic scenario addresses the
main question: **how much carbon can be saved, and what degradation in QoS is
required to achieve those savings?**

### 2.8 Assumptions and Limitations

- Shifting a job in time does not change its duration, resources, or power profile.
  This is a reasonable counterfactual assumption for the initial single-partition setting,
  but it should be reconsidered if DVFS, migration, or heterogeneous hardware are introduced.
- PM100 samples do not retain individual timestamps: the first sample is
  realigned to the simulated start time.
- The initial dataset includes completed jobs with no observed node-time overlaps
  and valid profiles. The potential bias introduced by these filters
  must be measured before the final experiments. One consequence is already
  visible: the simulated workload is a strict subset of what the machine really
  ran, so node utilisation reaches only about 20% and simulated FCFS waiting
  times come out _below_ the historical ones. The recorded waiting times were
  produced under contention with jobs that the filters removed, so trace replay
  is a fidelity check on start times and not a comparable performance baseline.
  Policies must be compared against each other on the same simulated workload.
- The entire input power to the nodes during execution is assigned to the job,
  rather than an incremental estimate relative to an idle baseline; the
  overall site consumption remains outside the model.
- The first version does not consider preemption, geographic migration, or embodied
  carbon.

These simplifications keep the model verifiable and sufficient to
build the first simulator. Extensions should be introduced only after
measuring the benefit of the benchmark and the carbon–QoS trade-off.

### 2.9 Essential References

- [Antici et al., _PM100: A Job Power Consumption Dataset of a Large-scale Production HPC System_](https://dl.acm.org/doi/10.1145/3624062.3624263)
- [Radovanović et al., _Carbon-Aware Computing for Datacenters_](https://arxiv.org/abs/2106.11750)
- [Wiesner et al., _Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud_](https://arxiv.org/abs/2110.13234)
- [Shim, _Quantifying job-level carbon efficiency in HPC_](https://doi.org/10.1186/s42162-025-00586-6)
