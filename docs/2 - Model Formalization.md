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

At present, carbon accounting, the two power models, and an actual-only
carbon-intensity provider are implemented. Forecast retrieval, the simulator,
scheduler, and ML models described in the following sections have not yet been
implemented.

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

To construct the perfect-information benchmark and evaluate the error of the
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

| Issue                | Initial Choice                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-node jobs      | In general, the power of the nodes actually allocated to the job is summed. In [PM100](https://dl.acm.org/doi/10.1145/3624062.3624263), this sum, including both power sockets, is already contained in `node_power_consumption`: the profile must therefore be used directly, without multiplying it by `num_nodes_alloc` and without adding CPU or memory profiles. Partitions are not summed. |
| Hardware differences | The first analysis uses only anonymized partition `1` and assumes a single execution class. The measured profiles incorporate the differences actually observed, but no node- or partition-specific coefficient is introduced yet. Explicit heterogeneity remains an extension.                                                                                                                  |
| ML target            | CO₂ is not predicted directly. Instead, $\hat d_j$ and $\hat{\bar P}_j$ are predicted, or $\hat E_j$ together with duration, using only features known at submission time. These quantities are then combined with the external carbon-intensity signal.                                                                                                                                         |
| Carbon intensity     | Actual historical and synthetic series are supported. Actual values and forecasts have separate access paths, and forecast retrieval remains part of Phase 7. Once available, historical actuals will be used for accounting, ex-post evaluation, and the perfect-information benchmark; synthetic values are used for controlled tests.                                                         |
| Granularity          | Initial choice: piecewise-constant carbon intensity at 15-minute resolution, explicitly requested from Electricity Maps. PM100 profiles remain at 20-second resolution; 5- and 60-minute resolutions will be used for sensitivity analysis.                                                                                                                                                      |
| Temporal model       | A discrete-event simulator with continuous timestamps is planned. Submission/eligibility, completion, and changes in carbon intensity are events; a fixed tick is not required.                                                                                                                                                                                                                  |
| Emissions boundary   | Only operational carbon assigned to the input power profile of the nodes is considered. Embodied carbon and overhead external to the profile are excluded.                                                                                                                                                                                                                                       |

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
partition are treated as equivalent. A start time is feasible only if
the job is eligible, enough nodes are available, and any maximum delay constraint is
satisfied. Initially, this delay is proposed to be measured from `eligible_time`;
this convention will need to be compared with submission time and the baseline.

By varying the maximum delay, the frontier between emissions and QoS can be estimated
without immediately introducing arbitrary weights between objectives. Tail statistics
and maxima of waiting time or bounded slowdown should also be reported: an improved
average may hide severely penalized jobs ([DIWS](preEditoriale.pdf), pp. 9–11). The full
MILP/CP formulation and core- or GPU-specific constraints belong to the subsequent
optimization phase.

### 2.7 Oracle, Predictions, and Evaluation

The following four configurations separate the sources of error. PM100 values
are historical ground truth reused at the new simulated start time; they are not
new measurements of the counterfactually shifted job.

| Configuration                 | Information Available at Decision Time                                | Purpose                                                  |
| ----------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| Perfect-information benchmark | actual duration, actual PM100 profile, actual future carbon intensity | Measures the potential without prediction errors         |
| Forecast-only error           | actual job data, carbon-intensity forecast                            | Isolates the effect of electricity-signal forecast error |
| Job-model-only error          | predicted duration and power, actual future carbon intensity          | Isolates the effect of job predictions                   |
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
  must be measured before the final experiments.
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
