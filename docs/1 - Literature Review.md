## 1. Revisione della letteratura mirata

Effettuare una ricognizione iniziale della letteratura, senza necessariamente leggere in dettaglio tutti gli articoli, ma per costruire un quadro chiaro del contesto e preparare la futura sezione di related work della tesi.

### Temi da esplorare

#### HPC job scheduling

- **FCFS**
- **EASY Backfilling**
- **Priority-based scheduling**

#### Power-aware / Energy-aware scheduling

- Obiettivi tipici
- Metriche tipiche
- Trade-off tra energia e prestazioni

#### Carbon-aware computing

- Differenza tra **energy-aware** e **carbon-aware**
- Uso dei segnali di **carbon intensity** della rete elettrica
- Impatto del momento di esecuzione dei job sulle emissioni

#### Optimization-based scheduling in HPC

- **MILP**
- **Constraint Programming (CP)**

### Domande guida per la letteratura

- La **carbon footprint** dipende solo dall’energia consumata oppure anche dal **momento temporale** in cui il job viene eseguito?
- Serve una **predizione diretta della CO₂ per job**, oppure basta combinare:
  - potenza/energia prevista,
  - durata prevista,
  - carbon intensity del sistema elettrico nel tempo?
- La differenza tra **scheduling carbon-aware** e **energy-aware** è probabilmente cruciale e potrebbe diventare un cardine della tesi.

# Targeted Literature Review

## Topics to Explore

### HPC Job Scheduling

This area provides the classical policies to use as baselines for evaluating a carbon-aware scheduler. The three main families considered are FCFS, EASY Backfilling, and priority-based scheduling.

#### FCFS

- **[Analysis of First-Come-First-Serve Parallel Job Scheduling](https://dl.acm.org/doi/10.5555/314613.315031)** — U. Schwiegelshohn and R. Yahyapour, 1998.

  **How it supports the thesis:** formally analyzes FCFS in parallel job scheduling. It can be used to present FCFS as a simple, predictable, and fairness-oriented baseline, while also highlighting its resource-fragmentation and head-of-line-blocking problems.

- **[Utilization, Predictability, Workloads, and User Runtime Estimates in Scheduling the IBM SP2 with Backfilling](https://ieeexplore.ieee.org/document/932708/)** — A. W. Mu’alem and D. G. Feitelson, 2001.

  **How it supports the thesis:** compares FCFS and backfilling by studying utilization, predictability, and the accuracy of user runtime estimates. It is a central reference for selecting the metrics used to compare classical and carbon-aware schedulers.

- **[Parallel Job Scheduling — A Status Report](https://link.springer.com/chapter/10.1007/11407522_1)** — D. G. Feitelson, L. Rudolph, and U. Schwiegelshohn, 2004.

  **How it supports the thesis:** provides an overview of parallel job scheduling, connecting algorithms, real workloads, metrics, and production systems. It is useful as an introductory reference for the entire related-work section on HPC scheduling.

#### EASY Backfilling

- **[The ANL/IBM SP Scheduling System](https://link.springer.com/chapter/10.1007/3-540-60153-8_35)** — D. A. Lifka, 1995.

  **How it supports the thesis:** is the original reference for the EASY scheduler. It introduces the principle that later jobs may be moved forward as long as they do not delay the reservation of the first job in the queue. EASY is one of the most important baselines for the simulation.

- **[The EASY — LoadLeveler API Project](https://link.springer.com/chapter/10.1007/BFb0022286)** — J. Skovira, W. Chan, H. Zhou, and D. Lifka, 1996.

  **How it supports the thesis:** describes the integration of EASY with IBM LoadLeveler. It shows that backfilling was adopted in a real HPC system rather than remaining only a theoretical algorithm.

- **[Utilization, Predictability, Workloads, and User Runtime Estimates in Scheduling the IBM SP2 with Backfilling](https://ieeexplore.ieee.org/document/932708/)** — A. W. Mu’alem and D. G. Feitelson, 2001.

  **How it supports the thesis:** enables a comparison among EASY Backfilling, conservative backfilling, and FCFS. It is useful for discussing the trade-off between resource utilization and waiting-time predictability.

- **[Characterization of Backfilling Strategies for Parallel Job Scheduling](https://www.mcs.anl.gov/~kettimut/publications/iwpp02.pdf)** — S. Srinivasan, R. Kettimuthu, V. Subramani, and P. Sadayappan, 2002.

  **How it supports the thesis:** compares aggressive and conservative backfilling strategies together with different queue-ordering criteria. It helps separate the effect of backfilling from that of the priority function.

#### Priority-Based Scheduling

- **[Supporting Priorities and Improving Utilization of the IBM SP Scheduler Using Slack-Based Backfilling](https://www.cs.huji.ac.il/~feit/papers/SlackBackfil99IPPS.pdf)** — D. Talby and D. G. Feitelson, 1999.

  **How it supports the thesis:** introduces priorities and temporal slack into backfilling. It is particularly relevant because a carbon-flexible job may be delayed within a maximum allowable margin, similarly to slack-based scheduling.

- **[Characterization of Backfilling Strategies for Parallel Job Scheduling](https://www.mcs.anl.gov/~kettimut/publications/iwpp02.pdf)** — S. Srinivasan et al., 2002.

  **How it supports the thesis:** compares FCFS with criteria such as Shortest Job First and Expansion Factor. It demonstrates that performance depends on the combination of the priority policy and the backfilling mechanism.

- **[Parallel Job Scheduling — A Status Report](https://link.springer.com/chapter/10.1007/11407522_1)** — D. G. Feitelson, L. Rudolph, and U. Schwiegelshohn, 2004.

  **How it supports the thesis:** provides the general framework needed to discuss priorities, fairness, reservations, slowdown, and the objectives of parallel scheduling. It can be used to motivate the introduction of a new priority factor based on carbon intensity.

---

### Power-Aware / Energy-Aware Scheduling

This literature introduces power and energy as variables in HPC scheduling. It provides the technical foundation for carbon-aware scheduling because estimating emissions first requires measuring or predicting job energy consumption.

#### Typical Objectives

The most common objectives are:

- minimizing total energy consumption;
- minimizing energy-to-solution;
- complying with a global power cap;
- reducing power peaks;
- maximizing throughput or utilization under a power limit;
- selecting more efficient frequencies, nodes, or partitions;
- reducing energy without exceeding slowdown or waiting-time limits.

#### Typical Metrics

The most common metrics are:

- total energy, in joules or kWh;
- average and peak power;
- energy-to-solution;
- performance per watt;
- Energy–Delay Product, EDP;
- Energy–Delay² Product, ED²P;
- power-cap violations;
- power-budget utilization;
- waiting time;
- turnaround time;
- slowdown and bounded slowdown;
- throughput and utilization.

#### Main Papers

- **[Energy-Aware Scheduling for High-Performance Computing Systems: A Survey](https://www.mdpi.com/1996-1073/16/2/890)** — B. Kocot, P. Czarnul, and J. Proficz, 2023.

  **How it supports the thesis:** is the main survey for classifying objectives, metrics, DVFS, power capping, heterogeneous systems, and optimization methods. It can be used to organize the entire energy-aware section of the related work.

- **[Optimizing Job Performance Under a Given Power Constraint in HPC Centers](https://ieeexplore.ieee.org/document/5598303/)** — M. Etinski, J. Corbalán, J. Labarta, and M. Valero, 2010.

  **How it supports the thesis:** studies the use of DVFS to optimize overall performance under a power budget. It shows that limiting the power of one job can allow more jobs to run concurrently, thereby reducing queueing time.

- **[Linear Programming Based Parallel Job Scheduling for Power Constrained Systems](https://ieeexplore.ieee.org/document/5999809/)** — M. Etinski, J. Corbalán, J. Labarta, and M. Valero, 2011.

  **How it supports the thesis:** formulates power-constrained HPC scheduling as an optimization problem. It is a direct precedent for a carbon-aware MILP formulation.

- **[Parallel Job Scheduling for Power Constrained HPC Systems](https://doi.org/10.1016/j.parco.2012.08.001)** — M. Etinski, J. Corbalán, J. Labarta, and M. Valero, 2012.

  **How it supports the thesis:** further investigates the joint management of parallel jobs, performance, and power budgets. It can be adapted by replacing or supplementing the energy cost with a time-dependent carbon cost.

- **[Practical Resource Management in Power-Constrained, High Performance Computing](https://dl.acm.org/doi/10.1145/2749246.2749262)** — T. Patki et al., 2015.

  **How it supports the thesis:** introduces RMAP, including power-aware backfilling, fair sharing of the power budget, and hardware overprovisioning. It is particularly important for linking EASY Backfilling to a future carbon-aware policy.

- **[Power Capping in High Performance Computing Systems](https://link.springer.com/chapter/10.1007/978-3-319-23219-5_37)** — A. Borghesi, F. Collina, M. Lombardi, M. Milano, and L. Benini, 2015.

  **How it supports the thesis:** presents CP and heuristic methods for dispatching under a power cap and evaluates them on CINECA’s Eurora supercomputer. It provides a precedent that is very close to the thesis’s application context.

- **[Scheduling-Based Power Capping in High Performance Computing Systems](https://www.sciencedirect.com/science/article/pii/S2210537917302317)** — A. Borghesi, A. Bartolini, M. Lombardi, M. Milano, and L. Benini, 2018.

  **How it supports the thesis:** combines a predictive power model with Constraint Programming while enforcing a global limit and preserving QoS. It provides a useful architectural pattern: predict job characteristics and then optimize the schedule.

#### Trade-Off Between Energy and Performance

- **[A Case Study of Energy Aware Scheduling on SuperMUC](https://link.springer.com/chapter/10.1007/978-3-319-07518-1_25)** — A. Auweter et al., 2014.

  **How it supports the thesis:** analyzes runtime and power at different frequencies on a real supercomputer. It shows how to design a policy that jointly considers energy savings and increased execution time.

- **[Practical Resource Management in Power-Constrained, High Performance Computing](https://dl.acm.org/doi/10.1145/2749246.2749262)** — T. Patki et al., 2015.

  **How it supports the thesis:** emphasizes that the trade-off must be evaluated at the system level rather than only at the individual-job level. Slightly slowing one job may enable greater concurrency and improve overall turnaround time.

- **[Energy-Aware Scheduling for High-Performance Computing Systems: A Survey](https://www.mdpi.com/1996-1073/16/2/890)** — B. Kocot, P. Czarnul, and J. Proficz, 2023.

  **How it supports the thesis:** surveys combined metrics such as EDP and multi-objective methods. It is useful for deciding how to jointly measure energy, runtime, slowdown, and utilization.

---

### Carbon-Aware Computing

Carbon-aware computing considers not only how much energy is consumed, but also the carbon intensity of the electricity at the time and location where that consumption occurs.

A simplified formulation of operational emissions is:

$$
CO_{2,j}
= \sum_t E_{j,t} \cdot CI_t
= \sum_t P_{j,t} \Delta t \cdot CI_t
$$

where:

- $P_{j,t}$ is the power consumption of job $j$;
- $\Delta t$ is the slot duration;
- $E_{j,t}$ is the energy consumed;
- $CI_t$ is the carbon intensity of the electricity grid.

#### Difference Between Energy-Aware and Carbon-Aware Scheduling

Energy-aware scheduling seeks to reduce:

$$
E_j = \sum_t P_{j,t} \Delta t
$$

Carbon-aware scheduling instead seeks to reduce:

$$
CO_{2,j} = \sum_t P_{j,t} \Delta t \cdot CI_t
$$

Two schedules may therefore consume the same amount of energy while producing different emissions if the energy is consumed at times with different carbon intensities.

#### Main Papers

- **[Energy and Carbon Aware Scheduling in Supercomputing](https://cris.vtt.fi/en/publications/energy-and-carbon-aware-scheduling-in-supercomputing/)** — M. Majanen, O. Mämmelä, and A. Giesler, 2012.

  **How it supports the thesis:** is one of the earliest works focused directly on supercomputing that distinguishes energy optimization from emissions optimization. It presents algorithms for HPC environments with multiple data centers.

- **[Carbon-Aware Computing for Datacenters](https://arxiv.org/abs/2106.11750)** — A. Radovanović et al., 2021; later published in _IEEE Transactions on Power Systems_.

  **How it supports the thesis:** is a fundamental modern reference on carbon-aware computing. It uses forecasts of carbon intensity and computational demand to shift temporally flexible workloads toward cleaner hours.

- **[Let’s Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud](https://arxiv.org/abs/2110.13234)** — P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, and L. Thamsen, 2021.

  **How it supports the thesis:** explicitly studies temporal shifting, flexibility windows, and the effect of forecast errors. It provides a basis for experiments in which PM100 jobs may be delayed within windows of 1, 3, 6, or 12 hours.

- **[Adaptive Carbon-Aware Scheduling Policies for HPC Systems](https://link.springer.com/chapter/10.1007/978-3-032-10507-3_4)** — A. Benhari and D. Trystram, JSSPP 2025, published in the proceedings in 2026.

  **How it supports the thesis:** is one of the references most directly aligned with carbon-aware HPC scheduling. It is useful for positioning the contribution against the most recent policies designed specifically for batch jobs and HPC systems.

#### Use of Carbon-Intensity Signals

- **[Carbon-Aware Computing for Datacenters](https://arxiv.org/abs/2106.11750)** — A. Radovanović et al.

  **How it supports the thesis:** justifies the use of day-ahead carbon-intensity forecasts for online decision-making. It suggests a modular architecture consisting of demand forecasting, an external grid signal, and schedule optimization.

- **[Let’s Wait Awhile](https://arxiv.org/abs/2110.13234)** — P. Wiesner et al.

  **How it supports the thesis:** shows that the benefit depends on the geographical region, grid variability, the available delay window, and forecast accuracy. It is useful for designing sensitivity analyses.

- **[Carbon-Aware Computing for Data Centers with Probabilistic Performance Guarantees](https://arxiv.org/abs/2410.21510)** — S. Hall et al., 2024.

  **How it supports the thesis:** introduces probabilistic performance guarantees when grid signals and demand are uncertain. It may serve as a future extension beyond an initial deterministic model.

#### Impact of Job Execution Time

- **[Carbon-Aware Computing for Datacenters](https://arxiv.org/abs/2106.11750)** — A. Radovanović et al.

  **How it supports the thesis:** demonstrates that execution time is a fundamental decision variable. Workloads with the same energy consumption may produce different emissions when executed at different hours.

- **[Let’s Wait Awhile](https://arxiv.org/abs/2110.13234)** — P. Wiesner et al.

  **How it supports the thesis:** quantifies the potential of temporal workload shifting and shows that there is not necessarily a universally optimal time window: the outcome depends on the local carbon-intensity time series.

- **[Energy and Carbon Aware Scheduling in Supercomputing](https://cris.vtt.fi/en/publications/energy-and-carbon-aware-scheduling-in-supercomputing/)** — M. Majanen et al.

  **How it supports the thesis:** connects emissions reduction to HPC metrics such as waiting time and turnaround time, supporting a multi-objective evaluation.

---

### Optimization-Based Scheduling in HPC

MILP and Constraint Programming make it possible to represent explicitly:

- job arrival and start times;
- job duration;
- node or partition capacity;
- power caps;
- deadlines;
- priorities;
- waiting time;
- slowdown;
- time-varying carbon intensity;
- total emissions.

These methods can be used either as schedulers or as offline oracles on reduced instances to evaluate how closely a heuristic policy approaches the optimal solution.

#### MILP

- **[Increasing Waiting Time Satisfaction in Parallel Job Scheduling via a Flexible MILP Approach](https://ieeexplore.ieee.org/document/7568331/)** — S. Schlagkamp, M. Hofmann, L. Eufinger, and R. Ferreira da Silva, 2016.

  **How it supports the thesis:** formulates parallel job scheduling through Mixed-Integer Linear Programming and uses a planning horizon to limit complexity. It is a direct reference for constructing a carbon-aware MILP on subsets of PM100.

- **[Linear Programming Based Parallel Job Scheduling for Power Constrained Systems](https://ieeexplore.ieee.org/document/5999809/)** — M. Etinski et al., 2011.

  **How it supports the thesis:** shows how to integrate resources, runtime, and power constraints into a mathematical programming model. The energy cost can be transformed into a carbon cost by multiplying the energy in each slot by $CI_t$.

- **[Parallel Job Scheduling for Power Constrained HPC Systems](https://doi.org/10.1016/j.parco.2012.08.001)** — M. Etinski et al., 2012.

  **How it supports the thesis:** provides a basis for jointly modeling the power budget, performance, and parallel scheduling. It can be extended with a multi-objective function that includes emissions, waiting time, and slowdown.

A possible objective function for the thesis is:

$$
\min \left(
\alpha \sum_j CO_{2,j}
+ \beta \sum_j \operatorname{waiting}_j
+ \gamma \sum_j \operatorname{boundedSlowdown}_j
\right)
$$

with:

$$
CO_{2,j}
= \sum_t P_{j,t} \Delta t \cdot CI_t
$$

#### Constraint Programming

- **[Power Capping in High Performance Computing Systems](https://link.springer.com/chapter/10.1007/978-3-319-23219-5_37)** — A. Borghesi et al., 2015.

  **How it supports the thesis:** uses CP for dispatching under a power cap on Eurora. It shows how to model cumulative resource and power constraints in a real HPC context.

- **[Constraint Programming-Based Job Dispatching for Modern HPC Applications](https://doi.org/10.1007/978-3-030-30048-7_26)** — C. Galleguillos, Z. Kiziltan, A. Sîrbu, and Ö. Babaoglu, 2019.

  **How it supports the thesis:** proposes online CP dispatchers that use job-duration predictions. It is useful for integrating runtime estimates derived from PM100 features into the model.

- **[A Job Dispatcher for Large and Heterogeneous HPC Systems Running Modern Applications](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2021.26)** — C. Galleguillos, Z. Kiziltan, and R. Soto, 2021.

  **How it supports the thesis:** extends CP to large, heterogeneous systems using a model whose size does not depend directly on the number of nodes. It is useful for a future extension toward hardware partitions with different efficiencies.

- **[Job Scheduling for HPC Clusters: Constraint Programming vs. Backfilling Approaches](https://dl.acm.org/doi/10.1145/3629104.3666038)** — A. V. Goponenko, K. Lamar, B. A. Allan, J. M. Brandt, and D. Dechev, 2024.

  **How it supports the thesis:** directly compares Constraint Programming and backfilling on HPC workloads. It is one of the most useful references for designing an experimental comparison among EASY, CP, and a carbon-aware policy.

- **[Scheduling-Based Power Capping in High Performance Computing Systems](https://www.sciencedirect.com/science/article/pii/S2210537917302317)** — A. Borghesi et al., 2018.

  **How it supports the thesis:** combines ML-based power prediction with CP-based scheduling. It provides a precedent closely aligned with the proposed architecture: predict power and duration, then use those estimates within the optimization model.

---

## References Specific to PM100 and Job-Level Modeling

### PM100: Reference Dataset

- **[PM100: A Job Power Consumption Dataset of a Large-Scale Production HPC System](https://dl.acm.org/doi/10.1145/3624062.3624263)** — F. Antici, M. Seyedkazemi Ardebili, A. Bartolini, and Z. Kiziltan, SC-W 2023.

  **How it supports the thesis:** is the official paper introducing the dataset used in the study. PM100 contains approximately 230,000 jobs from the Marconi100 system and associates scheduling metadata with node-, CPU-, and memory-level power measurements. The paper should be cited in the dataset and methodology sections.

  The dataset enables two evaluation modes:
  1. **Oracle evaluation**, using measured runtimes and power profiles.
  2. **Predictive evaluation**, training models to estimate duration, average power, or energy using only features available before execution.

  Its central contribution to the thesis is that it enables an emissions formulation based on real measurements:

  $$
  CO_{2,j}(s)
  = \sum_m
  \frac{P_j[m]}{1000}
  \Delta t_h
  \cdot \operatorname{CI}(s + m \Delta t)
  $$

  where $P_j[m]$ is the power profile of the job and $s$ is the start time selected by the scheduler.

### Job-Level Analysis of Carbon Efficiency

- **[Quantifying Job-Level Carbon Efficiency in HPC: An Empirical Study Based on the PM100 Dataset](https://link.springer.com/article/10.1186/s42162-025-00586-6)** — H. Shim, 2025.

  **How it supports the thesis:** directly uses PM100 to estimate job-level emissions, analyze the relationship between resource configuration and emissions, and introduce a **Carbon Efficiency Score, CES**. The paper is therefore the closest reference to the dataset modeling and analysis stage.

  In particular, the study:
  - integrates power time series to estimate the energy consumed by each job;
  - converts energy into emissions using an average factor for the Italian electricity grid;
  - trains an MLP model to predict emissions;
  - proposes CES to classify jobs into efficiency tiers;
  - identifies relationships among duration, memory, GPUs, resource configuration, and emissions.

  The paper is also useful as a **critical comparison point**. It uses a constant national carbon intensity of $0.233\ \text{kgCO}_2/\text{kWh}$ and therefore does not account for temporal variation in the grid. The authors themselves acknowledge this limitation and identify the integration of dynamic carbon intensity as a direction for future work.

  The thesis can therefore extend this work as follows:

  $$
  CO_{2,j}^{\mathrm{Shim}}
  = E_j \cdot CI_{\mathrm{average}}
  $$

  versus:

  $$
  CO_{2,j}^{\mathrm{thesis}}(s)
  = \sum_t E_{j,t} \cdot CI_t
  $$

  The difference is substantial:
  - Shim’s model primarily measures the effect of energy consumption and hardware configuration;
  - the model proposed in the thesis adds execution time as a decision variable;
  - with time-varying $CI_t$, the same job may produce different emissions without changing its total energy consumption.

  A second point to discuss concerns direct CO₂ prediction. In Shim’s work, the carbon target is derived by multiplying energy by a constant factor, and the model also uses power and execution information. For an online carbon-aware scheduler, it is more modular and interpretable to predict separately:

  $$
  \hat d_j, \qquad \hat P_j
  $$

  or:

  $$
  \hat E_j
  $$

  and then combine these predictions with the time-varying signal:

  $$
  \widehat{CO}_{2,j}(s)
  = \sum_t
  \hat P_{j,t} \Delta t \cdot \widehat{CI}_t
  $$

---

## Possible Positioning

The thesis can be positioned as follows:

> Carbon-aware scheduling extends energy-aware HPC scheduling by introducing the time-varying carbon intensity of the electricity grid into the objective function. Unlike approaches that estimate emissions using a constant average factor, the proposed model associates each job’s energy profile with a dynamic carbon-intensity time series. Execution time therefore becomes a decision variable. The PM100 dataset makes it possible to evaluate this model using real power profiles and to compare an oracle scheduler with a predictive version based on estimated duration, power, or energy.
