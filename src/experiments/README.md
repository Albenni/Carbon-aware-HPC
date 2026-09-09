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

| configuration | job_information | carbon_information | runtime_estimate | nodes | max_delay_hours | forecast_reach_hours | decision_granularity_minutes | emissions_saved_vs_easy | oracle_recovery | scheduler | jobs | total_emissions_tco2e | mean_emissions_gco2e | total_energy_mwh | peak_power_mw | average_model_gap | waiting_mean_s | waiting_median_s | waiting_p95_s | waiting_p99_s | waiting_max_s | turnaround_mean_s | turnaround_median_s | bounded_slowdown_mean | bounded_slowdown_median | bounded_slowdown_p95 | bounded_slowdown_max | total_nodes | peak_busy_nodes | utilisation | makespan_days | throughput_jobs_per_hour | delay_vs_trace_mean_s | delay_vs_trace_max_s | submit_waiting_mean_s | submit_waiting_median_s | submit_waiting_p95_s | submit_waiting_p99_s | submit_waiting_max_s | submit_turnaround_mean_s | submit_turnaround_median_s | submit_bounded_slowdown_mean | submit_bounded_slowdown_median | submit_bounded_slowdown_p95 | submit_bounded_slowdown_max | carbon_forecast_loss_share | job_model_loss_share | interaction_loss_share | combined_loss_share |
| ------------- | --------------- | ------------------ | ---------------- | ----- | --------------- | -------------------- | ---------------------------- | ----------------------- | --------------- | --------- | ---- | --------------------- | -------------------- | ---------------- | ------------- | ----------------- | -------------- | ---------------- | ------------- | ------------- | ------------- | ----------------- | ------------------- | --------------------- | ----------------------- | -------------------- | -------------------- | ----------- | --------------- | ----------- | ------------- | ------------------------ | --------------------- | -------------------- | --------------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ------------------------ | -------------------------- | ---------------------------- | ------------------------------ | --------------------------- | --------------------------- | -------------------------- | -------------------- | ---------------------- | ------------------- |
| easy_actual_jobs_none_carbon | actual | none | scheduling | 880 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | easy | 23560 | 17.471199695944442 | 741.5619565341444 | 67.52420508333333 | 0.8499497519169722 | -0.0002420468463871 | 422.90394736842103 | 0.0 | 0.0 | 15010.0 | 20021.0 | 3176.553310696095 | 2.0 | 2.1106431117946207 | 1.0 | 1.0 | 990.5 | 880 | 880 | 0.4751380821797033 | 7.614016203703704 | 128.9288911926865 | -2842.245713073005 | 19372.0 | 2423.7931239388795 | 0.0 | 3304.0 | 78038.0 | 509487.0 | 5177.442487266553 | 2.0 | 3.2885256861701557 | 1.0 | 3.287878787878788 | 2149.8883248730963 |  |  |  |  |
| easy_predicted_jobs_none_carbon | predicted | none | scheduling | 880 | 0.0 | 0.0 | 0.0 | -0.0002402457252344 | -0.0219043714286856 | easy | 23560 | 17.475397076986113 | 741.7401136241983 | 67.52420508333333 | 0.8443061244198385 | -0.0002538604030989 | 458.2618421052632 | 0.0 | 1341.0 | 14546.0 | 18720.0 | 3211.911205432937 | 2.0 | 3.0002097530506937 | 1.0 | 1.852334875893984 | 1445.6 | 880 | 880 | 0.4751380821797033 | 7.614016203703704 | 128.9288911926865 | -2806.887818336163 | 18526.0 | 2459.1510186757214 | 0.0 | 4340.0 | 78038.0 | 509487.0 | 5212.8003820033955 | 2.0 | 4.178096571908402 | 1.0 | 4.7862857142857145 | 2149.8883248730963 |  |  |  |  |
| carbon_actual_jobs_actual_carbon | actual | actual | scheduling | 880 | 6.0 | 23.0 | 15.0 | 0.0109679351455775 | 1.0 | carbon-aware-bounded | 23560 | 17.27957671076389 | 733.4285530884504 | 67.52420508333333 | 0.8459317802504518 | -0.0002832514568118 | 7740.895373514431 | 6059.5 | 21178.0 | 35344.0 | 39975.0 | 10494.544736842105 | 7404.0 | 459.7094449461425 | 49.7772054104055 | 1708.9 | 2158.6 | 880 | 880 | 0.4643366339648876 | 7.791134259259259 | 125.99791429598577 | 4475.745713073005 | 39428.0 | 9741.784550084889 | 6265.5 | 21515.0 | 85633.0 | 521165.0 | 12495.433913412564 | 7462.0 | 460.88938609437207 | 51.547995219306145 | 1710.4 | 2180.096446700508 |  |  |  |  |
| carbon_actual_jobs_forecast_carbon | actual | forecast | scheduling | 880 | 6.0 | 23.0 | 15.0 | 0.0045286628875436 | 0.412900224831263 | carbon-aware-forecast-bounded | 23560 | 17.392078522280556 | 738.2036724227739 | 67.52420508333333 | 0.8979896537032758 | -0.0001240665284174 | 7635.2475382003395 | 5161.5 | 20894.0 | 29184.0 | 32852.0 | 10388.896901528013 | 6948.0 | 508.3453769954825 | 56.53997212543554 | 1776.5 | 2156.6 | 880 | 880 | 0.4610660740284012 | 7.846400462962963 | 125.11044666919398 | 4370.0978777589135 | 32251.0 | 9636.1367147708 | 5592.5 | 21346.0 | 85220.0 | 521165.0 | 12389.786078098472 | 7085.0 | 509.52542850024855 | 58.5 | 1778.2 | 2234.9187817258885 |  |  |  |  |
| carbon_predicted_jobs_actual_carbon | predicted | actual | scheduling | 880 | 6.0 | 23.0 | 15.0 | 0.0051969066636733 | 0.4738272605275946 | carbon-aware-bounded | 23560 | 17.380403501822222 | 737.7081282607055 | 67.52420508333333 | 0.8563465235195132 | 0.0001674810650541 | 7848.782682512733 | 6204.5 | 21263.0 | 33231.0 | 37670.0 | 10602.432045840407 | 7515.0 | 462.8122430673278 | 48.7 | 1714.5 | 3342.3 | 880 | 880 | 0.4635275399693621 | 7.804733796296296 | 125.77836634639768 | 4583.633022071307 | 37078.0 | 9849.671859083192 | 6377.5 | 21560.0 | 84869.0 | 509487.0 | 12603.321222410868 | 7564.0 | 463.9921884600396 | 49.99499840915102 | 1716.3 | 3342.3 |  |  |  |  |
| carbon_predicted_jobs_forecast_carbon | predicted | forecast | scheduling | 880 | 6.0 | 23.0 | 15.0 | 0.0005736052352166 | 0.0522983795585247 | carbon-aware-forecast-bounded | 23560 | 17.461178124333333 | 741.136592713639 | 67.52420508333333 | 0.8977111424482367 | -0.0001205982886482 | 7743.46642614601 | 5555.5 | 20996.0 | 28541.0 | 32272.0 | 10497.115789473684 | 7048.5 | 510.09381677217607 | 58.3 | 1778.3 | 2794.1 | 880 | 880 | 0.4622306353987823 | 7.826631944444444 | 125.42645082007212 | 4478.316765704584 | 31658.0 | 9744.355602716469 | 5783.0 | 21456.0 | 86044.0 | 524765.0 | 12498.004966044142 | 7106.0 | 511.2739064772816 | 60.365289256198345 | 1778.3 | 2794.1 | 0.587099775168737 | 0.5261727394724053 | -0.1655708941996669 | 0.9477016204414752 |

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
