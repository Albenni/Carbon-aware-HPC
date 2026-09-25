# Pushing the carbon-QoS frontier past the delay-aware benefit rate

**Outcome: RESTORE.** No frozen finalist met either success criterion on the test cohort, so the round is reverted and the incumbent benefit rate stands.

## What was run

Three mechanisms, each answering one weakness of the incumbent, swept one at a time on the validation cohort and then combined only where a mechanism put a point on the overall Pareto frontier that kept at least 95% of the incumbent's saving:

- **uncertainty** — a deferral is charged for the forecast error its own residuals have shown, either `B = Δ − κσ` or `B = Δ + min(0, q_α)`;
- **scaling** — the threshold and the price are multiplied per job by `(1 h / duration)^a`, `(1 kWh / energy)^a` or the larger of the two, clipped to [1, 100];
- **hybrid** — the deferral is scored by `Δ / ((D + 15 min) / 1 h)^β − λ_j D` rather than by the ratio `Δ / D`.

The simulator, the accounting, the 880 nodes, the 15-minute decision grid, the receding horizon, the frozen job predictions and the fixed 6 h nominal delay budget are the previous round's, unchanged. No model was refitted.

### Cohorts and calibration

- validation: **17,656 jobs**, released far enough before the carbon cutoff to be scored on observed intensity;
- test: **23,560 jobs**;
- uncertainty calibration: **3,008 out-of-sample trajectories × 96 leads**, the carbon model's own 2020-01-01 to 2020-05-06 validation residuals, rebuilt from the archived forecast matrix and checked lead by lead against the metrics stored with it (max gap 6e-14). The residual spread runs from 9.9 to 24.0 gCO2e/kWh across the 24 h horizon. It was frozen before the validation sweep and reused unchanged on test.

## Baselines reproduced on validation

| policy                      | saving vs EASY | tCO2e   | wait mean s | wait p95 s | bsld mean | bsld p95 | held  | voluntary h | g per voluntary h |
| --------------------------- | -------------- | ------- | ----------- | ---------- | --------- | -------- | ----- | ----------- | ----------------- |
| congestion (price 1)        | 1.421%         | 14.9513 | 8,234       | 20,923     | 417.36    | 1962.50  | 72.2% | 38,660      | 5.6               |
| benefit-rate θ (historical) | 1.323%         | 14.9661 | 6,343       | 20,322     | 268.01    | 1688.30  | 65.5% | 29,138      | 6.9               |
| benefit-rate θ=0.0149       | 1.323%         | 14.9661 | 6,359       | 20,333     | 269.54    | 1690.40  | 65.5% | 29,213      | 6.9               |

The reproduction control at the full historical threshold (`receding|benefit-rate|theta=0.0149307|fixed=6h`) is reported separately from the θ = 0.0149 reference the round is judged against; the two are different numbers and are never conflated.

## Selection, frozen before the test

- promising mechanisms: **delay-cost**
- finalists (3): `receding|carbon-delay|lambda=0.255568|combined^1|fixed=6h`, `receding|hybrid|beta=0.5|lambda=0.122587|combined^1|fixed=6h`, `receding|carbon-delay|lambda=0.0149|combined^1|fixed=6h`
- principal configuration: `receding|carbon-delay|lambda=0.255568|combined^1|fixed=6h`
- ablations (2): `receding|carbon-delay|lambda=0.255568|fixed=6h`, `receding|benefit-rate|theta=0.0149|combined^1|fixed=6h`

Representatives were taken at 95.0%, 97.5%, 99.0% of the best new saving, choosing among the admissible configurations the one minimising the worst of the four QoS ratios to the benefit rate, with ties broken by the mean ratio, then emissions, then identifier. The rule applied afterwards was frozen with them:

> primary: a finalist Pareto-dominates the benefit rate in the realistic test cell on total_emissions_gco2e, waiting_mean_s, waiting_p95_s, bounded_slowdown_mean, bounded_slowdown_p95 (rtol 1e-08, atol 1e-09). practical: it keeps at least 95% of its saving, cuts both mean waiting and mean bounded slowdown by at least 20%, and worsens neither p95. Neither met: restore.

## Test results

| policy                          | saving vs EASY | tCO2e   | wait mean s | wait p95 s | bsld mean | bsld p95 | held  | voluntary h | g per voluntary h | finalist | ablation |
| ------------------------------- | -------------- | ------- | ----------- | ---------- | --------- | -------- | ----- | ----------- | ----------------- | -------- | -------- |
| benefit-rate θ=0.0149           | 0.548%         | 17.3797 | 2,194       | 18,121     | 15.37     | 38.96    | 24.7% | 11,743      | 8.1               | False    | False    |
| benefit-rate θ (historical)     | 0.548%         | 17.3797 | 2,194       | 18,121     | 15.37     | 38.96    | 24.7% | 11,743      | 8.1               | False    | False    |
| θ=0.0149 combined^1             | 0.545%         | 17.3802 | 1,923       | 17,629     | 5.08      | 19.25    | 20.0% | 10,002      | 9.5               | False    | True     |
| congestion (price 1)            | 0.432%         | 17.3999 | 6,590       | 20,783     | 432.67    | 1816.50  | 69.6% | 41,436      | 1.8               | False    | False    |
| Δ−λD λ=0.256 combined^1         | 0.410%         | 17.4037 | 1,774       | 16,099     | 4.48      | 13.56    | 14.6% | 8,553       | 8.4               | True     | False    |
| Δ−λD λ=0.256                    | 0.392%         | 17.4069 | 2,034       | 18,428     | 5.87      | 20.59    | 17.4% | 10,852      | 6.3               | False    | True     |
| Δ−λD λ=0.0149 combined^1        | 0.390%         | 17.4072 | 2,199       | 19,348     | 5.22      | 20.00    | 20.0% | 11,911      | 5.7               | True     | False    |
| hybrid β=0.5 λ=0.123 combined^1 | 0.385%         | 17.4081 | 1,780       | 15,718     | 4.59      | 15.04    | 14.9% | 8,589       | 7.8               | True     | False    |

### Against both carbon-aware baselines

- benefit rate (the reference): 0.548% saved, 2,194 s mean wait, 18,121 s p95, bsld 15.4 mean / 39.0 p95
- congestion, price 1 (the earlier incumbent): 0.432% saved, 6,590 s mean wait, 20,783 s p95, bsld 432.7 mean / 1816.5 p95
- benefit rate at the historical θ: 0.548% saved, 2,194 s mean wait, 18,121 s p95, bsld 15.4 mean / 39.0 p95

### The best finalist, measured against the reference

- saving retained: **74.9%** (criterion: ≥ 95%)
- mean waiting: **+19.2%** change (criterion: ≥ 20% reduction)
- mean bounded slowdown: **+70.8%** change (criterion: ≥ 20% reduction)
- p95 waiting worsened: **False**; p95 bounded slowdown worsened: **False**
- Pareto-dominates the reference on all five metrics: **False**

### Ablations

| policy              | saving vs EASY | tCO2e   | wait mean s | wait p95 s | bsld mean | bsld p95 | held  | voluntary h | g per voluntary h |
| ------------------- | -------------- | ------- | ----------- | ---------- | --------- | -------- | ----- | ----------- | ----------------- |
| θ=0.0149 combined^1 | 0.545%         | 17.3802 | 1,923       | 17,629     | 5.08      | 19.25    | 20.0% | 10,002      | 9.5               |
| Δ−λD λ=0.256        | 0.392%         | 17.4069 | 2,034       | 18,428     | 5.87      | 20.59    | 17.4% | 10,852      | 6.3               |

### What the voluntary delay bought

| policy                          | held     | saving < 0 | share | saving = 0 | share | < 1 g incl. | share | 0 < s < 1 g | share | 0 < s < 0.1% EASY | share |
| ------------------------------- | -------- | ---------- | ----- | ---------- | ----- | ----------- | ----- | ----------- | ----- | ----------------- | ----- |
| benefit-rate θ=0.0149           | 5,819.0  | 1,310.0    | 22.5% | 380.0      | 6.5%  | 2,962.0     | 50.9% | 1,272.0     | 21.9% | 138.0             | 2.4%  |
| benefit-rate θ (historical)     | 5,819.0  | 1,310.0    | 22.5% | 380.0      | 6.5%  | 2,962.0     | 50.9% | 1,272.0     | 21.9% | 138.0             | 2.4%  |
| θ=0.0149 combined^1             | 4,710.0  | 1,095.0    | 23.2% | 180.0      | 3.8%  | 2,024.0     | 43.0% | 749.0       | 15.9% | 129.0             | 2.7%  |
| congestion (price 1)            | 16,398.0 | 4,009.0    | 24.4% | 2,455.0    | 15.0% | 13,397.0    | 81.7% | 6,933.0     | 42.3% | 46.0              | 0.3%  |
| Δ−λD λ=0.256 combined^1         | 3,429.0  | 853.0      | 24.9% | 47.0       | 1.4%  | 989.0       | 28.8% | 89.0        | 2.6%  | 37.0              | 1.1%  |
| Δ−λD λ=0.256                    | 4,104.0  | 1,096.0    | 26.7% | 76.0       | 1.9%  | 1,709.0     | 41.6% | 537.0       | 13.1% | 44.0              | 1.1%  |
| Δ−λD λ=0.0149 combined^1        | 4,707.0  | 1,166.0    | 24.8% | 90.0       | 1.9%  | 1,912.0     | 40.6% | 656.0       | 13.9% | 41.0              | 0.9%  |
| hybrid β=0.5 λ=0.123 combined^1 | 3,522.0  | 891.0      | 25.3% | 55.0       | 1.6%  | 1,050.0     | 29.8% | 104.0       | 3.0%  | 49.0              | 1.4%  |

Among jobs actually held back by a policy decision, the columns separate a negative saving, an exactly zero one, everything under a gram inclusive of those two, the strictly positive savings under a gram, and the strictly positive ones under 0.1% of the job's EASY emissions. A share is left missing, with its reason recorded in the CSV, wherever no job was held.

## Computational cost

```
2026-09-21T15:29:26.131656+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py calibrate	1.2 s
2026-09-21T15:29:52.831218+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py smoke --workers 4	15.2 s
2026-09-21T16:15:23.610163+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py validation --workers 4	2678.4 s
2026-09-21T16:15:42.996280+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py validation --workers 4	2632.6 s
2026-09-21T16:47:09.162270+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py validation --workers 4	1756.3 s
2026-09-21T16:49:16.049125+00:00	/Users/alberto/Desktop/Repositories/Carbon-aware HPC/src/experiments/scheduler_frontier.py test --workers 4	117.5 s
```

## Figures

- `validation_frontier` — every validation configuration and the Pareto frontier;
- `mechanisms` — each mechanism alone against the reference;
- `test_frontier`, `test_ratios` — the frozen selection on test;
- `calibration` — the residual spread by lead;
- `duration_bands` — where the saving comes from.

All under `data/experiments/scheduler_frontier/figures/`.

## Limits

**Numerical dominance is not generalisation.** The decision rule is a Pareto test on one test cohort; a configuration can win it and still be a fit to that cohort's particular contention and weather.

- **The test cohort has been used before.** September–October 2020 is the carbon model's test period and this scheduler cohort was analysed in earlier rounds. The comparison is therefore **retrospective**: parameters were chosen on validation and the test was run once after the freeze, but this is not a holdout that has never been consulted.
- **The residuals are transferred across seasons.** The uncertainty is calibrated on 2020-01 to 2020-05 and applied to September–October. If the forecast's error structure differs between those periods — and the Italian grid's mix certainly does — the charge is misspecified in a direction this round cannot measure.
- **Trajectories are not independent.** The residual rows are hourly origins with overlapping 24 h windows, so the spread and the quantiles are estimated from correlated samples and their effective sample size is well below 3,008.
- **Only carbon uncertainty is modelled.** The duration and power predictions are treated as exact, as agreed. They are not, and their error feeds directly into both the predicted cost and the window the job is scored over.
- **Per-job savings against EASY include contention.** A job's `saved_vs_easy` is measured against EASY's own schedule, in which the machine's occupancy differs; part of any per-job difference is contention the policy did not choose. The isolated counterfactual is reported beside it, never instead of it.
- **The forecast reach still binds the budget.** A job may only move within `reach − duration`, so the longest jobs never receive the whole nominal 6 h. The effective budget is recorded per decision.

---

Generated by `python -m experiments.scheduler_frontier report` from the artifacts in `data/experiments/scheduler_frontier/`. Full tables: `report_tables.md` there.
