# Carbon intensity providers

This package provides a stable carbon-intensity API for accounting and simulation, independent of the data source.

The initial actual series uses `gCO2eq/kWh` averages on a 15-minute UTC grid. A sample at `10:00` applies to `[10:00, 10:15)`. All ranges are start-inclusive and end-exclusive.

`TimeSeriesCarbonIntensityProvider` is strict: missing buckets, out-of-range lookups, and timezone-naive timestamps raise errors. It never interpolates or extrapolates. Actual and forecast data remain separate: `get_forecast(...)` raises `ForecastUnavailableError` for actual-only series.

## Actual carbon intensity

```python
from datetime import datetime, timezone

from carbon_accounting import JobPowerProfile, account_emissions
from carbon_intensity import TimeSeriesCarbonIntensityProvider

provider = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/electricity_maps_it_no_15min.json"
)

start_time = datetime(2020, 5, 6, 7, 5, tzinfo=timezone.utc)
job = JobPowerProfile(
    duration_seconds=900,
    average_power_watts=1_000,
)

result = account_emissions(job, start_time, provider.get_actual)
```

The PM100 trace describes Marconi100 at CINECA. Its Electricity Maps bidding zone is North Italy, `IT-NO`.

The downloader uses the Electricity Maps v4 `past-range` endpoint with:

- 15-minute granularity;
- flow tracing;
- lifecycle emission factors.

Lifecycle factors apply to supplied electricity only. Embodied carbon from HPC hardware and the datacenter is outside the model.

```bash
.venv/bin/python scripts/download_carbon_intensity.py \
  --start 2020-04-30T00:00:00Z \
  --end 2020-11-02T00:00:00Z
```

The token is loaded from `.env` and is never written to the cache. `--end` is exclusive. Requests are split into adjacent end-exclusive intervals of at most two days and merged chronologically.

The range above produces 93 requests and 17,856 buckets when complete.

```bash
.venv/bin/python tests/check_carbon_intensity_client.py
```

References: [CINECA Bologna](https://www.hpc.cineca.it/about-us/contacts/cineca-bologna/), [Electricity Maps coverage](https://app.electricitymaps.com/coverage), [Electricity Maps API](https://app.electricitymaps.com/developer-hub/api/reference).

## Historical actual dataset

`history.py` extends the 2020 cache with 2016–2019 and January–April 2020. CLI handling remains in:

- `scripts/download_carbon_intensity.py`
- `scripts/prepare_carbon_history.py`

No additional dependencies are required.

```bash
.venv/bin/python scripts/prepare_carbon_history.py
```

Completed two-day requests are cached under:

```text
data/carbon_intensity/actual/chunks/
```

Reruns revalidate and reuse them. The existing April–October 2020 cache is reused without changing its values.

The command produces:

- `actual.json`: chronological observations, source paths and SHA-256 hashes, request metadata, retrieval timestamps, and yearly quality counts;
- `protocol.json`: UTC split boundaries, sample counts, observation-availability convention, workload path, and actual-data hash.

Merging requires consistent:

- `IT-NO`;
- `gCO2eq/kWh`;
- 15-minute UTC grid;
- API v4;
- lifecycle emission factors;
- flow tracing.

Flow tracing represents consumption including electricity exchanges. Matching settings do not establish an identical historical methodology version because the source caches do not provide one. Original provenance and retrieval dates are retained.

Missing buckets and conflicting duplicates raise errors. Identical overlaps are deduplicated and counted. API duplicates are counted for new downloads; legacy caches without a duplicate counter are marked accordingly.

Estimated and unlabelled observations are retained and reported, including estimation methods. Forecast caches are rejected as actual inputs.

The archive contains **169,536 buckets** from `2016-01-01T00:00Z` to `2020-11-01T00:00Z` exclusive:

| Year | Buckets |
| ---- | ------: |
| 2016 |  35,136 |
| 2017 |  35,040 |
| 2018 |  35,040 |
| 2019 |  35,040 |
| 2020 |  29,280 |

There are no gaps or overlaps. Four buckets on `2019-01-24`, from `09:00` through `09:45` UTC, are estimated with `TIME_SLICER_AVERAGE`; every other observation is marked not estimated. New downloads contain no API duplicates. The legacy 2020 cache has no API duplicate counter.

Offline reproduction:

```bash
.venv/bin/python scripts/prepare_carbon_history.py --offline
```

## Temporal forecasting protocol

`TemporalProtocol.from_workload(...)` reads only release and completion times from the cleaned PM100 workload.

| Partition  | Inclusive start  | Exclusive end    |
| ---------- | ---------------- | ---------------- |
| Train      | 2016-01-01 00:00 | 2020-01-01 00:00 |
| Validation | 2020-01-01 00:00 | 2020-05-06 07:00 |
| Test       | 2020-05-06 07:00 | 2020-10-13 05:00 |

The test interval contains the first release (`07:04:53`) and last completion (`04:47:04` on October 13). The first release is rounded down so the partially future `07:00` bucket is excluded from validation.

Actual observations after the test interval remain available for ex-post accounting of delayed schedules but do not enter model selection. No observations from 2021 or later are used.

For an `issue_time`, `history(...)` exposes only observations satisfying:

```text
bucket end + observation_delay <= issue_time
```

The default publication delay is zero. `--observation-delay-minutes` records any additional delay in the protocol. With zero delay, the `10:00–10:15` bucket is unavailable at `10:07` and becomes available at `10:15`.

`example(...)` separates past observations into `features` and future labels into `targets`. The entire target horizon must remain inside the requested partition. Training targets cannot enter validation, and validation targets cannot enter test. With a publication delay, training and validation labels must also become observable before their partition ends.

Validation is used for feature design, model selection, hyperparameters, history length, and update strategy.

```python
from datetime import datetime, timedelta, timezone

from carbon_intensity import TemporalProtocol, TimeSeriesCarbonIntensityProvider

actual = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/actual/actual.json"
)
protocol = TemporalProtocol.from_workload(
    "data/processed/pm100_clean.parquet"
)

example = protocol.example(
    actual,
    datetime(2019, 6, 1, tzinfo=timezone.utc),
    lookback=timedelta(days=7),
    horizon=timedelta(hours=24),
    partition="train",
)

protocol.save_training_metadata(
    "data/carbon_intensity/models/example_metadata.json",
    "model-name",
    [example],
)
```

`save_training_metadata(...)` revalidates all examples and rejects future feature observations or validation/test labels. It records:

- `training_cutoff`: latest timestamp used by features or targets;
- `training_available_at`: time at which the final training bucket becomes observable.

This is the common model-input and metadata contract.

The availability rule is a replay assumption. The archive contains historical actuals retrieved later and may include revisions. Exact historical replay would require archived publication vintages. See the [Electricity Maps methodology](https://www.electricitymaps.com/data/methodology).

```bash
.venv/bin/python tests/check_carbon_intensity_history.py
```

## Forecasting baselines

`BaselineCarbonIntensityProvider` implements three unfitted forecasts through:

```python
get_forecast(issue_time, horizon) -> CarbonIntensityForecast
```

Methods:

- `persistence`: repeat the latest available observation;
- `seasonal_daily`: use the same UTC time one day earlier;
- `seasonal_weekly`: use the same UTC time one week earlier.

All history access goes through `TemporalProtocol.history(...)`, including publication delay.

Seasonal forecasts repeat the last observable day or week when the horizon exceeds one cycle. If the corresponding delayed observation is unavailable, an earlier cycle is used. A complete source cycle is required; missing history raises an error. No filling or interpolation is performed. Calendar alignment is UTC.

```python
from datetime import datetime, timedelta, timezone

from carbon_intensity import (
    BaselineCarbonIntensityProvider,
    TemporalProtocol,
    TimeSeriesCarbonIntensityProvider,
)

actual = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/actual/actual.json"
)
protocol = TemporalProtocol.load(
    "data/carbon_intensity/actual/protocol.json"
)

provider = BaselineCarbonIntensityProvider(
    actual, protocol, "seasonal_daily"
)

forecast = provider.get_forecast(
    datetime(2020, 1, 1, tzinfo=timezone.utc),
    timedelta(hours=24),
)

assert len(forecast.samples) == 96
```

Forecasts contain a complete 15-minute trajectory and their issue timestamp. For unaligned issue times, the unfinished bucket containing the issue time is included.

Baselines require no fitting, so `training_cutoff` is empty.

Evaluation:

```bash
.venv/bin/python -m carbon_intensity.evaluate
```

The command restores the saved protocol, verifies the actual cache against its SHA-256, and evaluates 24-hour forecasts hourly on validation. `--horizon-hours` and `--cadence-minutes` are configurable.

Targets must remain inside the selected partition. Validation labels must become available before test begins. Test evaluation requires `--partition test`.

`evaluate_forecast(...)` accepts any implementation of the forecast API and verifies issue times and target grids before scoring.

MAE and RMSE weight every issue/target pair equally, so the same bucket is scored once for each overlapping forecast that predicts it.

Outputs under `data/carbon_intensity/forecasts/`:

- `validation_metrics.csv`
- `validation_horizon_metrics.csv`
- `validation_monthly_metrics.csv`
- `validation_metadata.json`

Test runs use `test_...` filenames.

```bash
.venv/bin/python tests/check_carbon_intensity_baselines.py
```

## Ridge forecast model

`forecast_features` builds 25 issue-time features from one week of completed observations and the UTC calendar:

- lags at 15 min, 30 min, 1 h, 3 h, 6 h, 12 h, 24 h, 48 h, and 7 d;
- rolling mean/min/max over 1 h, 6 h, and 24 h;
- sine/cosine pairs for hour, weekday, and month;
- weekend indicator.

Lag offsets are measured from:

```text
floor15(issue_time - observation_delay)
```

so publication delay shifts all lags consistently.

`target_features` adds two target-specific columns: the observations one day and one week before each target bucket. They use the same validated history and seasonal wrapping as the baselines.

These two columns reduce validation MAE from `19.6660` to `17.5232` (**10.9%**) and bias from `4.19` to `1.38`.

Both feature functions use `TemporalProtocol.validate_features(...)` and require a complete gap-free week ending exactly at the observable boundary.

`RidgeCarbonIntensityForecaster` fits one ridge equation for each 15-minute target position: 96 equations for a 24-hour horizon. Each uses 25 shared and 2 target-specific features. Predictions are direct multi-horizon estimates; outputs are never fed back as inputs.

Features are standardized using training rows only. Stored coefficients are converted back to original units. Negative predictions are clipped to zero.

```python
from carbon_intensity.forecasting import RidgeCarbonIntensityForecaster

model = RidgeCarbonIntensityForecaster.load(
    "data/carbon_intensity/forecasts/ridge_direct.json",
    actual,
    protocol,
)

forecast = model.get_forecast(
    datetime(2020, 6, 1, tzinfo=timezone.utc),
    timedelta(hours=24),
)
```

Hourly training origins run from the first point with a full week of history to the last 24-hour trajectory entirely inside train:

- training examples: **34,873**
- `training_cutoff`: `2019-12-31T23:45:00+00:00`

`load(...)` rejects incompatible protocols, feature layouts, or training timestamps. `get_forecast(...)` rejects issue times earlier than `training_available_at`.

### Validation

3,008 trajectories and 288,768 predicted buckets per model, in gCO2e/kWh:

| Model             |         MAE |        RMSE |   Bias |
| ----------------- | ----------: | ----------: | -----: |
| Persistence       |     27.3187 |     36.4016 | 0.2434 |
| Daily seasonal    |     23.6003 |     30.7095 | 0.2203 |
| Weekly seasonal   |     27.8895 |     35.0441 | 1.5142 |
| **Ridge, direct** | **17.5232** | **23.3359** | 1.3789 |

Against the best baseline, ridge reduces MAE by **25.8%** and RMSE by **24.0%**.

### Error by horizon

`validation_horizon_metrics.csv` contains one row per model and 15-minute lead.

| Lead  | Persistence |   Daily |  Weekly |       Ridge |
| ----- | ----------: | ------: | ------: | ----------: |
| +1 h  |      9.2566 | 23.7021 | 28.0688 |  **8.3902** |
| +3 h  |     19.5120 | 23.7168 | 28.0489 | **14.1257** |
| +6 h  |     26.7394 | 23.7108 | 27.9970 | **16.8051** |
| +12 h |     33.0918 | 23.6054 | 27.8617 | **18.6396** |
| +24 h |     23.4688 | 23.4688 | 27.7763 | **19.3228** |

Persistence degrades with lead before meeting the daily seasonal forecast at +24 h. Seasonal baselines are nearly flat. Ridge is best at every reported horizon.

```bash
.venv/bin/python tests/check_carbon_intensity_forecasting.py
```

## Model and history selection

`selection.py` selects training-history length, update strategy, and ridge penalty using validation only.

All candidates use the same features and direct multi-horizon design. Model matrices are built once per partition through `TemporalProtocol.example(...)`; candidates differ only by row mask and regularization.

Two families are evaluated:

- **frozen**: trained once using labels ending before validation;
- **30-day refit**: retrained during validation using only rows whose full 24-hour label window is observable by the refit time.

Validation results:

| Candidate                | Training rows |         MAE |        RMSE |    Bias |
| ------------------------ | ------------: | ----------: | ----------: | ------: |
| Full history, α=1        |        34,873 |     17.5300 |     23.3488 |  1.3881 |
| Full history, α=0.1      |        34,873 |     17.5235 |     23.3370 |  1.3816 |
| **Full history, α=0.01** |    **34,873** | **17.5232** | **23.3359** |  1.3789 |
| Recent 36 months         |        26,257 |     17.5264 |     23.3360 |  0.4962 |
| Recent 24 months         |        17,497 |     17.7679 |     23.6344 |  0.7896 |
| Recent 12 months         |         8,737 |     17.8829 |     23.7674 |  0.5396 |
| Recent 6 months          |         4,345 |     19.3022 |     25.4849 | -8.8472 |
| Full history, refit 30 d | 34,873→37,730 |     17.5259 |     23.3296 |  1.4123 |
| Recent 24 months, refit  | 17,497→17,474 |     17.7748 |     23.6701 |  0.6886 |
| Recent 12 months, refit  |   8,737→8,714 |     17.8471 |     23.6413 |  0.2950 |
| Recent 6 months, refit   |   4,345→4,322 |     18.9295 |     24.6923 | -1.2036 |

History shorter than 24 months degrades accuracy. Thirty-six months is effectively identical to the full four years, so the full history is retained.

Six months produces a strong negative spring bias because training covers only the second half of 2019. Refitting reduces the bias but not the error.

The predefined selection rule keeps the simpler full-history frozen model unless another candidate improves validation MAE by at least 1%. None does.

Exploratory nonlinear models on the same features and origins also underperform:

- `HistGradientBoostingRegressor` on levels: `18.4724` MAE;
- correction on top of ridge: `18.1494` MAE.

These exploratory candidates are not part of committed `CANDIDATES`.

```bash
.venv/bin/python -m carbon_intensity.selection
```

Outputs:

- `selection_metrics.csv`
- `selection_horizon_metrics.csv`
- `selection_metadata.json`
- `selected_model.json`

`selected_model.json` is authoritative for subsequent evaluation.

## Historical forecast snapshots

Electricity Maps does not provide an archive of historical 2020 forecasts, so `snapshots.py` reconstructs forecasts from the selected models using only information available at each issue time.

```bash
.venv/bin/python -m carbon_intensity.snapshots
```

Six archives are generated under `data/carbon_intensity/snapshots/`:

- persistence;
- daily seasonal;
- weekly seasonal;
- selected frozen ridge;
- walk-forward ridge;
- ridge refitted once at replay start.

Each contains **3,838 snapshots**, issued hourly from `2020-05-06T07:00Z` through `2020-10-13T04:00Z`.

Archive metadata includes model/version information, training timestamps, alpha, feature layout, history span, library versions, granularity, horizon, cadence, issue bounds, snapshot count, temporal protocol, and SHA-256 hashes of the actual cache and model file.

Each snapshot stores:

- `issue_time`;
- predicted interval;
- 96 forecast values.

Baselines have no training cutoff. Walk-forward archives also store every refit with its time, training cutoff, availability time, and row count.

Snapshots may predict beyond the evaluation partition because generation requires no future labels.

`verify()` checks:

- model availability before each issue;
- complete 15-minute trajectories;
- declared cadence without gaps;
- per-snapshot training cutoffs for walk-forward models.

Generation is deterministic and contains no wall-clock timestamp.

`ForecastArchive.get_forecast(issue_time)` returns the latest snapshot with:

```text
snapshot.issue_time <= issue_time
```

For example:

```python
from carbon_intensity.snapshots import ForecastArchive

archive = ForecastArchive.load(
    "data/carbon_intensity/snapshots/test_boosted_ridge.json"
)

forecast = archive.get_forecast(
    datetime(2020, 6, 1, 7, 4, tzinfo=timezone.utc)
)

assert forecast.issue_time == datetime(
    2020, 6, 1, 7, 0, tzinfo=timezone.utc
)
```

```bash
.venv/bin/python tests/check_carbon_intensity_snapshots.py
```

## Model updating during replay

`WalkForwardForecaster` tests whether the ridge should be refitted during the PM100 period.

A training row becomes eligible only once its complete 24-hour label window is observable:

```text
training_cutoff <= refit_at <= issue_time
```

`design`, `observable_rows`, `predict`, and `trajectory` are shared with the frozen ridge and selection code.

Update frequency is selected on validation using the model configuration from `selected_model.json`.

| Candidate  | Fits | Training rows |         MAE |        RMSE |    Bias |   Δ MAE |
| ---------- | ---: | ------------: | ----------: | ----------: | ------: | ------: |
| **Frozen** |    1 |    **34,873** | **17.5232** | **23.3359** | +1.3789 |       — |
| Refit 7 d  |   18 | 34,873→37,706 |     17.5080 |     23.3125 | +1.3663 | -0.087% |
| Refit 14 d |    9 | 34,873→37,538 |     17.5079 |     23.3124 | +1.3518 | -0.087% |
| Refit 30 d |    5 | 34,873→37,730 |     17.5257 |     23.3286 | +1.4100 | +0.014% |
| Refit 90 d |    2 | 34,873→37,010 |     17.5153 |     23.3233 | +1.3036 | -0.045% |

The best update frequency improves MAE by only `0.087%`, below the predefined 1% threshold. The selected model therefore remains frozen.

`selected_update.json` records 14 days as `comparison_refit_days` so the strongest validation update strategy is still carried into the PM100 comparison.

`ridge_refit_once` fits the same configuration once at replay start using the observations available by then.

```bash
.venv/bin/python -m carbon_intensity.walkforward
.venv/bin/python tests/check_carbon_intensity_walkforward.py
```

Outputs:

- `walkforward_metrics.csv`
- `walkforward_horizon_metrics.csv`
- `walkforward_metadata.json`
- `selected_update.json`

## Final forecast evaluation

Held-out evaluation uses the saved snapshot archives:

```bash
.venv/bin/python -m carbon_intensity.evaluate \
  --partition test \
  --snapshots data/carbon_intensity/snapshots
```

The evaluator also groups error by UTC target hour.

Results over **3,815 trajectories and 366,240 predicted buckets per model**, from `2020-05-06T07:00Z` to `2020-10-12T05:00Z`:

| Model                     |         MAE |        RMSE |    Bias |
| ------------------------- | ----------: | ----------: | ------: |
| Persistence               |     28.3124 |     36.5234 | -0.1573 |
| Daily seasonal            |     26.3997 |     34.9265 | -0.1455 |
| Weekly seasonal           |     30.3944 |     40.4541 | -0.4959 |
| **Ridge, direct, frozen** | **18.2084** | **23.9883** | +2.3422 |
| Ridge, refit every 14 d   |     18.2099 |     24.0033 | +2.1515 |
| Ridge, refitted in May    |     18.2188 |     24.0116 | +2.2343 |

The selected ridge moves from `17.5232` validation MAE to `18.2084` on test, a **3.9% degradation**, while remaining **31.0% below the best baseline in MAE** and **31.3% below it in RMSE**.

Updating does not improve held-out error.

### Test error by horizon

| Lead  | Persistence |   Daily |  Weekly |       Ridge |
| ----- | ----------: | ------: | ------: | ----------: |
| +1 h  |      8.9602 | 26.2524 | 30.2710 |  **8.1001** |
| +3 h  |     18.2427 | 26.2907 | 30.2836 | **13.5970** |
| +6 h  |     26.3953 | 26.3240 | 30.3211 | **16.4851** |
| +12 h |     34.8176 | 26.3709 | 30.3903 | **19.2301** |
| +24 h |     26.5056 | 26.5056 | 30.4907 | **21.0868** |

### Seasonal error

Frozen-ridge MAE remains between `15.4` and `18.1` from May through August, rises to `20.1` in September, and reaches `26.0` during the first twelve days of October.

Bias rises from `+0.40` in May to `+13.51` in October. A 14-day refit still scores `26.19` MAE and `+13.61` bias in October, indicating that the autumn error is not caused by stale weights.

Additional predictors such as weather or generation mix would be needed to represent that regime.

By UTC target hour, ridge MAE ranges from:

- `14.33` at 18:00;
- `14.54` at 17:00;
- `20.99` at 01:00;
- `21.15` at 00:00.

The highest mean intensity occurs around 21:00 UTC, so the largest errors correspond more closely to changes in the daily profile than to its absolute level.

### Figures

```bash
.venv/bin/python -m carbon_intensity.figures
```

Generated under `data/carbon_intensity/figures/`:

- `test_actual_vs_forecast.png`
- `test_error_by_horizon.png`
- `test_error_distribution.png`
- `test_error_by_hour.png`
- `test_frozen_vs_walkforward.png`

Forecast artifacts under `data/carbon_intensity/forecasts/`:

- `test_metrics.csv`
- `test_horizon_metrics.csv`
- `test_monthly_metrics.csv`
- `test_hour_metrics.csv`
- `test_metadata.json`

`matplotlib` is required only for figure generation.

## Boosted residual model

`boosted.py` extends the direct ridge with a gradient-boosted residual correction.

The ridge feature set is first widened with:

- rolling statistics up to 30 days;
- EWMAs;
- differences against lags and rolling levels;
- target-aligned daily and weekly levels;
- target-aligned daily and weekly shape changes.

The wider linear model reduces validation MAE from `17.5232` to `16.6799`.

The residual booster uses two constraints.

### Deviations instead of absolute levels

All level-carrying features are expressed relative to the latest observation before reaching the trees.

IT-NO annual mean intensity falls from about `392 gCO2eq/kWh` in 2016 to `289` in 2020. A tree trained on absolute levels cannot extrapolate this trend. Using absolute intensities gives `17.78` validation MAE, worse than the ridge.

### Separate lead bands

One booster is trained for each lead range:

- 0–1 h;
- 1–6 h;
- 6–24 h.

Residual scale increases with lead time. A single pooled booster degrades +1 h MAE from `7.48` to `8.54`; the banded version improves it to `7.33`.

The booster optimizes absolute error.

```bash
.venv/bin/python -m carbon_intensity.boosted \
  --partition validation \
  --baselines

.venv/bin/python -m carbon_intensity.boosted \
  --partition test
```

A 30-day refit improves validation MAE by `0.59%`, below the 1% selection threshold, so the boosted model remains frozen.

In the search harness, test results are:

- frozen: `16.9591` MAE, `+2.1084` bias;
- 30-day refit: `16.9778` MAE, `+2.7027` bias.

An online bias correction based only on already-published forecast errors was also rejected: validation MAE worsens by about `0.5`, although on test it would reduce bias from `+2.1084` to `+0.4028`.

Held-out comparison:

| Model                 |         MAE |        RMSE |    Bias |
| --------------------- | ----------: | ----------: | ------: |
| Ridge, direct, frozen |     18.2084 |     23.9883 | +2.3422 |
| Ridge, wider features |     17.1510 |     22.7845 | +2.0653 |
| **Boosted ridge**     | **17.0091** | **22.5920** | +2.2770 |

The boosted model reduces MAE by **6.6%** and RMSE by **5.8%** relative to the original ridge.

All 96 target buckets improve in both MAE and RMSE. MAE improvement ranges from `0.57` to `1.61 gCO2e/kWh`.

Selected leads:

| Lead    | Boosted | Original ridge |
| ------- | ------: | -------------: |
| +15 min |  7.2425 |         8.0871 |
| +3 h    | 12.6706 |        13.5970 |
| +6 h    | 15.4191 |        16.4851 |
| +24 h   | 19.9150 |        21.0868 |

The autumn overprediction remains.

## Issue cadence

With hourly issue times, the first four target buckets have nearly identical test MAE because all are already at least one hour beyond the latest closed observation.

Training and evaluating the same model on a 15-minute issue cadence separates them:

| Lead    | Hourly cadence | 15-minute cadence |
| ------- | -------------: | ----------------: |
| +15 min |         7.2400 |        **2.0390** |
| +30 min |         7.2480 |        **3.9630** |
| +45 min |         7.2440 |        **5.5940** |
| +1 h    |         7.2390 |            7.2680 |

Full-horizon MAE also improves:

| Partition  |  Hourly | 15-minute |
| ---------- | ------: | --------: |
| Validation | 16.1853 |   16.0383 |
| Test       | 16.9591 |   16.6933 |

A 15-minute cadence requires the model to be trained on that cadence and snapshots to be generated with:

```bash
--cadence-minutes 15
```

This produces roughly four times as many training rows and archive entries.

## Serving snapshots to a scheduler

`ArchiveCarbonIntensityProvider` combines an actual series with a forecast archive.

```python
from carbon_intensity import TimeSeriesCarbonIntensityProvider
from carbon_intensity.snapshots import (
    ArchiveCarbonIntensityProvider,
    ForecastArchive,
)

actual = TimeSeriesCarbonIntensityProvider.load(
    "data/carbon_intensity/actual/actual.json"
)

archive = ForecastArchive.load(
    "data/carbon_intensity/snapshots/test_boosted_ridge.json"
)

provider = ArchiveCarbonIntensityProvider(actual, archive)

forecast = provider.get_forecast(
    datetime(2020, 6, 1, 7, 4, tzinfo=timezone.utc),
    timedelta(hours=6),
)

assert forecast.issue_time == datetime(
    2020, 6, 1, 7, 0, tzinfo=timezone.utc
)
```

The provider enforces three rules:

1. the latest snapshot with `issue_time <= as_of` is used;
2. requested coverage is measured from the decision time, not the snapshot issue time;
3. missing forecast coverage raises `ForecastUnavailableError`; actual observations are never substituted.

`TimeSeriesCarbonIntensityProvider.get_forecast(...)` continues to reject actual-only caches.

## Scheduling on issued forecasts

`CarbonAwareScheduler(..., forecast=True)` uses forecast snapshots instead of future actuals.

The scheduling policy remains unchanged:

- same candidate-start grid;
- same constant-average-power cost;
- same earliest-start tie break.

At job release, the provider is queried for the full decision interval from the release bucket through the completion of the latest candidate. `CarbonSignal.from_samples(...)` converts the forecast trajectory into the same cumulative signal used by the oracle.

```python
from hpc_sim import CarbonAwareScheduler, Simulator, account_schedule

scheduler = CarbonAwareScheduler(
    provider,
    forecast=True,
    max_delay=timedelta(hours=12),
)

result = account_schedule(
    Simulator(jobs, cluster, scheduler).run(),
    jobs,
    actual,
)

scheduler.forecast_issue_times[job_id]
```

Actual carbon intensity is used only for ex-post accounting after simulation.

`forecast_issue_times` records the snapshot used for each scheduling decision.

For `max_delay=0`, there is only one admissible start and no carbon signal is queried, reproducing plain EASY.

Forecast horizon limits the feasible delay budget. A 24-hour forecast cannot cover a 20-hour delay plus a 10-hour job, and the scheduler raises rather than extrapolating.

Tests prevent forecast scheduling from reading actual future observations and verify that deliberately shifted forecasts change the resulting schedule.

```bash
.venv/bin/python -m unittest discover \
  -s tests \
  -p "check_hpc_sim_carbon_aware.py"
```

## Scheduling impact

`carbon_intensity.scheduling_impact` compares, on the same jobs, cluster, and delay budget:

- carbon-blind EASY;
- a carbon-aware oracle using actual observations;
- each archived forecast.

Durations and powers are measured PM100 values in every run. Emissions are always accounted ex-post using actual observations.

The main metric is oracle recovery:

$$
\text{Oracle recovery} =
\frac{C_{\text{easy}} - C_{\text{forecast}}}
     {C_{\text{easy}} - C_{\text{oracle}}}
$$

It measures the fraction of the oracle's available carbon saving retained when scheduling uses forecasts.

A value of:

- `1` matches the oracle;
- `0` recovers none of its saving;
- `< 0` performs worse than EASY.

An hourly 24-hour archive guarantees only 23 hours of future trajectory for a decision occurring anywhere within the issue interval.

The `reach` parameter therefore limits each job's delay budget to:

```text
reach - duration
```

Jobs close to 24 hours may receive no carbon-aware delay budget. The same reach limit is applied to the oracle so that oracle and forecast runs differ only in the signal available to the scheduler.

```bash
.venv/bin/python -m carbon_intensity.scheduling_impact
.venv/bin/python tests/check_carbon_intensity_scheduling_impact.py
```

The command writes:

```text
data/carbon_intensity/forecasts/test_scheduling_impact.csv
```

Each row contains the full `ScheduleMetrics` result together with carbon saving, oracle recovery, and distributions of forecast and realized starts relative to the oracle.

Forecast runs also assert that no job decision uses a snapshot issued after the job release time.
