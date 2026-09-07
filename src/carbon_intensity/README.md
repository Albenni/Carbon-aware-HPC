# Carbon intensity providers

This package gives the accounting code and simulator one stable API,
independent of where carbon intensity data came from. The project's initial
actual series uses `gCO2eq/kWh` averages on a 15 minute UTC grid. A sample at
`10:00` is valid on `[10:00, 10:15)`, and ranges are also start inclusive and
end exclusive.

`TimeSeriesCarbonIntensityProvider` is deliberately strict: a missing bucket,
an out-of-range lookup, or a timezone naive timestamp raises an error. It never
interpolates or extrapolates. `get_forecast(...)` is kept separate and raises
`ForecastUnavailableError` for an actual only series, so historical future
values cannot accidentally be used as a forecast.

## Use

After a successful download, load the resulting cache and pass the provider
directly to carbon accounting:

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

The PM100 trace describes Marconi100 at CINECA. Its
configured Electricity Maps bidding zone is North Italy, `IT-NO`. The downloader
requests the v4 `past-range` endpoint with a selected 15 minute granularity,
flow tracing, and lifecycle emission factors, then stores both samples and
provenance in a local JSON cache. Lifecycle factors concern the supplied
electricity; embodied carbon of the HPC hardware and datacenter remains outside
this model.

```bash
.venv/bin/python scripts/download_carbon_intensity.py \
  --start 2020-04-30T00:00:00Z \
  --end 2020-11-02T00:00:00Z
```

The token is read from `.env` and is never written to
the cache. `--end` is exclusive, so `2020-11-02T00:00:00Z` includes every
15 minute bucket on November 1. Long downloads are split into adjacent,
end exclusive requests of at most two days and merged into one chronological
cache. The range above produces 93 API requests and 17,856 buckets when the
source series is complete. Run the deterministic offline check with:

```bash
.venv/bin/python tests/check_carbon_intensity.py
```

Source details: [CINECA Bologna location](https://www.hpc.cineca.it/about-us/contacts/cineca-bologna/),
[Electricity Maps coverage](https://app.electricitymaps.com/coverage), and
[Electricity Maps API reference](https://app.electricitymaps.com/developer-hub/api/reference).

## Historical actual dataset

`history.py` extends the existing 2020 cache with the four complete years from
2016 to 2019, plus January–April 2020 for validation. Command-line arguments, defaults and execution live in
`scripts/download_carbon_intensity.py` and `scripts/prepare_carbon_history.py`.
The API client, historical merge and shared token loading remain in this package.
No new dependencies are required.

From the repository root:

```bash
.venv/bin/python scripts/prepare_carbon_history.py
```

Each completed two-day request is saved in `data/carbon_intensity/actual/chunks/`.
Rerunning revalidates and reuses those files. The existing April–October 2020
cache is reused without modifying its values. A successful run produces:

- `data/carbon_intensity/actual/actual.json`: the chronological actual series,
  source cache paths and SHA-256 hashes, original request metadata and retrieval
  timestamps, and quality counts by year.
- `data/carbon_intensity/actual/protocol.json`: UTC split boundaries, sample
  counts, observation availability convention, workload path and actual hash.

The merge requires matching `IT-NO`, `gCO2eq/kWh`, a 15-minute UTC grid,
API v4, lifecycle emission factors and flow tracing. The latter represents
consumption including electricity exchanges, as specified by the
[Electricity Maps API](https://app.electricitymaps.com/docs/reference/carbon-intensity/past-range).
The client checks these settings when present in the response, including
per-sample granularity and flow tracing. Matching settings do not establish an
identical historical methodology version: the source caches do not supply one.
Their original provenance and retrieval dates remain available for review.

Missing buckets and conflicting duplicate observations raise errors. Identical
overlaps are deduplicated and counted; API duplicates are also counted in new
downloads. Legacy sources without an API duplicate counter are explicitly
reported as such. Estimated and unlabelled observations are retained and counted,
with their estimation methods, rather than silently discarded or interpolated.
Forecast caches are rejected as actual inputs and stay separate from this archive.

The downloaded archive contains **169,536 buckets** from `2016-01-01T00:00Z` to
`2020-11-01T00:00Z` exclusive, with no gaps, overlaps or unlabelled estimates:
35,136 in the 2016 leap year, 35,040 in each of 2017, 2018 and 2019, and 29,280
in 2020. Four buckets on `2019-01-24`, from `09:00` through `09:45` UTC, are
marked estimated with method `TIME_SLICER_AVERAGE`; every other observation is
marked not estimated, and the added years contain none. New downloads contain no
API duplicates; the legacy 2020 cache has no API duplicate counter.

To reproduce the artifacts without network access:

```bash
.venv/bin/python scripts/prepare_carbon_history.py --offline
```

## Temporal forecasting protocol

`TemporalProtocol.from_workload(...)` reads only release and completion columns
from the cleaned PM100 workload. All splits are half-open, on complete UTC buckets:

| Partition  | Inclusive start  | Exclusive end    |
| ---------- | ---------------- | ---------------- |
| Train      | 2016-01-01 00:00 | 2020-01-01 00:00 |
| Validation | 2020-01-01 00:00 | 2020-05-06 07:00 |
| Test       | 2020-05-06 07:00 | 2020-10-13 05:00 |

These test bounds surround the first release (`07:04:53`) and last observed
completion (`04:47:04` on October 13). Rounding the first release down keeps the
partially future `07:00` bucket out of validation. Actuals after the test end
remain available for accounting delayed schedules, outside the model-selection
partitions. No 2021-or-later observations enter this experiment.

For each `issue_time`, `history(...)` supplies only complete observations whose
`bucket end + observation_delay <= issue_time`. The default publication delay is
zero; `--observation-delay-minutes` makes an additional delay explicit and records
it in the protocol. For example, at `10:07` the `10:00–10:15` bucket is unavailable;
at `10:15` it becomes available with zero additional delay.

`example(...)` keeps those source observations in `features` and future labels in
`targets`. The **entire** target horizon must fit in the requested partition.
Thus a December training origin cannot borrow January validation targets, and a
validation origin cannot borrow a target from the PM100 test period. With a
publication delay, training/validation labels must also become observable before
their partition ends. Use validation exclusively for features, model choice,
hyperparameters, history length and any update-strategy comparison.

```python
from datetime import datetime, timedelta, timezone
from carbon_intensity import TemporalProtocol, TimeSeriesCarbonIntensityProvider

actual = TimeSeriesCarbonIntensityProvider.load("data/carbon_intensity/actual/actual.json")
protocol = TemporalProtocol.from_workload("data/processed/pm100_clean.parquet")
example = protocol.example(
    actual, datetime(2019, 6, 1, tzinfo=timezone.utc),
    lookback=timedelta(days=7), horizon=timedelta(hours=24), partition="train",
)
# After fitting a model, pass exactly the examples used by that fit:
protocol.save_training_metadata(
    "data/carbon_intensity/models/example_metadata.json", "model-name", [example],
)
```

The last call validates every training example again, rejecting future feature
observations and any validation/test labels, and records `training_cutoff` as
the maximum timestamp actually used across features and targets.
`training_available_at` additionally records when the final bucket is observable.
This is the common input/metadata contract for subsequent forecast models; no
model is trained by the dataset preparation command.

The availability rule is an explicit replay assumption. These are historical
actuals retrieved today, which can include later revisions; they do not prove
which revision was published at a historical decision time. Electricity Maps
describes its historical revision process in its
[methodology](https://www.electricitymaps.com/data/methodology). Replaying those
revisions exactly would require an archive of historical publication vintages.

Run the small offline dataset and leakage check with:

```bash
.venv/bin/python -m carbon_intensity.check_history
```

## Forecasting baselines

`BaselineCarbonIntensityProvider` adds three unfitted forecasts through the
existing `get_forecast(issue_time, horizon) -> CarbonIntensityForecast` API:

- `persistence`: repeat the most recent available observation.
- `seasonal_daily`: reuse the observation at the same UTC time the previous day.
- `seasonal_weekly`: reuse the observation at the same UTC time the previous week.

All methods obtain observations through `TemporalProtocol.history`, including
its publication delay. Seasonal methods repeat the last observable day/week if
the requested horizon extends beyond that cycle. If a delayed observation is
unavailable, the same UTC time from an earlier cycle is used. They require a
complete source cycle and raise on missing history, with no fallback or filling.
Calendar alignment is UTC, without local daylight-saving adjustments.

```python
from datetime import datetime, timedelta, timezone
from carbon_intensity import (
    BaselineCarbonIntensityProvider, TemporalProtocol,
    TimeSeriesCarbonIntensityProvider,
)

actual = TimeSeriesCarbonIntensityProvider.load("data/carbon_intensity/actual/actual.json")
protocol = TemporalProtocol.load("data/carbon_intensity/actual/protocol.json")
provider = BaselineCarbonIntensityProvider(actual, protocol, "seasonal_daily")
forecast = provider.get_forecast(
    datetime(2020, 1, 1, tzinfo=timezone.utc), timedelta(hours=24),
)
assert len(forecast.samples) == 96
```

The forecast contains a complete 15-minute trajectory and its issue timestamp.
For an unaligned issue time, it includes the unfinished bucket containing that
instant, matching the temporal protocol. Prediction reads only available
history; the provider's actual methods continue to expose observations separately.
These methods require no fitting, so the evaluation table leaves their
`training_cutoff` empty.

Run the offline comparison from the repository root:

```bash
.venv/bin/python -m carbon_intensity.evaluate
```

The command restores the saved splits and publication delay, verifies the actual
cache against the protocol's SHA-256, then emits a 24-hour forecast every hour
on validation. Both `--horizon-hours` and `--cadence-minutes` are configurable.
Every target must remain inside the selected partition; validation labels must
also become available before test begins. Nothing is fitted or selected on test.
`--partition test` explicitly enables held-out evaluation when required.

`evaluate_forecast` accepts any callable implementing the same forecast API,
so subsequent models can reuse the exact origins, labels and scoring. It checks
the issue timestamp and complete target grid before scoring. MAE and RMSE give
equal weight to every issue/target pair; a bucket in overlapping trajectories
is therefore scored once for each forecast that predicts it.

The run writes `validation_metrics.csv`, `validation_horizon_metrics.csv`,
`validation_monthly_metrics.csv` and `validation_metadata.json` under
`data/carbon_intensity/forecasts/`, including input hash, protocol, configuration
and scored origin bounds. Test runs use separate `test_...` filenames. These are
forecast-error results; connecting issued forecasts to scheduling remains
separate work.

Run the short deterministic check with:

```bash
.venv/bin/python -m carbon_intensity.check_baselines
```

## Feature pipeline and the ridge forecast model

`forecast_features` turns a week of completed observations plus the UTC calendar
into 25 deterministic numbers shared by the whole trajectory: nine lags (15 and
30 minutes, 1, 3, 6, 12, 24 and 48 hours, 7 days), rolling mean/min/max over the
last 1, 6 and 24 hours, sine and cosine pairs for hour, weekday and month, and a
weekend indicator. Lag offsets are counted from
`floor15(issue_time - observation_delay)`, the last boundary that is observable
at issue time, so a publication delay moves every lag back together.

`target_features` adds the two columns that cannot be shared: for each target
bucket, the observation one day and one week before *that* bucket. The shared
vector describes the issue time alone, so without them a direct equation knows
yesterday at the issue hour but never yesterday at its own target hour. Both
columns are read from the same validated week and wrap to the last observable
day or week exactly as the seasonal baselines do, so they introduce no new data
source and stay available at issue time. They are worth 10.9% of validation MAE,
which drops from 19.6660 to 17.5232, and cut the positive bias from 4.19 to 1.38.

Causality is not assumed. Both functions go through
`TemporalProtocol.validate_features` and additionally require a complete,
gap-free week ending exactly at that boundary, so a history containing the
bucket in progress is rejected rather than silently used.

`RidgeCarbonIntensityForecaster` fits one ridge equation per 15-minute target
bucket, 96 in total, over the 25 shared and 2 target-specific columns: a direct
multi-horizon strategy, where the +30-minute and the +24-hour predictions each
learn their own weights and no prediction is fed back as an input. Features are
standardised on the training rows only; the stored coefficients are converted
back to original units, so a single equation stays readable. Predictions are
floored at zero.

```python
from carbon_intensity.forecasting import RidgeCarbonIntensityForecaster

model = RidgeCarbonIntensityForecaster.load(
    "data/carbon_intensity/forecasts/ridge_direct.json", actual, protocol,
)
forecast = model.get_forecast(datetime(2020, 6, 1, tzinfo=timezone.utc), timedelta(hours=24))
```

Training origins run hourly from the first instant with a full week of history
to the last whose 24-hour target window still ends inside train: 34,873 examples
from `2016-01-01` onwards, `training_cutoff` `2019-12-31T23:45:00+00:00`. `load` refuses a model whose
protocol, feature layout or training timestamps disagree with the current split,
and `get_forecast` refuses an issue time earlier than `training_available_at`,
so a model can never be used before the data it was fitted on existed.

Validation results, **3,008 trajectories / 288,768 predicted buckets** per model,
in gCO2e/kWh:

| Model           |         MAE |        RMSE |   Bias |
| --------------- | ----------: | ----------: | -----: |
| Persistence     |     27.3187 |     36.4016 | 0.2434 |
| Daily seasonal  |     23.6003 |     30.7095 | 0.2203 |
| Weekly seasonal |     27.8895 |     35.0441 | 1.5142 |
| Ridge, direct   | **17.5232** | **23.3359** | 1.3789 |

The ridge lowers MAE by 25.8% and RMSE by 24.0% against the best baseline, at
the cost of overpredicting by about 1.4 gCO2e/kWh on average.

### Multi-horizon reporting

Every model answers the same `get_forecast(issue_time, horizon)` call and returns
a `CarbonIntensityForecast` on the same 15-minute grid, so baselines and fitted
models are scored by the same evaluator over the same origins; `evaluate_forecast`
raises if a trajectory does not cover the requested grid exactly. Error is
accumulated separately per lead time in `validation_horizon_metrics.csv`, one row
per model and per 15-minute lead up to 24 hours, measured from the issue time to
the end of the target bucket. The command prints the five reference horizons:

| Lead  | Persistence | Daily seasonal | Weekly seasonal | Ridge, direct |
| ----- | ----------: | -------------: | --------------: | ------------: |
| +1 h  |      9.2566 |        23.7021 |         28.0688 |    **8.3902** |
| +3 h  |     19.5120 |        23.7168 |         28.0489 |   **14.1257** |
| +6 h  |     26.7394 |        23.7108 |         27.9970 |   **16.8051** |
| +12 h |     33.0918 |        23.6054 |         27.8617 |   **18.6396** |
| +24 h |     23.4688 |        23.4688 |         27.7763 |   **19.3228** |

Persistence degrades with lead time and recovers at +24 h, where it coincides
with the daily seasonal value by construction; the seasonal baselines are flat
in lead time. The ridge is best at every reference horizon, and its advantage is
largest in the first hours, where recent lags carry the most information.

Run the short deterministic check with:

```bash
.venv/bin/python -m carbon_intensity.check_forecasting
```

## Model and history selection

`selection.py` settles the remaining choices — how much history to train on,
whether to refit during the year, and the ridge penalty — on validation alone,
before the PM100 period is read. Every candidate shares the feature pipeline and
the direct multi-horizon strategy, so only the visible training rows and the
penalty differ, which is what makes the numbers comparable.

The model matrices are built once per partition, on the forecaster's own hourly
origin grid and through `TemporalProtocol.example`; a candidate is then a row
mask over them. Two families are compared:

- **frozen**: trained once on the rows whose labels end before validation starts,
  the configuration the snapshot archive uses;
- **refitting every 30 days**: retrained during validation, so it may use the
  2020 observations that would already have been published at that moment. A row
  enters training only once its whole 24-hour label window is observable, and it
  serves only origins at or after that refit, so the walk-forward is leak-free by
  construction rather than by convention.

Validation results, 3,008 trajectories and 288,768 predicted buckets each, in
gCO2e/kWh:

| Candidate                 | Training rows |         MAE |        RMSE |    Bias |
| ------------------------- | ------------: | ----------: | ----------: | ------: |
| Full history, α=1         |        34,873 |     17.5300 |     23.3488 |  1.3881 |
| Full history, α=0.1       |        34,873 |     17.5235 |     23.3370 |  1.3816 |
| **Full history, α=0.01**  |    **34,873** | **17.5232** | **23.3359** |  1.3789 |
| Recent 36 months          |        26,257 |     17.5264 |     23.3360 |  0.4962 |
| Recent 24 months          |        17,497 |     17.7679 |     23.6344 |  0.7896 |
| Recent 12 months          |         8,737 |     17.8829 |     23.7674 |  0.5396 |
| Recent 6 months           |         4,345 |     19.3022 |     25.4849 | -8.8472 |
| Full history, refit 30 d  | 34,873→37,730 |     17.5259 |     23.3296 |  1.4123 |
| Recent 24 months, refit   | 17,497→17,474 |     17.7748 |     23.6701 |  0.6886 |
| Recent 12 months, refit   |  8,737→ 8,714 |     17.8471 |     23.6413 |  0.2950 |
| Recent 6 months, refit    |  4,345→ 4,322 |     18.9295 |     24.6923 | -1.2036 |

Longer history wins, and the gain saturates. Cutting to 24 months costs 1.4% of
MAE and to 12 months 2.0%, while 36 months is statistically indistinguishable
from the full four years (17.5264 against 17.5232) — the fourth year is kept
because dropping it buys nothing either. Six months is decisively too short:
trained on the second half of 2019 only, the model carries a −8.85 gCO2e/kWh
bias into a spring it has never seen, and periodic refitting repairs the bias
but not the error. Extra seasonal cycles therefore matter more than the risk of
stale observations, at least back to 2016. The longer history does cost about
0.6 gCO2e/kWh of positive bias against the shortest windows, which MAE and RMSE
both say is worth paying.

The selection rule is declared before the results: the simplest configuration —
full history, frozen — is kept unless a candidate lowers validation MAE by at
least 1%. No candidate does; refitting every 30 days actually scores marginally
worse than leaving the weights alone (17.5259 against 17.5232). That also
answers, on validation only, whether the model needs to keep learning during the
replay: measurably, it does not.

Non-linear alternatives were measured on the same features and origins and
rejected. A `HistGradientBoostingRegressor` per target bucket scored 18.4724 MAE
predicting levels, and 18.1494 as a correction on top of the linear predictions:
both worse than the ridge's 17.5232 on its own, and both carrying about
4 gCO2e/kWh of positive bias. Nothing approached the 10.9% that the two seasonal
target columns give for two extra coefficients, so the trees buy nothing the
features have not already captured. Those runs were exploratory and are not part
of the committed pipeline; `CANDIDATES` is the list the command reproduces.

```bash
.venv/bin/python -m carbon_intensity.selection
```

The command writes `selection_metrics.csv`, `selection_horizon_metrics.csv`,
`selection_metadata.json` and `selected_model.json` under
`data/carbon_intensity/forecasts/`. The last file is binding: `evaluate` fits the
penalty and history span recorded there instead of an ad-hoc default, so the
configuration chosen on validation is the one the test period sees. Delete the
saved model to refit it after a change.

## Historical forecast snapshots

Electricity Maps supplies observations, not an archive of the forecasts it
published in 2020, so what a scheduler could have known has to be reconstructed.
`snapshots.py` issues one forecast per hour across the PM100 period from every
model and stores each trajectory with the provenance needed to replay it. A
snapshot's `issue_time` is the instant from which a scheduler may use it, never
the instant its targets occur. Six archives are written, each with **3,838
snapshots** from `2020-05-06T07:00Z` to `2020-10-13T04:00Z`: the three baselines,
the ridge selected on validation, and the two update variants described in the
next section.

```bash
.venv/bin/python -m carbon_intensity.snapshots
```

Each archive is a JSON file under `data/carbon_intensity/snapshots/` holding
shared metadata and the list of snapshots. The metadata carries the model name
and version, `training_cutoff` and `training_available_at`, alpha, feature
layout, history span, library versions, the granularity, horizon and cadence,
the issue bounds and snapshot count, plus the temporal protocol and the SHA-256
of both the actual cache and the model file. Each snapshot records its
`issue_time`, the start and end of the predicted interval and the 96 values.
Baselines record no training cutoff, because they fit nothing. A walk-forward
archive additionally records its whole refit schedule in `refits`, one entry per
fit with its `refit_at`, `training_cutoff`, `training_available_at` and row count.

Unlike an evaluation origin, a snapshot needs no labels, so its trajectory may
reach past the end of the partition — exactly what a scheduler deciding on the
final day requires. Causality is enforced upstream: every model receives its
observations through `TemporalProtocol.history`, which only releases buckets
complete at the issue time. `verify()` rechecks the archive itself, requiring
that the model was already trained (`training_available_at <= issue_time`), that
each trajectory starts at its issue time and covers a complete 15-minute grid,
and that issue times follow the declared cadence without a gap. When a refit
schedule is present, each snapshot is additionally matched to the fit that
actually served it, so `training_cutoff <= issue_time` is checked per snapshot
rather than only for the first fit. `load` runs it again on read.

Generation is deterministic: nothing records a wall-clock time, so the same
actual cache, protocol and model file produce a byte-identical archive.

`ForecastArchive.get_forecast(issue_time)` returns the most recent snapshot
already issued at that instant, which is how a replayed decision at 07:04 uses
the 07:00 forecast. Passing a horizon truncates the trajectory, measured from
the snapshot's own issue time; on the cadence grid the two coincide, so an
archive answers the provider call exactly as the model that produced it did and
can be scored by `evaluate_forecast` directly.

```python
from carbon_intensity.snapshots import ForecastArchive

archive = ForecastArchive.load("data/carbon_intensity/snapshots/test_ridge_direct.json")
forecast = archive.get_forecast(datetime(2020, 6, 1, 7, 4, tzinfo=timezone.utc))
assert forecast.issue_time == datetime(2020, 6, 1, 7, 0, tzinfo=timezone.utc)
```

Run the short deterministic check with:

```bash
.venv/bin/python -m carbon_intensity.check_snapshots
```

## Model updating during the replay

`walkforward.py` answers whether the model should keep learning during the PM100
period. `WalkForwardForecaster` refits the selected configuration at a fixed
period: `observable_rows` admits a training row only once its whole 24-hour label
window has been published, so `training_cutoff <= refit_at <= issue_time` holds by
construction rather than by convention. Every fit in a schedule is computed before
the first forecast is issued, so a trajectory never waits on training and the
whole calendar is known up front. `period=None` is the frozen configuration
itself, fitted once at the start of the replay.

`design`, `observable_rows`, `predict` and `trajectory` now live in
`forecasting.py` and are shared by the frozen ridge, the selection sweep and the
walk-forward, so all three build the same matrices and evaluate the same equation
and differ only in which rows are visible when.

The comparison runs on validation alone, with alpha and history span read from
`selected_model.json`, so retraining frequency is the only variable. Four explicit
frequencies against the frozen model, 3,008 trajectories and 288,768 predicted
buckets each, in gCO2e/kWh:

| Candidate       | Fits | Training rows |         MAE |        RMSE |    Bias | Δ MAE   |
| --------------- | ---: | ------------: | ----------: | ----------: | ------: | ------- |
| **frozen**      |    1 |    **34,873** | **17.5232** | **23.3359** | +1.3789 | —       |
| Refit 7 days    |   18 | 34,873→37,706 |     17.5080 |     23.3125 | +1.3663 | −0.087% |
| Refit 14 days   |    9 | 34,873→37,538 |     17.5079 |     23.3124 | +1.3518 | −0.087% |
| Refit 30 days   |    5 | 34,873→37,730 |     17.5257 |     23.3286 | +1.4100 | +0.014% |
| Refit 90 days   |    2 | 34,873→37,010 |     17.5153 |     23.3233 | +1.3036 | −0.045% |

Updating buys nothing. The best frequency, 14 days, gains 0.087% of MAE, eleven
times less than the 1% margin declared before the results, and refitting monthly
is marginally worse than not refitting at all. Four years of history make the few
added weeks irrelevant: halfway through validation a refit adds about 7% more rows
to a base of nearly 35,000, and those rows carry no new regime. The model stays
frozen.

The update is not dismissed without a trial on the real period. The strongest
validation candidate, 14 days, is recorded in `selected_update.json` as
`comparison_refit_days` and is archived over the PM100 period too, so the final
frozen-against-walk-forward comparison faces the best arm rather than a weakened
one. `ridge_refit_once` is the third variant: the same configuration refitted once
at the start of the replay, which is what "trained on the period preceding the
test" permits and the selected model does not use, since it stops at the end of
2019.

```bash
.venv/bin/python -m carbon_intensity.walkforward
.venv/bin/python -m carbon_intensity.check_walkforward
```

The command writes `walkforward_metrics.csv`, `walkforward_horizon_metrics.csv`,
`walkforward_metadata.json` and `selected_update.json`. `snapshots.py` reads the
last file, so the archived schedule is the one the comparison selected.

## Final forecast evaluation

The held-out evaluation scores the saved archives rather than rerunning the
models, so a table and the figure beside it cannot disagree:

```bash
.venv/bin/python -m carbon_intensity.evaluate \
  --partition test --snapshots data/carbon_intensity/snapshots
```

`evaluate_forecast` also groups error by the UTC hour of the target bucket, which
is the hour a scheduler would be placing work into, alongside the existing lead
time and issue month breakdowns.

Results over the PM100 period, **3,815 trajectories and 366,240 predicted
buckets** per model, from `2020-05-06T07:00Z` to `2020-10-12T05:00Z`, in
gCO2e/kWh:

| Model                            |         MAE |        RMSE |    Bias |
| -------------------------------- | ----------: | ----------: | ------: |
| Persistence                      |     28.3124 |     36.5234 | −0.1573 |
| Daily seasonal                   |     26.3997 |     34.9265 | −0.1455 |
| Weekly seasonal                  |     30.3944 |     40.4541 | −0.4959 |
| **Ridge, direct, frozen (used)** | **18.2084** | **23.9883** | +2.3422 |
| Ridge, walk-forward every 14 d   |     18.2099 |     24.0033 | +2.1515 |
| Ridge, frozen, refitted in May   |     18.2188 |     24.0116 | +2.2343 |

The model chosen on validation holds up on a period it never saw: MAE moves from
17.5232 to 18.2084, a 3.9% degradation, and stays 31.0% below the best baseline
with 31.3% less RMSE. The three ridge variants are indistinguishable, and ordered
the wrong way for anyone expecting a gain from updating: refitting every 14 days
costs 0.008% of MAE and refitting once in May, taking in the four validation
months, costs 0.06%. The validation conclusion is confirmed on the replay itself.

| Lead  | Persistence | Daily seasonal | Weekly seasonal |     Ridge |
| ----- | ----------: | -------------: | --------------: | --------: |
| +1 h  |      8.9602 |        26.2524 |         30.2710 |  **8.1001** |
| +3 h  |     18.2427 |        26.2907 |         30.2836 | **13.5970** |
| +6 h  |     26.3953 |        26.3240 |         30.3211 | **16.4851** |
| +12 h |     34.8176 |        26.3709 |         30.3903 | **19.2301** |
| +24 h |     26.5056 |        26.5056 |         30.4907 | **21.0868** |

The monthly breakdown shows the model's one serious limit. The frozen ridge stays
between 15.4 and 18.1 MAE from May to August, rises to 20.1 in September and to
26.0 over the first twelve days of October, and its bias climbs from +0.40 in May
to +13.51 in October: it systematically overpredicts carbon intensity in autumn.
This is not stale weights, which is why updating does not repair it — refitted
every 14 days the same equation scores 26.19 MAE and +13.61 bias in October,
slightly worse. The autumn regime is not reconstructible from the immediately
preceding weeks either; closing that gap needs features the vector does not have,
such as weather or mix composition, not a shorter refit period.

By hour of day the ridge is most accurate in the late UTC afternoon, 14.33 at
18:00 and 14.54 at 17:00, and least accurate overnight, 21.15 at midnight and
20.99 at 01:00. The baselines have the opposite and far sharper profile, with
persistence swinging between 23.2 and 33.3. The ridge's worst hours are not those
of highest mean intensity, which fall around 21:00 UTC, but those where the curve
changes slope.

Five figures are written to `data/carbon_intensity/figures/` from the same
archives:

```bash
.venv/bin/python -m carbon_intensity.figures
```

`test_actual_vs_forecast.png` shows one issued trajectory and a week of
day-ahead predictions, `test_error_by_horizon.png` MAE against lead time,
`test_error_distribution.png` the signed error density and box plot,
`test_error_by_hour.png` accuracy across the day with mean intensity overlaid,
and `test_frozen_vs_walkforward.png` the monthly and running error of the three
ridge variants. Those three are drawn with different dashes because on test they
lie exactly on top of one another, which is the result the figures exist to show.

Artifacts are `test_metrics.csv`, `test_horizon_metrics.csv`,
`test_monthly_metrics.csv`, `test_hour_metrics.csv` and `test_metadata.json` under
`data/carbon_intensity/forecasts/`, plus the five PNGs. `matplotlib` is required
for the figures only; every other command in this package runs without it. These
are forecast-error results: the effect of those errors on the scheduler stays
separate, later work.

## Serving the archive to a scheduler

`ArchiveCarbonIntensityProvider` binds one actual series to one forecast archive
and is the seam between the two. It is the object a replayed simulation holds:
`get_actual` still reaches the observations, because the oracle and the ex-post
accounting both need them, while `get_forecast(as_of, horizon)` answers only
with what had already been published at `as_of`.

```python
from carbon_intensity import TimeSeriesCarbonIntensityProvider
from carbon_intensity.snapshots import ArchiveCarbonIntensityProvider, ForecastArchive

actual = TimeSeriesCarbonIntensityProvider.load("data/carbon_intensity/actual/actual.json")
archive = ForecastArchive.load("data/carbon_intensity/snapshots/test_ridge_direct.json")
provider = ArchiveCarbonIntensityProvider(actual, archive)

forecast = provider.get_forecast(datetime(2020, 6, 1, 7, 4, tzinfo=timezone.utc), timedelta(hours=6))
assert forecast.issue_time == datetime(2020, 6, 1, 7, 0, tzinfo=timezone.utc)
```

Three rules make the answer safe to schedule on. The snapshot is the most recent
one with `issue_time <= as_of`, so a decision at 07:04 reads the 07:00 forecast
and never the 08:00 one. Coverage is measured from the decision instant rather
than from the issue time: fifty minutes into an hourly issue there are fifty
fewer minutes of trajectory left, so a window that runs past the end of the
snapshot raises `ForecastUnavailableError` instead of being answered short. And
nothing falls back to the observation it is meant to be predicting — an archive
that cannot cover the window is an error, not an invitation to read the actual
series. `TimeSeriesCarbonIntensityProvider` keeps refusing `get_forecast`
outright, so an actual-only cache still cannot masquerade as one.

## Scheduling on issued forecasts

`CarbonAwareScheduler(..., forecast=True)` in `hpc_sim` is the realistic counterpart of the
carbon-aware oracle. The policy is unchanged — the same candidate grid, the same
constant-average-power cost, the same tie-break towards the earliest start — and
only the signal differs. When a job is released the provider is asked for the
forecast covering its whole decision window, from the release bucket to the
completion of the latest candidate, and `CarbonSignal.from_samples` turns that
trajectory into the same cumulative integral the oracle builds from
observations. Every decision therefore consumes exactly what a scheduler running
at the time could have read, and the difference between the two runs is
attributable to forecast error and to nothing else.

```python
from hpc_sim import CarbonAwareScheduler, Simulator, account_schedule

scheduler = CarbonAwareScheduler(provider, forecast=True, max_delay=timedelta(hours=12))
result = account_schedule(Simulator(jobs, cluster, scheduler).run(), jobs, actual)
scheduler.forecast_issue_times[job_id]  # the snapshot that decided this job
```

The actual series is touched only by that last call, after the simulation, which
is what makes the emissions ex-post rather than assumed. `forecast_issue_times`
records which snapshot decided each job, so any single decision can be replayed
against the trajectory it actually saw. A job with no delay budget has one
admissible start, so the signal is never consulted at all and `max_delay=0`
reproduces plain EASY whatever the forecast says.

The horizon bounds what an archive can serve: a 24-hour trajectory cannot score
a 20-hour budget on top of a 10-hour job, and that combination raises rather
than inventing the tail. The tests seal the observations behind a provider whose
`get_actual` fails, run a schedule whose forecast is deliberately wrong by two
hours, and check that the policy follows the forecast — so a decision that
reached for the actual series would fail rather than quietly score better.

```bash
.venv/bin/python -m unittest discover -s tests -p "check_carbon_aware.py"
```

## What forecast error costs the schedule

`carbon_intensity.scheduling_impact` puts the whole question on one table. It
runs the carbon-blind EASY reference, the carbon-aware oracle reading the
observations it will later be scored against, and one run per archive in
`data/carbon_intensity/snapshots/`, all on the same cohort, the same cluster and
the same delay budget. Job durations and powers are the real measured ones
everywhere, so the only thing that changes between the oracle and an archive is
the signal the policy reads. Emissions are always charged ex-post against the
observations, whatever the policy planned with.

The headline is oracle recovery,

$$
\text{Oracle recovery} =
\frac{C_{\text{easy}} - C_{\text{forecast}}}{C_{\text{easy}} - C_{\text{oracle}}}
$$

the share of the saving available under perfect information that survives when
the future has to be predicted from what was published at decision time. It is
1 for the oracle by construction, 0 for a forecast that never moves a job, and
negative for one whose errors push jobs into dirtier hours than EASY found.

A 24-hour archive issued hourly guarantees only 23 hours of trajectory to a
decision taken anywhere inside the issue interval, and scoring a start time
needs the signal all the way to the end of the job. The `reach` argument
therefore caps each job's budget at `reach - duration`: the PM100 jobs that run
close to 24 hours get no carbon budget at all, which is the honest answer rather
than an invented tail. The same `reach` caps the oracle
identically even though it could read further, so the two runs differ in their
signal and in nothing else.

```bash
.venv/bin/python -m carbon_intensity.scheduling_impact
.venv/bin/python -m carbon_intensity.check_scheduling_impact
```

The comparison writes `test_scheduling_impact.csv` to
`data/carbon_intensity/forecasts/`, one row per configuration, carrying the full
`ScheduleMetrics` row alongside the saving, the recovery, and the distribution of
how far each run's targets and realised starts sit from the oracle's. Every
forecast run also asserts that no decision read a snapshot issued after the job
was released, so a leak fails the run instead of improving its score.
