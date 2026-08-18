# Carbon intensity providers

This package gives the accounting code and future simulator one stable API,
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
