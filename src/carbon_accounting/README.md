# Job-level carbon accounting

This directory implements the first operational-carbon calculation used by the
HPC scheduling project. Given a job power description, a timezone-aware start
time, and a carbon-intensity value or signal, it calculates the job's energy in
kWh and emissions in gCO2.

## Formal model

For a job with constant average whole-job power `P_avg` in watts and duration
`d` in seconds:

```text
energy_kWh = P_avg_W × d_seconds / (1000 W/kW × 3600 s/h)
           = P_avg_W × d_seconds / 3,600,000

emissions_gCO2 = energy_kWh × carbon_intensity_gCO2_per_kWh
```

With time-varying intensity, the execution is divided into segments:

```text
emissions_gCO2 = Σ ([P_i_W × d_i_seconds / 3,600,000]
                    × CI(start + offset_i))
```

The constant-average model uses the same power in every segment. The measured
model uses the real PM100 sample for each segment. This makes the distinction
between energy-aware and carbon-aware execution explicit: shifting a job does
not change its energy, but it can change its emissions when the grid intensity
changes.

## Public API

```python
from datetime import datetime, timezone

from carbon_accounting import JobPowerProfile, carbon_emissions

job = JobPowerProfile(
    job_id=42,
    duration_seconds=3_600,
    average_power_watts=1_000,
)

grams_co2 = carbon_emissions(
    job,
    datetime(2026, 1, 1, tzinfo=timezone.utc),
    400.0,  # gCO2/kWh
)

assert grams_co2 == 400.0
```

`account_emissions(...)` returns an `AccountingResult` containing both energy
and emissions. `carbon_emissions(...)` is the requested compact function and
returns only grams of CO2. Select `PowerModel.MEASURED` to integrate a supplied
`power_profile_watts`; the default is `PowerModel.AVERAGE`.

Carbon intensity can be either:

- a non-negative scalar in gCO2/kWh, treated as constant for the execution; or
- a callable receiving each segment's timezone-aware timestamp and returning
  gCO2/kWh.

For a callable, intensity is sampled at the start of each segment and treated as
piecewise constant for that segment. The job's `sample_interval_seconds`
controls the segmentation and defaults to the PM100 cadence of 20 seconds.
