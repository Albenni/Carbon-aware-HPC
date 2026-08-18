from __future__ import annotations

from collections.abc import Iterable, Mapping

from carbon_accounting import PowerModel, account_emissions
from carbon_intensity import CarbonIntensityProvider

from .models import Job, JobRecord, SimulationResult


class CoverageError(ValueError):
    """Raised when a schedule runs outside the provider's available data."""


def _coverage_bounds(provider: CarbonIntensityProvider) -> tuple[object, object] | None:
    start = getattr(provider, "coverage_start", None)
    end = getattr(provider, "coverage_end", None)
    if start is None or end is None:
        return None  # a provider without declared coverage self-reports gaps
    return start, end


def check_coverage(
    result: SimulationResult,
    provider: CarbonIntensityProvider,
) -> None:
    """Fail early and legibly when the schedule leaves the cached series.

    A carbon-aware policy can push jobs past the end of the downloaded window.
    Without this check the run would die deep inside an accounting loop with a
    bare missing-bucket error that says nothing about the cause.
    """

    bounds = _coverage_bounds(provider)
    if bounds is None:
        return
    coverage_start, coverage_end = bounds

    schedule_start = result.schedule_start
    schedule_end = result.schedule_end
    if schedule_start < coverage_start or schedule_end > coverage_end:
        raise CoverageError(
            f"the schedule spans {schedule_start.isoformat()} to "
            f"{schedule_end.isoformat()}, outside the carbon-intensity coverage "
            f"{coverage_start.isoformat()} to {coverage_end.isoformat()}"
        )


def account_schedule(
    result: SimulationResult,
    jobs: Iterable[Job],
    provider: CarbonIntensityProvider,
) -> SimulationResult:
    """Attach energy and emissions to every record of a finished run.

    Two figures are produced per job. The measured-profile figure is the ground
    truth used for evaluation: it integrates the real PM100 samples against the
    carbon intensity actually seen at the simulated execution times. The
    average-power figure is what a scheduler working from a single mean power
    would have modelled. Carrying both makes the model error of the simple
    representation a byproduct of every run rather than a separate experiment.

    A job carrying no measured profile — a synthetic job, or a purely predicted
    one — has no ground truth to integrate, so both figures come from its
    average power and are therefore equal.

    Accounting happens here, after the event loop, so the engine stays
    carbon-agnostic and a forecast provider can be substituted without touching
    it.
    """

    check_coverage(result, provider)
    by_id: Mapping[object, Job] = {job.job_id: job for job in jobs}

    accounted = []
    for record in result.records:
        job = by_id.get(record.job_id)
        if job is None:
            raise KeyError(f"no job supplied for record {record.job_id}")

        average = account_emissions(
            job.power,
            record.start_time,
            provider.get_actual,
            power_model=PowerModel.AVERAGE,
        )
        measured = (
            account_emissions(
                job.power,
                record.start_time,
                provider.get_actual,
                power_model=PowerModel.MEASURED,
            )
            if job.power.power_profile_watts is not None
            else average
        )
        accounted.append(
            record.with_accounting(
                energy_kwh=measured.energy_kwh,
                emissions_gco2e=measured.emissions_gco2,
                energy_kwh_average_model=average.energy_kwh,
                emissions_gco2e_average_model=average.emissions_gco2,
            )
        )

    return result.replace_records(tuple(accounted))


def total_energy_kwh(result: SimulationResult) -> float:
    return sum(
        record.energy_kwh
        for record in result.records
        if record.energy_kwh is not None
    )


def total_emissions_gco2e(result: SimulationResult) -> float:
    return sum(
        record.emissions_gco2e
        for record in result.records
        if record.emissions_gco2e is not None
    )
